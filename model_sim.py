import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union
from random import shuffle

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import pdb

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import check_min_version
from transformers import AutoTokenizer
from transformers.utils.dummy_pt_objects import AutoModelForMaskedLM


def unwrapped_preprocess_function(examples, tokenizer, context_name, choice_name, max_seq_length, data_args):
    # Examples is a dict with keys: translation, choices, answer, size is 1k?
    translation = [[context] * 4 for context in examples[context_name]]
    classic_poetry = [
        [c for c in choices] for choices in examples[choice_name]
    ]
    # Flatten out
    first_sentences = sum(translation, [])
    second_sentences = sum(classic_poetry, [])

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    results = {}
    results.update({k: [v[i: i + 4] for i in range(0, len(v), 4)]
                   for k, v in tokenized_examples.items()})
    results['labels'] = [answer for answer in examples['answer']]
    # print(results)
    # Un-flatten
    return results


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        # Keys of each item in features: attention_mask, input_ids, labels, token_type_ids
        label_name = "labels"
        # print(list(features[0].keys()))
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1)
                 for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


class SimModule(nn.Module):
    def __init__(self, model_args, config):
        super(SimModule, self).__init__()
        self.first_model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        self.second_model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        self.cosine_embedding_loss = nn.CosineEmbeddingLoss()

    def forward(self, first_input_ids, second_input_ids, first_attention_mask,
                second_attention_mask, target):
        """[summary]

        Args:
            first_input_ids ([type]): (batch_size, seq_len)
            second_input_ids ([type]): (batch_size, choice_num, seq_len)
            first_attention_mask ([type]): (batch_size, seq_len)
            second_attention_mask ([type]): (batch_size, choice_num, seq_len)
            labels ([type]): (batch_size, num_chioce), 1 for TRUE choice, -1 for FALSE choice
        """
        batch_size, num_chioce, seq_len = second_input_ids.shape
        first_output = self.first_model(
            input_ids=first_input_ids,
            attention_mask=first_attention_mask,
            output_hidden_states=True,
        )
        second_output = self.second_model(
            input_ids=second_input_ids,
            attention_mask=second_attention_mask,
            output_hidden_states=True,
        )
        # first_hidden_states: (batch_size, seq_len, hidden_size)
        # second_hidden_states: (batch_size, choice_num, seq_len, hidden_size)
        first_hidden_states, _ = first_output.hidden_states
        second_hidden_states, _ = second_output.hidden_states

        # reshape `first_hidden_states` to (batch_size, choice_num, seq_len, hidden_size)
        _, _, hidden_size = first_hidden_states.shape
        first_hidden_states = first_hidden_states.unsqueeze(1).expand(batch_size,
                                                                      num_chioce,
                                                                      seq_len,
                                                                      hidden_size)

        # reshape `first_hidden_states` and `second_hidden_states` to
        # (batch_size * choice_num, seq_len * hidden_size)
        first_hidden_states = first_hidden_states.reshape(
            batch_size * num_chioce, -1)
        second_hidden_states = second_hidden_states.reshape(
            batch_size * num_chioce, -1)

        # target (batch_size * chioce_num, )
        target = target.reshape(-1)
        loss = self.cosine_embedding_loss(
            input1=first_hidden_states, input2=second_hidden_states, target=target)
        return loss