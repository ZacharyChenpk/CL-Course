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
    AutoTokenizer,
    AutoModel,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import check_min_version


def unwrapped_preprocess_function(examples, tokenizer, context_name, choice_name, max_seq_length, data_args):
    # Examples is a dict with keys: translation, choices, answer, size is 1k?
    translation = [[context] for context in examples[context_name]]
    classic_poetry = [
        [c for c in choices] for choices in examples[choice_name]
    ]
    # Flatten out
    first_sentences = sum(translation, [])
    second_sentences = sum(classic_poetry, [])

    # Tokenize
    tokenized_first_examples = tokenizer(
        first_sentences,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    tokenized_second_examples = tokenizer(
        second_sentences,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    results = {}
    results.update(
        {'first_' + k: v for k, v in tokenized_first_examples.items()})
    results.update({'second_' + k: [v[i: i + 4] for i in range(0, len(v), 4)]
                   for k, v in tokenized_second_examples.items()})
    results['labels'] = [answer for answer in examples['answer']]
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
        # Keys of each item in features: first/second_attention_mask, first/second_input_ids, labels, first/second_token_type_ids
        label_name = "labels"
        labels = [feature.pop(label_name) for feature in features]
        first_features = [{k.split("first_", 1)[1]: v for k, v in feature.items(
        ) if k.startswith("first")} for feature in features]
        second_features = [{k.split("second_", 1)[1]: v for k, v in feature.items(
        ) if k.startswith("second")} for feature in features]

        batch_size = len(features)
        num_choices = len(features[0]["second_input_ids"])
        flattened_second_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in second_features
        ]
        flattened_second_features = sum(flattened_second_features, [])
        batch1 = self.tokenizer.pad(
            first_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch2 = self.tokenizer.pad(
            flattened_second_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch1 = {'first_' + k: v for k, v in batch1.items()}
        batch2 = {'second_' + k: v.view(batch_size, num_choices, -1)
                  for k, v in batch2.items()}
        batch = {}
        batch.update(batch1)
        batch.update(batch2)
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        # batch["targets"] = F.one_hot(
        #     batch["labels"], num_classes=num_choices).to(torch.float64) * 2 - 1
        return batch


class SimModule(nn.Module):
    def __init__(self, model_args, config):
        super(SimModule, self).__init__()
        self.first_model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        self.second_model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, first_input_ids,
                first_token_type_ids,
                first_attention_mask,
                second_input_ids,
                second_token_type_ids,
                second_attention_mask,
                labels):
        """[summary]

        Args:
            first_input_ids ([type]): (batch_size, seq_len1)
            first_token_type_ids ([type]): (batch_size, seq_len1)
            first_attention_mask ([type]): (batch_size, seq_len1)
            second_input_ids ([type]): (batch_size, choice_num, seq_len2)
            second_token_type_ids ([type]): (batch_size, choice_num, seq_len2)
            second_attention_mask ([type]): (batch_size, choice_num, seq_len2)
            labels ([type]): (batch_size, )
        """
        batch_size, num_chioce, _ = second_input_ids.shape
        first_output = self.first_model(
            input_ids=first_input_ids,
            token_type_ids=first_token_type_ids,
            attention_mask=first_attention_mask,
        )
        second_output = self.second_model(
            input_ids=second_input_ids.reshape(batch_size * num_chioce, -1),
            token_type_ids=second_token_type_ids.reshape(
                batch_size * num_chioce, -1),
            attention_mask=second_attention_mask.reshape(
                batch_size * num_chioce, -1),
        )
        # first_hidden_states: (batch_size, hidden_size)
        # second_hidden_states: (batch_size * choice_num, hidden_size)
        first_pooler_output = first_output.pooler_output
        second_pooler_output = second_output.pooler_output

        # reshape `first_hidden_states` to (batch_size, choice_num, hidden_size)
        _, hidden_size = first_pooler_output.shape
        first_pooler_output = first_pooler_output.unsqueeze(1).expand(batch_size,
                                                                      num_chioce,
                                                                      hidden_size)

        # reshape `second_hidden_states` to (batch_size, choice_num, hidden_size)
        second_pooler_output = second_pooler_output.reshape(
            batch_size, num_chioce, -1)

        # cos_sim : (batch_size, chioce_num)
        cos_sim = F.cosine_similarity(
            x1=first_pooler_output, x2=second_pooler_output, dim=2)
        cos_sim = (cos_sim + 1) / 2
        loss = self.loss_func(cos_sim, labels)
        # output: (batch_size, chioce_num)
        outputs = F.softmax(cos_sim, dim=1)
        # By default, all models return the loss in the first element.
        return loss, outputs
