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

import transformers
from transformers import (
    AutoModel,
    BertPreTrainedModel,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_pt_utils import (
    get_parameter_names
)
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import check_min_version

def unwrapped_preprocess_function(examples, tokenizer, context_name, choice_name, max_seq_length, data_args):
    # Examples is a dict with keys: translation, choices, answer, size is 1k?
    translation = [context for context in examples[context_name]]
    classic_poetry = [
        "[SEP]".join(choices) for choices in examples[choice_name]
    ]

    # Tokenize
    tokenized_examples = tokenizer(
        translation,
        classic_poetry,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    results = {}
    results.update(tokenized_examples)
    results['labels'] = [ answer for answer in examples['answer']]
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

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Add back labels
        # print('collate', batch['input_ids'].size())
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
    
MyTokenizer = lambda model_args, config: AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)

# MyModule = lambda model_args, config: AutoModelForMultipleChoice.from_pretrained(
#             model_args.model_name_or_path,
#             from_tf=bool(".ckpt" in model_args.model_name_or_path),
#             config=config,
#             cache_dir=model_args.cache_dir,
#             revision=model_args.model_revision,
#             use_auth_token=True if model_args.use_auth_token else None,
#         )
class MyModule(BertPreTrainedModel):
    def __init__(self, model_args, config):
        super(MyModule, self).__init__(config)
        self.model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        self.args = model_args
        self.loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, 4)

        # Initialize weights and apply final processing
        config.initializer_range = 0.02
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = 4

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss = self.loss(reshaped_logits, labels)
        if self.args.softmax_temperature is not None:
            reshaped_logits = reshaped_logits / self.args.softmax_temperature
        return {"loss": loss,
            "logits": reshaped_logits}
    
def MyOptimizer(model, args, multiplier=10):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    classifier_parameters = [name for name, _ in model.named_parameters() if "classifier" in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters and n not in classifier_parameters],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters and n not in classifier_parameters],
            "weight_decay": 0.0,
            "lr": args.learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters and n in classifier_parameters],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate * multiplier
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters and n in classifier_parameters],
            "weight_decay": 0.0,
            "lr": args.learning_rate * multiplier
        },
    ]
    optimizer_cls = torch.optim.AdamW
    optimizer_kwargs = {
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
    }
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)