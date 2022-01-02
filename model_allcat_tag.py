import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from transformers.trainer_pt_utils import (
    get_parameter_names
)
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import hanlp
from collections import Counter

CTB_TAGS = ['VA', 'DEC', 'NN', 'VV', 'AS', 'AD', 'SB', 'PU', 'PN', 'DEG', 'NT', 'M', 'P', 'LC', 'JJ', 'CC', 'VE', 'BA', 'CS', 'DER', 'NR', 'DT', 'SP', 'MSP', 'VC', 'CD', 'NOI', 'DEV', 'ON', 'OD', 'ETC', 'IJ', 'LB']
CTB_TAG_DICT = {k:i for i,k in enumerate(CTB_TAGS)}

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)

def char_level_tag(tags_and_words):
    tags, words = tags_and_words
    return sum([[tags[i]] * len(words[i]) for i in range(len(tags))], [])

def ratio_split(counter, ratio):
    tags, counter_nums = list(zip(*counter.items()))
    bound = sum(counter_nums) * (1-ratio)
    pivot = 0
    while bound > 0 and pivot < len(counter):
        bound -= counter_nums[pivot]
        pivot += 1
    return tags[pivot-1:]

def char_level_filtering(inputs):
    sentence, postags, filter = inputs
    # print(inputs)
    sentence = ''.join(sentence)
    return ''.join([c for i, c in enumerate(sentence) if postags[i] in filter])

def unwrapped_preprocess_function(examples, tokenizer, context_name, choice_name, max_seq_length, data_args):
    # Examples is a dict with keys: translation, choices, answer, size is 1k?
    translation = [context for context in examples[context_name]]
    classic_poetry = [
        choices for choices in examples[choice_name]
    ]

    hanout = HanLP(examples[context_name])
    seg_sents = hanout['tok/fine']
    postags = hanout['pos/ctb']
    postags = map(char_level_tag, zip(postags, seg_sents))

    ans_hanout = HanLP(sum(examples[choice_name], []))
    ans_seg_sents = ans_hanout['tok/fine']
    ans_postags = ans_hanout['pos/ctb']
    ans_postags = map(char_level_tag, zip(ans_postags, ans_seg_sents))
    ans_postags_counters = list(map(Counter, ans_postags))
    ans_postags_counters = [ans_postags_counters[i]+ans_postags_counters[i+1]+ans_postags_counters[i+2]+ans_postags_counters[i+3] for i in range(0, len(ans_postags_counters), 4)]

    filter_tags = map(lambda x: ratio_split(x, data_args.tagging_ratio), ans_postags_counters)
    filtered_sen = map(char_level_filtering, zip(seg_sents, postags, filter_tags))
    sentences = [t+'[SEP]'+f+'[SEP]'+"[SEP]".join(c) for t,f,c in zip(translation, filtered_sen, classic_poetry)]
    print(sentences[0])

    # Tokenize
    tokenized_examples = tokenizer(
        sentences,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    results = {}
    results.update(tokenized_examples)
    if 'answer' in examples:
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

class MyModule(nn.Module):
    def __init__(self, model_args, config):
        super(MyModule, self).__init__()
        config.num_labels = 4
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config
        )
        self.args = model_args
        if self.args.softmax_temperature is not None:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None, labels = None):
        # print(type(self.model))
        if 'guwenbert' in self.args.model_name_or_path:
            token_type_ids[:] = 0
        output = self.model(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids, 
            labels = labels, 
        )
        if self.args.softmax_temperature is None or self.args.softmax_temperature == 1.:
            return output
        logits = output.logits / self.args.softmax_temperature
        return {"logits": logits, "loss": self.loss(logits, labels)}
    
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