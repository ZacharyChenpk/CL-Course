import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union
from random import shuffle
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import transformers
from transformers import (
    AutoModelForMultipleChoice,
    AutoModel,
    AutoTokenizer
)
from transformers.trainer_pt_utils import (
    get_parameter_names
)
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import hanlp
import ipdb

CTB_TAGS = ['VA', 'DEC', 'NN', 'VV', 'AS', 'AD', 'SB', 'PU', 'PN', 'DEG', 'NT', 'M', 'P', 'LC', 'JJ', 'CC', 'VE', 'BA', 'CS', 'DER', 'NR', 'DT', 'SP', 'MSP', 'VC', 'CD', 'NOI', 'DEV', 'ON', 'OD', 'ETC', 'IJ', 'LB']
CTB_TAG_DICT = {k:i for i,k in enumerate(CTB_TAGS)}

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)

def wordlist_to_slices(words):
    len_list = [len(word) for word in words]
    cum = 0
    slices = []
    for i in range(len(len_list)):
        slices.append(slice(cum+1, cum+1+len_list[i])) # +1 because [CLS]
        cum += len_list[i]
    return slices

def tags_and_words_to_ids(tags_and_words, first_sent=True):
    tags, words = tags_and_words
    tags = list(map(CTB_TAG_DICT.__getitem__, tags))
    lens = [len(w) for w in words]
    ids = [0] + sum([[tag]*len for tag, len in zip(tags, lens)], [])
    if not first_sent:
        ids = ids + [0]
    ids = torch.LongTensor(ids)
    onehot = torch.zeros(len(ids), len(CTB_TAGS)).scatter_(1, ids.unsqueeze(1), 1)
    onehot[0] = 0
    if not first_sent:
        onehot[-1] = 0
    return ids, onehot # add [0] for [CLS] and [SEP]

def unwrapped_preprocess_function(examples, tokenizer, context_name, choice_name, max_seq_length, data_args):
    # Examples is a dict with keys: translation, choices, answer, size is 1k?
    hanout = HanLP(examples[context_name])
    seg_sents = hanout['tok/fine']
    postags = hanout['pos/ctb']
    postag_ids = map(tags_and_words_to_ids, zip(postags, seg_sents))
    postag_ids, postag_onehot = zip(*postag_ids)
    
    ans_hanout = HanLP(sum(examples[choice_name], []))
    ans_seg_sents = ans_hanout['tok/fine']
    ans_postags = ans_hanout['pos/ctb']
    ans_postag_ids = map(lambda x:tags_and_words_to_ids(x, first_sent=False), zip(ans_postags, ans_seg_sents))
    ans_postag_ids, ans_postag_onehot = zip(*ans_postag_ids)
    
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
    results.update({k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()})
    results['labels'] = [ answer for answer in examples['answer']]
    # postag_ids: n_sent * n_char
    # postag_ids: n_sent * n_char * 33
    # ans_postag_ids: n_sent * 4 * n_char
    # ans_postag_onehot: n_sent * 4 * n_char * 33
    results['postag_ids'] = list(postag_ids)
    results['postag_onehot'] = list(postag_onehot)
    results['ans_postag_ids'] = list([ans_postag_ids[i:i+4] for i in range(0, len(ans_postag_ids), 4)])
    results['ans_postag_onehot'] = list([ans_postag_onehot[i:i+4] for i in range(0, len(ans_postag_onehot), 4)])
    # print(results['postag_ids'], results['postag_onehot'])
    
    # Un-flatten
    # print(results.keys())
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
        # ipdb.set_trace()
        label_name = "labels"
        # print(list(features[0].keys()))
        labels = [feature.pop(label_name) for feature in features]
        # postag_ids: n_sent * n_char
        # postag_ids: n_sent * n_char * 33
        postag_ids = [feature.pop("postag_ids") for feature in features]
        postag_onehot = [feature.pop("postag_onehot") for feature in features]
        
        ans_postag_ids = sum([feature.pop("ans_postag_ids") for feature in features], [])
        ans_postag_ids = [torch.tensor(a+postag_ids[i//4]) for i,a in enumerate(ans_postag_ids)]
        ans_postag_onehot = sum([feature.pop("ans_postag_onehot") for feature in features], [])
        ans_postag_onehot = [torch.tensor(a+postag_onehot[i//4]) for i,a in enumerate(ans_postag_onehot)]
        
        postag_ids = pad_sequence(ans_postag_ids).T
        postag_onehot = pad_sequence(ans_postag_onehot).permute(1,0,2)
        
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
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        # batch["slices"] = slices
        batch["postag_ids"] = postag_ids
        batch["postag_onehot"] = postag_onehot
        batch["ans_postag_ids"] = postag_ids
        batch["ans_postag_onehot"] = postag_onehot
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
        self.model = AutoModelForMultipleChoice.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        self.pos_info_extractor = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config
        )
        self.pos_info_extractor.requires_grad_(False)

        self.pos_weights = nn.Parameter(torch.ones(len(CTB_TAGS), 1)/len(CTB_TAGS))
        self.dropout = nn.Dropout(0.1)
        self.pos_scorer = nn.Linear(config.hidden_size, 1)
        self.pos_scorer.reset_parameters()
        
        torch.nn.init.normal_(self.pos_weights)
        self.use_original_score = True
        if model_args.pos_info_factor == None:
            self.pos_info_factor = nn.Parameter(torch.rand(1))
        elif model_args.pos_info_factor < 0:
            self.use_original_score = False
        else:
            self.pos_info_factor = float(model_args.pos_info_factor)
        self.args = model_args
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels, postag_ids, postag_onehot, ans_postag_ids, ans_postag_onehot):
        # input_ids: n_sent * 4 * n_both_char
        # postag_ids: (n_sent*4) * n_char
        # postag_onehot: (n_sent*4) * n_char * 33
        # print(input_ids.size())
        if 'guwenbert' in self.args.model_name_or_path:
            token_type_ids[:] = 0
        output = self.model(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids, 
            labels = labels, 
            output_hidden_states = True
        )
        bsz = postag_onehot.size(0)
        pos_info = self.pos_info_extractor(
            input_ids = input_ids.view(bsz, -1), 
            attention_mask = attention_mask.view(bsz, -1), 
            token_type_ids = token_type_ids.view(bsz, -1),
            output_hidden_states = True
        )
        
        real_context_len = postag_onehot.size(1)
        # ipdb.set_trace()
        pos_info = pos_info.last_hidden_state.permute(0, 2, 1)[:,:,:real_context_len].bmm(postag_onehot)
        # pos_info: (n_sent*4) * hid_dim * 33
        pos_info = pos_info / (postag_onehot.sum(dim=1, keepdim=True)+1e-10)
        pos_info = pos_info.bmm(self.pos_weights.unsqueeze(0).repeat(pos_info.size(0), 1, 1)).squeeze(2)
        # pos_info: (n_sent*4) * hid_dim
        pos_scores = self.pos_scorer(self.dropout(pos_info)).view(-1, 4)
        # pos_scores: n_sent * 4
        if self.use_original_score:
            logits = output.logits + self.pos_info_factor * pos_scores
        else:
            logits = pos_scores
        if self.args.softmax_temperature is not None:
            logits = output.logits / self.args.softmax_temperature
        return {"logits": logits, "loss": self.loss(logits, labels)}
    
def MyOptimizer(model, args, multiplier=10):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    pos_parameters = [name for name, _ in model.named_parameters() if "pos_" in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters and n not in pos_parameters and p.requires_grad],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters and n not in pos_parameters and p.requires_grad],
            "weight_decay": 0.0,
            "lr": args.learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters and n in pos_parameters and p.requires_grad],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate * multiplier
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters and n in pos_parameters and p.requires_grad],
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