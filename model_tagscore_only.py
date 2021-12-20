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
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import hanlp

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

def tags_and_words_to_ids(tags_and_words):
    tags, words = tags_and_words
    tags = list(map(CTB_TAG_DICT.__getitem__, tags))
    lens = [len(w) for w in words]
    ids = [0] + sum([[tag]*len for tag, len in zip(tags, lens)], []) + [0]
    ids = torch.LongTensor(ids)
    onehot = torch.zeros(len(ids), len(CTB_TAGS)).scatter_(1, ids.unsqueeze(1), 1)
    onehot[0] = 0
    onehot[-1] = 0
    return ids, onehot # add [0] for [CLS] and [SEP]

def unwrapped_preprocess_function(examples, tokenizer, context_name, choice_name, max_seq_length, data_args):
    # Examples is a dict with keys: translation, choices, answer, size is 1k?
    hanout = HanLP(examples[context_name])
    seg_sents = hanout['tok/fine']
    postags = hanout['pos/ctb']
    # pool = Pool()
    # # slices = [pool.apply_async(wordlist_to_slices, words) for words in seg_sents]
    # # postag_lens = [pool.apply_async(len, words) for words in seg_sents]
    # postag_ids = [pool.apply_async(tags_and_words_to_ids, tags_and_words) for tags_and_words in zip(postags, seg_sents)]
    # pool.close()
    # pool.join()
    postag_ids = map(tags_and_words_to_ids, zip(postags, seg_sents))
    postag_ids, postag_onehot = zip(*postag_ids)
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
    results['postag_ids'] = list(postag_ids)
    results['postag_onehot'] = list(postag_onehot)
    # print(results['postag_ids'], results['postag_onehot'])
    
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
        label_name = "labels"
        # print(list(features[0].keys()))
        labels = [feature.pop(label_name) for feature in features]
        # postag_ids: n_sent * n_char
        # postag_ids: n_sent * n_char * 33
        postag_ids = [torch.tensor(feature.pop("postag_ids")) for feature in features]
        postag_onehot = [torch.tensor(feature.pop("postag_onehot")) for feature in features]
        # print(len(postag_onehot))
        # print(postag_onehot)
        postag_ids = pad_sequence(postag_ids).T
        postag_onehot = pad_sequence(postag_onehot).permute(1,0,2)
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
        # print(postag_onehot.size())
        # postag_masks = torch.zeros(batch.size(0), batch.size(2), len(CTB_TAGS))
        # ===
        # for sent_id in range(batch_size):
        #     for word_id in range(len(postag_ids[sent_id])):
        #         postag_masks[sent_id][slices[sent_id][word_id]][:,postag_ids[sent_id][word_id]] = 1.
        # ===
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

        self.pos_weights = nn.Parameter(torch.ones(len(CTB_TAGS), 1)/len(CTB_TAGS))
        self.pos_scorer = nn.Parameter(torch.eye(self.model.config.hidden_size))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels, postag_ids, postag_onehot):
        # input_ids: n_sent * 4 * n_both_char
        # postag_ids: n_sent * n_char
        # postag_onehot: n_sent * n_char * 33
        # print(input_ids.size())
        output = self.model(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids, 
            labels = labels, 
            output_hidden_states = True
        )
        bsz = postag_onehot.size(0)
        context_len = postag_onehot.size(1)
        pos_info = self.pos_info_extractor(
            input_ids = input_ids[:, 0, :context_len], 
            attention_mask = attention_mask[:, 0, :context_len], 
            token_type_ids = token_type_ids[:, 0, :context_len],
            output_hidden_states = True
        )
        # print(pos_info.last_hidden_state.size(), postag_onehot.size())
        pos_info = pos_info.last_hidden_state.permute(0, 2, 1).bmm(postag_onehot)
        # pos_info: n_sent * hid_dim * 33
        pos_info = pos_info / (postag_onehot.sum(dim=1, keepdim=True)+1e-10)
        pos_info = pos_info.bmm(self.pos_weights.unsqueeze(0).repeat(pos_info.size(0), 1, 1))
        # pos_info: n_sent * hid_dim * 1
        pos_scores = output.hidden_states[-1][:,0].view(bsz*4, -1).mm(self.pos_scorer).view(bsz, 4, -1).bmm(pos_info).squeeze(2)
        # pos_scores: n_sent * 4
        logits = pos_scores
        return {"logits": logits, "loss": self.loss(logits, labels)}