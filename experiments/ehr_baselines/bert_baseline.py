from __future__ import absolute_import, division, unicode_literals

import argparse
import sys
import numpy as np
import logging
import torch
from collections import OrderedDict
import pickle
import os
import code
import time

import torch
from transformers import *

# Set PATHs
PATH_TO_ENTEVAL = '../EntEval/'         #https://github.com/ZeweiChu/EntEval
"""
MODELS = {
  'bert-base-uncased': (BertModel, BertTokenizer),
  'bert-large-uncased': (BertModel, BertTokenizer),
  'bert-large-uncased-whole-word-masking': (BertModel, BertTokenizer),
  'roberta-base': (RobertaModel, RobertaTokenizer),
  'roberta-large': (RobertaModel, RobertaTokenizer)
}
"""
MODELS = {
  'bert-base-uncased': (BertModel, BertTokenizer),
  'bert-large-uncased': (BertModel, BertTokenizer),
  'bert-large-uncased-whole-word-masking': (BertModel, BertTokenizer),
  'roberta-base': (RobertaModel, RobertaTokenizer),
  'roberta-large': (RobertaModel, RobertaTokenizer),
  'allenai/biomed_roberta_base': (AutoModel.from_pretrained("allenai/biomed_roberta_base"), AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")),
  'monologg/biobert_v1.1_pubmed': (AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed"), AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")),
  'allenai/scibert_scivocab_uncased': (AutoModel.from_pretrained('allenai/scibert_scivocab_uncased'), AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')),
  'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext': (AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'), AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'))
}

BERT_CLS_TOKEN = '[CLS]'
BERT_SEP_TOKEN = '[SEP]'
BERT_PAD_TOKEN = '[PAD]'


# import EntEval
sys.path.insert(0, PATH_TO_ENTEVAL)   
import enteval

#sys.path.insert(0, '../enttype')
#sys.path.insert(0, '../enttype/resources')
#sys.path.insert(0, '../EntityTypingTransformer')

#import enttype, config_parser, constant, data_utils, model_utils


def prepare(params, samples):
  return


def json_to_enteval(example, is_context):
  if is_context:
    s = len(example['left_context'])
    e = s + len(example['mention_as_list'])
    ctx = example['left_context'] + example['mention_as_list'] + example['right_context']
    return (ctx, s, e, None)
  else:
    desc = example['left_context'] + example['mention_as_list'] + \
           example['right_context']
    s = len(example['left_context'])
    e = s + len(example['mention_as_list'])
    return (example['mention_as_list'], s, e, desc)


def format_cap(example):
  sentence = example[0]
  start_loc = example[1]
  end_loc = example[2]
  mention = sentence[start_loc:end_loc]
  left_contex = sentence[:start_loc]
  right_context = sentence[end_loc:]
  ex_json = {
    'ex_id': 0,  # dummy
    'left_context': left_contex,
    'right_context': right_context,
    'mention_as_list': mention,
    'word': ' '.join(mention),
    'sent1': ' '.join(left_contex + mention + right_context),
    'sent2': ' '.join(mention),
    'y_category': []
  }
  return ex_json


def format_conll_yago(example):
  if example[0] is None:
    description = example[3]
    if not description:
      description = ['empty', 'description', '.']
    d_start_loc = 0
    d_end_loc = 1
    sidx = 1
    if [w.lower() for w in description[:]] == ['this', 'is', 'a', 'list',
                                               'of']:
      d_start_loc = 5
      sidx = 6
    for j in range(sidx, len(description)):
      if description[j].lower() in [
        'is', 'was', 'are', 'were', ',', '.', '(', 'has', 'have', 'can', 'or'
      ]:
        d_end_loc = j
        break

    # Clip long mentions
    if d_end_loc - d_start_loc > 10:
      d_end_loc = d_start_loc + 10

    d_mention = description[d_start_loc:d_end_loc]
    d_left_contex = description[:d_start_loc]
    d_right_context = description[d_end_loc:]
    # print(d_mention, description[:10])
    ex_json = {
      'ex_id': 0,  # dummy
      'left_context': d_left_contex,
      'right_context': d_right_context,
      'mention_as_list': d_mention,
      'word': ' '.join(d_mention),
      'sent1': ' '.join(d_left_contex + d_mention + d_right_context),
      'sent2': ' '.join(d_mention),
      'y_category': []
    }
  else:
    sentence = example[0]
    start_loc = example[1]
    end_loc = example[2]
    mention = sentence[start_loc:end_loc]
    left_contex = sentence[:start_loc]
    right_context = sentence[end_loc:]
    ex_json = {
      'ex_id': 0,  # dummy
      'left_context': left_contex,
      'right_context': right_context,
      'mention_as_list': mention,
      'word': ' '.join(mention),
      'sent1': ' '.join(left_contex + mention + right_context),
      'sent2': ' '.join(mention),
      'y_category': []
    }
  return ex_json


def format_wlned(example):
  if isinstance(example, list):
    example_ = example[3]
    example_['sent1'] = ' '.join(
      example_['left_context'] + example_['mention_as_list'] +\
      example_['right_context'])
    example_['sent2'] = ' '.join(example_['mention_as_list'])
    return example_
  else:
    example_ = example
    example_['sent1'] = ' '.join(
      example_['left_context'] + example_['mention_as_list'] +\
      example_['right_context'])
    example_['sent2'] = ' '.join(example_['mention_as_list'])
    return example_

def format_ehr(example):
  # {"ex_id": "Q4688873_467", 
  #  "word": "blushing", 
  #  "right_context": [], 
  #  "left_context": ["Shame", "/", "Humiliation", "(", "reaction", "to", "failure", "/", "impulse", "to", "review", "behaviour", ")", "--", "eyes", "lowered", ",", "the", "head", "down", "and", "averted", ","], 
  #  "wikiurl": "blushing", 
  #  "categories": ["reflexes", "emotion"], 
  #  "ncandidates": 4, 
  #  "candidates": [
  #    ["Flushing (physiology)", "0.135211", [""], ["is", "to", "become", "markedly", "red", "in", "the", "face", "and", "often", "other", "areas", "of", "the", "skin,", "..."], "439380"], 
  #    ["Boy", "0.0028169", ["A"], ["is", "a", "young", "male", "human.", "The", "term", "is", "usually", "used", "for", "a", "child", "or", "an", "adolescent.", "..."], "194253"], 
  #    ["Blushing", "0.847887", [""], ["is", "the", "reddening", "of", "a", "person\\'s", "face", "due", "to", "psychological", "reasons.", "It", "is", ".."], "240889"], 
  #    ["Idiopathic craniofacial erythema", "0.0140845", [""], ["is", "a", "medical", "condition", "characterised", "by", "severe,", "uncontrollable,", "and", "..."], "1392763"]]}

  # print(example)   #do i need to do example[3] like above?
  # {'ex_id': 'Q178999_77', 'word': 'sensory neurons', 
  #  'right_context': ['(', 'pseudounipolar', 'neurons', ')', ',', 'such', 'as', 'those', 'for', 'touch', 'and', 'warmth', ',', 'the', 'axons', 'are', 'called', 'afferent', 'nerve', 'fibers', 'and', 'the', 'electrical', 'impulse', 'travels', 'along', 'these', 'from', 'the', 'periphery', 'to', 'the', 'cell', 'body', ',', 'and', 'from', 'the', 'cell', 'body', 'to', 'the', 'spinal', 'cord', 'along', 'another', 'branch', 'of', 'the', 'same', 'axon', '.'], 
  #  'left_context': ['In', 'certain'], 'wikiurl': 'sensory neuron', 'ncandidates': 2, 'mention_as_list': ['sensory', 'neurons'], 'y_category': ['afferent neurons', 'human cells', 'receptor cells']}

  if isinstance(example, list):
    example_ = example[3]
    example_['sent1'] = ' '.join(
      example_['left_context'] + example_['mention_as_list'] +\
      example_['right_context'])
    example_['sent2'] = ' '.join(example_['mention_as_list'])
  else:
    example_ = example
    example_["mention_as_list"] = str(example["word"]).split(" ")
    example_['y_category'] = example['y_category']   #vs exp_categories ?   TODO
    example_['sent1'] = ' '.join(
      example_['left_context'] + example_['mention_as_list'] +\
      example_['right_context'])
    example_['sent2'] = ' '.join(example_['mention_as_list'])
    
  return example_

def context_tokenizer_avg(batch):
  #print('context')
  batch = [json_to_enteval(ex, True) for ex in batch]
  #print('batch:', batch)
  tokenized_batch = []
  span_locations = []
  for i, (ctx, s, e, desc) in enumerate(batch):
    tokenized_sent = [BERT_CLS_TOKEN]
    offset = 0
    span_counter = 0
    for j, word in enumerate(ctx):
      tokenized_word = wordpiece_tokenizer.tokenize(word)
      if j < s:
        offset += len(tokenized_word)
      if s <= j < e:
        span_counter += len(tokenized_word)
        # print('span_counter =', span_counter)
      tokenized_sent += tokenized_word
      # hacky way to cut down long sentences
      if len(tokenized_sent) == 511:
        break
    tokenized_sent.append(BERT_SEP_TOKEN)
    tokenized_batch.append(tokenized_sent)
    #print(offset+1, offset+span_counter+1, ctx[s:e], tokenized_sent[
    # offset+1:offset+span_counter+1])
    span_locations.append(list(range(s+1, s+span_counter+1)))  # add 1 for [CLS])
  assert len(batch) == len(span_locations)
  #print('span_locations:', span_locations)
  max_len = max([len(sent) for sent in tokenized_batch])
  batch_size = len(tokenized_batch)
  mask = torch.zeros(batch_size, max_len)
  for k, span_loc in enumerate(span_locations):
    mask[k, span_loc] = 1.
  #print('tokenized_batch:', max_len,  [len(x) for x in tokenized_batch])
  #print('mask:', mask.shape)
  for k in range(batch_size):
    if len(tokenized_batch[k]) < max_len:
      for _ in range(max_len - len(tokenized_batch[k])):
        tokenized_batch[k].append(BERT_PAD_TOKEN)
  # print('tokenized_batch:', max_len, [len(x) for x in tokenized_batch])
  # print(tokenized_batch)
  return tokenized_batch, mask.unsqueeze(-1).float().cuda()


def description_tokenizer_avg(batch):
  #print('desc')
  batch = [json_to_enteval(ex, False) for ex in batch]
  tokenized_batch = []
  span_locations = []
  for i, (ctx, s, e, desc) in enumerate(batch):
    tokenized_sent = [BERT_CLS_TOKEN]
    offset = 0
    span_counter = 0
    for j, word in enumerate(desc):
      tokenized_word = wordpiece_tokenizer.tokenize(word.lower())
      if j < s:
        offset += len(tokenized_word)
      if s <= j < e:
        span_counter += len(tokenized_word)
        # print('span_counter =', span_counter)
      tokenized_sent += tokenized_word
      # hacky way to cut down long sentences
      if len(tokenized_sent) == 511:
        break
    tokenized_sent.append(BERT_SEP_TOKEN)
    tokenized_batch.append(tokenized_sent)
    #print(offset+1, offset+span_counter+1, ctx[s:e], tokenized_sent[
    # offset+1:offset+span_counter+1])
    span_locations.append(list(range(s+1, s+span_counter+1)))  # add 1 for [CLS])
  max_len = max([len(sent) for sent in tokenized_batch])
  batch_size = len(tokenized_batch)
  mask = torch.zeros(batch_size, max_len)
  for k, span_loc in enumerate(span_locations):
    mask[k, span_loc] = 1.
  #print('tokenized_batch:', max_len,  [len(x) for x in tokenized_batch])
  #print('mask:', mask.shape, mask)
  for k in range(batch_size):
    if len(tokenized_batch[k]) < max_len:
      for _ in range(max_len - len(tokenized_batch[k])):
        tokenized_batch[k].append(BERT_PAD_TOKEN)
  return tokenized_batch, mask.unsqueeze(-1).float().cuda()


# def description_tokenizer_avg(batch):
#   batch = [json_to_enteval(ex, False) for ex in batch]
#   tokenized_batch = []
#   for i, (ctx, s, e, desc) in enumerate(batch):
#     tokenized_sent = [BERT_CLS_TOKEN]
#     for j, word in enumerate(desc):
#       tokenized_word = wordpiece_tokenizer.tokenize(word)
#       tokenized_sent += tokenized_word
#       # hacky way to cut down long sentences
#       if len(tokenized_sent) == 511:
#         break
#     tokenized_sent.append(BERT_SEP_TOKEN)
#     tokenized_batch.append(tokenized_sent)
#   max_len = max([len(sent) for sent in tokenized_batch])
#   batch_size = len(tokenized_batch)
#   mask = torch.ones(batch_size, max_len)
#   for k in range(batch_size):
#     if len(tokenized_batch[k]) < max_len:
#       for _ in range(max_len - len(tokenized_batch[k])):
#         tokenized_batch[k].append(BERT_PAD_TOKEN)
#   return tokenized_batch, mask.unsqueeze(-1).float().cuda()


def context_tokenizer_cls(batch):
  batch = [json_to_enteval(ex, True) for ex in batch]
  # print('batch:', batch)
  tokenized_batch = []
  span_locations = []
  for i, (ctx, s, e, desc) in enumerate(batch):
    tokenized_sent = [BERT_CLS_TOKEN]
    for j, word in enumerate(ctx):
      tokenized_word = wordpiece_tokenizer.tokenize(word.lower())
      tokenized_sent += tokenized_word
      # hacky way to cut down long sentences
      if len(tokenized_sent) == 511:
        break
    tokenized_sent.append(BERT_SEP_TOKEN)
    for word in ctx[s:e]:
      tokenized_sent += wordpiece_tokenizer.tokenize(word.lower())
    tokenized_sent.append(BERT_SEP_TOKEN)
    tokenized_batch.append(tokenized_sent)
  max_len = max([len(sent) for sent in tokenized_batch])
  batch_size = len(tokenized_batch)
  # print('tokenized_batch:', max_len,  [len(x) for x in tokenized_batch])
  for k in range(batch_size):
    if len(tokenized_batch[k]) < max_len:
      for _ in range(max_len - len(tokenized_batch[k])):
        tokenized_batch[k].append(BERT_PAD_TOKEN)
  # print('tokenized_batch:', max_len, [len(x) for x in tokenized_batch])
  # print(tokenized_batch)
  return tokenized_batch, None


def description_tokenizer_cls(batch):
  batch = [json_to_enteval(ex, False) for ex in batch]
  tokenized_batch = []
  for i, (ctx, s, e, desc) in enumerate(batch):
    tokenized_sent = [BERT_CLS_TOKEN]
    for word in ctx:  # ctx = mention as list
      tokenized_sent += wordpiece_tokenizer.tokenize(word.lower())
    tokenized_sent.append(BERT_SEP_TOKEN)
    for j, word in enumerate(desc):
      tokenized_word = wordpiece_tokenizer.tokenize(word.lower())
      tokenized_sent += tokenized_word
      # hacky way to cut down long sentences
      if len(tokenized_sent) == 511:
        break
    tokenized_sent.append(BERT_SEP_TOKEN)
    tokenized_batch.append(tokenized_sent)
  max_len = max([len(sent) for sent in tokenized_batch])
  batch_size = len(tokenized_batch)
  for k in range(batch_size):
    if len(tokenized_batch[k]) < max_len:
      for _ in range(max_len - len(tokenized_batch[k])):
        tokenized_batch[k].append(BERT_PAD_TOKEN)
  return tokenized_batch, None


def encoder(input, average_hidden=False):
  #print("Encoding: ", input)

  # HANDLE TOKENSIZE > 512 ENCODING WARNING
  max_len = 512
  input_tokenized = bert_tokenizer.batch_encode_plus(input, add_special_tokens=False)   
  batch_input_lens = [ len(input[i]) for i in range(len(input))]
  batch_input_token_lens = [ len(input_tokenized['input_ids'][i]) for i in range(len(input))]

  if any([ b  > max_len for b in batch_input_token_lens ] ):
    print("Encoding: ", len(input), ",", len(input[0]), ",", type(input_tokenized))       #len(input) = 16, <class 'transformers.tokenization_utils.BatchEncoding'>
    print("Batch lens:",batch_input_lens)                            
    print("Batch Token lens:",batch_input_token_lens)                            
    print("Warning found tokenized input in batch that is greater than maxlen")
    for i in range(len(input)):
      overflow_len = len(input[i])- max_len
      input[i] = input[i][:-overflow_len]

  input_ids = bert_tokenizer.batch_encode_plus(input, add_special_tokens=False)
  input_ids = np.array(input_ids['input_ids'])
  input_ids = torch.from_numpy(input_ids).cuda()
  
  #print("input", input)
  # print("input_idds", input_ids)
  
  with torch.no_grad():
    if average_hidden:
      #error here  RuntimeError: CUDA error: no kernel image is available for execution on the device
      all_hidden_states = model(input_ids)[2][-4:]
      averaged_hidden_states = torch.zeros_like(all_hidden_states[0])
      for hidx, hid in enumerate(all_hidden_states):
        averaged_hidden_states += hid
      averaged_hidden_states /= hidx + 1
      return averaged_hidden_states
    else:
      last_hidden_states = model(input_ids)[0]
      return last_hidden_states


def batcher(params, batch):
  if args.task in ['cap_same', 'cap_next']:
    batch = [format_cap(ex) for ex in batch]
  elif args.task == 'conll_yago':
    batch = [format_conll_yago(ex) for ex in batch]
  elif args.task == 'wlned':
    batch = [format_wlned(ex) for ex in batch]
  elif args.task == 'ehr':
    batch = [format_ehr(ex) for ex in batch]
  else:
    print('ERROR: {} is an invalid task.'.format(args.task))
    raise
  use_ctx = False
  use_def = False

  #print("in Batcher with")
  #print("params:", params)
  #print("batch: ", batch)
  
  # this might need to change for ehr task
  #print(batch[0])
  if 'y_title' in batch[0]: # or 'wikiurl' in batch[0]:    #its possible I may need to add y_title in format_ehr or in ed.py loadFile <--- HERE
    use_ctx = True
    batch_ctx, mask_ctx = context_tokenizer_cls(batch)
    #batch_ctx, mask_ctx = context_tokenizer_avg(batch)
    #print("Context", type(batch_ctx), len(batch_ctx), batch_ctx[0])
  else:
    use_def = True
    batch_desc, mask_desc = description_tokenizer_cls(batch)
    #batch_desc, mask_desc = description_tokenizer_avg(batch)
    #print("Description", type(batch_desc), len(batch_desc), batch_desc[0])

  # 2020-08-11 13:39:35,332 : [('reg:1e-05', 67.5), ('reg:0.0001', 67.47), ('reg:0.001', 67.47), ('reg:0.01', 67.43)]
  # 2020-08-11 13:39:35,332 : Validation : best param found is reg = 1e-05 with score             67.5

  context_embedding = None
  def_embedding = None
  with torch.no_grad():
    if use_ctx:
      embeddings = encoder(batch_ctx, average_hidden=args.avg_hid)
      if mask_ctx is None:
        embeddings = embeddings[:, 0, :]
      else:
        embeddings = torch.sum(embeddings * mask_ctx, 1) / torch.sum(mask_ctx, 1)
      context_embedding = embeddings.cpu().data.numpy()
    if use_def:
      #error begins here
      embeddings = encoder(batch_desc, average_hidden=args.avg_hid)
      if mask_desc is None:
        embeddings = embeddings[:, 0, :]
      else:
        embeddings = torch.sum(embeddings * mask_desc, 1) / torch.sum(mask_desc, 1)
      context_embedding = embeddings.cpu().data.numpy()
  return context_embedding, def_embedding




# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":

  #args = config_parser.parser.parse_args()
  parser = argparse.ArgumentParser()
  parser.add_argument("-task", help="", choices=['conll_yago', 'wlned', 'ehr'],default="conll_yago")
  parser.add_argument("-model_name", help="", choices=["bert-base-uncased","bert-large-uncased", 'monologg/biobert_v1.1_pubmed', 'allenai/scibert_scivocab_uncased', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext' ], default="bert-base-uncased")   
  
  args = parser.parse_args()

  # Setup args
  args.eval_batch_size = 128
  args.avg_hid = True

  print('-' * 80)
  for k, v in vars(args).items():
    print(k, ':', v)
  print('-' * 80)

  # SWITCH BETWEEN base and large ( maybe add BioBert ? )
  model_name = args.model_name #'bert-base-uncased', 'bert-large-uncased'  TODO: add biobert

  # set location of data
  
  if args.task != 'ehr':
    PATH_TO_DATA = '../EntEval'    #location of CoNLL, WLNED and original wiki_descs data
  else:
    #PATH_TO_DATA = "/dccstor/redrug_ier/diego/data/entity_type_systems/0720lg2_v6/test/wiki_desc/test_ehr_examples_between_1_and_6_cands_v2_overlaping_with_wikidesc0720_3k_full_with_cand_desc.json"   #location of ehr examples  #
    PATH_TO_DATA = "/home/diego/biomed_fall20/check_ehr_data/test_ehr_examples_0824.json"   #location of ehr examples  <--- this is just pointing to test but it needs train/dev too?



  print('Using {}'.format(model_name))

  st = time.time()
  # Define BERT model
  model_class, tokenizer_class = MODELS[model_name]
  bert_tokenizer = tokenizer_class.from_pretrained(model_name)
  wordpiece_tokenizer = bert_tokenizer.wordpiece_tokenizer
  model = model_class.from_pretrained(model_name, output_hidden_states=True)
  assert model.config.output_hidden_states == True
  model.cuda()

  device = torch.device("cuda")

  # Set params for EntEval
  params_enteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'batch_size': 16}
  params_enteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 16, 'tenacity': 5, 'epoch_size': 4}
  params_enteval['softmask'] = False

  se = enteval.engine.SE(params_enteval, batcher, prepare)

  if args.task == 'cap_same':
    tasks = ["CAPsame"]
  elif args.task == 'cap_next':
    tasks = ["CAPnext"]
  elif args.task == 'conll_yago':
    tasks = ["ConllYago"]
  elif args.task == 'wlned':
    tasks = ["WLNED"]
  elif args.task == 'ehr':
    tasks = ["EHR"]     
  else:
    print('ERROR: {} is an invalid task.'.format(args.task))
    raise NotImplementedError

  results = se.eval(tasks)
  print("Done. Elapsed Time: ", time.time() - st )
  print("FINAL RESULTS!! ")
  for k, v in results.items():
    print(k, v)
