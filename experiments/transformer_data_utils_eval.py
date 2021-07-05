import argparse
import glob
import json
import numpy as np
import torch
import string
from random import shuffle
from tqdm import tqdm

import transformer_constant
from models import TransformerModel

def to_torch(batch, device):
  inputs_to_model = {k: v.to(device) for k, v in batch['inputs'].items()}
  targets = batch['targets'].to(device)
  return inputs_to_model, targets


def get_example(generator, batch_size, max_len, eval_data=False, tokenizer=None, answer_num=60000):
  # [cur_stream elements]
  # 0: example id, 1: left context, 2: right context, 3: mention word, 4: gold category
  cur_stream = [None] * batch_size
  no_more_data = False
  no_ans_count = 0
  while True:
    bsz = batch_size
    seq_length = max_len
    mention_length_limit = 10  # in words, not word pieces
    for i in range(batch_size):
      try:
        cur_stream[i] = list(next(generator))
      except StopIteration:
        no_more_data = True
        bsz = i
        break
    if no_more_data and bsz == 0:
      break
    ex_ids = np.zeros([bsz], np.object)
    targets = np.zeros([bsz, answer_num], np.float32)
    inputs = []
    sentence_len_wp = []
    for i in range(bsz):
      ex_ids[i] = cur_stream[i][0]
      left_seq = cur_stream[i][1]
      right_seq = cur_stream[i][2]
      mention_seq = cur_stream[i][3]
      if len(mention_seq) > mention_length_limit:
        mention_seq = mention_seq[:mention_length_limit]
      mention = ' '.join(mention_seq)
      context = ' '.join(left_seq + mention_seq + right_seq)
      len_after_tokenization = len(tokenizer.encode_plus(mention, context)['input_ids'])
      if len_after_tokenization > max_len:
        overflow_len = len_after_tokenization - max_len
        context = ' '.join(left_seq + mention_seq + right_seq[:-overflow_len])
      inputs.append([mention, context])
      len_after_tokenization = len(tokenizer.encode_plus(mention, context)['input_ids'])
      sentence_len_wp.append(len_after_tokenization)
      # Gold categories
      for answer_ind in cur_stream[i][4]:
        targets[i, answer_ind] = 1.0
    max_len_in_batch = max(sentence_len_wp)

    inputs = tokenizer.batch_encode_plus(
      inputs,
      add_special_tokens=True,
      max_length=min(max_len, max_len_in_batch),
      truncation_strategy='only_second',
      pad_to_max_length=True,
      return_tensors='pt'
    )
    # print(max_len, max_len_in_batch, inputs["input_ids"].size(), inputs["attention_mask"].size(),
    #       inputs["token_type_ids"].size())
    targets = torch.from_numpy(targets)
    #feed_dict = {"ex_ids": ex_ids, "inputs": inputs, "targets": targets}
    feed_dict = {"ex_ids": ex_ids, "inputs": inputs, "targets": targets}
    if bsz == 1:
      feed_dict['mention'] = mention
      feed_dict['context'] = context 

    if no_more_data:
      if eval_data and bsz > 0:
        yield feed_dict
      break
    yield feed_dict


class DatasetLoader(object):

  def __init__(self, filepattern, args, tokenizer):
      self._all_shards = [file for file in glob.glob(filepattern) if 'allwikipp_wiki' not in file]
      shuffle(self._all_shards)
      print('Found %d shards at %s' % (len(self._all_shards), filepattern))
      if args.goal == '60k':
          self.word2id = transformer_constant.ANS2ID_DICT_60K
      elif args.goal == 'ufet':
          self.word2id = transformer_constant.ANS2ID_DICT_UFET
      else:
          print('ERROR: Invalid input... ' + args.goal)
          raise
      self.tokenizer = tokenizer
      self.do_lower = args.do_lower
      self.context_window_size = args.context_window_size
      self.args = args

  def _load_npz(self, path):
      with open(path, 'rb') as f:
          data = np.load(f)
      return data

  def _load_shard(self, shard_name, eval_data):
      print('Loarding {}'.format(shard_name))
      with open(shard_name) as f:
        lines = [json.loads(line.strip()) for line in tqdm(f)]
        ex_ids = [line["ex_id"] for line in lines]
        mention_word = [line["word"].split() if not self.do_lower
                        else line["word"].lower().split() for line in lines]
        left_seq = [line['left_context'][-self.context_window_size:] if not self.do_lower
                    else [w.lower() for w in line['left_context']][-self.context_window_size:] for line in lines]
        right_seq = [line['right_context'][:self.context_window_size] if not self.do_lower
                     else [w.lower() for w in line['right_context']][:self.context_window_size] for line in lines]

        y_categories = [line['y_category'] for line in lines]
        y_category_ids = []
        for iid, y_strs in enumerate(y_categories):
            y_category_ids.append([self.word2id[x] for x in y_strs if x in self.word2id])

        # Ddebug
        if False:
            idx = 0
            print('ex_ids: ', ex_ids[idx])
            print('left_seq: ', left_seq[idx])
            print('right_seq: ', right_seq[idx])
            print('mention_word: ', mention_word[idx])
            print('y_category_ids: ', y_category_ids[idx])
            print('y_categories: ', y_categories[idx])

        print('0) ex_ids:', len(ex_ids), '1) left_seq:', len(left_seq), '2) right_seq:', len(right_seq),
             '3) mention_word:', len(mention_word), '4) y_category_ids:', len(y_category_ids))

        # 0: example id, 1: left context, 2: right context, 3: mention word, 4: gold category
        return zip(ex_ids, left_seq, right_seq, mention_word,  y_category_ids)

  def _get_sentence(self, epoch, eval_data):
    for i in range(0, epoch):
      for shard in self._all_shards:
        ids = self._load_shard(shard, eval_data)
        for current_ids in ids:
          yield current_ids

  def get_batch(self, batch_size, max_len, epoch, eval_data=True):
      return get_example(
          self._get_sentence(epoch, eval_data), batch_size=batch_size, max_len=max_len,
          eval_data=eval_data, tokenizer=self.tokenizer, answer_num=transformer_constant.ANSWER_NUM_DICT[self.args.goal]
      )


class DatasetLoaderForEntEval(object):

  def __init__(self, data, args, tokenizer):
      self.data = data
      if args.goal == '60k':
          self.word2id = transformer_constant.ANS2ID_DICT_60K
      elif args.goal == 'ufet':
          self.word2id = transformer_constant.ANS2ID_DICT_UFET
      elif args.goal == 'medwiki':
          self.word2id = transformer_constant.ANS2ID_MEDWIKI_DICT[args.env]   #for expanded cats
          #print("Loaded medwiki ", args.env ," type system with # terms =", len(self.word2id))  Loaded medwiki  0720_3k_full  type system with # terms = 19022
      else:
          print('ERROR: Invalid input... ' + args.goal)
          raise
      self.tokenizer = tokenizer
      self.do_lower = args.do_lower
      self.context_window_size = args.context_window_size
      self.args = args

  def _load_npz(self, path):
      with open(path, 'rb') as f:
          data = np.load(f)
      return data

  def _load_shard(self):
      lines = self.data
      ex_ids = [line["ex_id"] for line in lines]
      mention_word = [line["word"].split() if not self.do_lower
                      else line["word"].lower().split() for line in lines]

      if self.args.goal == "medwiki" and self.args.task not in ["conll_yago", "wlned", "ehr"]:
          left_seq = [line['left_context'].split()[-self.context_window_size:] if not self.do_lower
                      else [w.lower() for w in line['left_context'].split()[-self.context_window_size:]] for line in lines]
          right_seq = [line['right_context'].split()[:self.context_window_size] if not self.do_lower
                       else [w.lower() for w in line['right_context'].split()[:self.context_window_size]] for line in lines]
      else:
          left_seq = [line['left_context'][-self.context_window_size:] if not self.do_lower
                      else [w.lower() for w in line['left_context']][-self.context_window_size:] for line in lines]
          right_seq = [line['right_context'][:self.context_window_size] if not self.do_lower
                       else [w.lower() for w in line['right_context']][:self.context_window_size] for line in lines]

      if self.args.goal == 'medwiki' and self.args.task not in ["conll_yago", "wlned", "ehr"]:
          y_categories = [[c.lower() for c in line['categories']] for line in lines]    #lowercase categories for medwiki
      else:
          y_categories = [line['y_category'] for line in lines]

      y_category_ids = []
      for iid, y_strs in enumerate(y_categories):
          y_category_ids.append([self.word2id[x] for x in y_strs if x in self.word2id])

      # Ddebug
      if False:
          idx = 0
          print('ex_ids: ', ex_ids[idx])
          print('left_seq: ', left_seq[idx])
          print('right_seq: ', right_seq[idx])
          print('mention_word: ', mention_word[idx])
          print('y_category_ids: ', y_category_ids[idx])
          print('y_categories: ', y_categories[idx])

      # print('0) ex_ids:', len(ex_ids), '1) left_seq:', len(left_seq), '2) right_seq:', len(right_seq),
      #      '3) mention_word:', len(mention_word), '4) y_category_ids:', len(y_category_ids))

      # 0: example id, 1: left context, 2: right context, 3: mention word, 4: gold category
      return zip(ex_ids, left_seq, right_seq, mention_word,  y_category_ids)

  def _get_sentence(self):
      ids = self._load_shard()
      #print('ids', list(ids))
      for current_ids in ids:
        yield current_ids

  def get_batch(self, batch_size, max_len, epoch, eval_data=True):
      ind = self.args.goal if self.args.env == "" or self.args.env == "yasu" else self.args.env    #for expanded cats
      #if self.args.goal == 'medwiki':
      #  ind = "0720_3k_full_orig"   #for orig
      #print(ind)

      return get_example(
          self._get_sentence(), batch_size=batch_size, max_len=max_len,
          eval_data=eval_data, tokenizer=self.tokenizer, answer_num=transformer_constant.ANSWER_NUM_DICT[ind]
      )


def get_type_reduction_example(generator, batch_size, max_len, eval_data=False, tokenizer=None, answer_num=60000):
  # [cur_stream elements]
  # 0: example id, 1: left context_pos1, 2: right context_pos1, 3: mention word_pos1,
  #                4: left context_pos2, 5: right context_pos2, 6: mention word_pos2,
  #                7: left context_neg1, 8: right context_neg1, 9: mention word_neg1,
  #                10: left context_neg2, 11: right context_neg2, 12: mention word_neg2

  cur_stream = [None] * batch_size
  no_more_data = False
  while True:
    bsz = batch_size
    seq_length = max_len
    mention_length_limit = 10  # in words, not word pieces
    for i in range(batch_size):
      try:
        cur_stream[i] = list(next(generator))
      except StopIteration:
        no_more_data = True
        bsz = i
        break
    if no_more_data and bsz == 0:
      break
    ex_ids = np.zeros([bsz], np.object)
    inputs_pos1 = []
    inputs_pos2 = []
    inputs_neg1 = []
    inputs_neg2 = []
    sentence_len_wp_pos1 = []
    sentence_len_wp_pos2 = []
    sentence_len_wp_neg1 = []
    sentence_len_wp_neg2 = []
    for i in range(bsz):
      ex_ids[i] = cur_stream[i][0]

      # pos1
      left_seq_pos1 = cur_stream[i][1]
      right_seq_pos1 = cur_stream[i][2]
      mention_seq_pos1 = cur_stream[i][3]
      if len(mention_seq_pos1) > mention_length_limit:
        mention_seq_pos1 = mention_seq_pos1[:mention_length_limit]
      mention_pos1 = ' '.join(mention_seq_pos1)
      context_pos1 = ' '.join(left_seq_pos1 + mention_seq_pos1 + right_seq_pos1)
      inputs_pos1.append([mention_pos1, context_pos1])
      sentence_len_wp_pos1.append(len(tokenizer.encode_plus(mention_pos1, context_pos1)['input_ids']))

      # pos2
      left_seq_pos2 = cur_stream[i][4]
      right_seq_pos2 = cur_stream[i][5]
      mention_seq_pos2 = cur_stream[i][6]
      if len(mention_seq_pos2) > mention_length_limit:
        mention_seq_pos2 = mention_seq_pos2[:mention_length_limit]
      mention_pos2 = ' '.join(mention_seq_pos2)
      context_pos2 = ' '.join(left_seq_pos2 + mention_seq_pos2 + right_seq_pos2)
      inputs_pos2.append([mention_pos2, context_pos2])
      sentence_len_wp_pos2.append(len(tokenizer.encode_plus(mention_pos2, context_pos2)['input_ids']))

      # neg1
      left_seq_neg1 = cur_stream[i][7]
      right_seq_neg1 = cur_stream[i][8]
      mention_seq_neg1 = cur_stream[i][9]
      if len(mention_seq_neg1) > mention_length_limit:
        mention_seq_neg1 = mention_seq_neg1[:mention_length_limit]
      mention_neg1 = ' '.join(mention_seq_neg1)
      context_neg1 = ' '.join(left_seq_neg1 + mention_seq_neg1 + right_seq_neg1)
      inputs_neg1.append([mention_neg1, context_neg1])
      sentence_len_wp_neg1.append(len(tokenizer.encode_plus(mention_neg1, context_neg1)['input_ids']))

      # neg2
      left_seq_neg2 = cur_stream[i][10]
      right_seq_neg2 = cur_stream[i][11]
      mention_seq_neg2 = cur_stream[i][12]
      if len(mention_seq_neg2) > mention_length_limit:
        mention_seq_neg2 = mention_seq_neg2[:mention_length_limit]
      mention_neg2 = ' '.join(mention_seq_neg2)
      context_neg2 = ' '.join(left_seq_neg2 + mention_seq_neg2 + right_seq_neg2)
      inputs_neg2.append([mention_neg2, context_neg2])
      sentence_len_wp_neg2.append(len(tokenizer.encode_plus(mention_neg2, context_neg2)['input_ids']))

    # pos1
    max_len_in_batch_pos1 = max(sentence_len_wp_pos1)
    inputs_pos1 = tokenizer.batch_encode_plus(
      inputs_pos1,
      add_special_tokens=True,
      max_length=min(max_len, max_len_in_batch_pos1),
      truncation_strategy='only_second',
      pad_to_max_length=True,
      return_tensors='pt'
    )

    # pos2
    max_len_in_batch_pos2 = max(sentence_len_wp_pos2)
    inputs_pos2 = tokenizer.batch_encode_plus(
      inputs_pos2,
      add_special_tokens=True,
      max_length=min(max_len, max_len_in_batch_pos2),
      truncation_strategy='only_second',
      pad_to_max_length=True,
      return_tensors='pt'
    )

    # neg1
    max_len_in_batch_neg1 = max(sentence_len_wp_neg1)
    inputs_neg1 = tokenizer.batch_encode_plus(
      inputs_neg1,
      add_special_tokens=True,
      max_length=min(max_len, max_len_in_batch_neg1),
      truncation_strategy='only_second',
      pad_to_max_length=True,
      return_tensors='pt'
    )

    # neg2
    max_len_in_batch_neg2 = max(sentence_len_wp_neg2)
    inputs_neg2 = tokenizer.batch_encode_plus(
      inputs_neg2,
      add_special_tokens=True,
      max_length=min(max_len, max_len_in_batch_neg2),
      truncation_strategy='only_second',
      pad_to_max_length=True,
      return_tensors='pt'
    )

    # print(max_len, max_len_in_batch, inputs["input_ids"].size(), inputs["attention_mask"].size(),
    #       inputs["token_type_ids"].size())
    feed_dict = {
      "ex_ids": ex_ids, "inputs":
      {
        "pos1": inputs_pos1,
        "pos2": inputs_pos2,
        "neg1": inputs_neg1,
        "neg2": inputs_neg2
      }
    }

    if no_more_data:
      if eval_data and bsz > 0:
        yield feed_dict
      break
    yield feed_dict


class DatasetLoaderTypeReduction(object):

  def __init__(self, data, args, tokenizer):
      self.data = data
      if args.goal == '60k':
          self.word2id = transformer_constant.ANS2ID_DICT_60K
      elif args.goal == 'ufet':
          self.word2id = transformer_constant.ANS2ID_DICT_UFET
      else:
          print('ERROR: Invalid input... ' + args.goal)
          raise
      self.tokenizer = tokenizer
      self.do_lower = args.do_lower
      self.context_window_size = args.context_window_size
      self.args = args

  def _load_npz(self, path):
      with open(path, 'rb') as f:
          data = np.load(f)
      return data

  def _load_shard(self):
      lines = self.data
      ex_ids = list(range(len(lines)))

      mention_word_pos1 = [line['pos']['sent1']["word"].split() if not self.do_lower
                      else line['pos']['sent1']["word"].lower().split() for line in lines]
      left_seq_pos1 = [line['pos']['sent1']['left_context'][-self.context_window_size:] if not self.do_lower
                  else [w.lower() for w in line['pos']['sent1']['left_context']][-self.context_window_size:] for line in lines]
      right_seq_pos1 = [line['pos']['sent1']['right_context'][:self.context_window_size] if not self.do_lower
                   else [w.lower() for w in line['pos']['sent1']['right_context']][:self.context_window_size] for line in lines]

      mention_word_pos2 = [line['pos']['sent2']["word"].split() if not self.do_lower
                      else line['pos']['sent2']["word"].lower().split() for line in lines]
      left_seq_pos2 = [line['pos']['sent2']['left_context'][-self.context_window_size:] if not self.do_lower
                  else [w.lower() for w in line['pos']['sent2']['left_context']][-self.context_window_size:] for line in lines]
      right_seq_pos2 = [line['pos']['sent2']['right_context'][:self.context_window_size] if not self.do_lower
                   else [w.lower() for w in line['pos']['sent2']['right_context']][:self.context_window_size] for line in lines]

      mention_word_neg1 = [line['neg']['sent1']["word"].split() if not self.do_lower
                      else line['neg']['sent1']["word"].lower().split() for line in lines]
      left_seq_neg1 = [line['neg']['sent1']['left_context'][-self.context_window_size:] if not self.do_lower
                  else [w.lower() for w in line['neg']['sent1']['left_context']][-self.context_window_size:] for line in lines]
      right_seq_neg1 = [line['neg']['sent1']['right_context'][:self.context_window_size] if not self.do_lower
                   else [w.lower() for w in line['neg']['sent1']['right_context']][:self.context_window_size] for line in lines]

      mention_word_neg2 = [line['neg']['sent2']["word"].split() if not self.do_lower
                      else line['neg']['sent2']["word"].lower().split() for line in lines]
      left_seq_neg2 = [line['neg']['sent2']['left_context'][-self.context_window_size:] if not self.do_lower
                  else [w.lower() for w in line['neg']['sent2']['left_context']][-self.context_window_size:] for line
                       in lines]
      right_seq_neg2 = [line['neg']['sent2']['right_context'][:self.context_window_size] if not self.do_lower
                   else [w.lower() for w in line['neg']['sent2']['right_context']][:self.context_window_size] for
                        line in lines]

      # Ddebug
      if False:
          idx = 0
          print('ex_ids: ', ex_ids[idx])
          print('left_seq: ', left_seq[idx])
          print('right_seq: ', right_seq[idx])
          print('mention_word: ', mention_word[idx])

      print(
        '0) ex_ids:', len(ex_ids),
        '1) left_seq_pos1:', len(left_seq_pos1), '2) right_seq+pos1:', len(right_seq_pos1), '3) mention_word_pos1:',
        len(mention_word_pos1), '4) left_seq_pos2:', len(left_seq_pos2), '5) right_seq_pos2:', len(right_seq_pos2),
        '6) mention_word_pos2:', len(mention_word_pos2), '7) left_seq_neg1:', len(left_seq_neg1), '8) right_seq_neg1:',
        len(right_seq_neg1), '9) mention_word_neg1:', len(mention_word_neg1), '10) left_seq_neg2:', len(left_seq_neg2),
        '11) right_seq_neg2:', len(right_seq_neg2), '12) mention_word_neg2:', len(mention_word_neg2)
      )


      return zip(ex_ids, left_seq_pos1, right_seq_pos1, mention_word_pos1,
                         left_seq_pos2, right_seq_pos2, mention_word_pos2,
                         left_seq_neg1, right_seq_neg1, mention_word_neg1,
                         left_seq_neg2, right_seq_neg2, mention_word_neg2)

  def _get_sentence(self, epoch):
    for i in range(0, epoch):
      ids = self._load_shard()
      #print('ids', list(ids))
      for current_ids in ids:
        yield current_ids

  def get_batch(self, batch_size, max_len, epoch, eval_data=True):
      return get_type_reduction_example(
          self._get_sentence(epoch), batch_size=batch_size, max_len=max_len,
          eval_data=eval_data, tokenizer=self.tokenizer, answer_num=transformer_constant.ANSWER_NUM_DICT[self.args.goal]
      )

def get_type_reduction_LR_example(generator, batch_size, max_len, eval_data=False, tokenizer=None):
  # [cur_stream elements]
  # 0: example id, 1: left context_pos1, 2: right context_pos1, 3: mention word_pos1,
  #                4: left context_pos2, 5: right context_pos2, 6: mention word_pos2,
  #                7: labels

  cur_stream = [None] * batch_size
  no_more_data = False
  while True:
    bsz = batch_size
    seq_length = max_len
    mention_length_limit = 10  # in words, not word pieces
    for i in range(batch_size):
      try:
        cur_stream[i] = list(next(generator))
      except StopIteration:
        no_more_data = True
        bsz = i
        break
    if no_more_data and bsz == 0:
      break
    ex_ids = np.zeros([bsz], np.object)
    targets = np.zeros([bsz], np.float32)
    inputs_sent1 = []
    inputs_sent2 = []
    sentence_len_wp_sent1 = []
    sentence_len_wp_sent2 = []
    for i in range(bsz):
      ex_ids[i] = cur_stream[i][0]

      # sent1
      left_seq_sent1 = cur_stream[i][1]
      right_seq_sent1 = cur_stream[i][2]
      mention_seq_sent1 = cur_stream[i][3]
      if len(mention_seq_sent1) > mention_length_limit:
        mention_seq_sent1 = mention_seq_sent1[:mention_length_limit]
      mention_sent1 = ' '.join(mention_seq_sent1)
      context_sent1 = ' '.join(left_seq_sent1 + mention_seq_sent1 + right_seq_sent1)
      len_after_tokenization1 = len(tokenizer.encode_plus(mention_sent1, context_sent1)['input_ids'])
      if len_after_tokenization1 > max_len:
        overflow_len1 = len_after_tokenization1 - max_len
        context_sent1 = ' '.join(left_seq_sent1 + mention_seq_sent1 + right_seq_sent1[:-overflow_len1])
      inputs_sent1.append([mention_sent1, context_sent1])
      sentence_len_wp_sent1.append(len_after_tokenization1)

      # sent2
      left_seq_sent2 = cur_stream[i][4]
      right_seq_sent2 = cur_stream[i][5]
      mention_seq_sent2 = cur_stream[i][6]
      if len(mention_seq_sent2) > mention_length_limit:
        mention_seq_sent2 = mention_seq_sent2[:mention_length_limit]
      mention_sent2 = ' '.join(mention_seq_sent2)
      context_sent2 = ' '.join(left_seq_sent2 + mention_seq_sent2 + right_seq_sent2)
      len_after_tokenization2 = len(tokenizer.encode_plus(mention_sent2, context_sent2)['input_ids'])
      if len_after_tokenization2 > max_len:
        overflow_len2 = len_after_tokenization2 - max_len
        context_sent2 = ' '.join(left_seq_sent2 + mention_seq_sent2 + right_seq_sent2[:-overflow_len2])
      inputs_sent2.append([mention_sent2, context_sent2])
      sentence_len_wp_sent2.append(len_after_tokenization2)

      # label
      targets[i] = cur_stream[i][7]


    # sent1
    max_len_in_batch_sent1 = max(sentence_len_wp_sent1)
    inputs_sent1 = tokenizer.batch_encode_plus(
      inputs_sent1,
      add_special_tokens=True,
      max_length=min(max_len, max_len_in_batch_sent1),
      truncation_strategy='only_second',
      pad_to_max_length=True,
      return_tensors='pt'
    )

    # sent2
    max_len_in_batch_sent2 = max(sentence_len_wp_sent2)
    inputs_sent2 = tokenizer.batch_encode_plus(
      inputs_sent2,
      add_special_tokens=True,
      max_length=min(max_len, max_len_in_batch_sent2),
      truncation_strategy='only_second',
      pad_to_max_length=True,
      return_tensors='pt'
    )

    # print(max_len, max_len_in_batch, inputs["input_ids"].size(), inputs["attention_mask"].size(),
    #       inputs["token_type_ids"].size())
    feed_dict = {
      "ex_ids": ex_ids,
      "inputs":
        {
          "sent1": inputs_sent1,
          "sent2": inputs_sent2
        },
      "targets": torch.from_numpy(targets)
    }

    if no_more_data:
      if eval_data and bsz > 0:
        yield feed_dict
      break
    yield feed_dict


class DatasetLoaderTypeReductionLR(object):

  def __init__(self, data, args, tokenizer):
      self.data = data
      if args.goal == '60k':
          self.word2id = transformer_constant.ANS2ID_DICT_60K
      elif args.goal == 'ufet':
          self.word2id = transformer_constant.ANS2ID_DICT_UFET
      else:
          print('ERROR: Invalid input... ' + args.goal)
          raise
      self.tokenizer = tokenizer
      self.do_lower = args.do_lower
      self.context_window_size = args.context_window_size
      self.args = args

  def _load_npz(self, path):
      with open(path, 'rb') as f:
          data = np.load(f)
      return data

  def _load_shard(self):
      lines = self.data
      ex_ids = list(range(len(lines)))

      mention_word_sent1 = [line['sent1']["word"].split() if not self.do_lower
                      else line['sent1']["word"].lower().split() for line in lines]
      left_seq_sent1 = [line['sent1']['left_context'][-self.context_window_size:] if not self.do_lower
                  else [w.lower() for w in line['sent1']['left_context']][-self.context_window_size:] for line in lines]
      right_seq_sent1 = [line['sent1']['right_context'][:self.context_window_size] if not self.do_lower
                   else [w.lower() for w in line['sent1']['right_context']][:self.context_window_size] for line in lines]

      mention_word_sent2 = [line['sent2']["word"].split() if not self.do_lower
                      else line['sent2']["word"].lower().split() for line in lines]
      left_seq_sent2 = [line['sent2']['left_context'][-self.context_window_size:] if not self.do_lower
                  else [w.lower() for w in line['sent2']['left_context']][-self.context_window_size:] for line in lines]
      right_seq_sent2 = [line['sent2']['right_context'][:self.context_window_size] if not self.do_lower
                   else [w.lower() for w in line['sent2']['right_context']][:self.context_window_size] for line in lines]

      labels = [int(line['label']) for line in lines]

      # Ddebug
      if False:
        idx = 113
        print('ex_ids: ', ex_ids[idx])
        print('left_seq_sent1: ', left_seq_sent1[idx])
        print('right_seq_sent1: ', right_seq_sent1[idx])
        print('mention_word_sent1: ', mention_word_sent1[idx])
        print('left_seq_sent2: ', left_seq_sent2[idx])
        print('right_seq_sent2: ', right_seq_sent2[idx])
        print('mention_word_sent2: ', mention_word_sent2[idx])
        print('labels:', labels[idx])

      print(
        '0) ex_ids:', len(ex_ids),
        '1) left_seq_sent1:', len(left_seq_sent1), '2) right_seq_sent1:', len(right_seq_sent1), '3) mention_word_sent1:',
        len(mention_word_sent1), '4) left_seq_sent2:', len(left_seq_sent2), '5) right_seq_sent2:', len(right_seq_sent2),
        '6) mention_word_sent2:', len(mention_word_sent2), '7) labels', len(labels)
      )

      return zip(ex_ids, left_seq_sent1, right_seq_sent1, mention_word_sent1,
                 left_seq_sent2, right_seq_sent2, mention_word_sent2, labels)

  def _get_sentence(self, epoch):
    for i in range(0, epoch):
      ids = self._load_shard()
      #print('ids', list(ids))
      for current_ids in ids:
        yield current_ids

  def get_batch(self, batch_size, max_len, epoch, eval_data=True):
      return get_type_reduction_LR_example(
          self._get_sentence(epoch), batch_size=batch_size, max_len=max_len,
          eval_data=eval_data, tokenizer=self.tokenizer
      )


def get_type_reduction_BL4NED_example(generator, batch_size, max_len,
                                 eval_data=False, tokenizer=None):
  # [cur_stream elements]
  # 0: example id, 1: left context_pos1, 2: right context_pos1, 3: mention word_pos1,
  #                4: left context_pos2, 5: right context_pos2, 6: mention word_pos2,
  #                7: labels

  cur_stream = [None] * batch_size
  no_more_data = False
  while True:
    bsz = batch_size
    seq_length = max_len
    mention_length_limit = 10  # in words, not word pieces
    for i in range(batch_size):
      try:
        cur_stream[i] = list(next(generator))
      except StopIteration:
        no_more_data = True
        bsz = i
        break
    if no_more_data and bsz == 0:
      break
    ex_ids = np.zeros([bsz], np.object)
    targets = np.zeros([bsz], np.float32)
    inputs_sent1 = []
    inputs_sent2 = []
    sentence_len_wp_sent1 = []
    sentence_len_wp_sent2 = []
    for i in range(bsz):
      ex_ids[i] = cur_stream[i][0]

      # sent1
      left_seq_sent1 = cur_stream[i][1]
      right_seq_sent1 = cur_stream[i][2]
      mention_seq_sent1 = cur_stream[i][3]
      if len(mention_seq_sent1) > mention_length_limit:
        mention_seq_sent1 = mention_seq_sent1[:mention_length_limit]
      mention_sent1 = ' '.join(mention_seq_sent1)
      context_sent1 = ' '.join(left_seq_sent1 + mention_seq_sent1 + right_seq_sent1)
      len_after_tokenization1 = len(tokenizer.encode_plus(mention_sent1, context_sent1)['input_ids'])
      if len_after_tokenization1 > max_len:
        overflow_len1 = len_after_tokenization1 - max_len
        context_sent1 = ' '.join(left_seq_sent1 + mention_seq_sent1 + right_seq_sent1[:-overflow_len1])
      inputs_sent1.append([mention_sent1, context_sent1])
      sentence_len_wp_sent1.append(len_after_tokenization1)

      # sent2
      left_seq_sent2 = cur_stream[i][4]
      right_seq_sent2 = cur_stream[i][5]
      mention_seq_sent2 = cur_stream[i][6]
      if len(mention_seq_sent2) > mention_length_limit:
        mention_seq_sent2 = mention_seq_sent2[:mention_length_limit]
      mention_sent2 = ' '.join(mention_seq_sent2)
      context_sent2 = ' '.join(left_seq_sent2 + mention_seq_sent2 + right_seq_sent2)
      len_after_tokenization2 = len(tokenizer.encode_plus(mention_sent2, context_sent2)['input_ids'])
      if len_after_tokenization2 > max_len:
        overflow_len2 = len_after_tokenization2 - max_len
        context_sent2 = ' '.join(left_seq_sent2 + mention_seq_sent2 + right_seq_sent2[:-overflow_len2])
      inputs_sent2.append([mention_sent2, context_sent2])
      sentence_len_wp_sent2.append(len_after_tokenization2)

      # label
      targets[i] = cur_stream[i][7]


    # sent1
    max_len_in_batch_sent1 = max(sentence_len_wp_sent1)
    inputs_sent1 = tokenizer.batch_encode_plus(
      inputs_sent1,
      add_special_tokens=True,
      max_length=min(max_len, max_len_in_batch_sent1),
      truncation_strategy='only_second',
      pad_to_max_length=True,
      return_tensors='pt'
    )

    # sent2
    max_len_in_batch_sent2 = max(sentence_len_wp_sent2)
    inputs_sent2 = tokenizer.batch_encode_plus(
      inputs_sent2,
      add_special_tokens=True,
      max_length=min(max_len, max_len_in_batch_sent2),
      truncation_strategy='only_second',
      pad_to_max_length=True,
      return_tensors='pt'
    )

    # print(max_len, max_len_in_batch, inputs["input_ids"].size(), inputs["attention_mask"].size(),
    #       inputs["token_type_ids"].size())
    feed_dict = {
      "ex_ids": ex_ids,
      "inputs":
        {
          "sent1": inputs_sent1,
          "sent2": inputs_sent2
        },
      "targets": torch.from_numpy(targets)
    }

    if no_more_data:
      if eval_data and bsz > 0:
        yield feed_dict
      break
    yield feed_dict


class DatasetLoaderTypeReductionBL4NED(object):

  def __init__(self, data, args, tokenizer):
      self.data = data
      if args.goal == '60k':
          self.word2id = transformer_constant.ANS2ID_DICT_60K
      elif args.goal == 'ufet':
          self.word2id = transformer_constant.ANS2ID_DICT_UFET
      else:
          print('ERROR: Invalid input... ' + args.goal)
          raise
      self.tokenizer = tokenizer
      self.do_lower = args.do_lower
      self.context_window_size = args.context_window_size
      self.args = args

  def _load_npz(self, path):
      with open(path, 'rb') as f:
          data = np.load(f)
      return data

  def _load_shard(self):
      lines = self.data  # jsonlines
      ex_ids = list(range(len(lines)))

      mention_word_sent1 = [line['sent1']["word"].split() if not self.do_lower
                      else line['sent1']["word"].lower().split() for line in lines]
      left_seq_sent1 = [line['sent1']['left_context'][-self.context_window_size:] if not self.do_lower
                  else [w.lower() for w in line['sent1']['left_context']][-self.context_window_size:] for line in lines]
      right_seq_sent1 = [line['sent1']['right_context'][:self.context_window_size] if not self.do_lower
                   else [w.lower() for w in line['sent1']['right_context']][:self.context_window_size] for line in lines]

      mention_word_sent2 = [line['sent2']["word"].split() if not self.do_lower
                      else line['sent2']["word"].lower().split() for line in lines]
      left_seq_sent2 = [line['sent2']['left_context'][-self.context_window_size:] if not self.do_lower
                  else [w.lower() for w in line['sent2']['left_context']][-self.context_window_size:] for line in lines]
      right_seq_sent2 = [line['sent2']['right_context'][:self.context_window_size] if not self.do_lower
                   else [w.lower() for w in line['sent2']['right_context']][:self.context_window_size] for line in lines]

      labels = [int(line['label']) for line in lines]

      # Ddebug
      if False:
        idx = 113
        print('ex_ids: ', ex_ids[idx])
        print('left_seq_sent1: ', left_seq_sent1[idx])
        print('right_seq_sent1: ', right_seq_sent1[idx])
        print('mention_word_sent1: ', mention_word_sent1[idx])
        print('left_seq_sent2: ', left_seq_sent2[idx])
        print('right_seq_sent2: ', right_seq_sent2[idx])
        print('mention_word_sent2: ', mention_word_sent2[idx])
        print('labels:', labels[idx])

      print(
        '0) ex_ids:', len(ex_ids),
        '1) left_seq_sent1:', len(left_seq_sent1), '2) right_seq_sent1:', len(right_seq_sent1), '3) mention_word_sent1:',
        len(mention_word_sent1), '4) left_seq_sent2:', len(left_seq_sent2), '5) right_seq_sent2:', len(right_seq_sent2),
        '6) mention_word_sent2:', len(mention_word_sent2), '7) labels', len(labels)
      )

      return zip(ex_ids, left_seq_sent1, right_seq_sent1, mention_word_sent1,
                 left_seq_sent2, right_seq_sent2, mention_word_sent2, labels)

  def _get_sentence(self, epoch):
    for i in range(0, epoch):
      ids = self._load_shard()
      #print('ids', list(ids))
      for current_ids in ids:
        yield current_ids

  def get_batch(self, batch_size, max_len, epoch, eval_data=True):
      return get_type_reduction_BL4NED_example(
          self._get_sentence(epoch), batch_size=batch_size, max_len=max_len,
          eval_data=eval_data, tokenizer=self.tokenizer
      )


if __name__ == '__main__':

  # TEST
  def get_data_gen(dataname, mode, batch_size, args, tokenizer):
    data_path = transformer_constant.FILE_ROOT + dataname
    dataset = DatasetLoader(data_path, args, tokenizer)
    if mode == 'train':
      data_gen = dataset.get_batch(batch_size, args.max_position_embeddings, args.num_epoch, eval_data=False)
    else:  # test mode
      data_gen = dataset.get_batch(batch_size, 1, args.max_position_embeddings, eval_data=True)
    return data_gen

  def get_all_datasets(args, batch_size, tokenizer):
    train_gen_list = []
    if args.mode in ['train']:
      train_gen_list.append(get_data_gen(args.train_data, 'train', batch_size, args, tokenizer))
    return train_gen_list

  parser = argparse.ArgumentParser()
  parser.add_argument("-model_type", default='bert-large-uncased-whole-word-masking')
  parser.add_argument("-goal", help="category vocab size.", default="60k", choices=["60k", "ufet"])
  parser.add_argument("-mode", help="Whether to train or test", default="train", choices=["train", "test"])
  parser.add_argument("-train_data", help="Train data",
                      default="train/wiki_et_zeroshot_60k_ex_random/train_*.json")
  parser.add_argument("-dev_data", help="Dev data", default="validation/dev_zeroshot_10k.json")
  parser.add_argument("-per_gpu_train_batch_size", help="The batch size per GPU", default=4, type=int)
  parser.add_argument("-num_epoch", help="The number of epoch", default=5000, type=int)
  parser.add_argument("-hidden_dropout_prob", help="Dropout rate", default=.1, type=float)
  args = parser.parse_args()
  args.do_lower = True
  args.avg_pooling = True
  #args.model_type = 'roberta-large'
  device = torch.device("cuda")
  model = TransformerModel(args, transformer_constant.ANSWER_NUM_DICT[args.goal])
  model.to(device)
  args.max_position_embeddings = model.transformer_config.max_position_embeddings

  train_gen_list = get_all_datasets(args, args.per_gpu_train_batch_size, model.transformer_tokenizer)
  batch = next(train_gen_list[0])
  inputs, targets = to_torch(batch, device)
  loss, logits = model(inputs, targets)
  print(loss)
  print(logits)


