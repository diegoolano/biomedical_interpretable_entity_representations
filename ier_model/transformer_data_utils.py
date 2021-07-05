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

#import warnings
#warnings.filterwarnings("error")

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
        try:
          targets[i, answer_ind] = 1.0
        except Exception as e:
          print("Index out of range", e )
          targets[i, answer_ind - 1] = 1.0

    max_len_in_batch = max(sentence_len_wp)

    try: 
      #print("Max Len in batch, ",max_len_in_batch,", Inputs len: ",[ (len(i[0]), len(i[1])) for i in inputs ]) 
      inputs = tokenizer.batch_encode_plus(
        inputs,
        add_special_tokens=True,
        max_length=min(max_len, max_len_in_batch),
        truncation_strategy='only_second',
        pad_to_max_length=True,
        return_tensors='pt'
      )
      #print("Post tokenizer.batch_encolde_plus", max_len, max_len_in_batch, inputs["input_ids"].size(), inputs["attention_mask"].size(), inputs["token_type_ids"].size())
    except Warning as e:
      print("In get_example: Warning token size error", e)
      continue
    except Exception as e:
      print("In get_example: General error", e)
      continue

    targets = torch.from_numpy(targets)
    feed_dict = {"ex_ids": ex_ids, "inputs": inputs, "targets": targets}

    if no_more_data:
      if eval_data and bsz > 0:
        yield feed_dict
      break
    yield feed_dict


class DatasetLoader(object):

  def __init__(self, filepattern, args, tokenizer):
      print("DSL calling dataset loader with filepattern", filepattern)
      self._all_shards = [file for file in glob.glob(filepattern) if 'allwikipp_wiki' not in file]
      shuffle(self._all_shards)

      print('Found %d shards at %s in DSL' % (len(self._all_shards), filepattern))
      print("Load dict for task: ", args.goal)
      if args.goal == '60k':
          self.word2id = transformer_constant.ANS2ID_DICT_60K
      elif args.goal == 'ufet':
          self.word2id = transformer_constant.ANS2ID_DICT_UFET
      elif args.goal == 'medwiki':
          self.word2id = transformer_constant.ANS2ID_MEDWIKI_DICT[args.env]
          #self.word2id = transformer_constant.get_medwiki(ANS2ID_DICT_MEDWIKI
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
      print('Loading {} line in class DSL'.format(shard_name))
      with open(shard_name) as f:
        lines = [json.loads(line.strip()) for line in tqdm(f)]
        ex_ids = [line["ex_id"] for line in lines]
        mention_word = [line["word"].split() if not self.do_lower
                        else line["word"].lower().split() for line in lines]

        if self.args.goal == "medwiki" and 'wiki_desc' not in self.args.model_id:
            left_seq = [line['left_context'].split()[-self.context_window_size:] if not self.do_lower
                        else [w.lower() for w in line['left_context'].split()[-self.context_window_size:]] for line in lines]
            right_seq = [line['right_context'].split()[:self.context_window_size] if not self.do_lower
                         else [w.lower() for w in line['right_context'].split()[:self.context_window_size]] for line in lines]
        else:
            left_seq = [line['left_context'][-self.context_window_size:] if not self.do_lower
                        else [w.lower() for w in line['left_context']][-self.context_window_size:] for line in lines]
            right_seq = [line['right_context'][:self.context_window_size] if not self.do_lower
                         else [w.lower() for w in line['right_context']][:self.context_window_size] for line in lines]

        if self.args.goal == 'medwiki'and 'wiki_desc' not in self.args.model_id:  
            if 'orig' not in self.args.env and self.args.env != "0720_600k_full":
                y_categories = [[c.lower() for c in line['exp_categories']] for line in lines]    #lowercase categories for medwiki
            else:
                y_categories = [[c.lower() for c in line['categories']] for line in lines]    #600k orig 

        elif self.args.goal == 'medwiki'and 'wiki_desc' in self.args.model_id:
            y_categories = [[c.lower() for c in line['y_categories']] for line in lines]    #lowercase categories for medwiki
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
      ind = self.args.goal if self.args.env == "" else self.args.env
      return get_example(
          self._get_sentence(epoch, eval_data), batch_size=batch_size, max_len=max_len,
          eval_data=eval_data, tokenizer=self.tokenizer, answer_num=transformer_constant.ANSWER_NUM_DICT[ind]
      )


class DatasetLoaderForEntEval(object):

  def __init__(self, data, args, tokenizer):
      self.data = data
      print("Set Data in DatasetLoaderForEntEval ", data)
      if args.goal == '60k':
          self.word2id = transformer_constant.ANS2ID_DICT_60K
      elif args.goal == 'ufet':
          self.word2id = transformer_constant.ANS2ID_DICT_UFET
      elif args.goal == 'medwiki':
          self.word2id = transformer_constant.ANS2ID_MEDWIKI_DICT[args.env]
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
      left_seq = [line['left_context'][-self.context_window_size:] if not self.do_lower
                  else [w.lower() for w in line['left_context']][-self.context_window_size:] for line in lines]
      right_seq = [line['right_context'][:self.context_window_size] if not self.do_lower
                   else [w.lower() for w in line['right_context']][:self.context_window_size] for line in lines]

      if self.args.goal == 'medwiki':
          if 'exp_categories' in lines[0]:
              y_categories = [line['exp_categories'] for line in lines]
          else:
              y_categories = [line['categories'] for line in lines]
      else:
          y_categories = [line['y_category'] for line in lines]

      y_category_ids = []
      for iid, y_strs in enumerate(y_categories):
          y_category_ids.append([self.word2id[x] for x in y_strs if x in self.word2id])

      # Debug
      if 1 == 1:
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
      for current_ids in ids:
        yield current_ids

  def get_batch(self, batch_size, max_len, epoch, eval_data=True):
      ind = self.args.goal if self.args.env == "" else self.args.env
      return get_example(
          self._get_sentence(), batch_size=batch_size, max_len=max_len,
          eval_data=eval_data, tokenizer=self.tokenizer, 
          answer_num=transformer_constant.ANSWER_NUM_DICT[ind]
      )



if __name__ == '__main__':

  # TEST
  def get_data_gen(dataname, mode, batch_size, args, tokenizer):
    #data_path = transformer_constant.FILE_ROOT + dataname
    data_path = transformer_constant.get(args.env, 'FILE_ROOT') + dataname
    dataset = DatasetLoader(data_path, args, tokenizer)
    if mode == 'train':
      data_gen = dataset.get_batch(batch_size, args.max_position_embeddings, args.num_epoch, eval_data=False)
    else:  # test mode
      data_gen = dataset.get_batch(batch_size, 1, args.max_position_embeddings, eval_data=True)
    return data_gen

  def get_all_datasets(args, batch_size, tokenizer):
    train_gen_list = []
    if args.mode in ['train']:
      train_gen_list.append(get_data_gen(transformer_constant.get(args.env, 'TRAIN_DATA'), 'train', batch_size, args, tokenizer))
      #train_gen_list.append(get_data_gen(args.train_data, 'train', batch_size, args, tokenizer))
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


