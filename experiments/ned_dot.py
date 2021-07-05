import json
import numpy as np
import torch
import sys
import pickle as pkl
import time

from tqdm import tqdm

# Set PATHs
sys.path.insert(0, '../ier_model')

##import model_utils
import run_et
import transformer_constant

#import transformer_data_utils   
import transformer_data_utils_eval   
from models import TransformerModel, TRANSFORMER_MODELS


def get_data_gen(data, args, tokenizer):
  dataset = transformer_data_utils_eval.DatasetLoaderForEntEval(data, args, tokenizer)
  data_gen = dataset.get_batch(args.eval_batch_size, args.max_position_embeddings, 1, eval_data=True)
  return data_gen


def get_datasets(data, args, tokenizer):
  data_gen = get_data_gen(data, args, tokenizer)
  return data_gen


def load_json(path):
  with open(path) as f:
    return [json.loads(l.strip()) for l in tqdm(f)]


def save_json(data, save_to):
  with open(save_to, 'w') as f:
    for d in data:
      json.dump(d, f)
      f.write('\n')


def transform(v):
  v = np.log(v) + 15.
  v[v < 0.] = 0.
  v /= v.max()
  return v


def is_int(s):
  try:
    int(s)
    return True
  except:
    return False


def format_conll_yago(example, is_desc, ex_id, cand_id=None):
    if is_desc:
        description = example[2].split()
        title = example[1]
        pop_prior = float(example[0])
        if not description:
            description = ['empty', 'description', '.']
        d_start_loc = 0
        d_end_loc = 1
        sidx = 1
        if [w.lower() for w in description[:]] == ['this', 'is', 'a', 'list', 'of']:
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
        #print(d_mention, description[:10])
        ex_json = {
            'ex_id': str(ex_id) + '_ent_' + str(cand_id),
            'left_context': d_left_contex,
            'right_context': d_right_context,
            'mention_as_list': d_mention,
            'word': ' '.join(d_mention),
            'title': title,
            'pop_prior': pop_prior,
            'y_category': []
        }
    else:
        start_loc = int(example[0])
        end_loc = int(example[1])
        sentence = example[2].split()
        title = example[3]
        mention = sentence[start_loc:end_loc]
        left_contex = sentence[:start_loc]
        right_context = sentence[end_loc:]
        ex_json = {
            'ex_id': str(ex_id) + '_mic',
            'left_context': left_contex,
            'right_context': right_context,
            'mention_as_list': mention,
            'word': ' '.join(mention),
            'title': title,
            'y_category': []
        }
    return ex_json


def load_conll_data(path):
  #15^I16^ICRICKET - LEICESTERSHIRE TAKE OVER AT TOP AFTER INNINGS VICTORY . LONDON 1996-08-30 CRICKET - LEICESTERSHIRE TAKE OVER AT TOP AFTER INNINGS VICTORY .^ILEICESTERSHIRE^I0.0|||Leicestershire_County_Cricket_Club|||leicestershire county cricket club is one of eighteen first-class county clubs within the domestic cricket structure of england and wales . it represents the historic county of leicestershire . it has also been representative of the county of rutland . the club 's limited overs team is called the leicestershire foxes . founded in 1879 , the club had minor county status until 1894 when it was promoted to first-class status pending its entry into the county championship in 1895. since then , leicestershire have played in every top-level domestic cricket competition in england . the club is based at^I1|||Leicestershire|||leicestershire ( ( listen ) ; postal abbreviation leics . ) is a landlocked county in the english midlands . the county borders nottinghamshire to the north , lincolnshire to the north-east , rutland to the east , northamptonshire to the south-east , warwickshire to the south-west , staffordshire to the west , and derbyshire to the north-west . the border with most of warwickshire is watling street ( the a5 ) . leicestershire takes its name from the city of leicester ( unitary authority ) located at its centre and administered separately from the rest of the county .$
  examples = []
  with open (path) as f:
    for i, line in enumerate(f):
      line = line.strip().split('\t')
      context = line[:4]
      candidates = [c.split('|||') for c in line[4:]]
      context_json = format_conll_yago(context, False, i)
      ex_json = [context_json]
      for j, cand in enumerate(candidates):
        ex_json.append(format_conll_yago(cand, True, i, cand_id=j))
      examples.append(ex_json)
  return examples


def load_wlned_data(path):
  # {  "context": {
  #       "ex_id": "3_232493", "wikiId": 12627, "y_title": "Gospel", 
  #       "left_context": ["endured", "great", "pain", "on", "the", "cross", ".", "a", "he", "died", "for", "thea"], 
  #       "right_context": ["a", "brings", "about", "true", "freedom", ",", "then", "his", "message", "brings", "pastors"], 
  #       "mention_as_list": ["gospel"], "word": "gospel", 
  #       "y_category": ["gospels", "2nd-century christian texts", "christian terminology", "greek-language books", "1st-century christian texts", "christian genres"]
  #        }, 
  #     "candidates": [
  #       {"wikiId": 12627, "title": "Gospel", "score": 0.8954545454545455, 
  #          "description": {"ex_id": 0, "left_context": [], "right_context": ["originally", "meant", "the", "Christian", "message", "building"], "mention_as_list": ["Gospel"], "word": "Gospel", "y_category": []}}, 
  #       {"wikiId": 167391, "title": "Gospel music", "score": 0.09181818181818181, 
  #          "description": {"ex_id": 0, "left_context": [], "right_context": ["is", "a", "genre", "of", "Christian", "music", ".", "as"], "mention_as_list": ["Gospel", "music"], "word": "Gospel music", "y_category": []}}, 
  #       {"wikiId": 2261497, "title": "The gospel", "score": 0.012727272727272728, 
  #         "description": {"ex_id": 0, "left_context": ["In", "Christianity", ","], "right_context": ["(", ";", ";", ")", ",", "or", "the", "Good."], "mention_as_list": ["the", "gospel"], "word": "the gospel", "y_category": []}}
  #       ]
  # }
  examples = []
  with open (path) as f:
    for line in f:
      ex = json.loads(line.strip())
      ex_json = [ex['context']]
      for cand in ex['candidates']:
        desc = cand['description']
        desc['wikiId'] = cand['wikiId']
        desc['title'] = cand['title']
        desc['score'] = cand['score']
        ex_json.append(desc)
      examples.append(ex_json)
  return examples

def load_ehr_data(path):
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
  examples = []
  err = 0
  debug = False
  cnt = 0
  found_trues = 0
  with open (path) as f:
    for line in f:
      ex = json.loads(line.strip())
      ex_dic = { k:ex[k] for k in ex if k not in ['candidates', 'categories']}
      ex_dic["mention_as_list"] = str(ex_dic["word"]).split(" ")
      ex_dic['y_category'] = ex['categories']
      ex_json = [ex_dic]
      
      # Important [ Correct Candidate must be first in list ] 
      true_ent = ex_dic['wikiurl'].lower().strip()
      found_true_in_cands = 0
      for cand in ex['candidates']:
        if len(cand) < 5:
          print("Current cand has less than 5 items ", cand, "\n found in ", ex)
          err += 1
        elif str(cand[0]).lower().strip() == true_ent and found_true_in_cands == 0:
          found_true_in_cands = 1
          found_trues += 1
          desc = {}
          desc["word"] = str(cand[0])
          desc["mention_as_list"] = str(cand[0]).split(" ")
          desc['score'] = float(cand[1])
          desc["left_context"] = cand[2]
          desc["right_context"] = cand[3]
          desc['wikiId'] = cand[4]   
          desc['ex_id'] = "ex_"+str(cand[4])+"_"+str(cnt)   
          desc["y_category"] = []
          ex_json.append(desc)

      if found_true_in_cands == 0:
        print("Didn't find true ent: ",true_ent," in candidates for ", [ str(cand[0]).lower().strip() for cand in ex['candidates'] ])
        continue
      else:
        for cand in ex['candidates']:      # link, score, left context, right context, wikiId   (0 - 4)
          if len(cand) < 5:
            #print("Current cand has less than 5 items ", cand, "\n found in ", ex)
            err += 1
          elif str(cand[0]).lower().strip() != true_ent:
            desc = {}
            desc["word"] = str(cand[0])
            desc["mention_as_list"] = str(cand[0]).split(" ")
            desc['score'] = float(cand[1])
            desc["left_context"] = cand[2]
            desc["right_context"] = cand[3]
            desc['wikiId'] = cand[4]   
            desc['ex_id'] = "ex_"+str(cand[4])+"_"+str(cnt)   
            desc["y_category"] = []
            ex_json.append(desc)
  
        if debug:
          print("EX:",ex)
          print("EX DIC:",ex_dic)
          print("EX_JSON:",ex_json)
          import sys; sys.exit(0)
  
        examples.append(ex_json)
        cnt += 1

  print("Found ",err," bad candidates in ",cnt," successfully processed examples")   # Found  2033  bad candidates in  7600  successfully processed examples
  print("Found true ent in candidates for ",found_trues," examples")                 # Found true ent in candidates for  8307  examples
  return examples


def get_topk_types(prob, args, k=100):
  id2ans = transformer_constant.ID2ANS_MEDWIKI_DICT[args.env]
  sorted_prob = sorted( [(p, id2ans[i]) for i, p in enumerate(prob)], key=lambda x: x[0], reverse=True)
  return sorted_prob[:k] + [(k, v) for k, v in sorted_prob if k <= 0.]

def make_pred(p1, p2, args, mask=None):
  include_dense = 0
  if type(p1) == list:
    include_dense = 1
    tmp_p1, tmp_p2 = p1, p2
    p1, p1_dense = tmp_p1
    p2, p2_dense = tmp_p2
    mask_dense = np.ones(len(p1_dense))
    p1_dense = np.array(p1_dense) * mask_dense
    p2_dense = np.array(p2_dense) * mask_dense
    enc1_dense, enc2_dense = np.array(p1_dense), np.array(p2_dense)

  if mask is None:
    mask = np.ones(len(p1))
  
  enc1, enc2 = np.array(p1), np.array(p2)
  p1 = np.array(p1) * mask
  p2 = np.array(p2) * mask
  p1xp2 = p1 * p2
  enc1 = transform(enc1) * mask
  enc2 = transform(enc2) * mask
  enc1xenc2 = enc1 * enc2
  score1 = np.sum(enc1xenc2)
  score2 = np.sum(p1xp2)
  sim1 = np.sum(enc1xenc2) / (np.sqrt(np.sum(enc1 ** 2)) * np.sqrt(np.sum(enc2 ** 2)))
  sim2 = np.sum(p1xp2) / (np.sqrt(np.sum(p1 ** 2)) * np.sqrt(np.sum(p2 ** 2)))

  if include_dense == 1:
    score1 += np.sum(enc1_dense * enc2_dense)
    score2 += np.sum(p1_dense * p2_dense)
    sim1 += np.sum(enc1_dense * enc2_dense) / (np.sqrt(np.sum(enc1_dense ** 2)) * np.sqrt(np.sum(enc2_dense ** 2)))
    sim2 += np.sum(p1_dense * p2_dense) / (np.sqrt(np.sum(p1_dense ** 2)) * np.sqrt(np.sum(p2_dense ** 2)))


  ## ADD INTEPRETABILITY TO SCORE2 and SIM2  ( ie, what are the non-zero components on both sides p1 and p2, and the final dot product .. which lead to score/prediction! )
  if args.show_types == 'yes':
    #get ordered list of most salient properties ( name/prob ) for both sides
    p1 = p1.flatten()
    p2 = p2.flatten()
    p1xp2 = p1xp2.flatten()
    p1_types = get_topk_types(p1, args, k=100)
    p2_types = get_topk_types(p2, args, k=100)
    p1xp2_types = get_topk_types(p1xp2, args, k=100)
    print("$$$ P1 types: ", p1_types)
    print("$$$ P2 types: ", p2_types)
    print("$$$ P1xP2 types: ", p1xp2_types)

  return score1, score2, sim1, sim2


def to_torch(batch, device):
  inputs_to_model = {k: v.to(device) for k, v in batch['inputs'].items()}
  return inputs_to_model, batch['ex_ids'], batch['targets']


def main(args, model, desc_model, data_path, saveout = 'conll_yago_bert_large.pred', debug = False):
  print("Processing ", data_path, "and saving preds to ", saveout)

  if args.task == 'conll_yago':
    data = load_conll_data(data_path)
  elif args.task == 'wlned':
    data = load_wlned_data(data_path)
  elif args.task == 'ehr':
    data = load_ehr_data(data_path)

  total_count = 0
  correct_count1 = 0
  correct_count2 = 0
  correct_count3 = 0
  correct_count4 = 0
  correct_pop_prior = 0

  total_errs = 0

  with open(saveout, 'w') as f:
    for d in data:
      data_gen = get_datasets(d, args, model.transformer_tokenizer)
      scores1 = []
      scores2 = []
      sims1 = []
      sims2 = []
      pop_priors = []
      pred_json = {}

      #if debug or total_count == 0:
      #  print(len(scores1),". Looking at ex",d)

      if args.task == 'conll_yago':
        assert 'pop_prior' not in d[0], d[0]  # The first json is context
      else:
        #elif args.task == 'wlned':
        assert 'score' not in d[0], d[0]  # The first json is context

      try:
        pred_json['ex_id'] = int(d[0]['ex_id'].split('_')[0])
      except Exception as e:
        pred_json['ex_id'] = str(d[0]['ex_id']).split('_')[0]
        

      for ex_ in d[1:]:
        if args.task == 'conll_yago':
          pop_priors.append(ex_['pop_prior'])
        else:
          #elif args.task == 'wlned':
          pop_priors.append(ex_['score'])

      sent = ' '.join(d[0]['left_context'] + d[0]['mention_as_list'] + d[0]['right_context']).lower()
      mention = d[0]['word'].lower()

       
      if debug or total_count == 0:
        print("len(d):",len(d))
        #print("Looking at ex",d)
        print("mention:",mention)
        print("sent", sent)
        #print("pred: ", pred_json)
        print("pop prior: ", pop_priors)
        #print(data_gen)

      """
      Looking at ex [
      {'ex_id': 'Q4688873_467', 'word': 'blushing', 'right_context': [], 'left_context': ['Shame', '/', 'Humiliation', '(', 'reaction', 'to', 'failure', '/', 'impulse', 'to', 'review', 'behaviour', ')', '--', 'eyes', 'lowered', ',', 'the', 'head', 'down', 'and', 'averted', ','], 'wikiurl': 'blushing', 'ncandidates': 4, 'mention_as_list': ['blushing'], 'y_category': ['reflexes', 'emotion']}, 
      {'word': 'Blushing', 'mention_as_list': ['Blushing'], 'score': 0.847887, 'left_context': [''], 'right_context': ['is', 'the', 'reddening', 'of', 'a', "person\\'s", 'face', 'due', 'to', 'psychological', 'reasons.', 'It', 'is', 'normally', 'involuntary', 'and', 'triggered', 'by', 'emotional', 'stress', 'associated', 'with', 'passion,', 'embarrassment,', 'shyness,', 'anger,', 'or', 'romantic', 'stimulation.<p>Severe', 'blushing', 'is', 'common', 'in', 'people', 'who', 'suffer', 'social', 'anxiety', 'in', 'which', 'the', 'person', 'experiences', 'extreme', 'and', 'persistent', 'anxiety', 'in', 'social', 'and', 'performance', 'situation.<p><h2>', 'Summary', '</h2><p>Blushing', 'is', 'generally', 'distinguished,', 'despite', 'a', 'close', 'physiological', 'relation,', 'from', 'flushing,', 'which', 'is', 'more', 'intensive', 'and', 'extends', 'over', 'more', 'of', 'the', 'body,', 'and', 'seldom', 'has', 'a', 'mental', 'source.', 'If', 'redness', 'persists', 'for', 'abn'], 'wikiId': '240889', 'ex_id': 'ex_240889_0', 'y_category': []}, 
      {'word': 'Flushing (physiology)', 'mention_as_list': ['Flushing', '(physiology)'], 'score': 0.135211, 'left_context': [''], 'right_context': ['is', 'to', 'become', 'markedly', 'red', 'in', 'the', 'face', 'and', 'often', 'other', 'areas', 'of', 'the', 'skin,', 'from', 'various', 'physiological', 'conditions.', 'Flushing', 'is', 'generally', 'distinguished,', 'despite', 'a', 'close', 'physiological', 'relation', 'between', 'them,', 'from', 'blushing,', 'which', 'is', 'milder,', 'generally', 'restricted', 'to', 'the', 'face,', 'cheeks', 'or', 'ears,', 'and', 'generally', 'assumed', 'to', 'reflect', 'emotional', 'stress,', 'such', 'as', 'embarrassment,', 'anger,', 'or', 'romantic', 'stimulation.', 'Flushing', 'is', 'also', 'a', 'cardinal', 'symptom', 'of', 'carcinoid', 'syndrome\\xe2\\x80\\x94the', 'syndrome', 'that', 'results', 'from', 'hormones', '(often', 'serotonin', 'or', 'histamine)', 'being', 'secreted', 'into', 'systemic', 'circulation.<p><h2>Causes</h2><p><u'], 'wikiId': '439380', 'ex_id': 'ex_439380_0', 'y_category': []}, 
      {'word': 'Boy', 'mention_as_list': ['Boy'], 'score': 0.0028169, 'left_context': ['A'], 'right_context': ['is', 'a', 'young', 'male', 'human.', 'The', 'term', 'is', 'usually', 'used', 'for', 'a', 'child', 'or', 'an', 'adolescent.', 'When', 'a', 'male', 'human', 'reaches', 'adulthood,', 'he', 'is', 'described', 'as', 'a', 'man.<p><h2>Definition,', 'etymology,', 'and', 'use</h2><p>According', 'to', 'the', 'Merriam-Webster', 'Dictionary,', 'a', 'boy', 'is', '"a', 'male', 'child', 'from', 'birth', 'to', 'adulthood".<p>The', 'word', '"boy"', 'comes', 'from', 'Middle', 'English', '<em>boi,', 'boye</em>', '("boy,', 'servant"),', 'related', 'to', 'other', 'Germanic', 'words', 'for', '<em>boy</em>,', 'namely', 'East', 'Frisian', '<em>boi</em>', '("boy,', 'young', 'man")'], 'wikiId': '194253', 'ex_id': 'ex_194253_0', 'y_category': []}, 
      {'word': 'Idiopathic craniofacial erythema', 'mention_as_list': ['Idiopathic', 'craniofacial', 'erythema'], 'score': 0.0140845, 'left_context': [''], 'right_context': ['is', 'a', 'medical', 'condition', 'characterised', 'by', 'severe,', 'uncontrollable,', 'and', 'frequently', 'unprovoked,', 'facial', 'blushing.<p>Blushing', 'can', 'occur', 'at', 'any', 'time', 'and', 'is', 'frequently', 'triggered', 'by', 'even', 'mundane', 'events,', 'such', 'as,', 'talking', 'to', 'friends,', 'paying', 'for', 'goods', 'in', 'a', 'shop,', 'asking', 'for', 'directions', 'or', 'even', 'simply', 'making', 'eye', 'contact', 'with', 'another', 'person.<p>For', 'many', 'years,', 'the', 'cause', 'of', 'the', 'condition', 'was', 'thought', 'to', 'be', 'an', 'anxiety', 'problem,', 'caused', 'by', 'a', 'mental', 'health', 'disorder.', 'However,', 'in', 'recent', 'years', 'experts', 'in', 'the', 'field', 'of', 'the', 'disorder', 'believe'], 'wikiId': '1392763', 'ex_id': 'ex_1392763_0', 'y_category': []}
      ]
      mention: blushing
      sent shame / humiliation ( reaction to failure / impulse to review behaviour ) -- eyes lowered , the head down and averted , blushing
      pred:  {'ex_id': 'Q4688873'}
      pop prior:  [0.847887, 0.135211, 0.0028169, 0.0140845]

      """

      """
      mention: leicestershire
      sent cricket - leicestershire take over at top after innings victory . london 1996-08-30 cricket - leicestershire take over at top after innings victory .
      pred:  {'ex_id': 0}
      pop prior:  [0.0, 1.0]
      <generator object get_example at 0x2ab01cd9fdd0>
      """

      if args.include_dense != "no":
        dense_model, dense_tokenizer = TRANSFORMER_MODELS[args.model_type]
        dense_model.eval()

      for i, ex in enumerate(data_gen):
        #if debug:
          #print(i, "... In enumerate(data_gen)",ex)
          #print("************************",i,"Current Input: ", ex['mention'], ex['context'])

        inputs_to_model, _, _ = to_torch(ex, args.device)
        if i == 0:
          if debug:
            #  print("inputs to model", inputs_to_model)
            print("***",total_count,"--",i,"Current Input: ", ex['mention'], ex['context'])

          _, output_logits = model(inputs_to_model)
          context_probs = model.sigmoid_fn(output_logits[:, :]).data.cpu().clone().numpy()

          if args.include_dense != "no":
            dense_tokens = dense_tokenizer.tokenize(ex['context']) #, return_tensors='pt')   #TypeError: 'BertTokenizer' object is not callable
            indexed_tokens = dense_tokenizer.convert_tokens_to_ids(dense_tokens)  # Convert token to vocabulary indices
            tokens_tensor = torch.tensor([indexed_tokens])
            #tokens_tensor = tokens_tensor.to(args.device)

            #if debug:
              #print("************************",i,"Current Input: ", ex['mention'], ex['context'])
              #print(dense_tokens, len(dense_tokens))
              #print(indexed_tokens, len(indexed_tokens))
              #print(tokens_tensor, tokens_tensor.shape)

            #dense_context_output = dense_model(tokens_tensor)   #RuntimeError: Expected object of device type cuda but got device type cpu for argument #1 'self' in call to _th_index_select
            dense_context_output = dense_model(tokens_tensor)   
            dense_context_emb_reals = dense_context_output[1].data.cpu().clone().numpy()
            dense_context_emb = 1/(1 + np.exp(-1 * dense_context_emb_reals))

            #if debug:
              #print("Dense context output: ", type(dense_context_output), len(dense_context_output))
              #print(type(dense_context_output[0]), dense_context_output[0].shape, type(dense_context_output[1]), dense_context_output[1].shape)

        else:
          _, output_logits = desc_model(inputs_to_model)   
          cand_probs = model.sigmoid_fn(output_logits[:, :]).data.cpu().clone().numpy()

          # Debug: Apply same changes as context vec
          if debug:
            print("XXX",total_count,"--",i, "CANDIDATE: ", ex['mention'], ex['context'])

          if args.include_dense == "no":
            score1, score2, sim1, sim2 = make_pred( context_probs, cand_probs, args, mask=type_mask)
          else:
            ## concat dense pretrained rep to bier probs and predict with those
            #encoded_input = dense_model.transformer_tokenizer.tokenize(ex['context'], return_tensors='pt')
            #dense_cand_output = dense_model(**encoded_input)
            dense_tokens = dense_tokenizer.tokenize(ex['context']) #, return_tensors='pt')   #TypeError: 'BertTokenizer' object is not callable
            indexed_tokens = dense_tokenizer.convert_tokens_to_ids(dense_tokens)  # Convert token to vocabulary indices
            tokens_tensor = torch.tensor([indexed_tokens])
            dense_cand_output = dense_model(tokens_tensor)   
            dense_cand_emb_reals = dense_cand_output[1].data.cpu().clone().numpy()
            dense_cand_emb = 1/(1 + np.exp(-1 * dense_cand_emb_reals))

            """
            if debug:
              print("Dense Section with") 
              print("Inputs:",type(ex['inputs']), ex['inputs'])
              print("Dense context emb:",type(dense_context_emb), dense_context_emb.shape)
              print("Dense cand emb:",type(dense_cand_emb), dense_cand_emb.shape)
              print("BIER mention_context output:",type(context_probs), context_probs.shape)
              print("BIER rightt side output:",type(cand_probs), context_probs.shape)
              # Dense context emb: <class 'torch.Tensor'> torch.Size([1, 768])
              # Dense cand emb: <class 'torch.Tensor'> torch.Size([1, 768])
              # BIER mention_context output: <class 'numpy.ndarray'> (1, 19022)
              # BIER rightt side output: <class 'numpy.ndarray'> (1, 19022)
            """

            new_context_probs = np.concatenate((context_probs, dense_context_emb),1)
            new_cand_probs = np.concatenate((cand_probs, dense_cand_emb),1)

            #if debug:
              #print("New Context", new_context_probs.shape, "New Cand:", new_context_probs.shape)

            if args.include_dense == "concat":
              score1, score2, sim1, sim2 = make_pred( new_context_probs, new_cand_probs, args, mask=type_mask)    
            elif args.include_dense == "dotdot":
              score1, score2, sim1, sim2 = make_pred( [context_probs, dense_context_emb_reals], [cand_probs, dense_cand_emb_reals], args, mask=type_mask)

          scores1.append(score1)
          scores2.append(score2)
          sims1.append(sim1)
          sims2.append(sim2)

      if debug:
        print("Done WITH MENT/CONTXT/CANDS: ",i)
        print("log dot (1):",scores1)
        print("raw dot (2):",scores2)
        print("log cos (3)",sims1)
        print("raw cos (4):",sims2)
        print("pop prior: ", pop_priors)
      

      assert len(scores1) == len(scores2) == len(pop_priors), (len(scores1), len(scores2), len(pop_priors))
      try:
        pred1 = np.argmax(scores1)
        pred2 = np.argmax(scores2)
        pred3 = np.argmax(sims1)
        pred4 = np.argmax(sims2)
        pop_prior_guess = np.argmax(pop_priors)
        pred_json['prob_pred'] = int(pred2)
        pred_json['prob_scores'] = list([float(x) for x in scores2])
        pred_json['logprob_pred'] = int(pred1)
        pred_json['logprob_scores'] = list([float(x) for x in scores1])
        pred_json['prob_pred_cos'] = int(pred4)
        pred_json['prob_scores_cos'] = list([float(x) for x in sims2])
        pred_json['logprob_pred_cos'] = int(pred3)
        pred_json['logprob_scores_cos'] = list([float(x) for x in sims1])
        pred_json['pop_prior_pred'] = int(pop_prior_guess)
        pred_json['pop_prior_scores'] = list([float(x) for x in pop_priors])

        if int(pred1) == 0:
          correct_count1 += 1.
        if int(pred2) == 0:
          correct_count2 += 1.
        if int(pred3) == 0:
          correct_count3 += 1.
        if int(pred4) == 0:
          correct_count4 += 1.
        if int(pop_prior_guess) == 0:
          correct_pop_prior += 1.
        total_count += 1.
        if total_count % 500 == 0:
          print(total_count)
  
        if debug:
          print("Pred_Json", pred_json)
          print(d)
  
        if args.task != "ehr":
          print("Pred:",total_count,", Mention:", mention, "|| Sent:", sent, "|| Candidates:", [ c['title'] if 'title' in c else c['y_title'] for c in d ],"|| Pred1:",int(pred1),", Pred2: ",int(pred2),", Pred3: ",int(pred3),", Pred4: ",int(pred4),", PopPrior: ",int(pop_prior_guess))
        else:
          if all(['wikiurl' in c for c in d]): 
            outcs = [c['wikiurl'] for c in d]
          elif type(d[0]) == list:
            outcs = [c[0] for c in d]
          else:
            outcs = [c['word'] for c in d]
          print("Pred:",total_count,", Mention:", mention, "|| Sent:", sent, "|| Candidates:", outcs,"|| Pred1:",int(pred1),", Pred2: ",int(pred2),", Pred3: ",int(pred3),", Pred4: ",int(pred4),", PopPrior: ",int(pop_prior_guess))
        json.dump(pred_json, f)
        f.write('\n')
      except Exception as e:
        print("Error ",e," on example",ex)
        total_errs +=1 

  print("Using include_dense: ", args.include_dense, "and datafile: ", args.datafile)
  print('Acc -- log-prob: {:.4f}, raw-prob: {:.4f}, log-prob-cos: {:.4f}, raw-prob-cos: {:.4f}'.format(
    correct_count1 / total_count,
    correct_count2 / total_count,
    correct_count3 / total_count,
    correct_count4 / total_count
  ))
  print('Popular Prior: {:.4f}'.format(correct_pop_prior / total_count))
  print('# {} examples, total count {}, total errs{}'.format(len(data), total_count, total_errs))


def load_and_save_conll_data(path):
  examples = []
  with open (path) as f:
    i = 0
    for line in f:
      line = line.strip().split('\t')
      context = line[:4]
      candidates = [c.split('|||') for c in line[4:]]
      context_json = format_conll_yago(context, False, i)
      ex_json = {'ex_id': i, 'context': context_json}
      cand_json = []
      for j, cand in enumerate(candidates):
        cand_json.append(format_conll_yago(cand, True, i, cand_id=j))
      ex_json['candidates'] = cand_json
      examples.append(ex_json)
      i += 1
  save_to = path.rstrip('.txt') + '.json'
  save_json(examples, save_to)


if __name__ == "__main__":

  device = torch.device("cuda")
  run_et.parser.add_argument("-task", default=None, type=str, choices=[ 'cap', 'crep', 'ehr', 'ert', 'wikisrs', 'kore', 'conll_yago', 'rare', 'wlned'], help="The task name: cpa / crep / ert")
  run_et.parser.add_argument("-datafile", default=None, type=str, help="The datafile to process for eval purposes.  Usually hardset, but can be overwritten")
  run_et.parser.add_argument("-include_dense", default="no", type=str, choices=['no','concat', 'dotdot'], help="When doing dot product, use none, concated dense base model rep or dot dot approach to sparse IER rep")
  run_et.parser.add_argument("-show_types", default="no", type=str, choices=['no','yes'], help="Show interpretable components used during prediction?")
  args = run_et.parser.parse_args()

  # Setup args
  args.mode = 'test'

  # Now force passing of env and goal
  if args.model_type == "":
    args.model_type = 'bert-large-uncased-whole-word-masking'

  if args.reload_model_name == "":
    args.reload_model_name = 'bert_large_wwm_2_best'

  args.load = True
  args.eval_batch_size = 1

  args.do_lower = True if 'uncased' in args.model_type else False
  args.device = device

  args.max_position_embeddings = 512 if 'bert' in args.model_type else 514

  print('-' * 80)
  for k, v in vars(args).items():
    print(k, ':', v)
  print('-' * 80)

  args.exp_root = transformer_constant.get(args.env, 'EXP_ROOT')
  #############################
  # Load a typing model
  print('== ENT MODEL ==')
  st = time.time()
  if args.goal == "medwiki":
    model = TransformerModel(args, transformer_constant.ANSWER_NUM_DICT[args.env])    
  else:
    model = TransformerModel(args, transformer_constant.ANSWER_NUM_DICT[args.goal])
  model.to(device)
  model.eval()

  print("Model statedict size:", len(model.state_dict()))
  bio_ier_model_id = ""
  print(args.exp_root, args.reload_model_name)
  run_et.load_model(args.reload_model_name, args.exp_root, bio_ier_model_id, model)    
  print("loaded in ", time.time() - st)


  print('== DESC MODEL ==')
  st = time.time()
  #if args.goal == "medwiki":
  #  desc_model = TransformerModel(args, transformer_constant.ANSWER_NUM_DICT[args.env])
  #else:

  #Need to retrain the desc model if i'm not using the 60k goal
  if '_bert_' in args.model_type:
    args.model_type = 'bert-large-uncased-whole-word-masking'                        #WHY is this hardcoded?
  
  if args.goal == "medwiki":
    desc_model = TransformerModel(args, transformer_constant.ANSWER_NUM_DICT[args.env])   
  else:
    desc_model = TransformerModel(args, transformer_constant.ANSWER_NUM_DICT[args.goal])

  desc_model.to(device)
  desc_model.eval()
  # /EntEval/wiki_desc/  | wiki_desc data location  # entity descriptions for 2million  #<-- how many of these are we covering do i need to remake/retrain this model?
  
  print(args.reload_model_name_desc)
  run_et.load_model(args.reload_model_name_desc, args.exp_root, bio_ier_model_id, desc_model)
  print("loaded in ", time.time() - st)

  mask_path = '' # Path to your trained weights. This should be a numpy array. 
  print('Load mask: {}'.format(mask_path))

  if mask_path == '':
    type_mask = None
    #type_mask = np.array([1 if i < 5000 else 0 for i in range(60000)])
  else:
    type_mask = np.load(mask_path)

  #print('mask: {} types'.format(sum(type_mask)))
  print('type_mask:', type_mask)

  if args.task == 'conll_yago':
    datafile = './EntEval/conll-yago/testa.final.1st.txt'
    if args.datafile != None:
      datafile = args.datafile
    

    if args.model_id:
      saveout = args.model_id
    else:
      saveout = '0803conll.pred'

    st = time.time()
    print("Running Main on file:", datafile)
    main(args, model, desc_model, datafile, saveout, debug=False)
    print("Elapsed Time: ", time.time() - st)

  elif args.task == 'wlned':
    datafile = './EntEval/WLNED/dev.json'
    if args.datafile != None:
      datafile = args.datafile

    if args.model_id:
      saveout = args.model_id
    else:
      saveout = '0803wlned.pred'

    st = time.time()
    print("Running Main on file:", datafile)
    main(args, model, desc_model, datafile, saveout, debug=True)
    print("Elapsed Time: ", time.time() - st)
    
  elif args.task == 'ehr':
    basefile = transformer_constant.get("0720_600k_full_orig", "FILE_ROOT")
    datafile = basefile + "test/wiki_desc/test_ehr_examples_between_1_and_6_cands_v2_overlaping_with_wikidesc0720_3k_full_with_cand_desc.json"

    if args.model_id:
      saveout = args.model_id
    else:
      saveout = "0805ehr"+ args.env +".pred"

    st = time.time()
    print("Running Main on file:", datafile)
    main(args, model, desc_model, datafile, saveout, debug=True)
    print("Elapsed Time: ", time.time() - st)
  else:
    print('Invalid task.')
