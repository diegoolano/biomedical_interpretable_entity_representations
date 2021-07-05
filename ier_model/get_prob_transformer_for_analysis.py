import torch
import run_et
import transformer_constant
import transformer_data_utils
from models import TransformerModel


def get_data_gen(data, args, tokenizer):
  #print("Get data with args", args)
  dataset = transformer_data_utils.DatasetLoaderForEntEval(data, args, tokenizer)
  data_gen = dataset.get_batch(
    args.eval_batch_size, args.max_position_embeddings, 1, eval_data=True
  )
  return data_gen


def get_datasets(data, args, tokenizer):
  data_gen = get_data_gen(data, args, tokenizer)
  return data_gen


def write_example(ex, types, _f):
  sentence = ' '.join(ex['left_context'] + ['*['] + ex['mention_as_list'] +
                      [']*'] + ex['right_context'])
  _f.write('\n\nTitle: {}\n\n'.format(ex['title']))
  _f.write('\n\n{}\n\n'.format(sentence))
  for t in types:
    _f.write('{:>100} : {:.4f}\n'.format(t[1], t[0]))
  _f.write('\n')


def get_topk_types(prob, k=100):
  #assert len(prob) == len(transformer_constant.ID2ANS_DICT_60K), (len(prob),len(transformer_constant.ID2ANS_DICT_60K))
  assert len(prob) == len(transformer_constant.ID2ANS_DICT_UFET), (len(prob),len(transformer_constant.ID2ANS_DICT_UFET))
  sorted_prob = sorted(
    [(p, transformer_constant.ID2ANS_DICT_60K[i]) for i, p in enumerate(prob)],
    key=lambda x: x[0],
    reverse=True
  )
  return sorted_prob[:k] + [(k, v) for k, v in sorted_prob if k <= 0.]


def to_torch(batch, device):
  inputs_to_model = {k: v.to(device) for k, v in batch['inputs'].items()}
  return inputs_to_model, batch['ex_ids'], batch['targets']


def main(data, args, save_to):
  print(data)
  data_gen = get_datasets(data, args, model.transformer_tokenizer)
  ex = data[0]
  with open(save_to, 'w') as f:
    for batch_num, batch_np in enumerate(data_gen):  # this loop actually go only once
      eval_batch, ex_ids, targets = to_torch(batch_np, device)
      _, output_logits = model(eval_batch)
      probs = model.sigmoid_fn(output_logits[:, :]).data.cpu().clone().numpy()
      types = get_topk_types(probs[0])
      write_example(ex, types, f)


if __name__ == "__main__":

  device = torch.device("cuda")

  run_et.parser.add_argument("-task", default=None, type=str, choices=[
    'cap', 'crep', 'ert', 'wikisrs', 'kore', 'conll_yago', 'rare', 'wlned'],
                             help="The task name: cpa / crep / ert")

  args = run_et.parser.parse_args()

  # Setup args
  args.model_type = 'bert-large-uncased-whole-word-masking'
  args.mode = 'test'

  # Decide which Type System and Model to Use (wiki '60k' or 'ufet' 10k )
  #args.goal = '60k'
  #args.reload_model_name = 'bert_large_wwm_2_best'

  args.goal = 'ufet'
  args.reload_model_name = 'bert_large_wwm_ec_6k_best' 


  args.load = True
  args.eval_batch_size = 1

  args.do_lower = True if 'uncased' in args.model_type else False
  args.device = device
  args.max_position_embeddings = 512 if 'bert' in args.model_type else 514

  print('-' * 80)
  for k, v in vars(args).items():
    print(k, ':', v)
  print('-' * 80)

  #############################
  # Load a typing model
  print('== ENT MODEL ==')
  model = TransformerModel(args, transformer_constant.ANSWER_NUM_DICT[args.goal])
  model.to(device)
  model.eval()
  run_et.load_model(args.reload_model_name, transformer_constant.EXP_ROOT, args.model_id, model)

  """
  data = [
    {
      'ex_id': '0',
      'title': 'Hinton',
      'mention_as_list': ['Hinton'],
      'word': 'Hinton',
      'left_context': [],
      'right_context': ', one of the pioneers of deep learning , spoke about '
                       'the limits of CNNs at the AAAI conference .'.split(),
      'y_category': []

    }
  ]
  """
  #line numbers from input.txt

  # 1.  Glesatinib is a dual inhibitor of c-Met and SMO that is under phase II clinical trial for non-small cell lung cancer.     #sample1.txt
  right_ctxt = 'is a dual inhibitor of c-Met and SMO that is under phase II clinical trial for non-small cell lung cancer.'     #sample1.txt
  sample1 = { 'ex_id': '0', 'title': 'Glesatinib', 'mention_as_list': ['Glesatinib'], 'word': 'Glesatinib', 'left_context': [], 'right_context': right_ctxt.split(), 'y_category': [] }

  # 2. Indibulin, ....
  right_ctxt = ', a synthetic inhibitor of tubulin assembly, has shown promising anticancer activity with a minimal neurotoxicity in preclinical animal studies and in Phase I clinical trials for cancer chemotherapy.'   #example2 sentence
  sample2 = { 'ex_id': '0', 'title': 'Indibulin', 'mention_as_list': ['Indibulin'], 'word': 'Indibulin', 'left_context': [], 'right_context': right_ctxt.split(), 'y_category': [] }

  # 2. The data indicated a possibility that indibulin may reduce chemotherapy-induced peripheral neuropathy in cancer patients.
  left_ctxt = 'The data indicated a possibility that'
  right_ctxt = 'may reduce chemotherapy-induced peripheral neuropathy in cancer patients.'
  sample3 = { 'ex_id': '0', 'title': 'indibulin', 'mention_as_list': ['indibulin'], 'word': 'indibulin', 'left_context': left_ctxt.split(), 'right_context': right_ctxt.split(), 'y_category': [] }

  # 3. Skin field cancerization (SFC) is a process that occurs in areas of the skin that have undergone genomic alterations induced by ultraviolet radiation. Actinic keratosis (AK) is a sign of its activity. To evaluate the effectiveness and safety of 0·5% colchicine (COL) cream vs methyl aminolaevulinate photodynamic therapy (MAL-PDT) in the treatment of AK and SFC, we conducted a randomized, open, intrasubject controlled trial. 
  left_ctx = 'Skin field cancerization (SFC) is a process that occurs in areas of the skin that have undergone genomic alterations induced by ultraviolet radiation. Actinic keratosis (AK) is a sign of its activity. To evaluate the effectiveness and safety of 0·5% colchicine (COL) cream vs'
  right_ctx = 'in the treatment of AK and SFC, we conducted a randomized, open, intrasubject controlled trial.'
  sample4 = { 'ex_id': '0', 'title': 'methyl aminolaevulinate photodynamic therapy (MAL-PDT)', 'mention_as_list': ['methyl aminolaevulinate photodynamic therapy (MAL-PDT)'], 'word': 'methyl aminolaevulinate photodynamic therapy (MAL-PDT)', 'left_context': left_ctxt.split(), 'right_context': right_ctxt.split(), 'y_category': [] }

  # 6. Colchicine, a natural product of Colchicum autumnae currently used for gout treatment, is a tubulin targeting compound which inhibits microtubule formation by targeting fast dividing cells. This tubulin-targeting property has lead researchers to investigate the potential of colchicine and analogs as possible cancer therapies.
  right_ctxt = ', a natural product of Colchicum autumnae currently used for gout treatment, is a tubulin targeting compound which inhibits microtubule formation by targeting fast dividing cells. This tubulin-targeting property has lead researchers to investigate the potential of colchicine and analogs as possible cancer therapies.'
  sample5 = { 'ex_id': '0', 'title': 'Colchicine', 'mention_as_list': ['Colchicine'], 'word': 'Colchicine', 'left_context': [], 'right_context': right_ctxt.split(), 'y_category': [] }

  # 7. After treatment of leukemia and tuberculosis, Colchicine and enalapril therapy resulted in an improvement of clinical symptoms.
  left_ctx = 'After treatment of leukemia and tuberculosis,'
  right_ctx = 'and enalapril therapy resulted in an improvement of clinical symptoms.'
  sample6 = { 'ex_id': '0', 'title': 'Colchicine', 'mention_as_list': ['Colchicine'], 'word': 'Colchicine', 'left_context': left_ctxt.split(), 'right_context': right_ctxt.split(), 'y_category': [] }

  # 8. In both tumor models, the high-dose group of ENMD-1198 showed antitumor activity equivalent to that of cyclophosphamide. ENMD-1198 was selected as the lead molecule in this analogue series and is currently in a phase I clinical trial in patients with refractory solid tumors.
  left_ctx = 'In both tumor models, the high-dose group of'
  right_ctx = 'showed antitumor activity equivalent to that of cyclophosphamide. ENMD-1198 was selected as the lead molecule in this analogue series and is currently in a phase I clinical trial in patients with refractory solid tumors.'
  sample7 = { 'ex_id': '0', 'title': 'ENMD-1198', 'mention_as_list': ['ENMD-1198'], 'word': 'ENMD-1198', 'left_context': left_ctxt.split(), 'right_context': right_ctxt.split(), 'y_category': [] }

  # 9. Evaluation of ABT-751 against childhood cancer models in vivo. ABT-751 is a novel antimitotic agent that binds tubulin at the colchicine binding site. ABT-751 is undergoing Phase I trials in children, but has not been evaluated against a range of pediatric tumor models in vivo. ABT-751 was evaluated against 27 subcutaneously implanted xenograft models of childhood cancer including neuroblastoma [4], osteosarcoma [4], Ewing sarcoma [2] rhabdomyosarcoma [8], medulloblastoma [1] and eight kidney cancer lines (six Wilms tumors, two rhabdoid). ABT-751 was administered at 100 mg/kg P.O. on a schedule of 5 days on, 5 days off, 5 days on, repeating the cycle at 21 days. Tumor diameters were measured at 7 day intervals for a period of 12 weeks. Three measures of antitumor activity were used: (1) clinical response criteria [e.g., partial response (PR), complete response (CR), etc.]; (2) treated to control (T/C) tumor volume at day 21; and (3) a time to event measure based on the median event free survival (EFS) of treated and control lines. ABT-751 induced regression in 4 of 25 models (16%) including models of neuroblastoma that are refractory to vincristine and paclitaxel. Other regressions occurred in rhabdomyosarcoma and Wilms tumor models. ABT-751 significantly increased event free survival (EFS > 2.0) in eight models (33%) in addition to those with objective responses. ABT-751 demonstrated intermediate activity against this tumor panel. Neuroblastoma models appear somewhat more sensitive to this agent, with objective regressions also in rhabdomyosarcoma and Wilms tumor. ABT-751 was also active in several tumor lines intrinsically refractory to vincristine or paclitaxel.
  left_ctx = 'Evaluation of ABT-751 against childhood cancer models in vivo.'
  right_ctx = 'is a novel antimitotic agent that binds tubulin at the colchicine binding site. ABT-751 is undergoing Phase I trials in children, but has not been evaluated against a range of pediatric tumor models in vivo. ABT-751 was evaluated against 27 subcutaneously implanted xenograft models of childhood cancer including neuroblastoma [4], osteosarcoma [4], Ewing sarcoma [2] rhabdomyosarcoma [8], medulloblastoma [1] and eight kidney cancer lines (six Wilms tumors, two rhabdoid). ABT-751 was administered at 100 mg/kg P.O. on a schedule of 5 days on, 5 days off, 5 days on, repeating the cycle at 21 days. Tumor diameters were measured at 7 day intervals for a period of 12 weeks. Three measures of antitumor activity were used: (1) clinical response criteria [e.g., partial response (PR), complete response (CR), etc.]; (2) treated to control (T/C) tumor volume at day 21; and (3) a time to event measure based on the median event free survival (EFS) of treated and control lines. ABT-751 induced regression in 4 of 25 models (16%) including models of neuroblastoma that are refractory to vincristine and paclitaxel. Other regressions occurred in rhabdomyosarcoma and Wilms tumor models. ABT-751 significantly increased event free survival (EFS > 2.0) in eight models (33%) in addition to those with objective responses. ABT-751 demonstrated intermediate activity against this tumor panel. Neuroblastoma models appear somewhat more sensitive to this agent, with objective regressions also in rhabdomyosarcoma and Wilms tumor. ABT-751 was also active in several tumor lines intrinsically refractory to vincristine or paclitaxel.'
  sample8 = { 'ex_id': '0', 'title': 'ABT-751', 'mention_as_list': ['ABT-751'], 'word': 'ABT-751', 'left_context': left_ctxt.split(), 'right_context': right_ctxt.split(), 'y_category': [] }


  #9  using above with less much less context
  right_ctxt = 'is a novel antimitotic agent that binds tubulin at the colchicine binding site. ABT-751 is undergoing Phase I trials in children, but has not been evaluated against a range of pediatric tumor models in vivo.'
  sample9 = { 'ex_id': '0', 'title': 'ABT-751', 'mention_as_list': ['ABT-751'], 'word': 'ABT-751', 'left_context': [], 'right_context': right_ctxt.split(), 'y_category': [] }
  

  main([sample1], args, './ufet_sample1_out.txt')
  main([sample2], args, './ufet_sample2_out.txt')
  main([sample3], args, './ufet_sample3_out.txt')
  main([sample4], args, './ufet_sample4_out.txt')
  main([sample5], args, './ufet_sample5_out.txt')
  main([sample6], args, './ufet_sample6_out.txt')
  main([sample7], args, './ufet_sample7_out.txt')
  main([sample8], args, './ufet_sample8_out.txt')
  main([sample9], args, './ufet_sample9_out.txt')
