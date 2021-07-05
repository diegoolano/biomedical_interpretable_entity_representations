def load_vocab_dict(vocab_file_name, vocab_max_size=None, start_vocab_count=None, common_vocab_file_name=None):
  with open(vocab_file_name) as f:
    text = [x.strip() for x in f.readlines()]
    if vocab_max_size:
      text = text[:vocab_max_size]
    if common_vocab_file_name:
        print('==> adding common training set types')
        print('==> before:', len(text))
        with open(common_vocab_file_name, 'r') as fc:
            common = [x.strip() for x in fc.readlines()]
        print('==> common:', len(common))
        text = list(set(text + common))
        print('==> after:', len(text))
    if start_vocab_count:
      file_content = dict(zip(text, range(0 + start_vocab_count, len(text) + start_vocab_count)))
    else:
      file_content = dict(zip(text, range(0, len(text))))
  return file_content


CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
PAD_TOKEN = '[PAD]'


def get(env,var):
	dataroot = '/home/diego/biomed_fall20/data/'

        if env == "0720_600k_full_orig":
		#37,607,135 examples and type system
		FILE_ROOT = dataroot + 'entity_type_systems/0721_600k_v7/'   		 
		EXP_ROOT = dataroot + 'entity_type_systems/0721_600k_v7/model/'         
		TRAIN_DATA = "train/training_*.json"
		DEV_DATA = "validation/0721_600k_all_validation_examples.json"
		EVAL_DATA = "test/0721_600k_all_test_examples.json"
		WIKI_TRAIN_DATA = "train/wiki_desc_full_orig/0830_examples_0720_600k_full_orig*.json"
		WIKI_DEV_DATA = "validation/wiki_desc_full_orig/0830_examples_0720_600k_full_orig_docs_dev.json"


	get_dict = {'FILE_ROOT': FILE_ROOT, 'EXP_ROOT': EXP_ROOT, 
		    'TRAIN_DATA': TRAIN_DATA, 'DEV_DATA': DEV_DATA, 'EVAL_DATA': EVAL_DATA, 
		    'WIKI_TRAIN_DATA': WIKI_TRAIN_DATA, 'WIKI_DEV_DATA': WIKI_DEV_DATA}

	return get_dict[var]

BASE_PATH = '/home/diego/biomed_fall20/biomed_ier/ier_model'
ONTOLOGY_DICT = { '0720_600k_full_orig': BASE_PATH +"/ontology/medwiki_types_721_600k_full.txt", }              
ONTOLOGY_DICT_FILTERED = { '0720_600k_full_orig': BASE_PATH +"/ontology/medwiki_types_721_600k_full_filtered.txt", }              

#BIOMED DICTS
ANS2ID_DICT_MEDWIKI_720_600ko = load_vocab_dict(ONTOLOGY_DICT["0720_600k_full_orig"])   # 68,304
TYPES_2_MENTIONS = {
	'0720_600k_full': "0721_600k_all_training_examples_with_concepts_and_categories_uniq_mentions_links.tsv",
}              

ANSWER_NUM_DICT = { '0720_600k_full_orig': len(ANS2ID_DICT_MEDWIKI_720_600ko), }              
ANS2ID_MEDWIKI_DICT = { '0720_600k_full_orig': ANS2ID_DICT_MEDWIKI_720_600ko, }              

ID2ANS_DICT_MEDWIKI_720_600ko = {v: k for k, v in ANS2ID_DICT_MEDWIKI_720_600ko.items()}
ID2ANS_MEDWIKI_DICT = {
	'0720_600k_full_orig': ID2ANS_DICT_MEDWIKI_720_600ko,
}              
