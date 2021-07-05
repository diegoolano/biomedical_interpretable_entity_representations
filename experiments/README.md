# Downstream Experiments for evaluation of Biomed IERs


## 1.  Named Entity Disambiguation using EHR Med Wiki dataset

See Colab for how to get BIER numbers [ [here](https://colab.research.google.com/drive/1_OOJ97GDfgZhB-K98B3P3-5ZJygXHazj) ] recommended.

## 1b. or run models locally ( older version )
<pre>
- We need to first train our Wiki Description model 
   CALL: . ../ier_model/train_wiki_desc.sh
   The bash script contains variables you may set
   
   Additionally, you'll need to set the path for the following in ../ier_model/transformer_constant.py
	WIKI_TRAIN_DATA = "train/wiki_desc/0804_examples_0720_3k_full_docs*.json"
	WIKI_DEV_DATA = "validation/0804_examples_0720_3k_full_docs_dev_1k.json"   


- Now with our EHR training data and WikiDesc model trained we can run our trained Biomed IER model on the data
   CALL: . run_ned_dot.py

   In the above you'll need to set the following variables depending on the task and entity type set you are using..

	export ENTSET="0720_600k_full"           			  # 
	export IERMODEL="0722_e3_v100_run_bert_large_7d_best"             # IERMODEL checkpoint to use ( BIER-PubMedBERT, BIER-BioBERT, BIER-SciBERT check point without .pt )
	export WIKIDESCMODEL="0804_bert_large_3k_full_wiki_desc_t2_best"  # WIKIDESCMODEL checkpoint to use ( trainined on WikiDescription data described in prior section ) 
	export SAVEOUT="0805_${TASK}_biomed_nv100_${ENTSET}"              # SAVEOUT where to save model predictions 
	export LOGFILE="0805_e1_${TASK}_biomed_nv100_${ENTSET}"           # LOGFILE

   Additionally for the ehr task, you need to set the training/validation/test data in ned_dot.py

</pre>
## 1.3 For baselines ( PubMEDBert, BioBERT, SciBERT locally)
To run code, go to ehr_baselines/, edit and run run_bert_baseline.sh with the baseline model of your choice and paths as needed.
       

## 2. Entity Label Classification on Cancer Genetics Data using BIER
See Colab available at: [ [Colab url](https://colab.research.google.com/drive/1CDwTG71UkTKLxMhk7uDm4DHX2YABYbEf) ]

