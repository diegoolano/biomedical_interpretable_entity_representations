#!/usr/bin/env bash

export PY_PATH="/home/diego/biomed_fall20/biomed_env/bin/"
export PROJ_BASE="/home/diego/biomed_fall20/"
export OUTPUT_PATH="${PROJ_BASE}data/model_out/"
export SCRIPT_PATH="${PROJ_BASE}biomed_ier/ier_model/"

#1. SELECT MODEL BASE
#export MODELTYPE="bert-large-uncased-whole-word-masking"
#export MODELTYPE="bert-base-uncased"
#export MODELTYPE="roberta-base"
#export MODELTYPE="allenai/biomed_roberta_base"
#export MODELTYPE="monologg/biobert_v1.1_pubmed"
export MODELTYPE="allenai/scibert_scivocab_uncased"
#export MODELTYPE="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

#2.  SELECT DATASET / TYPE SYSTEM  
# "data sub for medwiki", choices=["0720_3k_full","0720_3k_full_orig","0720_3k_drugs","0720_600k_full","0720_600k_full_orig","0720_600k_drugs"])
export MEDWIKI_ENV="0720_600k_full"   


#3.  SELECT OUTPUT NAME for metric result storage
#export OUTHANDLE="0726_e1_v100_run_bert_base_7d_test_results_k2"
#export OUTHANDLE="0923_e2_run_0720_600k_full_allenai_scibert_scivocab_uncased.results"
export OUTHANDLE="0923_e1_run_0720_600k_full_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext.results"

#IMPORTANT the model ckpts path is set in transformer_constant.py 

#4.  SELECT MODEL CHECKPOINT TO LOAD AND USE FOR TESTING (with no .pt at end )
#export SAVEDMODEL="0718_e1_v100_run_bert_large_word_3d_best"  
#export SAVEDMODEL="0923_e2_run_0720_600k_full_allenai_scibert_scivocab_uncased"   #no .pt
export SAVEDMODEL="0923_e1_run_0720_600k_full_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

export LOG_PERIOD=100
export TRAIN_ASSESS=100
export EVAL_ASSESS=5000
export SAVE_MODEL_PER=100000

export GPU=0

launch_cmd="CUDA_VISIBLE_DEVICES=${GPU} CUDA_CACHE_PATH='/home/diego/.cache/' ${PY_PATH}python3 -u ${SCRIPT_PATH}run_et.py \
-model_id ${OUTHANDLE} \
-model_type ${MODELTYPE} \
-load \
-reload_model_name ${SAVEDMODEL} \
-mode val \
-examples_limit 2000 \
-goal medwiki \
-env ${MEDWIKI_ENV} \
-learning_rate_enc 2e-5 \
-learning_rate_cls 1e-3 \
-per_gpu_train_batch_size 8 \
-per_gpu_eval_batch_size 8 \
-gradient_accumulation_steps 4 \
-log_period ${LOG_PERIOD} \
-eval_period ${TRAIN_ASSESS} \
-eval_after ${EVAL_ASSESS} \
-save_period ${SAVE_MODEL_PER} \
| tee log/${OUTHANDLE}.log"

outfile_f="${OUTPUT_PATH}${OUTHANDLE}"

#echo "Launching testing job and saving results to $OUTPUT_PATH and outfiles $outfile_f"
echo $launch_cmd
