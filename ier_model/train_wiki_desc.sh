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
#export MODELTYPE="allenai/scibert_scivocab_uncased"
export MODELTYPE="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

#2. SELECT DATA SET / TYPE SYSTEM
# "data sub for medwiki", choices=["0720_3k_full","0720_3k_drugs","0720_600k_full","0720_600k_drugs"])
#export MEDWIKI_ENV="0720_600k_full_orig"   
export MEDWIKI_ENV="0720_600k_full"   

#3. SELECT OUTPUT NAME
export OUTHANDLE="1105_e3_blurb_600k_full_wiki_desc_expanded"


export LOG_PERIOD=100000
export TRAIN_ASSESS=100000
export EVAL_ASSESS=1000000
export SAVE_MODEL_PER=200000

launch_cmd="CUDA_VISIBLE_DEVICES=1 CUDA_CACHE_PATH='/home/diego/.cache/' ${PY_PATH}python3 -u ${SCRIPT_PATH}run_et.py \
-model_id ${OUTHANDLE} \
-model_type ${MODELTYPE} \
-mode train \
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
-save_period ${SAVE_MODEL_PER}
| tee log/${OUTHANDLE}.log"

outfile_f="${OUTPUT_PATH}${OUTHANDLE}"

#echo "Launching job and saving results to $OUTPUT_PATH and outfiles $outfile_f"
echo $launch_cmd
