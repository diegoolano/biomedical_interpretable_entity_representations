# all the params are hard set in ned_dot for now

export PY_PATH="/home/diego/biomed_fall20/biomed_env/bin/"
export PROJ_BASE="/home/diego/biomed_fall20/biomed_ier/experiments/ehr_baselines/"
export LOG_BASE="/home/diego/biomed_fall20/biomed_ier/experiments/log_ehr/"

export TASK="ehr"                    # conll_yago or wlned or ehr
export MODELTYPE="bert-base-uncased"        # bert-base-uncased, monologg/biobert_v1.1_pubmed, allenai/scibert_scivocab_uncased, microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

export SAVEOUT="0824_${TASK}_${MODELTYPE}_ehr_baseline"


launch_cmd="CUDA_VISIBLE_DEVICES=2 ${PY_PATH}python -u ${PROJ_BASE}bert_baseline.py -task ${TASK} -model_name ${MODELTYPE}"
outfile_f="${LOG_BASE}${SAVEOUT}"

echo ""
echo "Launching job and saving logs to $outfile_f"
echo $launch_cmd
