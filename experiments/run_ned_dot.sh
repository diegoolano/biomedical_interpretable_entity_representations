# all the params are hard set in ned_dot for now

export PY_PATH="/home/diego/biomed_fall20/biomed_env/bin/"
export PROJ_BASE="/home/diego/biomed_fall20/biomed_ier/experiments/"
export LOG_BASE="log_ehr/"


#1.  CHOOSE EXPERIMENT

##################Yasu
#export TASK="ehr"                              # conll_yago or wlned or ehr
#export ENTSET="yasu"                           # yasu, 0720_3k_full, 0720_3k_full_orig or 0720_3k_drugs
#export IERMODEL="bert_large_wwm_2_best"        # for yasu: bert_large_wwm_2_best,   for 0720_3k_full_orig ( pre-expand ): 0722_e3_v100_run_bert_large_7d_best
#export WIKIDESCMODEL="bert_large_wwm_4_best"   # for yasu: bert_large_wwm_4_best    for 0720_3k_full_orig ( pre-expand ): 0804_bert_large_3k_full_wiki_desc_t2_best
#export GOAL="60k"                              # 60k for yasu  and medwiki for all else


##################3k Orig
export TASK="ehr"                                                  # conll_yago or wlned or ehr
export ENTSET="0720_3k_full_orig"                                  # yasu, 0720_3k_full, 0720_3k_full_orig or 0720_3k_drugs
#export IERMODEL="0722_e3_v100_run_bert_large_7d_best"              # for yasu: bert_large_wwm_2_best, for 0720_3k_full_orig: 0722_e3_v100_run_bert_large_7d_best       , for 0720_3k_full: 0809_e3_v100_run_biobert_7d_3k_full_filter_expanded
#export IERMODEL="0722_e3_v100_run_bert_base_7d_best"              # for yasu: bert_large_wwm_2_best, for 0720_3k_full_orig: 0722_e3_v100_run_bert_large_7d_best       , for 0720_3k_full: 0809_e3_v100_run_biobert_7d_3k_full_filter_expanded
export IERMODEL="0722_e3_v100_run_biobert_base_7d_best"
#export WIKIDESCMODEL="0804_bert_large_3k_full_wiki_desc_t2_best"   # for yasu: bert_large_wwm_4_best  for 0720_3k_full_orig: 0804_bert_large_3k_full_wiki_desc_t2_best , for 0720_3k_full: 0809_bert_large_3k_full_wiki_desc_expanded
#export WIKIDESCMODEL="0807_bert_large_3k_full_wiki_desc_best"   # for yasu: bert_large_wwm_4_best  for 0720_3k_full_orig: 0804_bert_large_3k_full_wiki_desc_t2_best , for 0720_3k_full: 0809_bert_large_3k_full_wiki_desc_expanded
export WIKIDESCMODEL="0808_biobert_3k_full_wiki_desc_v4_best"
export GOAL="medwiki"                                              # 60k for yasu  and medwiki for all else


##################3k Drugs
#export TASK="ehr"                                       
#export ENTSET="0720_3k_drugs"                             
#export IERMODEL="0726_e2_v100_run_bertbase_7d_3k_drugs2_best"
#export IERMODEL="0726_e2_v100_run_biobert_base_7d_3k_drugs2_best"
#export WIKIDESCMODEL="0804_bert_large_3k_drugs_wiki_desc_best"   
#export GOAL="medwiki"                                     


##################3k expanded
#export TASK="ehr"                                                               # conll_yago or wlned or ehr
#export ENTSET="0720_3k_full"                                                    # yasu, 0720_3k_full, 0720_3k_full_orig or 0720_3k_drugs
#export IERMODEL="0809_e2_v100_run_bert_large_7d_3k_full_filter_expanded_best"   # for 0720_3k_full: 0809_e3_v100_run_biobert_7d_3k_full_filter_expanded_best OR 0809_e2_v100_run_bert_large_7d_3k_full_filter_expanded_best
#export IERMODEL="0809_e3_v100_run_biobert_7d_3k_full_filter_expanded_best"    #above was giving me issues so trying this which works but gives terrible results
#export IERMODEL="0810_e2_v100_run_scibert_7d_3k_full_filter_expanded_best"     # no scibert wikidesc yet
#export WIKIDESCMODEL="0809_bert_large_3k_full_wiki_desc_expanded_best"          # for yasu: bert_large_wwm_4_best  for 0720_3k_full_orig: 0804_bert_large_3k_full_wiki_desc_t2_best , for 0720_3k_full: 0809_bert_large_3k_full_wiki_desc_expanded
#export WIKIDESCMODEL="0808_biobert_3k_full_wiki_desc_v4_best"    #see if this is the right size
#export GOAL="medwiki"                                                          # 60k for yasu  and medwiki for all else


# TRYING 3k expanded
#1ERROR: size mismatch for encoder.embeddings.word_embeddings.weight: copying a param with shape torch.Size([30522, 768]) from checkpoint, the shape in current model is torch.Size([28996, 768]).
#   - this was due to using IER model that was scibert and using MODELTYPE biobert

#2ERROR RuntimeError: Error(s) in loading state_dict for TransformerModel: 
#    size mismatch for encoder.embeddings.word_embeddings.weight: copying a param with shape torch.Size([30522, 768]) from checkpoint, the shape in current model is torch.Size([31090, 768]).
#   - this was due to IERmodel and WIKIDESC MODEL BEING DIFFERENT! ( ie i used scibert for ier, and bert_larget for wiki )   

#3ERROR RuntimeError: Error(s) in loading state_dict for TransformerModel:
#	size mismatch for encoder.embeddings.word_embeddings.weight: copying a param with shape torch.Size([28996, 768]) from checkpoint, the shape in current model is torch.Size([31090, 768]).
#    -  IERMODEL="0809_e3_v100_run_biobert_7d_3k_full_filter_expanded_best"  and WIKIDESCMODEL="0808_biobert_3k_full_wiki_desc_v4_best"  ... modeltype was incorrectly set to scibert

#4ERROR RuntimeError: Error(s) in loading state_dict for TransformerModel:
#  size mismatch for classifier.linear.weight: copying a param with shape torch.Size([19022, 768]) from checkpoint, the shape in current model is torch.Size([24431, 768]).
#    -  same as three, except using biobert for modeltype and desc_model  .. the issue is that the IER model has 24431, but the WIKI is not expanded and has 19022
#       -- 0722_e3_v100_run_biobert_base_7d_best.pt  as ier model is orig
# instead go with bert everwhere

#5ERROR  using all berts gives RuntimeError: storage has wrong size: expected -4856915331333032005 got 1024

#6ERROR 	size mismatch for classifier.linear.weight: copying a param with shape torch.Size([24431, 1024]) from checkpoint, the shape in current model is torch.Size([24431, 768]).
#  export IERMODEL="0809_e3_v100_run_biobert_7d_3k_full_filter_expanded_best"    #above was giving me issues so trying this which works but gives terrible results
#  export WIKIDESCMODEL="0809_bert_large_3k_full_wiki_desc_expanded_best"
#  they are of the same vocab size, but different lengths!

# NEED TO START TRAINING WIKIDESC MODELS!
#7 DO 3K ( not extended ) with biobert


##################600k
#export TASK="ehr"
#export ENTSET="0720_600k_full_orig"
#export IERMODEL="0727_e3_v100_run_biobert_base_7d_600k_full_best"
#export WIKIDESCMODEL="0809_bert_large_3k_full_wiki_desc_expanded_best"          #TODO 
#export GOAL="medwiki"                                                          

##################600k drugs
#export TASK="ehr"
#export ENTSET="0720_600k_durgs"
#export IERMODEL="0727_e3_v100_run_biobert_base_7d_600k_full_best"
#export WIKIDESCMODEL="0809_bert_large_3k_full_wiki_desc_expanded_best"          #TODO 
#export GOAL="medwiki"                                                          


#TODO 600k expanded


##### CHOOSE MODELTYPE

#export MODELTYPE="bert-large-uncased-whole-word-masking"     
#export MODELTYPE="bert-base-uncased"
#export MODELTYPE="roberta-base"
#export MODELTYPE="allenai/biomed_roberta_base"
export MODELTYPE="monologg/biobert_v1.1_pubmed"                                 #<-- 600k model
#export MODELTYPE="allenai/scibert_scivocab_uncased"
#TODO add blurb

# if empty use defaults set in ned_dot.py otherwise evaluate on file provided
#export DATAFILE="-datafile /home/diego/biomed_fall20/check_ehr_data/test_ehr_examples_between_1_and_6_cands_v2_overlaping_with_wikidesc0720_3k_full_with_cand_desc_60pt.json"
#export DATAFILE="-datafile /home/diego/biomed_fall20/check_ehr_data/test_ehr_examples_between_1_and_6_cands_v2_overlaping_with_wikidesc0720_3k_full_with_cand_desc_60pt_3candsup.json"
#export DATAFILE="-datafile /home/diego/biomed_fall20/check_ehr_data/test_ehr_examples_between_1_and_6_cands_v2_overlaping_with_wikidesc0720_3k_full_with_cand_desc_60pt_3candsup_no_incompletes_dupes.json"
#export DATAFILE="-datafile /home/diego/biomed_fall20/check_ehr_data/test_ehr_examples_0824_sub10.json"
export DATAFILE="-datafile /home/diego/biomed_fall20/check_ehr_data/test_ehr_examples_0824.json"
#export DATAFILE="-datafile /home/diego/biomed_fall20/data/entity_type_systems/0720lg2_v6/test/wiki_desc/test_ehr_examples_between_1_and_6_cands_v2_overlaping_with_wikidesc0720_3k_full_with_cand_desc.json"

export INCLUDE_DENSE="no"          # no, concat, dotdot
export SHOW_TYPES="yes"            # no, yes

export BASEOUT="1022_e1_test_ehr_examples_0824_include_dense_${INCLUDE_DENSE}_full"
export SAVEOUT="${BASEOUT}_${TASK}_${ENTSET}_ier_${IERMODEL}_wikidesc_${WIKIDESCMODEL}"
export LOGFILE="${BASEOUT}_${TASK}_${ENTSET}_ier_${IERMODEL}_wikidesc_${WIKIDESCMODEL}"

#next compare 1017_e1_test_ehr_examples_0824_sub10_include_dense to 1015 sub with no dense
#and compare 1015 all no dense with 1017 all with dense

outfile_f="${PROJ_BASE}${LOG_BASE}${LOGFILE}"

launch_cmd="CUDA_VISIBLE_DEVICES=1 ${PY_PATH}python -u ${PROJ_BASE}ned_dot.py -task ${TASK} -env ${ENTSET} -reload_model_name ${IERMODEL} -reload_model_name_desc ${WIKIDESCMODEL} -goal ${GOAL} -model_id ${SAVEOUT} -model_type ${MODELTYPE} -include_dense ${INCLUDE_DENSE} -show_types ${SHOW_TYPES} ${DATAFILE} | tee ${outfile_f}"

echo ""
#echo "Launching job and saving logs to $outfile_f"
echo $launch_cmd
