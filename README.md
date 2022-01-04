# Biomedical Interpretable Entity Representations

> [**Biomedical Interpretable Entity Representations**](https://arxiv.org/abs/2106.09502)<br/>
> Diego Garcia-Olano, Yasumasa Onoe, Ioana Baldini, Joydeep Ghosh, Byron Wallace, Kush Varshney<br/>
> Findings of ACL 2021

- Paper: [ [https://aclanthology.org/2021.findings-acl.311/](https://aclanthology.org/2021.findings-acl.311/) ]
- ACL slides: [ [pdf](https://github.com/diegoolano/biomedical_interpretable_entity_representations/blob/main/docs/BIERs%20-%20ACL-IJCNLP%202021%20presentation.pdf) ]

``` bibtex
@inproceedings{garcia-olano-etal-2021-biomedical,
    title = "Biomedical Interpretable Entity Representations",
    author = "Garcia-Olano, Diego  and
      Onoe, Yasumasa  and
      Baldini, Ioana  and
      Ghosh, Joydeep  and
      Wallace, Byron  and      
      Varshney, Kush",
    booktitle = "Findings of the 59th Annual Meeting of the Association for Computational Linguistics",
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```

To use pre-trained models without re-training BIERS, see colab notebooks in "Replicating downstream tasks" section at bottom.

### Installing Dependencies
```bash
$ git clone https://github.com/diegoolano/biomedical_interpretable_entity_representations.git
$ virtualenv --python=~/envs/py37/bin/python biomed_env
$ source biomed_env/bin/activate
$ pip install -r requirements.txt 
```

### How to train BioMed IER models
See ier\_model/train.sh  
<pre>
   Make sure to: 
   - set goal to "medwiki", 
   - set training and dev sets, 
   - set paths in transformers\_constants.py appropriately, 
   - make sure to use a GPU with a lot of memory ( ie v100 has 32GB) or lower the batch size.
   - set the intervals on which you'd like to get training acc, eval acc on dev, etc
   - set log location
</pre>

### BIER training data and best models
   BIER triples can be found [ [here](https://drive.google.com/drive/folders/18crQtSPVLNlVqiBEKV8qgObHAcCvMXXn?usp=sharing) ]
   
   Model files:
   - BIER-PubMedBERT: [ [model ckpt](https://drive.google.com/uc?id=1-3b2VRkGOkyoBGCm68_4d5m0aZGwQB0J) ]
   - BIER-SciBERT: [ [model ckpt](https://drive.google.com/uc?id=1-FSaNYQ17T8yhyLgCUZJFNuTrmcpbdLo) ]
   - BIER-BioBERT: [ [model ckpt](https://drive.google.com/uc?id=1-KYz3nK7HxA3sXJ-QtLOCHxKTKBSSdjT) ]
    
   See prior section for how to train BIER models using training data
   
   See Colabs below for how to load and use models on downstream tasks
   

### Replicating downstream task results 
   See experiments/README.md for baselines

  - Clinical NED task using EHR dataset:  [ ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_OOJ97GDfgZhB-K98B3P3-5ZJygXHazj?usp=sharing)
  - Entity Linking Classification on Cancer Genetics dataset:  [ ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CDwTG71UkTKLxMhk7uDm4DHX2YABYbEf?usp=sharing)

> <br/>
### Connecting PubMed entities to Wiki Categories through UMLS to generate training data
![](figs/biers3.png?raw=true)
- after generating (mention, context, categories) triples we then learn BIERs as follows:

><br/>
><br/>

### BioMed IER architecture for learning biomed entity representations with interpretable components
![](figs/biers2.png?raw=true)
- after learning BIERs we can test their efficacy in a Zeroshot capacity for different biomed tasks

><br/>
><br/>

### Zeroshot results for varying amounts of supervision 
![](figs/biers1.png?raw=true)
