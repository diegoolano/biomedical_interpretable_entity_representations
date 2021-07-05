# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
Entity Disembiguation task
'''
from __future__ import absolute_import, division, unicode_literals

import codecs
import os
import io
import copy
import json
import logging
import numpy as np
import code
import time
from tqdm import tqdm

from enteval.tools.validation import SplitMultiClassClassifier
from enteval.tools.validation import SplitClassifier, SplitClassifierWithSoftMask
from enteval.tools.validation import SplitClassifierWithLayerWeights

BASEOUT = "/home/diego/biomed_fall20/biomed_ier/experiments/ehr_baselines/"

class RareEval(object):
    def __init__(self, taskpath, use_name=False, seed=1111):
        logging.debug('***** Transfer task : Entity Disembiguation prediction *****\n\n')
        self.seed = seed
        self.use_name = use_name
        if self.use_name:
            logging.debug("***** Use entity names to compute embedding *****")
        else:
            logging.debug("***** Use entity descriptions to compute embedding *****")

        train_labels, train_context, train_desc = self.loadFile(os.path.join(taskpath, 'train.txt'))
        valid_labels, valid_context, valid_desc = self.loadFile(os.path.join(taskpath, 'valid.txt'))
        test_labels, test_context, test_desc = self.loadFile(os.path.join(taskpath, 'test.txt'))
        self.data = {'train': (train_labels, train_context, train_desc),
                     'valid': (valid_labels, valid_context, valid_desc),
                     'test': (test_labels, test_context, test_desc)}
        self.samples = [sent[0] for sents in train_context for sent in sents] + \
                        [sent for sents in train_desc for sent in sents] + \
                        [sent[0] for sents in valid_context for sent in sents] + \
                        [sent for sents in valid_desc for sent in sents] + \
                        [sent[0] for sents in test_context for sent in sents] + \
                        [sent for sents in test_desc for sent in sents] 

        # code.interact(local=locals())

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def process_desc(self, desc):
        sents = desc.split("\t")
        for sent in sents:
            if "**blank**" in sent:
                words = sent.split()
                break

        index = words.index("**blank**")
        words = words[max(0, index-150):index+150]
        index = words.index("**blank**")
        return words, index

    def loadFile(self, fpath):
        labels, contexts, descs = [], [], []
        data = []
        with open(fpath, 'r') as f:
            for line in tqdm(f):
                if line.strip() == "":
                    # code.interact(local=locals())
                    contexts.append([])
                    descs.append([])
                    words, index = self.process_desc(data[0])
                    for context in data[1:5]:
                        # print(context)
                        entity, desc = context.split("\t")
                        entity = entity.split()
                        desc = desc.split()
                        new_words = words[:index] + entity + words[index+1:]
                        contexts[-1].append([new_words, index, index+len(entity)])
                        if self.use_name:
                            descs[-1].append(entity)
                        else:
                            descs[-1].append(desc)
                    labels.append(int(data[5]))
                    data = []
                else:
                    data.append(line.strip())
        return labels, contexts, descs

    def run(self, params, batcher):
        print('run -- batch size: {}'.format(params.batch_size))
        self.X, self.y = {}, {}
        for key in self.data:
            print('Data key: {}'.format(repr(key)))
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            labels, contexts, descs = self.data[key]
            enc_input = []
            n_labels = len(labels)
            for ii in tqdm(range(0, n_labels, params.batch_size)):
                batch_context = [b for batch in contexts[ii:ii + params.batch_size] for b in batch]
                batch_desc = [b for batch in descs[ii:ii + params.batch_size] for b in batch]
                batch = [a + [b] for a,b in zip(batch_context, batch_desc)]
                if len(batch) > 0:  # == params.batch_size:
                    context_enc, desc_enc = batcher(params, batch)
                    # code.interact(local=locals())
                    if isinstance(context_enc, list) and isinstance(desc_enc, list):
                        assert len(context_enc) == len(desc_enc)
                        assert context_enc is not None and desc_enc is not None
                        n_layers = len(context_enc)
                        batch_size, n_dim = context_enc[0].shape
                        tmp = np.zeros((batch_size, n_layers, n_dim * 4))
                        for iii, ctxdesc in enumerate(zip(context_enc, desc_enc)):
                            ctx, desc = ctxdesc
                            tmp[:, iii, :] = np.hstack(
                                (ctx, desc, ctx * desc, np.abs(ctx - desc))
                            )
                        enc_input.append(tmp)
                    else:
                        enc_input.append(
                        np.hstack((
                            context_enc, desc_enc,
                            context_enc * desc_enc,
                            np.abs(context_enc - desc_enc)
                        )))
                if (ii) % (20000*params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / n_labels))
            self.X[key] = np.vstack(enc_input)
            self.X[key] = self.X[key].reshape(-1, 4, self.X[key].shape[-1])
            self.y[key] = np.array(labels) #[:len(self.X[key])])
            # code.interact(local=locals())
        logging.debug("Training data shape: {}".format(self.X["train"].shape))
        config = {'nclasses': 4, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'nhid': 2000, 'noreg': False}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config_classifier['nhid'] = 0#2000
        config['classifier'] = config_classifier

        clf = SplitMultiClassClassifier(self.X, self.y, config)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1} for Rare Entity Prediction\n'
                      .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}


class ConllYagoEval(object):
    def __init__(self, taskpath, use_name=False, seed=1111):
        logging.debug('***** Transfer task : Conll Yago Entity Linking *****\n\n')
        self.seed = seed
        self.use_name = use_name
        if self.use_name:
            logging.debug("***** Use entity names to compute embedding *****")
        else:
            logging.debug("***** Use entity descriptions to compute embedding *****")

        print('Loading {}'.format(os.path.join(taskpath, 'train.final.txt')))
        train_context, train_desc = self.loadFile(os.path.join(taskpath, 'train.final.txt'))
        print('Loading {}'.format(os.path.join(taskpath, 'testa.final.txt')))
        valid_context, valid_desc = self.loadFile(os.path.join(taskpath, 'testa.final.txt'))
        print('Loading {}'.format(os.path.join(taskpath, 'testb.final.txt')))
        test_context, test_desc = self.loadFile(os.path.join(taskpath, 'testb.final.txt'))
        self.data = {'train': (train_context, train_desc),
                     'valid': (valid_context, valid_desc),
                     'test': (test_context, test_desc)}
        self.samples = [ins[0] for ins in train_context] + \
                        [ins[0] for ins in valid_context] + \
                        [ins[0] for ins in test_context] + \
                        [cand[0] for cands in train_desc for cand in cands] + \
                        [cand[0] for cands in valid_desc for cand in cands] + \
                        [cand[0] for cands in test_desc for cand in cands]

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath):
        contexts, descs = [], []
        with open(fpath, 'r') as f:
            for line in f:
                s, e, sent, _, entities = line.strip().split("\t", 4)
                entities = entities.split("\t")
                # code.interact(local=locals())
                _descs = []
                for entity in entities:
                    # print(entity)
                    prior, entity_name, desc = entity.split("|||", 2)
                    if self.use_name:
                        entity_name = entity_name.split("_")
                        _descs.append([entity_name, float(prior), 0])
                    else:
                        desc = desc.split()
                        _descs.append([desc, float(prior), 0])
                _descs[0][-1] = 1  # Gold
                
                contexts.append([sent.split(), int(s), int(e)])
                descs.append(_descs)
                
        # context: [[sent, s, e] * N]
        # descs: [[[desc, prior, 0/1] * M] * N]
        return contexts, descs

    def run(self, params, batcher):
        print('run -- batch size: {}'.format(params.batch_size))
        self.X, self.y, self.priors = {}, {}, {}
        for key in self.data:
            print('Data key: {}'.format(repr(key)))
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            contexts, descs = self.data[key]
            n_contexts = len(contexts)
            enc_contexts = []
            print('Processing context...')
            for ii in tqdm(range(0, n_contexts, params.batch_size)):
                try:
                    batch_context = [item + [None] for item in contexts[ii:ii + params.batch_size]]
                    # code.interact(local=locals())
                    enc_context, _ = batcher(params, batch_context)
                    enc_contexts.append(enc_context)
                except Exception as e:
                    print("Error on batch: ", ii)
                    print(batch_context)
                    print(e)
                    
            enc_contexts = np.vstack(enc_contexts)

            enc_descs = []
            labels = [desc[-1] for _descs in descs for desc in _descs]
            priors = [desc[-2] for _descs in descs for desc in _descs]
            num_descs = [len(_descs) for _descs in descs]
            descs = [desc[0] for _descs in descs for desc in _descs]
            n_descs = len(descs)
            print('Processing descriptions...')
            for ii in tqdm(range(0, n_descs, params.batch_size)):
                batch_descs = [[None, None, None] + [item] for item in descs[ii:ii + params.batch_size]]
                enc_desc, _ = batcher(params, batch_descs) # our version
                #_, enc_desc = batcher(params, batch_descs)
                enc_descs.append(enc_desc)
            enc_descs = np.vstack(enc_descs)

            enc_contexts = [enc_contexts[i] for i in range(len(num_descs)) for j in range(num_descs[i])]
            enc_contexts = np.vstack(enc_contexts)
            labels = np.array(labels).astype("int64")
            priors = np.array(priors).astype("float32")
            print(enc_contexts.shape, enc_descs.shape)
#            self.X[key]["context"] = enc_contexts.reshape(enc_descs.shape)
#            self.X[key]["desc"] = enc_descs

            self.X[key] = np.concatenate([enc_contexts, enc_descs, enc_contexts*enc_descs, np.abs(enc_contexts-enc_descs)], 1)
            #self.X[key] = np.concatenate([enc_contexts, enc_descs], 1)
            self.y[key] = labels
            self.priors[key] = priors

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'noreg': False}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier
        config['saveout']=   BASEOUT + 'model_out/conll_yago_baseline_' + time.strftime("_%Y-%m-%d_%H:%M:%S", time.localtime()) + '.model'

        clf = SplitClassifier(self.X, self.y, config)
        devacc, testacc, _devacc = clf.run(return_score=True)
        testacc, all_logits = testacc
        _,  dev_logits = _devacc
        _, descs = self.data[key]
        num_descs = [len(_descs) for _descs in descs]
        cums = [0] + np.cumsum(num_descs).tolist()

        dev_logits_v2 = dev_logits + self.priors["valid"]
        devpreds_v2 = np.array([np.argmax(dev_logits_v2[cums[i]:cums[i+1]]) for i in range(len(cums)-1)])
        devacc_v2 = (devpreds_v2 == 0).sum() / len(devpreds_v2)

        all_logits_v1 = all_logits + self.priors["test"] - all_logits * self.priors["test"]
        # code.interact(local=locals()) 
        preds_v1 = np.array([np.argmax(all_logits_v1[cums[i]:cums[i+1]]) for i in range(len(cums)-1)])
        testacc_v1 = (preds_v1==0).sum()/len(preds_v1)
        all_logits_v2 = all_logits + self.priors["test"]
        preds_v2 = np.array([np.argmax(all_logits_v2[cums[i]:cums[i+1]]) for i in range(len(cums)-1)])
        testacc_v2 = (preds_v2==0).sum()/len(preds_v2)
        all_logits_v3 = all_logits * self.priors["test"]
        preds_v3 = np.array([np.argmax(all_logits_v3[cums[i]:cums[i+1]]) for i in range(len(cums)-1)])
        testacc_v3 = (preds_v3==0).sum()/len(preds_v3)
        all_logits_prior = self.priors["test"]
        preds_prior = np.array([np.argmax(all_logits_prior[cums[i]:cums[i+1]]) for i in range(len(cums)-1)])
        testacc_prior = (preds_prior==0).sum()/len(preds_prior)

        logging.debug('Dev acc : {0} Test acc v1: {1} Test acc v2: {2} Test acc v3: {3} Test acc prior: {4} for ConNLL Yago Entity Linking\n'
                      .format(devacc, testacc_v1, testacc_v2, testacc_v3, testacc_prior))
        return {'devacc': devacc, 'devacc_v2': devacc_v2, 'binarytestacc': testacc, 'testaccv1': testacc_v1,
                'testaccv2': testacc_v2, 'testaccv3': testacc_v3, 'testacc_prior': testacc_prior, 
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}


class WLNEDEval(object):
    def __init__(self, taskpath, use_name=False, seed=1111):
        logging.debug('***** Transfer task : WLNED Entity Linking *****\n\n')
        self.seed = seed
        self.use_name = False  # Never use this
        logging.debug("***** Use entity descriptions to compute embedding *****")


        train_context, train_desc = self.loadFile(os.path.join(taskpath, 'train.json'))
        print('Loading {}: {} - {}'.format(os.path.join(taskpath, 'train.json'), len(train_context), len(train_desc)))
        valid_context, valid_desc = self.loadFile(os.path.join(taskpath, 'dev.json'))
        print('Loading {}: {} - {}'.format(os.path.join(taskpath, 'dev.json'), len(valid_context), len(valid_desc)))
        test_context, test_desc = self.loadFile(os.path.join(taskpath, 'test.json'))
        print('Loading {}: {} - {}'.format(os.path.join(taskpath, 'test.json'), len(test_context), len(test_desc)))
        self.data = {'train': (train_context, train_desc),
                     'valid': (valid_context, valid_desc),
                     'test': (test_context, test_desc)}
        self.samples = [ins for ins in train_context] + \
                       [ins for ins in valid_context] + \
                       [ins for ins in test_context] + \
                       [cand[0] for cands in train_desc for cand in cands] + \
                       [cand[0] for cands in valid_desc for cand in cands] + \
                       [cand[0] for cands in test_desc for cand in cands]

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath):
        contexts, descs = [], []
        with open(fpath, 'r') as f:
            for line in f:
                ex = json.loads(line.strip())
                entities = ex['candidates']
                _descs = []
                for entity in entities:
                    prior = entity['score']
                    desc = entity['description']
                    _descs.append([desc, float(prior), 0])
                _descs[0][-1] = 1  # Gold
                contexts.append(ex['context'])
                descs.append(_descs)
        # context: [[sent, s, e] * N]
        # descs: [[[desc, prior, 0/1] * M] * N]
        return contexts, descs

    def run(self, params, batcher):
        print('run -- batch size: {}'.format(params.batch_size))
        self.X, self.y, self.priors = {}, {}, {}
        for key in self.data:
            print('Data key: {}'.format(repr(key)))
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            contexts, descs = self.data[key]
            n_contexts = len(contexts)
            enc_contexts = []
            use_layer_weights = False
            print('Processing context...')
            for ii in tqdm(range(0, n_contexts, params.batch_size)):
                batch_context = [item for item in contexts[ii:ii + params.batch_size]]
                # code.interact(local=locals())
                enc_context, _ = batcher(params, batch_context)
                if isinstance(enc_context, list):
                    assert enc_context is not None
                    use_layer_weights = True
                    n_layers = len(enc_context)
                    batch_size, n_dim = enc_context[0].shape
                    tmp = np.zeros((batch_size, n_layers, n_dim))
                    #print('ctx', n_layers, batch_size, n_dim)
                    for iii, ctx in enumerate(enc_context):
                        tmp[:, iii, :] = ctx
                    enc_contexts.append(tmp)
                else:
                    enc_contexts.append(enc_context)
            enc_contexts = np.vstack(enc_contexts)

            enc_descs = []
            labels = [desc[-1] for _descs in descs for desc in _descs]
            priors = [desc[-2] for _descs in descs for desc in _descs]
            num_descs = [len(_descs) for _descs in descs]
            descs = [desc[0] for _descs in descs for desc in _descs]
            n_descs = len(descs)
            print('Processing descriptions...')
            for ii in tqdm(range(0, n_descs, params.batch_size)):
                batch_descs = [[None, None, None] + [item] for item in descs[ii:ii + params.batch_size]]
                enc_desc, _ = batcher(params, batch_descs)  # our version
                # _, enc_desc = batcher(params, batch_descs)
                if isinstance(enc_desc, list):
                    assert enc_desc is not None
                    n_layers = len(enc_desc)
                    batch_size, n_dim = enc_desc[0].shape
                    #print('desc', n_layers, batch_size, n_dim)
                    tmp = np.zeros((batch_size, n_layers, n_dim))
                    for iii, desc in enumerate(enc_desc):
                        tmp[:, iii, :] = desc
                    enc_descs.append(tmp)
                else:
                    enc_descs.append(enc_desc)
            enc_descs = np.vstack(enc_descs)
            print(enc_contexts.shape, enc_descs.shape)
            if use_layer_weights:
                tmp = np.zeros_like(enc_descs)
                n_examples, n_layers, dim = enc_contexts.shape
                assert n_examples == len(num_descs), (n_examples, len(num_descs))
                desc_idx = 0
                for ctx_idx in range(n_examples):
                    for _ in range(num_descs[ctx_idx]):
                        tmp[desc_idx, :, :] = enc_contexts[ctx_idx, :, :]
                        desc_idx += 1
                    # Assume # desc = 3
                    print(tmp[desc_idx-3:desc_idx, :, :], tmp[desc_idx-3:desc_idx, :, :].shape)
                assert desc_idx == enc_descs.shape[0], (desc_idx, enc_descs.shape[0])
                enc_contexts = tmp
            else:
                enc_contexts = [enc_contexts[i] for i in range(len(num_descs)) for j in range(num_descs[i])]
                enc_contexts = np.vstack(enc_contexts)
            labels = np.array(labels).astype("int64")
            priors = np.array(priors).astype("float32")
            #print(enc_descs)
            print(enc_contexts.shape, enc_descs.shape)
            #            self.X[key]["context"] = enc_contexts.reshape(enc_descs.shape)
            #            self.X[key]["desc"] = enc_descs
            if use_layer_weights:
                self.X[key] = np.concatenate( [enc_contexts, enc_descs, enc_contexts * enc_descs, np.abs( enc_contexts - enc_descs)], 2)
            else:
                self.X[key] = np.concatenate( [enc_contexts, enc_descs, enc_contexts * enc_descs, np.abs( enc_contexts - enc_descs)], 1)
            print('X.shape:', self.X[key].shape)
            self.y[key] = labels
            self.priors[key] = priors

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'noreg': False}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier
        config['saveout']=   BASEOUT + 'model_out/wlned_baseline_' + time.strftime("_%Y-%m-%d_%H:%M:%S", time.localtime()) + '.model'

        if not params['softmask']:
            if all(self.X[key].ndim == 2 for key in self.data):
                print('Using SplitClassifier')
                clf = SplitClassifier(self.X, self.y, config)
            elif all(self.X[key].ndim == 3 for key in self.data):
                print('Using SplitClassifierWithLayerWeights')
                clf = SplitClassifierWithLayerWeights(self.X, self.y, config)
            else:
                print(
                    'Invalid X dim:',
                    [self.X[key].ndim == 2 for key in self.data]
                )
                raise NotImplementedError
        else:
            print('Using SplitClassifierWithSoftMask')
            clf = SplitClassifierWithSoftMask(self.X, self.y, config)

        devacc, testacc, _ = clf.run(return_score=True)
        testacc, all_logits = testacc
        _, descs = self.data[key]
        num_descs = [len(_descs) for _descs in descs]
        cums = [0] + np.cumsum(num_descs).tolist()
        all_logits_v1 = all_logits + self.priors["test"] - all_logits * self.priors["test"]
        # code.interact(local=locals())
        preds_v1 = np.array([np.argmax(all_logits_v1[cums[i]:cums[i + 1]]) for i in range(len(cums) - 1)])
        testacc_v1 = (preds_v1 == 0).sum() / len(preds_v1)
        all_logits_v2 = all_logits + self.priors["test"]
        preds_v2 = np.array([np.argmax(all_logits_v2[cums[i]:cums[i + 1]]) for i in range(len(cums) - 1)])
        testacc_v2 = (preds_v2 == 0).sum() / len(preds_v2)
        all_logits_v3 = all_logits * self.priors["test"]
        preds_v3 = np.array([np.argmax(all_logits_v3[cums[i]:cums[i + 1]]) for i in range(len(cums) - 1)])
        testacc_v3 = (preds_v3 == 0).sum() / len(preds_v3)
        all_logits_v4 = all_logits + np.log(self.priors["test"])
        preds_v4 = np.array([np.argmax(all_logits_v4[cums[i]:cums[i + 1]]) for i in range(len(cums) - 1)])
        testacc_v4 = (preds_v4 == 0).sum() / len(preds_v4)
        all_logits_prior = self.priors["test"]
        preds_prior = np.array([np.argmax(all_logits_prior[cums[i]:cums[i + 1]]) for i in range(len(cums) - 1)])
        testacc_prior = (preds_prior == 0).sum() / len(preds_prior)


        logging.debug(
            'Dev acc : {0} Test acc v1: {1} Test acc v2: {2} Test acc v3: {3} Test acc v4: {4} Test acc prior: {5} for ConNLL Yago Entity Linking\n'.format(devacc, testacc_v1, testacc_v2, testacc_v3, testacc_v4, testacc_prior))

        return {'devacc': devacc, 'binarytestacc': testacc, 'testaccv1': testacc_v1,
                'testaccv2': testacc_v2, 'testaccv3': testacc_v3, 'testacc_v4': testacc_v4,
                'testacc_prior': testacc_prior,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}

class EHREval(object):
    def __init__(self, taskpath, use_name=False, seed=1111):
        logging.debug('***** Transfer task : EHR Entity Linking *****\n\n')
        self.seed = seed
        self.use_name = False  # Never use this
        logging.debug("***** Use entity descriptions to compute embedding *****")

        #ent_sys_file_base = "/dccstor/redrug_ier/diego/data/entity_type_systems/0720lg2_v6/"
        ent_sys_file_base = "/home/diego/biomed_fall20/check_ehr_data/"

        """
        EHR_TRAIN_DATA = ent_sys_file_base + "train/wiki_desc/train_ehr_examples_between_1_and_6_cands_v2_overlaping_with_wikidesc0720_3k_full_with_cand_desc.json"    
        EHR_DEV_DATA = ent_sys_file_base + "validation/wiki_desc/dev_ehr_examples_between_1_and_6_cands_v2_overlaping_with_wikidesc0720_3k_full_with_cand_desc.json"  
        EHR_EVAL_DATA = ent_sys_file_base + "test/wiki_desc/test_ehr_examples_between_1_and_6_cands_v2_overlaping_with_wikidesc0720_3k_full_with_cand_desc.json"
        """
        EHR_TRAIN_DATA = ent_sys_file_base + "train_ehr_examples_0824.json"
        EHR_DEV_DATA = ent_sys_file_base + "dev_ehr_examples_0824.json"
        EHR_EVAL_DATA = ent_sys_file_base + "test_ehr_examples_0824.json"

        train_context, train_desc = self.loadFile(EHR_TRAIN_DATA)
        print('Loading {}: {} - {}'.format(EHR_TRAIN_DATA, len(train_context), len(train_desc)))
        valid_context, valid_desc = self.loadFile(EHR_DEV_DATA)
        print('Loading {}: {} - {}'.format(EHR_DEV_DATA, len(valid_context), len(valid_desc)))
        test_context, test_desc = self.loadFile(EHR_EVAL_DATA)
        print('Loading {}: {} - {}'.format(EHR_EVAL_DATA, len(test_context), len(test_desc)))
        self.data = {'train': (train_context, train_desc),
                     'valid': (valid_context, valid_desc),
                     'test': (test_context, test_desc)}
        self.samples = [ins for ins in train_context] + \
                       [ins for ins in valid_context] + \
                       [ins for ins in test_context] + \
                       [cand[0] for cands in train_desc for cand in cands] + \
                       [cand[0] for cands in valid_desc for cand in cands] + \
                       [cand[0] for cands in test_desc for cand in cands]

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath):
        cnt, errs, nofind = 0, 0, 0
        contexts, descs = [], []
        with open(fpath, 'r') as f:
            for line in tqdm(f):
                ex = json.loads(line.strip())
                #{"ex_id": "Q178999_77", "word": "sensory neurons", "right_context": ["(", "pseudounipolar", "neurons", ")", ",", "such", "as", "those", "for", "touch", "and", "warmth", ",", "the", "axons", "are", "called", "afferent", "nerve", "fibers", "and", "the", "electrical", "impulse", "travels", "along", "these", "from", "the", "periphery", "to", "the", "cell", "body", ",", "and", "from", "the", "cell", "body", "to", "the", "spinal", "cord", "along", "another", "branch", "of", "the", "same", "axon", "."], "left_context": ["In", "certain"], "wikiurl": "sensory neuron", "categories": ["afferent neurons", "human cells", "receptor cells"], "ncandidates": 2, 
                #"candidates": [
                #    ["Sensory neuron", "0.940887", [""], [",", "also", "known", "as", "<b>afferent", "neurons</b>,", "are", "neurons", "in", "the", "central", "nervous", "system,", "that", "convert", "a", "specific", "type", "of", "stimulus,", "via", "their", "receptors,", "into", "action", "potentials", "or", "graded", "potentials.", "This", "process", "is", "called", "sensory", "transduction.", "The", "cell", "bodies", "of", "the", "sensory", "neurons", "are", "located", "in", "the", "dorsal", "ganglia", "of", "the", "spinal", "cord.<p>This", "sensory", "information", "travels", "", "along", "afferent", "nerve", "fibers", "in", "an", "afferent", "or", "sensory", "nerve,", "to", "the", "brain", "via", "the", "spinal", "cord.", "The", "stimulus", "can", "come", "from", "<em>extoreceptors</em>", "outside", "the", "body,", "for", "example", "light", "and", "soun"], "859926"], 
                #    ["Sensory nerve", "0.0591133", ["A"], [",", "also", "called", "an", "<b>afferent", "nerve</b>,", "is", "a", "nerve", "that", "carries", "sensory", "information", "toward", "the", "central", "nervous", "system", "(CNS)", "and", "", "all", "those", "nerves", "which", "can", "sense", "or", "recognise", "the", "", "stimulie", "(Internal", "or", "External)", "are", "known", "as", "sensory", "nerves.<p>It", "is", "a", "cable-like", "bundle", "of", "the", "afferent", "nerve", "fibers", "coming", "from", "sensory", "receptors", "in", "the", "peripheral", "nervous", "system", "(PNS).", "A", "motor", "nerve", "carries", "information", "from", "the", "CNS", "to", "the", "PNS,", "and", "both", "types", "of", "nerve", "are", "called", "<b>peripheral", "nerves</b>.<p>Afferent", "nerve", "fibers", "link", "the", "sens"], "3480477"]
                #    ], 
                #"exp_categories": ["afferent neurons", "human cells", "cells", "human", "receptor cells", "receptor"]}
               
                ex_dic = { k:ex[k] for k in ex if k not in ['candidates', 'categories']}
                ex_dic["mention_as_list"] = str(ex_dic["word"]).split(" ")
                ex_dic['y_category'] = ex['categories']
                ex_dic['y_title'] = ex['wikiurl']
                contexts.append(ex_dic)

                entities = ex['candidates']
                _descs = []
                #for entity in entities:
                    #prior = entity['score']
                    #desc = entity['description']
                #    prior = entity[1]
                #    desc = 
                #    _descs.append([desc, float(prior), 0])
                # _descs[0][-1] = 1  # Gold
                # descs.append(_descs)

                true_ent = ex_dic['wikiurl'].lower().strip()
                found_true_in_cands = 0
                for cand in ex['candidates']:
                    if len(cand) < 5:
                        #print("Current cand has less than 5 items ", cand, "\n found in ", ex)
                        errs += 1
                    elif str(cand[0]).lower().strip() == true_ent and found_true_in_cands == 0:
                        found_true_in_cands = 1
                        desc = {}
                        desc["word"] = str(cand[0])
                        desc["mention_as_list"] = str(cand[0]).split(" ")
                        desc['score'] = float(cand[1])
                        desc["left_context"] = cand[2]
                        desc["right_context"] = cand[3]
                        desc['wikiId'] = cand[4]   
                        desc['ex_id'] = "ex_"+str(cand[4])+"_"+str(cnt)   
                        desc["y_category"] = []
                        _descs.append([desc, desc['score'], 0])
                        cnt += 1
            
                if found_true_in_cands == 0:
                    #print("Didn't find true ent: ",true_ent," in candidates for ", [ str(cand[0]).lower().strip() for cand in ex['candidates'] ])
                    nofind += 1 
                    continue
                else:
                    for cand in ex['candidates']:      # link, score, left context, right context, wikiId   (0 - 4)
                        if len(cand) < 5:
                            #print("Current cand has less than 5 items ", cand, "\n found in ", ex)
                            errs += 1
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
                            _descs.append([desc, desc['score'], 0])
                            cnt += 1

                _descs[0][-1] = 1  # Gold
                descs.append(_descs)

        # context: [[sent, s, e] * N]
        # descs: [[[desc, prior, 0/1] * M] * N]

        print("cnt:",cnt,", errs:",errs,", nofind: ",nofind)
        return contexts, descs

    def run(self, params, batcher):
        print('run -- batch size: {}'.format(params.batch_size))
        self.X, self.y, self.priors = {}, {}, {}
        for key in self.data:
            print('Data key: {}'.format(repr(key)))
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            contexts, descs = self.data[key]
            n_contexts = len(contexts)
            enc_contexts = []
            use_layer_weights = False

           
            print('Processing context...')
            for ii in tqdm(range(0, n_contexts, params.batch_size)):
                batch_context = [item for item in contexts[ii:ii + params.batch_size]]
                # code.interact(local=locals())
                enc_context, _ = batcher(params, batch_context)
                if isinstance(enc_context, list):
                    assert enc_context is not None
                    use_layer_weights = True
                    n_layers = len(enc_context)
                    batch_size, n_dim = enc_context[0].shape
                    tmp = np.zeros((batch_size, n_layers, n_dim))
                    #print('ctx', n_layers, batch_size, n_dim)
                    for iii, ctx in enumerate(enc_context):
                        tmp[:, iii, :] = ctx
                    enc_contexts.append(tmp)
                else:
                    enc_contexts.append(enc_context)
            enc_contexts = np.vstack(enc_contexts)
            

            enc_descs = []
            labels = [desc[-1] for _descs in descs for desc in _descs]
            priors = [desc[-2] for _descs in descs for desc in _descs]
            num_descs = [len(_descs) for _descs in descs]
            descs = [desc[0] for _descs in descs for desc in _descs]
            n_descs = len(descs)

            print('Processing descriptions...', n_descs)
            for ii in tqdm(range(0, n_descs, params.batch_size)):
                batch_descs = [[None, None, None] + [item] for item in descs[ii:ii + params.batch_size]]
                #print("BATCH descs", batch_descs)
                enc_desc, _ = batcher(params, batch_descs)  # our version
                # _, enc_desc = batcher(params, batch_descs)
                if isinstance(enc_desc, list):
                    assert enc_desc is not None
                    n_layers = len(enc_desc)
                    batch_size, n_dim = enc_desc[0].shape
                    #print('desc', n_layers, batch_size, n_dim)
                    tmp = np.zeros((batch_size, n_layers, n_dim))
                    for iii, desc in enumerate(enc_desc):
                        tmp[:, iii, :] = desc
                    enc_descs.append(tmp)
                else:
                    enc_descs.append(enc_desc)
            enc_descs = np.vstack(enc_descs)

            print(enc_contexts.shape, enc_descs.shape)
            if use_layer_weights:
                tmp = np.zeros_like(enc_descs)
                n_examples, n_layers, dim = enc_contexts.shape
                assert n_examples == len(num_descs), (n_examples, len(num_descs))
                desc_idx = 0
                for ctx_idx in range(n_examples):
                    for _ in range(num_descs[ctx_idx]):
                        tmp[desc_idx, :, :] = enc_contexts[ctx_idx, :, :]
                        desc_idx += 1
                    # Assume # desc = 3
                    print(tmp[desc_idx-3:desc_idx, :, :], tmp[desc_idx-3:desc_idx, :, :].shape)
                assert desc_idx == enc_descs.shape[0], (desc_idx, enc_descs.shape[0])
                enc_contexts = tmp
            else:
                enc_contexts = [enc_contexts[i] for i in range(len(num_descs)) for j in range(num_descs[i])]
                enc_contexts = np.vstack(enc_contexts)
            labels = np.array(labels).astype("int64")
            priors = np.array(priors).astype("float32")
            #print(enc_descs)
            print(enc_contexts.shape, enc_descs.shape)
            #            self.X[key]["context"] = enc_contexts.reshape(enc_descs.shape)
            #            self.X[key]["desc"] = enc_descs
            if use_layer_weights:
                self.X[key] = np.concatenate( [enc_contexts, enc_descs, enc_contexts * enc_descs, np.abs( enc_contexts - enc_descs)], 2)
            else:
                self.X[key] = np.concatenate( [enc_contexts, enc_descs, enc_contexts * enc_descs, np.abs( enc_contexts - enc_descs)], 1)
            print('X.shape:', self.X[key].shape)
            self.y[key] = labels
            self.priors[key] = priors

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'noreg': False}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier
        config['saveout']=   BASEOUT + 'model_out/ehr_baseline_' + time.strftime("_%Y-%m-%d_%H:%M:%S", time.localtime()) + '.model'

        if not params['softmask']:
            if all(self.X[key].ndim == 2 for key in self.data):
                print('Using SplitClassifier')
                clf = SplitClassifier(self.X, self.y, config)
            elif all(self.X[key].ndim == 3 for key in self.data):
                print('Using SplitClassifierWithLayerWeights')
                clf = SplitClassifierWithLayerWeights(self.X, self.y, config)
            else:
                print(
                    'Invalid X dim:',
                    [self.X[key].ndim == 2 for key in self.data]
                )
                raise NotImplementedError
        else:
            print('Using SplitClassifierWithSoftMask')
            clf = SplitClassifierWithSoftMask(self.X, self.y, config)

        devacc, testacc, _ = clf.run(return_score=True)
        testacc, all_logits = testacc
        _, descs = self.data[key]
        num_descs = [len(_descs) for _descs in descs]
        cums = [0] + np.cumsum(num_descs).tolist()
        all_logits_v1 = all_logits + self.priors["test"] - all_logits * self.priors["test"]
        # code.interact(local=locals())
        preds_v1 = np.array([np.argmax(all_logits_v1[cums[i]:cums[i + 1]]) for i in range(len(cums) - 1)])
        testacc_v1 = (preds_v1 == 0).sum() / len(preds_v1)
        all_logits_v2 = all_logits + self.priors["test"]
        preds_v2 = np.array([np.argmax(all_logits_v2[cums[i]:cums[i + 1]]) for i in range(len(cums) - 1)])
        testacc_v2 = (preds_v2 == 0).sum() / len(preds_v2)
        all_logits_v3 = all_logits * self.priors["test"]
        preds_v3 = np.array([np.argmax(all_logits_v3[cums[i]:cums[i + 1]]) for i in range(len(cums) - 1)])
        testacc_v3 = (preds_v3 == 0).sum() / len(preds_v3)
        all_logits_v4 = all_logits + np.log(self.priors["test"])
        preds_v4 = np.array([np.argmax(all_logits_v4[cums[i]:cums[i + 1]]) for i in range(len(cums) - 1)])
        testacc_v4 = (preds_v4 == 0).sum() / len(preds_v4)
        all_logits_prior = self.priors["test"]
        preds_prior = np.array([np.argmax(all_logits_prior[cums[i]:cums[i + 1]]) for i in range(len(cums) - 1)])
        testacc_prior = (preds_prior == 0).sum() / len(preds_prior)


        logging.debug(
            'Dev acc : {0} Test acc v1: {1} Test acc v2: {2} Test acc v3: {3} Test acc v4: {4} Test acc prior: {5} for EHR NED\n'.format(devacc, testacc_v1, testacc_v2, testacc_v3, testacc_v4, testacc_prior))

        return {'devacc': devacc, 'binarytestacc': testacc, 'testaccv1': testacc_v1,
                'testaccv2': testacc_v2, 'testaccv3': testacc_v3, 'testacc_v4': testacc_v4,
                'testacc_prior': testacc_prior,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
