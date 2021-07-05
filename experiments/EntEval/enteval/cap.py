# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
SNLI - Entailment
'''
from __future__ import absolute_import, division, unicode_literals

import codecs
import os
import io
import copy
import logging
import numpy as np

from enteval.tools.validation import SplitClassifier
from enteval.tools.validation import SplitClassifierWithSoftMask
from enteval.tools.validation import SplitClassifierWithLayerWeights
from tqdm import tqdm


class CAPEval(object):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : Coreference Arc Prediction binary Classification*****')
        self.seed = seed
        logging.debug('***** Task path: {}*****\n\n'.format(taskpath))
        train = self.loadFile(os.path.join(taskpath, 'train.txt'))
        valid = self.loadFile(os.path.join(taskpath, 'dev.txt'))
        test = self.loadFile(os.path.join(taskpath, 'test.txt'))

        
        self.samples = [item[0][0] for item in train] + \
                        [item[1][0] for item in train] + \
                        [item[0][0] for item in valid] + \
                        [item[1][0] for item in valid] + \
                        [item[0][0] for item in test] + \
                        [item[1][0] for item in test]
        self.data = {'train': train,
                     'valid': valid,
                     'test': test}


    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)


    def loadFile(self, fpath):
        data = []
        item = []
        i = 1
        with codecs.open(fpath, 'rb', 'utf-8') as fin:
            for line in fin:
                if line.strip() == "":
                    data.append(item)
                    item = []
                    i = 0
                else:
                    if i < 3:
                        # print(line)
                        sentence, start, end = line.strip().split("\t")
                        start = int(start)
                        end = int(end)
                        words = sentence.split()
                        item.append([words, start, end])
                    else:
                        item.append(int(line.strip()))
                i += 1
        return data


    def run(self, params, batcher):
        self.X, self.y = {}, {}
        for key in self.data:
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            input1 = [item[0] + [None] for item in self.data[key]]
            input2 = [item[1] + [None] for item in self.data[key]]
            labels = np.array([item[2] for item in self.data[key]])
            
            enc_input = []
            n_labels = len(labels)
            for ii in tqdm(range(0, n_labels, params.batch_size)):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1, _ = batcher(params, batch1)
                    enc2, _ = batcher(params, batch2)
                    if isinstance(enc1, list) and isinstance(enc2, list):
                        assert len(enc1) == len(enc2)
                        n_layers = len(enc1)
                        batch_size, n_dim = enc1[0].shape
                        tmp = np.zeros((batch_size, n_layers, n_dim * 4))
                        for iii, e1e2 in enumerate(zip(enc1, enc2)):
                            e1, e2 = e1e2
                            tmp[:, iii, :] = np.hstack(
                                (e1, e2, e1 * e2, np.abs(e1 - e2))
                            )
                        #print(tmp, tmp.shape)
                        enc_input.append(tmp)  # batch size, n layers, dim
                    else:
                        enc_input.append(np.hstack((enc1, enc2, enc1 * enc2,
                                                np.abs(enc1 - enc2))))
                    #enc_input.append({'enc1': enc1, 'enc2': enc2})

                if (ii*params.batch_size) % (20000*params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / n_labels))


            self.X[key] = np.vstack(enc_input)
            print(self.X[key].shape)
            #self.X[key] = enc_input
            self.y[key] = labels

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'nhid': params.nhid, 'noreg': True,
                  'file_header': params.file_header}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier

        if not params['softmask']:
            if all(self.X[key].ndim == 2 for key in self.data):
                print('SplitClassifier')
                clf = SplitClassifier(self.X, self.y, config)
            elif all(self.X[key].ndim == 3 for key in self.data):
                print('SplitClassifierWithLayerWeights')
                clf = SplitClassifierWithLayerWeights(self.X, self.y, config)
            else:
                print(
                    'Invalid X dim:',
                    [self.X[key].ndim == 2 for key in self.data]
                )
                raise NotImplementedError
        else:
            print('SplitClassifierWithSoftMask')
            clf = SplitClassifierWithSoftMask(self.X, self.y, config)

        devacc, testacc, _ = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1} for PreCo\n'
                      .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid']),
                'ntest': len(self.data['test'])}

if __name__ == "__main__":
    tpath = '/scratch/cluster/yasumasa/entity/data/EntEval'
    cap_same = CAPEval(tpath + '/CAP/next')
    idx = 0
    ex = cap_same.data['train'][idx]
    s1 = ex[0][0]
    s2 = ex[1][0]
    y = ex[2]
    e1 = s1[ex[0][1]:ex[0][2]]
    e2 = s2[ex[1][1]:ex[1][2]]

    print('s1:', s1)
    print('s2:', s2)
    print('e1:', e1, ', e2:', e2)
    print('y', y)
