# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Pytorch Classifier class in the style of scikit-learn
Classifiers include Logistic Regression and MLP
"""

from __future__ import absolute_import, division, unicode_literals

import numpy as np
import copy
from enteval import utils

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


class PyTorchClassifier(object):
    def __init__(self, inputdim, nclasses, l2reg=0., batch_size=64, seed=1111,
                 cudaEfficient=False, is_enttype=None, l1_coefficient=0.01):
        # fix seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.inputdim = inputdim
        self.nclasses = nclasses
        self.l2reg = l2reg
        self.batch_size = batch_size
        self.cudaEfficient = cudaEfficient
        self.is_enttype = True if is_enttype is None else is_enttype
        self.l1_coefficient = l1_coefficient

    def prepare_split(self, X, y, validation_data=None, validation_split=None):
        # Preparing validation data
        assert validation_split or validation_data
        if validation_data is not None:
            trainX, trainy = X, y
            devX, devy = validation_data
        else:
            permutation = np.random.permutation(len(X))
            trainidx = permutation[int(validation_split * len(X)):]
            devidx = permutation[0:int(validation_split * len(X))]
            trainX, trainy = X[trainidx], y[trainidx]
            devX, devy = X[devidx], y[devidx]

        device = torch.device('cpu') if self.cudaEfficient else torch.device('cuda')
        #print('trainX', trainX)
        trainX = torch.from_numpy(trainX).to(device, dtype=torch.float32)
        trainy = torch.from_numpy(trainy).to(device, dtype=torch.int64)
        devX = torch.from_numpy(devX).to(device, dtype=torch.float32)
        devy = torch.from_numpy(devy).to(device, dtype=torch.int64)

        return trainX, trainy, devX, devy


    def fit(self, X, y, validation_data=None, validation_split=None,
            early_stop=True):
        self.nepoch = 0
        bestaccuracy = -1
        stop_train = False
        early_stop_count = 0

        # Preparing validation data
        trainX, trainy, devX, devy = self.prepare_split(X, y, validation_data,
                                                        validation_split)

        # Training
        while not stop_train and self.nepoch <= self.max_epoch:
            self.trainepoch(trainX, trainy, epoch_size=self.epoch_size)
            accuracy = self.score(devX, devy)
            if accuracy > bestaccuracy:
                bestaccuracy = accuracy
                bestmodel = copy.deepcopy(self.model)
            elif early_stop:
                if early_stop_count >= self.tenacity:
                    stop_train = True
                early_stop_count += 1
        self.model = bestmodel
        return bestaccuracy

    def trainepoch(self, X, y, epoch_size=1):
        self.model.train()
        for _ in range(self.nepoch, self.nepoch + epoch_size):
            permutation = np.random.permutation(len(X))
            all_costs = []
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = torch.from_numpy(permutation[i:i + self.batch_size]).long().to(X.device)

                Xbatch = X[idx]
                ybatch = y[idx]

                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                output = self.model(Xbatch)
                # loss
                loss = self.loss_fn(output, ybatch)

                # # add L1 loss
                # if 'softmask' in dict(self.model.named_parameters()):
                #     l1_reg = self.l1_coefficient * torch.norm(
                #         dict(self.model.named_parameters())['softmask'],
                #         1
                #     )
                #     loss += l1_reg

                all_costs.append(loss.data.item())
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += epoch_size

    def score(self, devX, devy, test=False, return_score=False):
        self.model.eval()
        correct = 0
        if not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
            devX = torch.FloatTensor(devX).cuda()
            devy = torch.LongTensor(devy).cuda()

        all_preds = []
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                ybatch = devy[i:i + self.batch_size]
                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                output = self.model(Xbatch)
                pred = output.data.max(1)[1]
                correct += pred.long().eq(ybatch.data.long()).sum().item()
                all_preds.append(pred.long().data.cpu().numpy().reshape(-1))
                all_logits.append(F.softmax(output, -1).data.cpu()[:, -1].numpy().reshape(-1))
            accuracy = 1.0 * correct / len(devX)
        if test:
            all_preds = np.concatenate(all_preds, 0)
            golden_labels = devy.data.cpu().numpy().reshape(-1)
            cm = confusion_matrix(golden_labels, all_preds)
            diag = [cm[i,i] for i in range(len(cm))]
            print(diag)
            print("number of instances", len(all_preds))
        if return_score:
            all_logits = np.concatenate(all_logits, 0)
            return accuracy, all_logits
        else:
            return accuracy

    def predict(self, devX):
        self.model.eval()
        if not isinstance(devX, torch.cuda.FloatTensor):
            devX = torch.FloatTensor(devX).cuda()
        yhat = np.array([])
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                output = self.model(Xbatch)
                yhat = np.append(yhat,
                                 output.data.max(1)[1].cpu().numpy())
        yhat = np.vstack(yhat)
        return yhat

    def predict_proba(self, devX):
        self.model.eval()
        probas = []
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                vals = F.softmax(self.model(Xbatch).data.cpu().numpy())
                if not probas:
                    probas = vals
                else:
                    probas = np.concatenate(probas, vals, axis=0)
        return probas


"""
MLP with Pytorch (nhid=0 --> Logistic Regression)
"""

class MLP(PyTorchClassifier):
    def __init__(self, params, inputdim, nclasses, l2reg=0., batch_size=64,
                 seed=1111, cudaEfficient=False):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, cudaEfficient)
        """
        PARAMETERS:
        -nhid:       number of hidden units (0: Logistic Regression)
        -optim:      optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
        -tenacity:   how many times dev acc does not increase before stopping
        -epoch_size: each epoch corresponds to epoch_size pass on the train set
        -max_epoch:  max number of epoches
        -dropout:    dropout for MLP
        """

        self.nhid = 0 if "nhid" not in params else params["nhid"]
        self.optim = "adam" if "optim" not in params else params["optim"]
        self.tenacity = 5 if "tenacity" not in params else params["tenacity"]
        self.epoch_size = 4 if "epoch_size" not in params else params["epoch_size"]
        self.max_epoch = 200 if "max_epoch" not in params else params["max_epoch"]
        self.dropout = 0. if "dropout" not in params else params["dropout"]
        self.batch_size = 64 if "batch_size" not in params else params["batch_size"]

        if params["nhid"] == 0:
            print(self.inputdim, self.nclasses)
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, self.nclasses),
            ).cuda()
        elif params["n_types"] > 0 and params["nhid"] > 0:
            self.model = EmbMLP(
                params['n_types'],
                self.inputdim,
                params["nhid"],
                self.nclasses
            ).cuda()
        else:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, params["nhid"]),
                nn.Dropout(p=self.dropout),
                nn.Sigmoid(),
                nn.Linear(params["nhid"], self.nclasses),
            ).cuda()

        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False

        optim_fn, optim_params = utils.get_optimizer(self.optim)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = self.l2reg

    # Override super class methods
    # def prepare_split(self, X, y, validation_data=None, validation_split=None):
    #     # Preparing validation data
    #     assert validation_split or validation_data
    #     if validation_data is not None:
    #         trainX, trainy = X, y
    #         devX, devy = validation_data
    #     else:
    #         permutation = np.random.permutation(len(X))
    #         trainidx = permutation[int(validation_split * len(X)):]
    #         devidx = permutation[0:int(validation_split * len(X))]
    #         trainX, trainy = X[trainidx], y[trainidx]
    #         devX, devy = X[devidx], y[devidx]
    #
    #     device = torch.device('cpu') if self.cudaEfficient else torch.device('cuda')
    #     trainX = self.to_torch(trainX, device)
    #     trainy = torch.from_numpy(trainy).to(device, dtype=torch.int64)
    #     devX = self.to_torch(devX, device)
    #     devy = torch.from_numpy(devy).to(device, dtype=torch.int64)
    #
    #     return trainX, trainy, devX, devy
    #
    # def to_torch(self, X, device):
    #     for i in range(len(X)):
    #         if isinstance(X[i]['enc1'], np.ndarray):
    #             X[i]['enc1'] = torch.from_numpy(X[i]['enc1']).to(device, dtype=torch.int64)
    #         if isinstance(X[i]['enc2'], np.ndarray):
    #             X[i]['enc2'] = torch.from_numpy(X[i]['enc2']).to(device, dtype=torch.int64)
    #     return X
    #
    # def fit(self, X, y, validation_data=None, validation_split=None,
    #         early_stop=True):
    #     self.nepoch = 0
    #     bestaccuracy = -1
    #     stop_train = False
    #     early_stop_count = 0
    #
    #     # Preparing validation data
    #     trainX, trainy, devX, devy = self.prepare_split(X, y, validation_data,
    #                                                     validation_split)
    #
    #     # Training
    #     while not stop_train and self.nepoch <= self.max_epoch:
    #         self.trainepoch(trainX, trainy, epoch_size=self.epoch_size)
    #         accuracy = self.score(devX, devy)
    #         if accuracy > bestaccuracy:
    #             bestaccuracy = accuracy
    #             bestmodel = copy.deepcopy(self.model)
    #         elif early_stop:
    #             if early_stop_count >= self.tenacity:
    #                 stop_train = True
    #             early_stop_count += 1
    #     self.model = bestmodel
    #     return bestaccuracy
    #
    # def trainepoch(self, X, y, epoch_size=1):
    #     self.model.train()
    #     for _ in range(self.nepoch, self.nepoch + epoch_size):
    #         all_costs = []
    #         start_idx = 0
    #         for i in range(0, len(X)):
    #             # forward
    #             Xbatch = X[i]
    #             ybatch = y[start_idx:start_idx + self.batch_size]
    #             permutation = np.random.permutation(Xbatch['enc1'].size()[0])
    #             idx = torch.from_numpy(permutation).long().to(y.device)
    #             Xbatch = {'enc1': Xbatch['enc1'][idx], 'enc2': Xbatch['enc2'][idx]}
    #             ybatch = ybatch[idx]
    #
    #             start_idx += Xbatch['enc1'].size()[0]
    #
    #             if self.cudaEfficient:
    #                 Xbatch = {'enc1': Xbatch['enc1'].cuda(), 'enc2': Xbatch['enc2'].cuda()}
    #                 ybatch = ybatch.cuda()
    #             output = self.model(Xbatch)
    #
    #             # loss
    #             loss = self.loss_fn(output, ybatch)
    #             all_costs.append(loss.data.item())
    #             # backward
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             # Update parameters
    #             self.optimizer.step()
    #     self.nepoch += epoch_size
    #
    # def score(self, devX, devy, test=False, return_score=False):
    #     self.model.eval()
    #     correct = 0
    #     if not isinstance(devX[0]['enc1'], torch.cuda.LongTensor) or self.cudaEfficient:
    #         devX = [{'enc1': torch.LongTensor(x['enc1']).cuda(),
    #                  'enc2': torch.LongTensor(x['enc2']).cuda()} for x in devX]
    #         devy = torch.LongTensor(devy).cuda()
    #
    #     all_preds = []
    #     all_logits = []
    #     with torch.no_grad():
    #         start_idx = 0
    #         for i in range(0, len(devX)):
    #             # forward
    #             Xbatch = devX[i]
    #             ybatch = devy[start_idx:start_idx + self.batch_size]
    #             permutation = np.random.permutation(Xbatch['enc1'].size()[0])
    #             idx = torch.from_numpy(permutation).long().to(devy.device)
    #             Xbatch = {'enc1': Xbatch['enc1'][idx], 'enc2': Xbatch['enc2'][idx]}
    #             ybatch = ybatch[idx]
    #
    #             start_idx += Xbatch['enc1'].size()[0]
    #             if self.cudaEfficient:
    #                 Xbatch = {'enc1': Xbatch['enc1'].cuda(), 'enc2': Xbatch['enc2'].cuda()}
    #                 ybatch = ybatch.cuda()
    #             output = self.model(Xbatch)
    #             pred = output.data.max(1)[1]
    #             correct += pred.long().eq(ybatch.data.long()).sum().item()
    #             all_preds.append(pred.long().data.cpu().numpy().reshape(-1))
    #             all_logits.append(F.softmax(output, -1).data.cpu()[:, -1].numpy().reshape(-1))
    #         accuracy = 1.0 * correct / len(devy)
    #     if test:
    #         all_preds = np.concatenate(all_preds, 0)
    #         golden_labels = devy.data.cpu().numpy().reshape(-1)
    #         cm = confusion_matrix(golden_labels, all_preds)
    #         diag = [cm[i, i] for i in range(len(cm))]
    #         print(diag)
    #         print("number of instances", len(all_preds))
    #     if return_score:
    #         all_logits = np.concatenate(all_logits, 0)
    #         return accuracy, all_logits
    #     else:
    #         return accuracy
    #
    # def predict(self, devX):
    #     self.model.eval()
    #     if not isinstance(devX, torch.cuda.FloatTensor):
    #         devX = torch.FloatTensor(devX).cuda()
    #     yhat = np.array([])
    #     with torch.no_grad():
    #         for i in range(0, len(devX), self.batch_size):
    #             Xbatch = devX[i:i + self.batch_size]
    #             output = self.model(Xbatch)
    #             yhat = np.append(yhat,
    #                              output.data.max(1)[1].cpu().numpy())
    #     yhat = np.vstack(yhat)
    #     return yhat
    #
    # def predict_proba(self, devX):
    #     self.model.eval()
    #     probas = []
    #     with torch.no_grad():
    #         for i in range(0, len(devX), self.batch_size):
    #             Xbatch = devX[i:i + self.batch_size]
    #             vals = F.softmax(self.model(Xbatch).data.cpu().numpy())
    #             if not probas:
    #                 probas = vals
    #             else:
    #                 probas = np.concatenate(probas, vals, axis=0)
    #     return probas


"""
MLP with Pytorch with SoftMask (nhid=0 --> Logistic Regression)
"""

class MaskMLP(PyTorchClassifier):
    def __init__(self, params, inputdim, nclasses, l2reg=0., batch_size=64,
                 seed=1111, cudaEfficient=False, l1_coefficient=0.01, hardmask=None):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, cudaEfficient)
        """
        PARAMETERS:
        -nhid:       number of hidden units (0: Logistic Regression)
        -optim:      optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
        -tenacity:   how many times dev acc does not increase before stopping
        -epoch_size: each epoch corresponds to epoch_size pass on the train set
        -max_epoch:  max number of epoches
        -dropout:    dropout for MLP
        """

        self.nhid = 0 if "nhid" not in params else params["nhid"]
        self.optim = "adam" if "optim" not in params else params["optim"]
        self.tenacity = 5 if "tenacity" not in params else params["tenacity"]
        self.epoch_size = 4 if "epoch_size" not in params else params["epoch_size"]
        self.max_epoch = 200 if "max_epoch" not in params else params["max_epoch"]
        self.dropout = 0. if "dropout" not in params else params["dropout"]
        self.batch_size = 64 if "batch_size" not in params else params["batch_size"]
        self.n_types = 10331 if "n_types" not in params else params["n_types"]
        self.seed_types = np.array([1. if i < 100 else 0. for i in range(self.n_types)])\
            if "seed_types" not in params else params["seed_types"]
        self.l1_coefficient = l1_coefficient
        self.hardmask = hardmask

        # TODO: clean up here --> single mpdel is needed...
        assert params["nhid"] == 0, params["nhid"]
        if self.hardmask is not None:
            print('HardMaskMLP:', self.inputdim, self.nclasses)
            assert self.n_types == len(self.hardmask), (self.n_types, len(self.hardmask))
            self.model = HardMaskMlpModel(
                self.inputdim, self.nclasses, self.n_types, self.hardmask
            ).cuda()
        else:
            print('SoftMaskMLP:', self.inputdim, self.nclasses)
            self.model = SoftMaskMlpModel(
                self.inputdim, self.nclasses, self.n_types, self.seed_types
            ).cuda()

        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False

        optim_fn, optim_params = utils.get_optimizer(self.optim)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = self.l2reg


class SoftMaskMlpModel(nn.Module):
    def __init__(self, inputdim, nclasses, n_types, seed_types=None):
        super(SoftMaskMlpModel, self).__init__()
        self.linear = nn.Linear(inputdim, nclasses)
        self.softmask = nn.Parameter(torch.zeros(n_types))
        #self.softmask = nn.Parameter(torch.ones(n_types))
        self.seedmask = torch.zeros(n_types).cuda() if seed_types is None else torch.from_numpy(seed_types).cuda()
        print('self.seedmask:', sum(self.seedmask))
        self.n_types = n_types
        self.seed_types = seed_types
        self.n_repeat = 4  # Assume input size is 4 * n_types

    def forward(self, x):
        for i in range(self.n_repeat):
            x[:, i * self.n_types:(i + 1) * self.n_types] = \
                x[:, i * self.n_types:(i + 1) * self.n_types] * self.seedmask \
                    + (1. - self.seedmask) * x[:, i * self.n_types:(i + 1) * self.n_types] * self.softmask
        out = self.linear(x)
        return out


class HardMaskMlpModel(nn.Module):
    def __init__(self, inputdim, nclasses, n_types, hardmask):
        super(HardMaskMlpModel, self).__init__()
        self.linear = nn.Linear(inputdim, nclasses)
        self.hardmask = torch.from_numpy(hardmask).cuda()
        print('self.hardmask:', sum(self.hardmask))
        self.n_types = n_types
        self.n_repeat = 4  # Assume input size is 4 * n_types

    def forward(self, x):
        for i in range(self.n_repeat):
            x[:, i * self.n_types:(i + 1) * self.n_types] *= self.hardmask
        out = self.linear(x)
        return out


"""
MLP with Pytorch (nhid=0 --> Logistic Regression)
"""

class MultiClassMLP(PyTorchClassifier):
    def __init__(self, params, inputdim, nclasses, l2reg=0., batch_size=64,
                 seed=1111, cudaEfficient=False):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, cudaEfficient)
        """
        PARAMETERS:
        -nhid:       number of hidden units (0: Logistic Regression)
        -optim:      optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
        -tenacity:   how many times dev acc does not increase before stopping
        -epoch_size: each epoch corresponds to epoch_size pass on the train set
        -max_epoch:  max number of epoches
        -dropout:    dropout for MLP
        """

        self.nhid = 0 if "nhid" not in params else params["nhid"]
        self.optim = "adam" if "optim" not in params else params["optim"]
        self.tenacity = 5 if "tenacity" not in params else params["tenacity"]
        self.epoch_size = 4 if "epoch_size" not in params else params["epoch_size"]
        self.max_epoch = 200 if "max_epoch" not in params else params["max_epoch"]
        self.dropout = 0. if "dropout" not in params else params["dropout"]
        self.batch_size = 64 if "batch_size" not in params else params["batch_size"]

        if params["nhid"] == 0:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, self.nclasses),
            ).cuda()
        else:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, params["nhid"]),
                nn.Dropout(p=self.dropout),
                nn.Sigmoid(),
                nn.Linear(params["nhid"], self.nclasses),
            ).cuda()

        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False

        optim_fn, optim_params = utils.get_optimizer(self.optim)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = self.l2reg


class EmbMLP(nn.Module):
    def __init__(self, n_types, inputdim, nhid, nclasses, dropout=0.):
        super(EmbMLP, self).__init__()
        self.n_types = n_types
        self.embedding = nn.Embedding(n_types + 1, inputdim, padding_idx=n_types) # add pad idx
        self.linear1 = nn.Linear(inputdim * 4, nhid)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(nhid, nclasses)

    def forward(self, feed_dict):
        enc1 = feed_dict['enc1']
        enc2 = feed_dict['enc2']
        b_size, max_len = enc1.size()
        n_type1 = torch.sum(enc1 != self.n_types, dim=1)  # n_batch
        n_type2 = torch.sum(enc2 != self.n_types, dim=1)  # n_batch
        type_emb1 = self.embedding(enc1)  # n_batch x max_len x emb dim
        type_emb2 = self.embedding(enc2)  # n_batch x max_len x emb dim
        type_emb1 = torch.sum(type_emb1, dim=1) / n_type1.unsqueeze(1)  # n_batch x emb dim
        type_emb2 = torch.sum(type_emb2, dim=1) / n_type2.unsqueeze(1)  # n_batch x emb dim
        final_emb = torch.cat((type_emb1, type_emb2,
                               type_emb1 * type_emb2,
                               torch.abs(type_emb1 - type_emb2)), dim=1)
        out = self.linear1(final_emb)
        out = self.sigmoid(self.dropout(out))
        out = self.linear2(out)
        return out


"""
MLP with Pytorch (nhid=0 --> Logistic Regression)
"""

class MLPLayerWeighst(PyTorchClassifier):
    def __init__(self, params, inputdim, nclasses, l2reg=0., batch_size=64,
                 seed=1111, cudaEfficient=False, n_layers=13):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, cudaEfficient)
        """
        PARAMETERS:
        -nhid:       number of hidden units (0: Logistic Regression)
        -optim:      optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
        -tenacity:   how many times dev acc does not increase before stopping
        -epoch_size: each epoch corresponds to epoch_size pass on the train set
        -max_epoch:  max number of epoches
        -dropout:    dropout for MLP
        """

        self.nhid = 0 if "nhid" not in params else params["nhid"]
        self.optim = "adam" if "optim" not in params else params["optim"]
        self.tenacity = 5 if "tenacity" not in params else params["tenacity"]
        self.epoch_size = 4 if "epoch_size" not in params else params["epoch_size"]
        self.max_epoch = 200 if "max_epoch" not in params else params["max_epoch"]
        self.dropout = 0. if "dropout" not in params else params["dropout"]
        self.batch_size = 64 if "batch_size" not in params else params["batch_size"]

        assert params["nhid"] == 0, params["nhid"]
        self.model = MlpLayerWeightsModel(
            self.inputdim, self.nclasses, n_layers
        ).cuda()

        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False

        optim_fn, optim_params = utils.get_optimizer(self.optim)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = self.l2reg


class MlpLayerWeightsModel(nn.Module):
    def __init__(self, inputdim, nclasses, n_layers):
        super(MlpLayerWeightsModel, self).__init__()
        self.linear = nn.Linear(inputdim, nclasses)
        self.layer_weights = nn.Parameter(torch.randn(n_layers) * .01)
        print('self.layer_weights:', self.layer_weights.size())

    def forward(self, x):  # x.shape = (batch size, n layers, dim)
        x = (x * self.layer_weights.unsqueeze(0).unsqueeze(2)).sum(1)
        out = self.linear(x)
        return out