# %load_ext autoreload
# %autoreload 2
from __future__ import division
from __future__ import print_function

import sys, os
sys.path.insert(0, '..')
import models, graph, coarsening, utils
# from utils import model_perf

import tensorflow as tf
import numpy as np
import time
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import scipy.sparse as sp
import pickle as pkl

# from tensorflow.examples.tutorials.mnist import input_data

# %matplotlib inline
flags = tf.app.flags
FLAGS = flags.FLAGS

# neural network setting
# Graphs.
flags.DEFINE_integer('number_edges', 8, 'Graph: minimum number of edges per vertex.')
flags.DEFINE_string('metric', 'euclidean', 'Graph: similarity measure (between features).')
# TODO: change cgcnn for combinatorial Laplacians.
flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
flags.DEFINE_integer('coarsening_levels', 4, 'Number of coarsened graphs.')

# Directories.
# flags.DEFINE_string('dir_data', os.path.join('..', 'data', 'mnist'), 'Directory to store data.')

results_auc = dict()
results_accuracy = dict()
aucs = list()
accuracies = list()
results = list()

class model_perf(object):

    def __init__(self, i_fold, pairs_label):
        self.i_fold = i_fold
        self.pairs_label = pairs_label
        self.names, self.params = set(), {}
        self.fit_auc, self.fit_accuracy, self.fit_losses, self.fit_time = {}, {}, {}, {}
        self.train_auc, self.test_accuracy, self.train_loss = {}, {}, {}
        self.test_auc, self.train_accuracy, self.test_loss = {}, {}, {}
        self.s_represent = dict()
        self.s_count = dict()

    def test(self, model, name, params, data, records, train_data, train_recs, train_labels, val_data, val_recs, val_labels, test_data, test_recs, test_labels, train_pairs, test_pairs):
        self.params[name] = params
        self.fit_auc[name], self.fit_accuracy[name], self.fit_losses[name], self.fit_time[name] = \
                model.fit(train_data, train_recs, train_labels, val_data, val_recs, val_labels)
        del val_data, val_labels

        n, m, f = data.shape
        test_data = np.zeros([test_pairs.shape[0], m, f, 2])
        test_data[:, :, :, 0] = data[test_pairs[:,0], :, :]
        test_data[:, :, :, 1] = data[test_pairs[:,1], :, :]

        string, self.test_auc[name], self.test_accuracy[name], self.test_loss[name], _, test_represent, test_prob = \
                model.evaluate(test_data, test_recs, test_labels)
        print('test  {}'.format(string))
        self.names.add(name)

    def save(self, data_type):
        results = list()
        for name in sorted(self.names):
            results.append([name, self.test_accuracy[name], self.train_accuracy[name],
            self.test_f1[name], self.train_f1[name], self.test_loss[name],
            self.train_loss[name], self.fit_time[name]*1000])

        if os.path.exists(data_type + '_results.csv'):
            old = pd.read_csv(data_type + '_results.csv', header=None)
            new = pd.DataFrame(data=results)
            r = pd.concat([old, new], ignore_index=True)
            r.to_csv(data_type + '_results.csv', index=False, header=['method', 'test_acc',
            'train_acc', 'test_f1', 'train_f1', 'test_loss', 'train_loss', 'fit_time'])
        else:
            r = pd.DataFrame(data=results)
            r.to_csv(data_type + '_results.csv', index=False, header=['method', 'test_acc',
            'train_acc', 'test_f1', 'train_f1', 'test_loss', 'train_loss', 'fit_time'])


    def fin_result(self, data_type, i_fold=None):
        for name in sorted(self.names):
            if name not in results_auc:
                results_auc[name] = 0
            if name not in results_accuracy:
                results_accuracy[name] = 0
            results_auc[name] += self.test_auc[name]
            results_accuracy[name] += self.test_accuracy[name]
            aucs.append(self.test_auc[name])
            accuracies.append(self.test_accuracy[name])
            results.append([i_fold, self.test_auc[name], self.test_accuracy[name]])
        if i_fold == 4:
            for name in sorted(self.names):
                results_auc[name] /= 5
                results_accuracy[name] /= 5
                print('{:5.2f}  {}'.format(
                    results_auc[name], name))
                print('{:5.2f}  {}'.format(
                    results_accuracy[name], name))
            std_auc = np.std(np.array(aucs))
            std_accuracy = np.std(np.array(accuracies[:-1]))
            results.append([name, results_auc[name], std_auc, results_accuracy[name], std_accuracy])
            r = pd.DataFrame(data=results)
            r.to_csv(data_type + '_fin_results', index=False, header=['method', 'test_auc', 'std_auc', 'test_accuracy', 'std_accuracy'])


    def show(self, fontsize=None):
        if fontsize:
            plt.rc('pdf', fonttype=42)
            plt.rc('ps', fonttype=42)
            plt.rc('font', size=fontsize)         # controls default text sizes
            plt.rc('axes', titlesize=fontsize)    # fontsize of the axes title
            plt.rc('axes', labelsize=fontsize)    # fontsize of the x any y labels
            plt.rc('xtick', labelsize=fontsize)   # fontsize of the tick labels
            plt.rc('ytick', labelsize=fontsize)   # fontsize of the tick labels
            plt.rc('legend', fontsize=fontsize)   # legend fontsize
            plt.rc('figure', titlesize=fontsize)  # size of the figure title
        print('  auc      loss        time [ms]  name')
        print('test  train   test  train   test     train')
        for name in sorted(self.names):
            print('{:5.2f} {:5.2f}   {:.2e} {:.2e}   {:3.0f}   {}'.format(
                    self.test_auc[name], self.train_auc[name],
                    self.test_loss[name], self.train_loss[name], self.fit_time[name]*1000, name))


def grid_graph(m, corners=False):
    z = graph.grid(m)
    dist, idx = graph.distance_sklearn_metrics(z, k=FLAGS.number_edges, metric=FLAGS.metric)
    A = graph.adjacency(dist, idx)

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        import scipy.sparse
        A = A.toarray()
        A[A < A.max()/1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    print("{} > {} edges".format(A.nnz//2, FLAGS.number_edges*m**2//2))
    return A

def get_pair_label(pairs, labels):
    train_pairs, val_pairs, test_pairs = pairs
    train_labels, val_labels, test_labels = labels
    pairs = train_pairs.tolist() + test_pairs.tolist()
    labels = train_labels.tolist() + test_labels.tolist()
    pairs = [str(p[0]) + '_' + str(p[1]) for p in pairs]
    pair_label = dict(zip(pairs, labels))
    # pos_pairs = [pairs[i] for i in range(len(labels)) if labels[i] == 1]
    # print (len(pos_pairs))
    return pair_label

def get_feed_data(data, pairs, labels, method):
    train_pairs, val_pairs, test_pairs = pairs
    train_labels, val_labels, test_labels = labels
    n, m, f = data.shape
    # f = 1 # whether f can be deleted
    if 'GCN' in method:
        # dti data
        train_x = np.zeros([train_pairs.shape[0], m, f, 2])
        val_x = np.zeros([val_pairs.shape[0], m, f, 2])
        test_x = np.zeros([test_pairs.shape[0], m, f, 2])
        # store dti pairs
        train_x[:,:,:,0] = data[train_pairs[:,0], :, :]
        train_x[:,:,:,1] = data[train_pairs[:,1], :, :]
        val_x[:,:,:,0] = data[val_pairs[:,0], :, :]
        val_x[:,:,:,1] = data[val_pairs[:,1], :, :]
        test_x[:,:,:,0] = data[test_pairs[:,0], :, :]
        test_x[:,:,:,1] = data[test_pairs[:,1], :, :]

    train_y = train_labels
    val_y = val_labels
    test_y = test_labels

    print (train_x.shape)
    print (val_x.shape)
    print (test_x.shape)
    return train_x, train_y, val_x, val_y, test_x, test_y

def get_feed_records(records, pairs, mem_size, code_size, method):
    train_pairs, val_pairs, test_pairs = pairs
    # f = 1 # whether f can be deleted
    if 'GCN' in method:
        # clinical records
        train_r = np.zeros([train_pairs.shape[0], mem_size, code_size, 2])
        val_r = np.zeros([val_pairs.shape[0], mem_size, code_size, 2])
        test_r = np.zeros([test_pairs.shape[0], mem_size, code_size, 2])

        # store clinical pairs
        train_r[:,:,:,0] = records[train_pairs[:,0], :, :]
        train_r[:,:,:,1] = records[train_pairs[:,1], :, :]
        val_r[:,:,:,0] = records[val_pairs[:,0], :, :]
        val_r[:,:,:,1] = records[val_pairs[:,1], :, :]
        test_r[:,:,:,0] = records[test_pairs[:,0], :, :]
        test_r[:,:,:,1] = records[test_pairs[:,1], :, :]

    return train_r, val_r, test_r

def train(modality, method, data_type, distance, k, fdim, nhops, mem_size, code_size, n_words, edim, n_epoch, batch_size, pairs, labels, coords, data, records, i_fold):
    str_params = '_' + modality + '_' + distance + '_k' + str(k) + '_fdim' + str(fdim) + '_nhops' + str(nhops) + '_memsize' + str(mem_size) + '_codesize' + str(code_size) + '_nwords' + str(n_words) + '_edim' + str(edim)

    print ('Construct ROI graphs...')
    t_start = time.process_time()
    coo1, coo2, coo3 = coords.shape # coo2 is the roi dimension
    features = np.zeros([coo1*coo3, coo2])
    for i in range(coo3):
        features[coo1*i:coo1*(i+1), :] = coords[:, :, i]
    dist, idx = graph.distance_scipy_spatial(np.transpose(features), k=10, metric='euclidean')
    A = graph.adjacency(dist, idx).astype(np.float32)
    if method == '2gcn':
        graphs, perm = coarsening.coarsen(A, levels=FLAGS.coarsening_levels, self_connections=False)
        L = [graph.laplacian(A, normalized=True) for A in graphs]
        data = coarsening.perm_data1(data, perm)
    else:
        graphs = list()
        graphs.append(A)
        L = [graph.laplacian(A, normalized=True)]
    print ('The number of GCN layers: ', len(L))
    print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
    # graph.plot_spectrum(L)
    del A

    print ('Set parameters...')
    mp = model_perf(i_fold, get_pair_label(pairs, labels))
    # Architecture.
    common = {}
    common['dir_name']       = 'ppmi/'
    common['num_epochs']     = n_epoch
    common['batch_size']     = batch_size
    common['eval_frequency'] = 5 * common['num_epochs']
    common['patience']       = 5
    common['regularization'] = 1e-2
    common['dropout']        = 1
    common['learning_rate']  = 5e-3
    common['decay_rate']     = 0.95
    common['momentum']       = 0.9
    common['init_std']       = 5e-2

    print ('Get feed pairs and labels...')
    train_pairs, val_pairs, test_pairs = pairs
    train_x, train_y, val_x, val_y, test_x, test_y = get_feed_data(data, pairs, labels, method)
    train_r, val_r, test_r = get_feed_records(records, pairs, mem_size, code_size, method)
    C = max(train_y)+1
    common['decay_steps']    = train_x.shape[0] / common['batch_size']

    if method == 'MemGCN':
        # str_params += ''
        name = 'cgconv_softmax'
        params = common.copy()
        params['method'] = method
        params['p']              = [1] # pooling size
        params['M']              = [C]
        params['K']              = k    # support number
        params['nhops']          = nhops # hop number
        params['fdim']           = fdim # filters dimension
        params['edim']           = edim # embeddings dimension
        params['mem_size']       = mem_size # the length of sequential records
        params['code_size']      = code_size # the size of one record
        params['n_words']        = n_words # feature dimension
        params['distance']       = distance
        params['fin'] = train_x.shape[2]
        params['dir_name'] += name
        params['filter'] = 'chebyshev5'
        params['brelu'] = 'b2relu'
        params['pool'] = 'apool1'
        mp.test(models.siamese_cgcnn_mem(L, **params), name, params,
                        data, records, train_x, train_r, train_y,
                        val_x, val_r, val_y, test_x, test_r, test_y,
                        train_pairs, test_pairs)

    # mp.save(data_type)
    method_type = method + '_'
    mp.fin_result(method_type + data_type + str_params, i_fold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('modality', type=str)
    parser.add_argument('method', type=str)
    parser.add_argument('data_type', type=str)
    parser.add_argument('distance', type=str)
    parser.add_argument('K', type=int)
    parser.add_argument('fdim', type=int)
    parser.add_argument('nhops', type=int)
    parser.add_argument('mem_size', type=int)
    parser.add_argument('code_size', type=int)
    parser.add_argument('n_words', type=int)
    parser.add_argument('edim', type=int)
    parser.add_argument('n_epoch', type=int)
    parser.add_argument('batch_size', type=int)
    args = parser.parse_args()
    print ('-----------------START-------------------')
    print (args.method)
    # See function train for all possible parameter and there definition.
    data, subj, coords, records, pairs, labels = utils.load_data(data_type=args.data_type,
                                                                 mem_size=args.mem_size,
                                                                 code_size=args.code_size)
    print ("5-fold cross validation ...")
    for l in range(5): # 5-fold cross validation
        print ("********* The %d fold ... *********" %(l+1))
        train(modality=args.modality,
              method=args.method,
              data_type=args.data_type,
              distance=args.distance,
              k=args.K,
              fdim=args.fdim,
              nhops=args.nhops,
              mem_size=args.mem_size,
              code_size=args.code_size,
              n_words=args.n_words,
              edim=args.edim,
              n_epoch=args.n_epoch,
              batch_size=args.batch_size,
              pairs=pairs[l],
              labels=labels[l],
              coords=coords,
              data=data,
              records=records,
              i_fold=l)
    print ('-----------------DONE-------------------')
