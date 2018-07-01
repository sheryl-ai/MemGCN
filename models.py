import tensorflow as tf
import sklearn
import scipy.sparse
import numpy as np
import os, time, collections, shutil
import graph


# Common methods for all models
class base_model(object):

    def __init__(self):
        self.regularizers = []

    # High-level interface which runs the constructed computational graph.
    def predict(self, data, recs, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2], 2))
            batch_recs = np.zeros((self.batch_size, recs.shape[1], recs.shape[2], 2))
            tmp_data = data[begin:end, :, :, :]
            tmp_recs = recs[begin:end, :, :, :]

            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
                tmp_recs = tmp_recs.toarray()
            batch_data[:end-begin] = tmp_data
            batch_recs[:end-begin] = tmp_recs
            feed_dict = {self.ph_data: batch_data, self.ph_recs: batch_recs, self.ph_dropout: 1}

            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros(self.batch_size)
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)

            predictions[begin:end] = batch_pred[:end-begin]
            represent[begin:end, :] = batch_rep[:end-begin, :]
            prob[begin:end, :] = batch_prob[:end-begin, :]

        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions


    def evaluate(self, data, recs, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        recs: size N x Q
            N: number of signals (samples)
            Q: number of timestamps (sequence length)
        labels: size N
            N: number of signals (samples)
        """
        t_process, t_wall = time.process_time(), time.time()
        predictions, loss = self.predict(data, recs, labels, sess)

        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, predictions)
        auc = 100 * sklearn.metrics.auc(fpr, tpr)
        ncorrects = sum(predictions == labels)
        accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        string = 'auc: {:.2f}, accuracy: {:.2f} ({:d} / {:d}), loss: {:.2e}'.format(auc, accuracy, ncorrects, len(labels), loss)

        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall)
        # return string, auc, loss, predictions
        return string, auc, accuracy, loss, predictions


    def fit(self, train_data, train_recs, train_labels, val_data, val_recs, val_labels):
        t_process, t_wall = time.process_time(), time.time()
        sess = tf.Session(graph=self.graph)
        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)

        # Training.
        count = 0
        bad_counter = 0
        accuracies = []
        aucs = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        estop = False  # early stop

        for step in range(1, num_steps+1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]
            count += len(idx)
            batch_data, batch_recs, batch_labels = train_data[idx, :, :, :], train_recs[idx, :, :, :], train_labels[idx]

            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
                batch_recs = batch_recs.toarray()
            feed_dict = {self.ph_data: batch_data, self.ph_recs: batch_recs, self.ph_labels: batch_labels, self.ph_dropout: self.dropout}
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)


            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                print ('Seen samples: %d' % count)
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                string, auc, accuracy, loss, predictions = self.evaluate(val_data, val_recs, val_labels, sess)
                aucs.append(auc)
                accuracies.append(accuracy)
                losses.append(loss)
                print('  validation {}'.format(string))
                print(predictions.tolist()[:50])
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall))

                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validataion/auc', simple_value=auc)
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)

                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

                if len(aucs) > (self.patience+5) and auc > np.array(aucs).max():
                    bad_counter = 0

                if len(aucs) > (self.patience+5) and auc <= np.array(aucs)[:-self.patience].max():
                    bad_counter += 1
                    if bad_counter > self.patience:
                        print('Early Stop!')
                        estop = True
                        break
            if estop:
                break
        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
        print('validation auc: peak = {:.2f}, mean = {:.2f}'.format(max(aucs), np.mean(aucs[-10:])))
        writer.close()
        sess.close()
        t_step = (time.time() - t_wall) / num_steps
        return aucs, accuracies, losses, t_step

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    # Methods to construct the computational graph with memory network.
    def build_gcn_graph_mem(self, M_0):
        """Build the computational graph with memory network of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder(tf.float32, (self.batch_size, M_0, self.fin, 2), 'data')
                self.ph_recs = tf.placeholder(tf.int32, (self.batch_size, self.mem_size, self.code_size, 2), 'recs') # clinical records
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')

            # Model.
            op_logits = self.inference(self.ph_data, self.ph_recs, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)
            # self.op_loss, self.op_loss_average, self.op_var_loss, self.op_mean_loss, self.op_same_var, self.op_diff_var = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                    self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)

        self.graph.finalize()


    def inference(self, data, recs, dropout):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout.
            True: the model is built for training.
            False: the model is built for evaluation.
        """
        # TODO: optimizations for sparse data
        logits, represent, prob = self._inference(data, recs, dropout)
        return logits, represent, prob

    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=1)
            return prediction

    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.to_int64(labels)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cross_entropy = tf.reduce_mean(cross_entropy)
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = cross_entropy + regularization

            # Summaries for TensorBoard.
            tf.summary.scalar('loss/cross_entropy', cross_entropy)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, regularization, loss])
                tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average


    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum == 0:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


class siamese_cgcnn_mem(base_model):
    """
    Graph CNN which uses the Chebyshev approximation.

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        Q: Number of sequences.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.

    L: List of Graph Laplacians. Size M x M. One per coarsening level.

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.

    The following are choices of implementation for various blocks.
        filter: filtering operation, e.g. chebyshev5, lanczos2 etc.
        brelu: bias and relu, e.g. b1relu or b2relu.
        pool: pooling, e.g. mpool1.

    Training parameters:
        num_epochs:    Number of training epochs.
        learning_rate: Initial learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """
    def __init__(self, L, fdim, K, p, M, fin, n_words, mem_size, code_size, edim, nhops, distance, method='GCN', filter='chebyshev5', brelu='b1relu', pool='mpool1',
                num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                regularization=0, dropout=0, batch_size=100, eval_frequency=200, patience=10, init_std=0.05,
                dir_name=''):
        super().__init__()

        # Verify the consistency
        assert fdim == edim

        # Keep the useful Laplacians only. May be zero.
        M_0 = L[0].shape[0]
        j = 0
        self.L = []
        for pp in p:
            self.L.append(L[j])
            j += int(np.log2(pp)) if pp > 1 else 0
        L = self.L

        # Store attributes and bind operations.
        self.distance = distance
        self.L, self.fdim, self.K, self.p, self.M, self.fin = L, fdim, K, p, M, fin # gcn hyper-parameters
        self.n_words, self.mem_size, self.code_size, self.edim = n_words, mem_size, code_size, edim # memory hyper-parameters
        self.n_nodes = M_0
        self.num_epochs, self.learning_rate, self.patience = num_epochs, learning_rate, patience
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.method = method
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)
        self.init_std = init_std
        self.nhops = nhops

        # Build the computational graph with memory network.
        self.build_gcn_graph_mem(M_0)

    def build_var(self):
        self.A = tf.Variable(tf.random_normal([self.n_words, self.edim], stddev=self.init_std))
        self.B = tf.Variable(tf.random_normal([self.n_words, self.edim], stddev=self.init_std))
        self.C = tf.Variable(tf.random_normal([self.edim, self.edim], stddev=self.init_std))

    def chebyshev5(self, x, L, Fout, K):
        print ('chebnet')
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def b2relu(self, x):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def mpool1(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            #tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def apool1(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.avg_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def _inference_gcn(self, x_0, x_1, dropout, ihop):
        with tf.variable_scope("siamese") as scope:
            with tf.variable_scope('conv1{}'.format(ihop+1)):
                with tf.name_scope('filter'):
                    x_0 = self.filter(x_0, self.L[0], self.fdim, self.K)
                with tf.name_scope('bias_relu'):
                    x_0 = self.brelu(x_0)
                with tf.name_scope('pooling'):
                    x_0 = self.pool(x_0, self.p[0])
                    print (x_0.get_shape())

            with tf.variable_scope('conv2{}'.format(ihop+1)):
                with tf.name_scope('filter'):
                    x_1 = self.filter(x_1, self.L[0], self.fdim, self.K)
                with tf.name_scope('bias_relu'):
                    x_1 = self.brelu(x_1)
                with tf.name_scope('pooling'):
                    x_1 = self.pool(x_1, self.p[0])
                    print (x_1.get_shape())
        return x_0, x_1

    def build_memory(self, recs_0, recs_1):
        with tf.variable_scope("memory"):
            Ain_0 = tf.nn.embedding_lookup(self.A, recs_0) # recs_0, recs_1 size is (batch_size, mem_size, n_words)
            Ain_0 = tf.reduce_sum(Ain_0, 2)
            Ain_1 = tf.nn.embedding_lookup(self.A, recs_1)
            Ain_1 = tf.reduce_sum(Ain_1, 2)

            Bin_0 = tf.nn.embedding_lookup(self.B, recs_0)
            Bin_0 = tf.reduce_sum(Bin_0, 2)
            Bin_1 = tf.nn.embedding_lookup(self.B, recs_1)
            Bin_1 = tf.reduce_sum(Bin_1, 2)
        return Ain_0, Ain_1, Bin_0, Bin_1

    def _inference_memory(self, y_0, y_1, Ain_0, Ain_1, Bin_0, Bin_1):
        # compute weights for attention
        hid3dim_0 = tf.reshape(y_0, [-1, self.n_nodes, self.edim])
        Aout_0 = tf.matmul(hid3dim_0, Ain_0, adjoint_b=True)

        Aout3dim_0 = tf.reshape(Aout_0, [-1, self.n_nodes, self.mem_size])
        P_0 = tf.nn.softmax(Aout3dim_0) # batch_size x n_nodes x mem_size

        hid3dim_1 = tf.reshape(y_1, [-1, self.n_nodes, self.edim])
        Aout_1 = tf.matmul(hid3dim_1, Ain_1, adjoint_b=True)
        Aout3dim_1 = tf.reshape(Aout_1, [-1, self.n_nodes, self.mem_size])
        P_1 = tf.nn.softmax(Aout3dim_1)

        # output memory
        probs3dim_0 = tf.reshape(P_0, [-1, self.n_nodes, self.mem_size])
        Bout_0 = tf.matmul(probs3dim_0, Bin_0) # Bout_0 size is (batch_size, n_nodes, edim)
        Bout3dim_0 = tf.reshape(Bout_0, [-1, self.n_nodes, self.edim])

        probs3dim_1 = tf.reshape(P_1, [-1, self.n_nodes, self.mem_size])
        Bout_1 = tf.matmul(probs3dim_1, Bin_1) # Bout_0 size is (batch_size, n_nodes, edim)
        Bout3dim_1 = tf.reshape(Bout_1, [-1, self.n_nodes, self.edim])

        # compute the output
        batch, n_nodes, edim = y_0.get_shape() # (batch, n_nodes, edim)
        y_0 = tf.reshape(y_0, [int(batch * n_nodes), self.edim])
        Cout_0 = tf.matmul(y_0, self.C)
        Cout_0 = tf.reshape(Cout_0, [-1, self.n_nodes, self.edim])
        Dout_0 = tf.add(Cout_0, Bout3dim_0)

        batch, n_nodes, edim = y_1.get_shape()
        y_1 = tf.reshape(y_1, [int(batch * n_nodes), self.edim])
        Cout_1 = tf.matmul(y_1, self.C)
        Cout_1 = tf.reshape(Cout_1, [-1, self.n_nodes, self.edim])
        Dout_1 = tf.add(Cout_1, Bout3dim_1)

        return Dout_0, Dout_1

    def _inference_distance(self, u_0, u_1):
        # Dot product layer
        n, m, f = u_0.get_shape()
        u_0 = tf.reshape(u_0, [int(n * m), int(f)])
        u_1 = tf.reshape(u_1, [int(n * m), int(f)])
        u_0 = tf.nn.l2_normalize(u_0, dim=1, epsilon=1e-12, name=None)
        u_1 = tf.nn.l2_normalize(u_1, dim=1, epsilon=1e-12, name=None)

        if self.distance == 'in':
            u_ = tf.multiply(u_0, u_1)
            u_ = tf.reduce_sum(u_, 1, keep_dims=True)
            u_ = tf.reshape(u_, [int(n), int(m), 1])

        elif self.distance == 'bi':
            with tf.variable_scope("bilinear"):
                W = tf.get_variable("W", shape=[f, f], initializer=tf.contrib.layers.xavier_initializer())
                self.regularizers.append(tf.nn.l2_loss(W))
                transform_left = tf.matmul(u_0, W)
                u_  = tf.reduce_sum(tf.multiply(transform_left, u_1), 1, keep_dims=True)
                u_ = tf.reshape(u_, [int(n), int(m), 1])
        return u_

    def _inference(self, x, recs, dropout):
        self.build_var()
        u_0 = tf.squeeze(x[:, :, :, 0])
        u_1 = tf.squeeze(x[:, :, :, 1])
        recs_0 = tf.squeeze(recs[:, :, :, 0])
        recs_1 = tf.squeeze(recs[:, :, :, 1])
        Ain_0, Ain_1, Bin_0, Bin_1 = self.build_memory(recs_0, recs_1)

        for ihop in range(self.nhops):
            y_0, y_1 = self._inference_gcn(u_0, u_1, dropout, ihop) # y_0, y_1 size is (batch_size, n_nodes, fdim)
            u_0, u_1 = self._inference_memory(y_0, y_1, Ain_0, Ain_1, Bin_0, Bin_1) # u_0, u_1 size is (batch_size, n_nodes, fdim)

        u_ = self._inference_distance(u_0, u_1)

        # Fully connected hidden layers.
        n, m, f = u_.get_shape() # f = 1 here
        u_ = tf.reshape(u_, [int(n), int(m*f)])  # n x m

        for i, M in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                u_ = self.fc(u_, M)
                u_ = tf.nn.dropout(u_, dropout)

        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            prob = self.fc(u_, self.M[-1], relu=False)

        return prob
