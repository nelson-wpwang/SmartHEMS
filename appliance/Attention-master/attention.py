from __future__ import division
from __future__ import print_function

from datetime import datetime
from utils import *

import tensorflow as tf
import numpy as np
import sys
import os
import pprint

pp = pprint.PrettyPrinter().pprint

def density_mixture_output(input):
    coef, mu, sigma_sqrt = tf.split(input, 3, axis=1)
    coef = tf.nn.softmax(coef, axis=1)
    return coef, mu, sigma_sqrt


class AttentionNN(object):
    def __init__(self, sess, **config):

        name = config.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        self.sess = sess
        allowed_config = ['hidden_size', 'num_layers', 'batch_size', 'max_size', 'dropout',
                          'epochs', 'minval', 'maxval', 'lr_init', 'max_grad_norm', 'source_size',
                          'emb_size', 'is_test', 'name', 'checkpoint_dir', 'target_size', 'obser_dev',
                          'sample', 'train_data_path', 'test_data_path', 'random_seed', 'pred_dev',
                          'attention', 'GMM']
        for key in config.keys():
            if key not in allowed_config:
                raise ValueError('%s is not an allowed argument'%key)
        self.GMM           = config['GMM']
        self.atten         = config['attention']
        self.random_seed   = config['random_seed']
        self.hidden_size   = config['hidden_size']
        self.num_layers    = config['num_layers']
        self.batch_size    = config['batch_size']
        self.max_size      = config['max_size']
        self.init_dropout  = config['dropout']
        self.epochs        = config['epochs']
        self.minval        = config['minval']
        self.maxval        = config['maxval']
        self.lr_init       = config['lr_init']
        self.max_grad_norm = config['max_grad_norm']
        self.emb_size      = config['emb_size']
        self.is_test       = config['is_test']
        self.obser_dev     = np.array(config['obser_dev'])
        self.pred_dev      = config['pred_dev']
        self.checkpoint_dir = config['checkpoint_dir']

        self.train_data_path  = config['train_data_path']
        self.test_data_path  = config['test_data_path']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_init)

        self.source_size = 25 * len(self.obser_dev)
        if not self.GMM:
            self.target_size = 12
        else:
            self.target_size = 1

        self.train_iters = 0

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)


        self.source     = tf.placeholder(tf.float32, [self.batch_size, self.max_size, self.source_size], name="source")
        self.target     = tf.placeholder(tf.float32, [self.batch_size, self.target_size], name="target")
        self.dropout    = tf.placeholder(tf.float32, name="dropout")

        self.build_variables()
        self.build_model()

    def build_variables(self):
        #self.lr = tf.Variable(self.lr_init, trainable=False, name="lr")
        initializer = tf.random_uniform_initializer(self.minval, self.maxval)

        with tf.variable_scope("encoder"):
            self.s_emb_W = tf.get_variable('s_emb_W', shape=[self.source_size, self.emb_size],
                                         initializer=initializer)
            self.s_emb_b = tf.get_variable('s_emb_b', shape=[self.emb_size],
                                         initializer=initializer)
            self.s_proj_W = tf.get_variable("s_proj_W", shape=[self.emb_size, self.hidden_size],
                                            initializer=initializer)
            self.s_proj_b = tf.get_variable("s_proj_b", shape=[self.hidden_size],
                                            initializer=initializer)
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1-self.dropout))
            self.encoder = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

        with tf.variable_scope("decoder"):
            self.t_emb_W = tf.get_variable('t_emb_W', shape=[1, self.emb_size],
                                         initializer=initializer)
            self.t_emb_b = tf.get_variable('t_emb_b', shape=[self.emb_size],
                                         initializer=initializer)
            self.t_proj_W = tf.get_variable("t_proj_W", shape=[self.emb_size, self.hidden_size],
                                            initializer=initializer)
            self.t_proj_b = tf.get_variable("t_proj_b", shape=[self.hidden_size],
                                            initializer=initializer)
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1-self.dropout))
            self.decoder = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

            # projection
            self.proj_W = tf.get_variable("W", shape=[self.hidden_size, self.emb_size],
                                          initializer=initializer)
            self.proj_b = tf.get_variable("b", shape=[self.emb_size],
                                          initializer=initializer)
            self.proj_Wo = tf.get_variable("Wo", shape=[self.emb_size, self.target_size],
                                           initializer=initializer)
            self.proj_bo = tf.get_variable("bo", shape=[self.target_size],
                                           initializer=initializer)

            # attention
            self.v_a = tf.get_variable("v_a", shape=[self.hidden_size, self.target_size],
                                       initializer=initializer)
            self.W_a = tf.get_variable("W_a", shape=[2*self.hidden_size, self.hidden_size],
                                       initializer=initializer)
            self.b_a = tf.get_variable("b_a", shape=[self.hidden_size],
                                       initializer=initializer)
            self.W_c = tf.get_variable("W_c", shape=[2*self.hidden_size, self.hidden_size],
                                       initializer=initializer)
            self.b_c = tf.get_variable("b_c", shape=[self.hidden_size],
                                       initializer=initializer)

        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.trainable_vars = {var.name: var for var in trainable_variables}

    def build_model(self):
        with tf.variable_scope("encoder"):
            source_xs = tf.split(self.source, self.max_size, 1)
            for t in range(self.max_size):
                if t > 0: tf.get_variable_scope().reuse_variables()
                source_xs[t] = tf.squeeze(source_xs[t], [1])
                source_xs[t] = tf.matmul(source_xs[t], self.s_emb_W) + self.s_emb_b


        s = self.encoder.zero_state(self.batch_size, tf.float32)
        encoder_hs = []
        with tf.variable_scope("encoder"):
            for t in range(self.max_size):
                if t > 0: tf.get_variable_scope().reuse_variables()
                x = source_xs[t]
                x = tf.matmul(x, self.s_proj_W) + self.s_proj_b
                h, s = self.encoder(x, s)
                encoder_hs.append(h)
        encoder_hs = tf.stack(encoder_hs[:-1])

        with tf.variable_scope("decoder"):
            if self.atten:
                h_tld = self.attention(h, encoder_hs)
            else:
                h_tld = h
            oemb  = tf.matmul(h_tld, self.proj_W) + self.proj_b
            if not self.GMM:
                logit = tf.nn.relu(tf.matmul(oemb, self.proj_Wo) + self.proj_bo)
                self.prob = tf.nn.softmax(logit, axis=1)
                if self.is_test:
                    prob = tf.nn.softmax(logit, axis=1)


                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target, logits=logit))

                # self.probs = tf.transpose(tf.stack(probs), [1, 0, 2])

                # if not self.grads_clip:
                # self.opt_op = self.optimizer.minimize(self.loss)
                # else:


            else:
                self.pred = tf.matmul(oemb, self.proj_Wo) + self.proj_bo
                target = tf.cast(self.target, tf.float32)
                self.loss = tf.reduce_mean(tf.pow((self.pred - target), 2))

        clipped = [(tf.clip_by_norm(grads if grads is not None else tf.zeros_like(vars), self.max_grad_norm), vars) for grads, vars in self.optimizer.compute_gradients(self.loss)]
        # self.clipped = clipped
        self.opt_op = self.optimizer.apply_gradients(clipped)



        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()

    def attention(self, h_t, encoder_hs):
        #scores = [tf.matmul(tf.tanh(tf.matmul(tf.concat(1, [h_t, tf.squeeze(h_s, [0])]),
        #                    self.W_a) + self.b_a), self.v_a)
        #          for h_s in tf.split(0, self.max_size, encoder_hs)]
        #scores = tf.squeeze(tf.pack(scores), [2])
        scores = tf.reduce_sum(tf.multiply(encoder_hs, h_t), 2)
        a_t    = tf.nn.softmax(tf.transpose(scores))
        a_t    = tf.expand_dims(a_t, 2)
        c_t    = tf.matmul(tf.transpose(encoder_hs, perm=[1,2,0]), a_t)
        c_t    = tf.squeeze(c_t, [2])
        h_tld  = tf.tanh(tf.matmul(tf.concat([h_t, c_t], axis=1), self.W_c) + self.b_c)

        return h_tld

    def get_model_name(self):
        date = datetime.now()
        return "{}-{}-{}-{}-{}".format(self.name, self.dataset, date.month, date.day, date.hour)

    def train(self, epoch):
        #if epoch > 10 and epoch % 5 == 0 and self.lr_init > 0.00025:
        #    self.lr_init = self.lr_init*0.75
        #    self.lr.assign(self.lr_init).eval()

        total_loss = 0.
        i = 0

        data_gen = iterate_data(self.train_data_path)

        for data in data_gen:
            list = data
            if len(list) != self.batch_size:
                continue
            input_list = [(x[self.obser_dev, :(self.max_size), :].transpose([1, 0, 2])).reshape(self.max_size, -1) for x in list]
            batch_input = np.stack(input_list, axis=0)
            # print(batch_input.shape)  #should be batch, time, features
            output_list = [x[self.pred_dev, -1, :].reshape([1, -1]) for x in list]
            batch_output = np.squeeze(np.stack(output_list, axis=0), axis=1)

            batch_output = batch_output[:, :-1]
            batch_num = np.sum(batch_output, axis=1)
            if not self.GMM:
                batch_class = np.zeros((self.batch_size, self.target_size))
                batch_class[np.arange(self.batch_size), batch_num] = 1

    # for dsource, slen, dtarget, tlen in iterator:
                prob, loss, _= self.sess.run([self.prob, self.loss, self.opt_op],
                                        feed_dict={self.source: batch_input,
                                                   self.target: batch_class,
                                                   self.dropout: self.init_dropout})
            else:
                batch_num = np.expand_dims(batch_num.astype(float), axis=-1)
                pred, loss, _ = self.sess.run([self.pred, self.loss, self.opt_op],
                                        feed_dict={self.source: batch_input,
                                                   self.target: batch_num,
                                                   self.dropout: self.init_dropout})



            # for grads, vars in grad:
            #     print('grads are {}'.format(grads))
            #     print('vars are {}'.format(vars))
            # print(prob)



            itr  = self.train_iters*epoch + i
            total_loss += loss
            # if itr % 2 == 0:
            #     print("[Train] [Time: {}] [Epoch: {}] [Iteration: {}] [lr: {}] [Loss: {}] [Perplexity: {}]"
            #           .format(datetime.now(), epoch, itr, self.lr_init, loss, np.exp(loss)))
            #     sys.stdout.flush()
            i += 1
            self.train_iters = i
        return total_loss/i

    def test(self):
        total_loss = 0
        i = 0
        data_gen = iterate_data(self.test_data_path)

        for data in data_gen:
            list = data
            if len(list) != self.batch_size:
                continue
            input_list = [(x[self.obser_dev, :(self.max_size), :].transpose([1, 0, 2])).reshape(self.max_size, -1) for x in list]
            batch_input = np.stack(input_list, axis=0)
            # print(batch_input.shape)  #should be batch, time, features
            output_list = [x[self.pred_dev, -1, :].reshape([1, -1]) for x in list]
            batch_output = np.squeeze(np.stack(output_list, axis=0), axis=1)

            batch_output = batch_output[:, :-1]
            batch_num = np.sum(batch_output, axis=1)
            if not self.GMM:
                batch_class = np.zeros((self.batch_size, self.target_size))
                batch_class[np.arange(self.batch_size), batch_num] = 1

    # for dsource, slen, dtarget, tlen in iterator:
                prob, loss = self.sess.run([self.prob, self.loss],
                                        feed_dict={self.source: batch_input,
                                                   self.target: batch_class,
                                                   self.dropout: 0.0})

                pp('Groud truth {}'.format(np.argmax(batch_class, axis=1)))
                pp('Predicted {}'.format(np.argmax(prob, axis=1)))
            else:
                batch_num = np.expand_dims(batch_num.astype(float), axis=-1)
                pred, loss= self.sess.run([self.pred, self.loss],
                                        feed_dict={self.source: batch_input,
                                                   self.target: batch_num,
                                                   self.dropout: 0.0})

                pp('Groud truth {}'.format(batch_num))
                pp('Predicted {}'.format(pred))



            total_loss += loss
            i += 1

        total_loss /= i
        return total_loss

    def sample(self, source_data_path):
        data_gen = iterate_data(self.test_data_path)
        sample = []
        truth = []
        true_dis = []
        wods = []

        for data in data_gen:
            list = data
            if len(list) != self.batch_size:
                continue
            input_list = [(x[self.obser_dev, :(self.max_size), :].transpose([1, 0, 2])).reshape(self.max_size, -1) for x in list]
            batch_input = np.stack(input_list, axis=0)
            # print(batch_input.shape)  #should be batch, time, features
            output_list = [x[self.pred_dev, -1, :].reshape([1, -1]) for x in list]
            batch_output = np.squeeze(np.stack(output_list, axis=0), axis=1)

            wods.append(batch_output[:, -1])
            batch_output = batch_output[:, :-1]
            batch_num = np.sum(batch_output, axis=1)
            if not self.GMM:
                batch_class = np.zeros((self.batch_size, self.target_size))
                batch_class[np.arange(self.batch_size), batch_num] = 1

    # for dsource, slen, dtarget, tlen in iterator:
                prob, loss, _= self.sess.run([self.prob, self.loss, self.opt_op],
                                        feed_dict={self.source: batch_input,
                                                   self.target: batch_class,
                                                   self.dropout: self.init_dropout})
                samples = np.argmax(prob, axis=1)
                sample.append(samples)
                truth.append(batch_num)
                true_dis.append(batch_output)
            else:
                batch_num = np.expand_dims(batch_num.astype(float), axis=-1)
                pred, loss, _ = self.sess.run([self.pred, self.loss, self.opt_op],
                                        feed_dict={self.source: batch_input,
                                                   self.target: batch_num,
                                                   self.dropout: self.init_dropout})
                pred = pred.astype(int)
                pred[np.where(pred < 0)] = 0
                sample.append(pred)
                truth.append(batch_num)
                true_dis.append(batch_output)

        return sample, truth, true_dis, wods

    def run(self):

        best_valid_loss = float("inf")
        for epoch in range(self.epochs):
            train_loss = self.train(epoch)
            valid_loss = self.test()
            print("[Train] [Avg. Loss: {}] [Avg. Perplexity: {}]".format(train_loss, np.exp(train_loss)))
            print("[Valid] [Loss: {}] [Perplexity: {}]".format(valid_loss, np.exp(valid_loss)))
            # self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.name + ".epoch" + str(epoch)))
            if epoch == 0 or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save()
                # self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.name + ".bestvalid"))

    def save(self):
        saved_vars = {}
        if not self.sess:
            raise AttributeError("TensorFlow session not provided.")
        for key, value in self.trainable_vars.items():
            saved_vars[key] = value.eval(session=self.sess)

        if not os.path.exists('./tmp'):
            os.makedirs('./tmp')
        save_path = "tmp/%s.pickle" % self.name
        f = open(save_path, 'wb')
        pickle.dump(saved_vars, f)
        print("Model saved in file: %s" % save_path)

    def load(self):
        print("[*] Reading checkpoints...")
        if not self.sess:
            raise AttributeError("TensorFlow session not provided.")
        save_path = "tmp/%s.pickle" % self.name
        f = open(save_path, 'rb')
        saved_vars = pickle.load(f)

        for key, item in self.trainable_vars.items():
            self.sess.run(tf.assign(item, saved_vars[key]))
