from __future__ import division
from __future__ import print_function

from datetime import datetime
from data import data_iterator_len
from data import read_vocabulary

import tensorflow as tf
import numpy as np
import sys
import os


class AttentionNN(object):
    def __init__(self, sess, **config):
        name = config.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        self.sess = sess
        allowed_config = ['hidden_size', 'num_layers', 'batch_size', 'max_size', 'dropout',
                          'epochs', 'minval', 'maxval', 'lr_init', 'max_grad_norm', 'source_size',
                          'emb_size', 'is_test', 'name', 'checkpoint_dir', 'target_size']
        for key in arg.keys():
            if key not in allowed_config:
                raise ValueError('%s is not an allowed argument'%key)
        self.hidden_size   = config.get(hidden_size)
        self.num_layers    = config.get(num_layers)
        self.batch_size    = config.get(batch_size)
        self.max_size      = config.get(max_size)
        self.init_dropout  = config.get(dropout)
        self.epochs        = config.get(epochs)
        self.minval        = config.get(minval)
        self.maxval        = config.get(maxval)
        self.lr_init       = config.get(lr_init)
        self.max_grad_norm = config.get(max_grad_norm)
        self.dataset       = config.get(dataset)
        self.emb_size      = config.get(emb_size)
        self.is_test       = config.get(is_test)

        self.source_data_path  = config.source_data_path
        self.target_data_path  = config.target_data_path
        self.source_vocab_path = config.source_vocab_path
        self.target_vocab_path = config.target_vocab_path
        self.checkpoint_dir    = config.checkpoint_dir

        self.train_iters = 0

        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        self.source     = tf.placeholder(tf.int32, [self.batch_size, self.max_size, self.source_size], name="source")
        self.target     = tf.placeholder(tf.int32, [self.batch_size, self.max_size, self.target_size], name="target")
        self.target_len = tf.placeholder(tf.int32, [self.batch_size], name="target_len")
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
            self.t_emb_W = tf.get_variable('t_emb_W', shape=[self.target_size, self.emb_size],
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
            self.proj_Wo = tf.get_variable("Wo", shape=[self.emb_size, self.t_nwords],
                                           initializer=initializer)
            self.proj_bo = tf.get_variable("bo", shape=[self.t_nwords],
                                           initializer=initializer)


    def build_model(self):
        with tf.variable_scope("encoder"):
            source_xs = tf.split(1, self.max_size, self.source)
            for t in range(self.max_size):
                if t > 0: tf.get_variable_scope().reuse_variables()
                source_xs[t] = tf.squeeze(source_xs[t], [1])
                source_xs[t] = tf.matmul(source_xs[t], self.s_emb_W) + self.s_emb_b

        with tf.variable_scope("decoder"):
            target_xs = tf.split(1, self.max_size, self.target)
            for t in range(self.max_size):
                if t > 0: tf.get_variable_scope().reuse_variables()
                target_xs[t] = tf.squeeze(target_xs[t], [1])
                target_xs[t] = tf.matmul(target_xs[t], self.t_emb_W) + self.t_emb_b

        s = self.encoder.zero_state(self.batch_size, tf.float32)
        encoder_hs = []
        with tf.variable_scope("encoder"):
            for t in range(self.max_size):
                if t > 0: tf.get_variable_scope().reuse_variables()
                x = source_xs[t]
                x = tf.matmul(x, self.s_proj_W) + self.s_proj_b
                h, s = self.encoder(x, s)
                encoder_hs.append(h)
        encoder_hs = tf.stack(encoder_hs)

        s = self.decoder.zero_state(self.batch_size, tf.float32)
        logits = []
        probs  = []
        with tf.variable_scope("decoder"):
            for t in range(self.max_size):
                if t > 0: tf.get_variable_scope().reuse_variables()
                if not self.is_test or t == 0:
                    x = target_xs[t]
                x = tf.matmul(x, self.t_proj_W) + self.t_proj_b
                h_t, s = self.decoder(x, s)

                oemb  = tf.matmul(h_t, self.proj_W) + self.proj_b
                logit = tf.nn.relu(tf.matmul(oemb, self.proj_Wo) + self.proj_bo)
                prob  = tf.nn.softmax(logit)
                logits.append(logit)
                probs.append(prob)
                if self.is_test:
                    x = tf.cast(tf.argmax(prob, 1), tf.int32)

        logits = tf.stack(logits, axis=1)[:-1]
        masks = tf.expand_dims(tf.sequence_mask(self.target_len - 1, self.max_size - 1,
                                                dtype=tf.float32), axis=-1)
        dim = logits.get_shape().as_list()
        logits = tf.reshape(logits * masks, shape=[-1, dim[-1]])
        targets = tf.reshape(self.targets[:, 1:, :] * masks, shape=[-1])
        labels = tf.one_hot(targets, depth=2, axis=-1)


        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                            logits=logits
                                                            )

        self.probs = tf.transpose(tf.stack(probs), [1, 0, 2])

        self.optim = tf.contrib.layers.optimize_loss(self.loss, None,
                self.lr_init, "SGD", clip_gradients=5.,
                summaries=["learning_rate", "loss", "gradient_norm"])

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()


    def get_model_name(self):
        date = datetime.now()
        return "{}-{}-{}-{}-{}".format(self.name, self.dataset, date.month, date.day, date.hour)

    def train(self, epoch, merged_sum, writer):
        #if epoch > 10 and epoch % 5 == 0 and self.lr_init > 0.00025:
        #    self.lr_init = self.lr_init*0.75
        #    self.lr.assign(self.lr_init).eval()

        total_loss = 0.
        i = 0
        iterator = data_iterator_len(self.source_data_path,
                                     self.target_data_path,
                                     read_vocabulary(self.source_vocab_path),
                                     read_vocabulary(self.target_vocab_path),
                                     self.max_size, self.batch_size)
        for dsource, slen, dtarget, tlen in iterator:
            outputs = self.sess.run([self.loss, self.optim, merged_sum],
                                    feed_dict={self.source: dsource,
                                               self.target: dtarget,
                                               self.target_len: tlen,
                                               self.dropout: self.init_dropout})
            loss = outputs[0]
            itr  = self.train_iters*epoch + i
            total_loss += loss
            if itr % 2 == 0:
                writer.add_summary(outputs[-1], itr)
            if itr % 10 == 0:
                print("[Train] [Time: {}] [Epoch: {}] [Iteration: {}] [lr: {}] [Loss: {}] [Perplexity: {}]"
                      .format(datetime.now(), epoch, itr, self.lr_init, loss, np.exp(loss)))
                sys.stdout.flush()
            i += 1
        self.train_iters = i
        return total_loss/i

    def test(self, source_data_path, target_data_path):
        iterator = data_iterator_len(source_data_path,
                                     target_data_path,
                                     read_vocabulary(self.source_vocab_path),
                                     read_vocabulary(self.target_vocab_path),
                                     self.max_size, self.batch_size)

        total_loss = 0
        i = 0
        for dsource, slen, dtarget, tlen in iterator:
            loss, = self.sess.run([self.loss],
                                  feed_dict={self.source: dsource,
                                             self.target: dtarget,
                                             self.target_len: tlen,
                                             self.dropout: 0.0})
            total_loss += loss
            i += 1

        total_loss /= i
        return total_loss

    def sample(self, source_data_path):
        source_vocab = read_vocabulary(self.source_vocab_path)
        target_vocab = read_vocabulary(self.target_vocab_path)
        inv_target_vocab = {v:k for k,v in target_vocab.iteritems()}
        iterator = data_iterator_len(source_data_path,
                                     source_data_path,
                                     source_vocab,
                                     target_vocab,
                                     self.max_size, self.batch_size)
        samples = []
        for dsource, slen, dtarget, tlen in iterator:
            dtarget = [[target_vocab["<s>"]] + [target_vocab["<pad>"]]*(self.max_size-1)]
            dtarget = dtarget*self.batch_size
            probs, = self.sess.run([self.probs],
                                   feed_dict={self.source: dsource,
                                              self.target: dtarget,
                                              self.dropout: 0.0})
            for b in range(self.batch_size):
                samples.append([inv_target_vocab[np.argmax(p)] for p in probs[b]])

        return samples

    def run(self, valid_source_data_path, valid_target_data_path):
        merged_sum = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("./logs/{}".format(self.get_model_name()),
                                        self.sess.graph)

        best_valid_loss = float("inf")
        for epoch in range(self.epochs):
            train_loss = self.train(epoch, merged_sum, writer)
            valid_loss = self.test(valid_source_data_path, valid_target_data_path)
            print("[Train] [Avg. Loss: {}] [Avg. Perplexity: {}]".format(train_loss, np.exp(train_loss)))
            print("[Valid] [Loss: {}] [Perplexity: {}]".format(valid_loss, np.exp(valid_loss)))
            self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.name + ".epoch" + str(epoch)))
            if epoch == 0 or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.name + ".bestvalid"))

    def load(self):
        print("[*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("[!] No checkpoint found")
