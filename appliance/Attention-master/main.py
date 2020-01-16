from __future__ import division
from __future__ import print_function

from attention import AttentionNN
from utils import *


import os
import time
import pprint
import random
import numpy as np
import tensorflow as tf
import pickle


pp = pprint.PrettyPrinter().pprint

flags = tf.app.flags

device_number = 5
house_name = 'Dataport_1464'

config = {"GMM":True,
          "attention":False,
          "obser_dev":[device_number],
          "pred_dev":[device_number], #predict 1 [8 elements], predict 4[10 elements]
          "max_size": 7,
          "batch_size": 8,
          "random_seed": 123,
          "epochs": 128, #original 50
          "hidden_size": 64,
          "emb_size": 64,
          "num_layers": 4,
          "dropout": 0.2, # original 0.1
          "minval": -2,
          "maxval": 2,
          "lr_init": 0.005, #original 0.005
          "max_grad_norm": 20.0, #original 5.0
          "checkpoint_dir": './tmp/',
          "name": '(%d)_%dr'%(device_number, device_number),
          "is_test": False,
          "sample": True,
          "train_data_path": './shuffled/1464_shuffled_train.pkl',
          "test_data_path": './shuffled/1464_shuffled_test.pkl' }
 

random.seed(config['random_seed'])



def main():

    pp(config)
    with tf.Session() as sess:
        attn = AttentionNN(sess, **config)
        if config['sample']:
            attn.load()
            samples, truth, true_dis, wods = attn.sample(config['sample'])
            filename = house_name + attn.name + '_valid.pickle'
            with open(os.path.join('./data', filename), 'wb') as f:
                pickle.dump([samples, truth, true_dis, wods], f)
            # print_samples(samples)
        else:
            if not config['is_test']:
                attn.run()
            else:
                attn.load()
                loss = attn.test()
                print("[Test] [Loss: {}] [Perplexity: {}]".format(loss, np.exp(loss)))
                samples = attn.sample(test_source_data_path)
                # get_bleu_score(samples, data_config.test_target_data_path)

if __name__ == '__main__':
    main()
