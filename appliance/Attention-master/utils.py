import os
import numpy as np
from tqdm import tqdm
import pickle
from random import shuffle
import math
import tensorflow as tf



def batch_shuffle(data_dir, indices_dir, batch_size=8, max_size=8):
    assert os.path.isfile(data_dir), '%s is not valid data path'%data_dir
    assert os.path.isfile(indices_dir), '%s is not valid data path'%indices_dir

    with open(data_dir, 'rb') as f:
        dt = np.array(pickle.load(f))
    with open(indices_dir, 'rb') as f:
        idx_dt = pickle.load(f)

    print(dt.shape)
    
    # idx_dt = idx_dt[indices, :]
    indices = np.arange(len(idx_dt))
    train_idx = math.floor(len(indices) * 0.75)

    train_idx_dt = idx_dt[:train_idx, :]
    train_indices = np.arange(train_idx)
    shuffle(train_indices)
    train_idx_dt = train_idx_dt[train_indices, :]
    test_idx_dt = idx_dt[train_idx:, :]


    train_idx_batches = [train_idx_dt[i:i+batch_size] for i in range(0, len(train_idx_dt), batch_size)]
    test_idx_batches = [test_idx_dt[i:i+batch_size] for i in range(0, len(test_idx_dt), batch_size)]

    print(len(train_idx_batches))

    shuffled_dir = './shuffled'
    if not os.path.exists(shuffled_dir):
        os.mkdir(shuffled_dir)

    shuffled_train = os.path.join(shuffled_dir, 'dataport_3577_shuffled_train.pkl')
    shuffled_test = os.path.join(shuffled_dir, 'dataport_3577_shuffled_test.pkl')

    train_dt = []
    for train_idx_batch in train_idx_batches:
        batch_data = []
        for batch_idx in range(train_idx_batch.shape[0]):
            indices = train_idx_batch[batch_idx, :]
            single_entry = dt[:, indices, :]
            batch_data.append(single_entry)
        train_dt.append(batch_data)

    print(len(train_dt))

    with open(shuffled_train, 'wb') as f:
        pickle.dump(train_dt, f)

    test_dt = []
    for test_idx_batch in test_idx_batches:
        batch_data = []
        for batch_idx in range(test_idx_batch.shape[0]):
            indices = test_idx_batch[batch_idx, :]
            single_entry = dt[:, indices, :]
            batch_data.append(single_entry)
        test_dt.append(batch_data)

    with open(shuffled_test, 'wb') as f:
        pickle.dump(test_dt, f)
    print(test_dt[0])

def sample_usage(dev_num, wod, times, dis_path='./data/pre-process/Dataport/3577/3577_distribution.pkl'):
    with open(dis_path, 'rb') as f:
        dis = pickle.load(f)
    device_dis = dis[dev_num, wod, :]
    if times != 0:
        time_indices = np.random.choice(25, times, p=device_dis)
    return time_indices


def iterate_data(data_path):
    assert os.path.isfile(data_path)
    with open(data_path, 'rb') as f:
        dt = pickle.load(f)
    for batch in dt:
        yield batch


# def metrics_calculate(cor_usage, usage, dist, true_states, wods, iters=1):
def metrics_calculate(usage, dist, true_states, wods, iters=1):
    PREC_cor = 0
    TPR_cor = 0
    F1_cor = 0

    PREC = 0
    TPR = 0
    F1 = 0
    
    for i in range(iters):
        output_prediction = list()
        output_prediction_cor = list()
        P = 0
        N = 0
        FP_cor = 0
        TP_cor = 0
        FP = 0
        TP = 0
        for batch_idx in range(len(usage)):
            batch_w = wods[batch_idx]
            batch_pattern = true_states[batch_idx]
            if usage[batch_idx].shape[0] != 8:
                continue
            for ele_idx in range(usage[batch_idx].shape[0]):
                # cor_u = cor_usage[batch_idx][ele_idx]
                print(batch_idx)
                print(ele_idx)
                u = usage[batch_idx][ele_idx]
                w = batch_w[ele_idx]
                true_pattern = batch_pattern[ele_idx, :]
                true_on = np.where(true_pattern == 1)[0]
                true_off = np.where(true_pattern == 0)[0]
                P += len(true_on)
                N += len(true_off)
                # cor_u = min(cor_u, len(np.where(dist[w, :] != 0)[0]))
                u = min(u, len(np.where(dist[w, :] != 0)[0]))
                # cor_on = sorted(np.random.choice(len(true_pattern), cor_u, p=dist[w, :], replace = False))
                # output_prediction_cor.append(cor_on)
                print('True_pattern length: ', len(true_pattern))
                print('Distribution: ', dist[w, :])
                if np.isnan(dist[w, :]).any():
                    on = []
                else:
                    on = sorted(np.random.choice(len(true_pattern), u, p=dist[w, :], replace = False))
                output_prediction.append(on)
                # TP_cor += len(set(cor_on) & set(true_on))
                # FP_cor += len(set(cor_on) & set(true_off))
                
                TP += len(set(on) & set(true_on))
                FP += len(set(on) & set(true_off))
    #     print('Positive [%d], Negative [%d], TP_cor[%d], FP_cor[%d], TP[%d], FP[%d]'%(P, N, TP_cor, FP_cor, TP, FP))
        print('Positive [%d], Negative [%d], TP[%d], FP[%d]'%(P, N, TP, FP))
    #     PREC_cor += TP_cor / (TP_cor + FP_cor)
    #     TPR_cor += TP_cor / P
    #     F1_cor += 2 * PREC_cor * TPR_cor / (PREC_cor + TPR_cor)

        PREC += TP / (TP + FP)
        TPR += TP / P
        F1 += 2 * PREC * TPR / (PREC + TPR)

    # PREC_cor /= iters
    # TPR_cor /= iters
    # F1_cor /= iters

    PREC /= iters
    TPR /= iters
    F1 /= iters

    # print('PREC_cor [%.4f], TPR_cor [%.4f], F1_cor [%.4f], PREC [%.4f], TPR [%.4f], F1 [%.4f]'%(PREC_cor, TPR_cor, F1_cor, PREC, TPR, F1))
    print('PREC [%.4f], TPR [%.4f], F1 [%.4f]'%(PREC, TPR, F1))

    return output_prediction





            # for row_idx in range(cor_usage[batch_idx].shape[0]):
            #     w_row = true_states[batch_idx][row_idx, -1]
            #     pattern_row = true_states[batch_idx][row_idx, :-1]
            #     for idx, u in enumerate(cor_usage[batch_idx][row_idx, :]):
            #         w = w_row[idx]
            #         true_pattern = pattern_row[idx]
            #         cor_state = np.random.choice(true_pattern.shape[-1], u, p=dist[w, :])
            #         state = np.random.choice(true_pattern.shape[-1], usage[batch_idx][row_idx, idx], p=dis[w, :])
            #         print(w)
            #         print(true_pattern)
            #         print(cor_state)
            #         print(state)



def get_coordinates(cor_usage, usage, profile, true_states):
    cor_on = np.random.choice(len(profile), cor_usage, p=profile, replace=False)
    cor_ys = np.zeros_like(profile)
    cor_ys[cor_on] = 1
    cor_xs = np.arange(len(profile))
    on = np.random.choice(len(profile), usage, p=profile, replace=False)
    on_ys = np.zeros_like(profile)
    on_ys[on] = 1
    true_on = np.where(true_states)[0]
    true_ys = np.zeros_like(profile)
    true_ys[true_on] = 1
    return cor_xs, cor_ys, cor_xs, on_ys, cor_xs, true_ys

def get_lines(on):

    return np.array(sorted(line_segments)), np.array(state_segments)

def process_dis(dis):
    #dis = dis[:, :, :-1]

    norm_factor = np.sum(dis, axis=-1, keepdims=True)
    new_dis = dis / norm_factor
    return new_dis



def cross_entropy_with_logits(logits, labels):
    probs = tf.nn.softmax(logits, axis=1)
    loss = tf.reduce_mean(-tf.log(tf.reduce_sum(tf.multiply(probs, labels), axis=1)))
    return loss

if __name__ == '__main__':
    batch_shuffle('./data/pre-process/Dataport/3577/3577_dataset.pkl', './data/pre-process/Dataport/3577/3577_indexing.pkl', batch_size=8, max_size=8)
