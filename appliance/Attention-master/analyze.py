from utils import *
import os
import numpy as np
import pylab as pl
from matplotlib import collections as mc
import pprint
import matplotlib.pyplot as plt

device_number = 5
house_name = 'Dataport_1464'

pp = pprint.PrettyPrinter().pprint
# Load data to be analyzed
model1 = './data/%s(%d)_%dr_valid.pickle'%(house_name, device_number, device_number)
with open(model1, 'rb') as f:
    model1 = pickle.load(f)
usages1, true_usages, true_pattern, wods = model1


# model2 = './data/(3)_3r_valid.pickle'
# with open(model2, 'rb') as f:
#     model2 = pickle.load(f)
# usages2, _, _, _  = model2

dis_dir = './data/pre-process/Dataport/1464/1464_distribution.pkl'
with open(dis_dir, 'rb') as f:
    dis = pickle.load(f)
print(dis.shape)
dis = process_dis(dis)
dis = dis[device_number, :, :]
print('Distribution sizing in analyze.py: ', dis.shape)


# predicted_usage_cor, predicted_usage = metrics_calculate(usages1, usages2, dis, true_pattern, wods, iters=20)
predicted_usage = metrics_calculate(usages1, dis, true_pattern, wods, iters=20)
print(true_pattern)
print(predicted_usage)

model1 = './data/%s_%d_true_usage.pickle'%(house_name, device_number)
with open(model1, 'wb') as f:
    pickle.dump(true_pattern, f)
f.close()

# model1 = './data/(2,5,3)_3r_prediction.pickle'
# with open(model1, 'wb') as f:
#     pickle.dump(predicted_usage_cor, f)
# f.close()

model2 = './data/%s(%d)_%dr_prediction.pickle'%(house_name, device_number, device_number)
with open(model2, 'wb') as f:
    pickle.dump(predicted_usage, f)
f.close()
# for idx, _ in enumerate(usages1):
#     pp([np.squeeze(usages1[idx], axis=-1), np.squeeze(usages2[idx], axis=-1), np.squeeze(true_usages[idx], axis=-1)])

# # metrics_calculate(usages1, usages2, dev1_dis, true_pattern, wods)
#
#
# indices = [(2, 0),(5, 4)]
# for idx in indices:
#     time1 = usages1[idx[0]][idx[1]]
#     time2 = usages2[idx[0]][idx[1]]
#     true_time = true_usages[idx[0]][idx[1]]
#     wod = wods[idx[0]][idx[1]]
#     print(wod)
#     true_p = true_pattern[idx[0]][idx[1]]
#
#     a, b, c, d, e, f = get_coordinates(time1, time2, dis[3, wod, :], true_p)
#
#     plt.subplot(3, 1, 1)
#     plt.ylim(-0.5, 1.5)
#     plt.xlim(0, 23)
#     plt.xlabel('Time (hour)')
#     plt.ylabel('State')
#     plt.xticks([0, 4, 8, 12, 16, 20, 23], ['0', '4', '8', '12', '16', '20', '23'])
#     plt.yticks([0, 1], ['off', 'on'])
#     plt.step(a, b, linestyle=':')
#
#
#     plt.subplot(3, 1, 2)
#     plt.ylim(-0.5, 1.5)
#     plt.xlim(0, 23)
#     plt.xlabel('Time (hour)')
#     plt.ylabel('State')
#     plt.xticks([0, 4, 8, 12, 16, 20, 23], ['0', '4', '8', '12', '16', '20', '23'])
#     plt.yticks([0, 1], ['off', 'on'])
#     plt.step(c, d, linestyle='-.')
#
#     plt.subplot(3, 1, 3)
#     plt.ylim(-0.5, 1.5)
#     plt.xlim(0, 23)
#     plt.xlabel('Time (hour)')
#     plt.ylabel('State')
#     plt.xticks([0, 4, 8, 12, 16, 20, 23], ['0', '4', '8', '12', '16', '20', '23'])
#     plt.yticks([0, 1], ['off', 'on'])
#     plt.step(e, f, linestyle='--')
#
#     plt.show()
#     break

    # c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])
    #
    # lc = mc.LineCollection([a, c, e], colors=c, linewidths=2)
    # fig, ax = pl.subplots()
    # ax.add_collection(lc)
    # ax.autoscale()
    # ax.margins(0.1)
