import pickle
import numpy as np

small_penalty = 0.1
large_penalty = 0.2

device_number = 5
house_name = 'Dataport_1464'

#input: 3 array, power, ground truth, predicted
#data frame: Number of devices * days * times per day
power_path = '/Users/nelson/Academia/Research/Work/SmartHEMS/appliance/Attention-master/data/pre-process/Dataport/1464/1464_power.pkl'
ground_truth_path = '/Users/nelson/Academia/Research/Work/SmartHEMS/appliance/Attention-master/data/%s_%d_true_usage.pickle'%(house_name, device_number)
predicted_path = '/Users/nelson/Academia/Research/Work/SmartHEMS/appliance/Attention-master/data/%s(%d)_%dr_prediction.pickle'%(house_name, device_number, device_number)

file = open(power_path, 'rb')
power = pickle.load(file)
file.close()

file = open(ground_truth_path, 'rb')
ground_truth = pickle.load(file)
file.close()

file = open(predicted_path, 'rb')
predicted = pickle.load(file)
file.close()

ground_truth = np.array(ground_truth)
ground_truth = ground_truth.reshape(-1, 24)
predicted = np.array(predicted)


print(ground_truth.shape)
# print(ground_truth)
print(predicted.shape)
print(predicted)

power = np.array(power)
power = power[device_number, int(power.shape[1]*0.75):int(power.shape[1]*0.75) + ground_truth.shape[0], :]

sum_stby = list()
sum_saved = list()
sum_small_penalty = list()
sum_large_penalty = list()

saved_ratio = list()
small_penalty_ratio = list()
large_penalty_ratio = list()

#energy saving rule: if in use but predicted negative, small penalty = 0.25* active power, large penalty = 0.5*active power; 
#					 if not in use but predicted active, no saving. 

tmp_sum_stby = 0
tmp_sum_saved = 0
tmp_small_penalty = 0
tmp_large_penalty = 0
sum_right = 0
sum_wrong_active = 0
for days in range(power.shape[0]):
	for time in range(power.shape[1]):
		if ground_truth[days, time] == 0:
			tmp_sum_stby += power[days, time]
			if time not in predicted[days]:
				tmp_sum_saved += power[days, time]
				sum_right += 1
		if ground_truth[days, time] == 1 and time not in predicted[days]:
			tmp_small_penalty += power[days, time] * small_penalty
			tmp_large_penalty += power[days, time] * large_penalty
			sum_wrong_active += 1
sum_stby.append(tmp_sum_stby)
sum_saved.append(tmp_sum_saved)
saved_ratio.append(float(tmp_sum_saved/tmp_sum_stby))
sum_small_penalty.append(tmp_sum_saved - tmp_small_penalty)
small_penalty_ratio.append(float((tmp_sum_saved - tmp_small_penalty)/tmp_sum_stby))
sum_large_penalty.append(tmp_sum_saved - tmp_large_penalty)
large_penalty_ratio.append(float((tmp_sum_saved - tmp_large_penalty)/tmp_sum_stby))


print('Tested device: ', device_number)
print('Total standby enegy consumed: ', sum_stby)
print('Total saved energy in idle', sum_saved)
print('Total saved energy with small penalty', sum_small_penalty)
print('Total saved energy with large penalty', sum_large_penalty)
print('Total saved energy with small penalty', small_penalty_ratio)
print('Total saved energy with large penalty', large_penalty_ratio)
print(sum_right, sum_wrong_active)
