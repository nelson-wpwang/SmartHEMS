# from LDA_Prediction import *
from Data_Prep import *
# from sklearn.externals import joblib


state_data, power_data, dev_list = get_data()
print(dev_list)
chosen_data, Q, dev_daily_data, daily_op_sum, distribution = get_selected_data(state_data)
get_selected_power_data(power_data)
# X_data, Q = prepare_data(chosen_data)
#X = change_frame(chosen_data)

X = np.array(chosen_data)
print(X.shape)
dev_daily_data = np.array(dev_daily_data)
print(dev_daily_data.shape)
Q = np.array(Q)
print(Q.shape)
daily_op_sum = np.array(daily_op_sum)
print(daily_op_sum.shape)

#print(distribution.shape)
index = training_testing_index(dev_daily_data)

# print(index)

#get device association rules
# pat, rul, count, associ_rslt, associ_rule = dev_association(X)
# print(count)
# print(pat)
# # print(associ_rslt)
# for item in associ_rule:
# 	print(item)


#write data to training and testing
# N = LSTM_prep(X, Q)
# print(N.shape)
# chopt_data(N)


