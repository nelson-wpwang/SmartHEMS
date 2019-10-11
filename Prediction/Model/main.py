from LDA_Prediction import *
from LDA_prepare_data import *


state_data, dev_list = get_data()
chosen_data = get_selected_data(chosen_start_date, chosen_end_date, state_data)
X_data, Q = prepare_data(chosen_data)
X = change_frame(X_data)

X = np.array(X)
Q = np.array(Q)
# for item in X:
# 	print(item)

model = algo(1, [1,1,1,1,1,1,1], 1, 1, 20, X, Q, False)
model.train(3000)
# model.infer()
