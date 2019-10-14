from LDA_Prediction import *
from LDA_prepare_data import *
from sklearn.externals import joblib


state_data, dev_list = get_data()
chosen_data = get_selected_data(chosen_start_date, chosen_end_date, state_data)
X_data, Q = prepare_data(chosen_data)
X = change_frame(X_data)

X = np.array(X)
Q = np.array(Q)
# for item in X:
# 	print(item)
# for i in range(Q.shape[0]):
# 	if np.argmax(Q[i, :]) == 0:
# 		print(X[i, :])

model = algo(0.5, [1,1,1,1,1,1,1], 1, 5, 10, X, Q, True)
model.train(500001)

for i in range(X.shape[0]):
	model.infer(i)

filename = 'LDA_model.sav'
joblib.dump(model, filename)