from LDA_Prediction import *
from LDA_prepare_data import *


state_data, dev_list = get_data()
chosen_data = get_selected_data(chosen_start_date, chosen_end_date, state_data)
X_data, Q_data = prepare_data(chosen_data)
X = change_frame(X_data)

X = np.array(X)
#print(X.shape)

model = algo(1, [1,1,1,1,1,1,1], 1, 1, 50, X, False)
model.train()
