# from hmmlearn import hmm
import numpy as np
import scipy as sp
import copy
import pandas as pd
import os
import csv
import math
import matplotlib.pyplot as plt
import datetime
import time
import pickle
import scipy.io as sio
from tqdm import tqdm
import itertools
from math import ceil
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_random_state
import pyfpgrowth as fpgrowth
from apyori import apriori

#Set the threshold to mark the active status
threshold_microwave = 4
threshold_kitchenapp = 3
threshold_oven = 5
threshold_clotheswasher = 75
threshold_dishwasher = 5
threshold_lightsplug = 100
threshold_3577_furnace = 10
threshold_3577_dryg = 30
threshold_114_drye = 30
threshold_114_bedroom = 20
time_interval = 60 #min of time interval
freq_interval = int(60/time_interval)
chosen_start_date = '2013-02-05'
chosen_end_date = '2014-02-04'

house_num = 1464
# f = open('prepare_data_log', 'w')

starttime = datetime.datetime.strptime(chosen_start_date, '%Y-%m-%d')
endtime = datetime.datetime.strptime(chosen_end_date, '%Y-%m-%d')

starttime = datetime.datetime(starttime.year, starttime.month, starttime.day, 0, 0)
endtime = datetime.datetime(endtime.year, endtime.month, endtime.day, 23, 59)
def week_of_month(dt):
    #This will add the first day of the month weekday & present day of month
    first_day = dt.replace(day=1)
    day_of_month = dt.day
    #print day_of_month
    adj_day_of_month = day_of_month + first_day.weekday()
    return int(ceil(adj_day_of_month/7.0))
    

def get_data(path = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/Dataport %d'%house_num):
    if not os.path.exists(path):
        raise('Directory not exists')

    gt_data = list()# L*(N*1440)*5
    power_data = list()
    file_list = list()

    for file in os.listdir(path):
        if file.startswith('.'):
            continue
        file_list.append(file)
        file_path = os.path.join(path, file)
        if not os.path.isfile(file_path):
            continue
         
        gt_read_data = list()
        power_read_data = list()
        
        if file == 'microwave1.csv':
        	power_threshold = threshold_microwave
        if file == "oven1.csv":
        	power_threshold = threshold_oven
        if file == 'kitchenapp1.csv':
        	power_threshold = threshold_kitchenapp
        if file == 'lights_plugs1.csv':
        	power_threshold = threshold_lightsplug
        if file == 'clotheswasher1.csv':
        	power_threshold = threshold_clotheswasher
        if file == 'dishwasher1.csv':
        	power_threshold = threshold_dishwasher
        if file == '3577-dryg1.csv':
            power_threshold = threshold_3577_dryg
        if file == '3577-furnace1.csv':
            power_threshold = threshold_3577_furnace
        if file == '114-bedroom1.csv':
            power_threshold = threshold_114_bedroom
        if file == '114-drye1.csv':
            power_threshold = threshold_114_drye
        
        prev_time = None
        prev_gt = None
        
        datareader = pd.read_csv(file_path)
        datas = datareader.values.tolist()
        for row in tqdm(datas, desc = "Reading power data from %s"%file):
            dataid, time, power = row
            time = time[:-3]
            dttime = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

            if dttime < starttime:
                continue 

            if dttime > endtime:
                break
            
            power = round(power*1000, 2)

            if power > power_threshold:
                gt = 1
            else:
                gt = 0

            #fill the gap of the missing data
            if prev_time == None:
                y = dttime.year
                m = dttime.month
                d = dttime.day
                prev_time = datetime.datetime(y, m, d, 0, 0)

            if dttime - datetime.timedelta(minutes = 1) != prev_time:
                time_diff = round(datetime.timedelta.total_seconds(dttime - prev_time)/60)
                if time_diff < 1:
                	continue
                for i in range(1, time_diff):
                    miss_time = prev_time + datetime.timedelta(minutes = i)
                    days = datetime.datetime.weekday(miss_time)
                    weeks = week_of_month(miss_time)
                    months = miss_time.month
                    gt_read_data.append([miss_time, prev_gt, months, weeks, days])
                    power_read_data.append([miss_time, prev_power, months, weeks, days])

            prev_time = dttime
            prev_gt = gt
            prev_power = power
                
            days = datetime.datetime.weekday(dttime)
            weeks = week_of_month(dttime)
            months = dttime.month
            
            gt_read_data.append([dttime, gt, months, weeks, days])
            power_read_data.append([dttime, power, months, weeks, days])
        gt_data.append(gt_read_data)
        power_data.append(power_read_data)
        
    return gt_data, power_data, file_list


def get_selected_data(data):
    selected_data = list()
    selected_data_1 = list() # state data with days info at the last
    selected_data_2 = list() # hours of operation per day data 
    for device in data:
        dev_data = list()
        day_info = list()
        daily_data = list()
        daily_data_1 = list()
        dev_daily_data = list()
        dev_daily_count = list()
        accumulated_states = 0
        base_day = None
        base_hour = None
        count = 0
        for items in device:
            time, state, months, weeks, days = items
            # if time <= endtime and time >= starttime:
            if base_day == None:
                # base_day = time
                # base_hour = time
                base_day_days = days
                y = time.year
                m = time.month
                d = time.day
                base_day = datetime.datetime(y, m, d, 0, 0)
                base_hour = datetime.datetime(y, m, d, 0, 0)

            if time >= base_day and time < base_day + datetime.timedelta(days = 1):
                if time >= base_hour and time < base_hour + datetime.timedelta(minutes = time_interval):
                    if state:
                        accumulated_states = 1
                        continue
                else:
                    daily_data.append(accumulated_states)
                    daily_data_1.append(accumulated_states)
                    base_hour = base_hour + datetime.timedelta(minutes = time_interval)
                    accumulated_states = state
                    if state == 1:
                        count += 1
                    continue

            else:
                #print(days)
                daily_data.append(accumulated_states)
                daily_data_1.append(accumulated_states)
                dev_data.append(daily_data)
                day_info.append(base_day_days)
                daily_data_1.append(base_day_days)
                dev_daily_data.append(daily_data_1)
                dev_daily_count.append(count)
                count = 0
                daily_data = list()
                daily_data_1 = list()
                base_day = datetime.datetime(time.year, time.month, time.day, 0, 0)
                base_hour = datetime.datetime(time.year, time.month, time.day, 0, 0)
                base_day_days = days
                accumulated_states = state
                print(time)

                continue
            # else:
                # continue
        daily_data.append(accumulated_states)
        daily_data_1.append(accumulated_states)
        dev_data.append(daily_data)
        day_info.append(base_day_days)
        daily_data_1.append(base_day_days)
        dev_daily_data.append(daily_data_1)
        dev_daily_count.append(count)
        selected_data.append(dev_data)
        selected_data_1.append(dev_daily_data)
        selected_data_2.append(dev_daily_count)
        count = 0

    # create one-hot day info
    day_week_lst = list()
    for day in day_info:
        tmp_lst = list()
        for i in range(7):
            if i == day:
                tmp_lst.append(1)
            else:
                tmp_lst.append(0)
        day_week_lst.append(tmp_lst)
    

    array_select = np.array(selected_data)
    #print(array_select.shape)

    sum_dist = np.zeros((array_select.shape[0], 7, array_select.shape[2]) )
    sum_days = np.zeros((array_select.shape[0], 7))
    # print(array_select.shape)

    for a in range(array_select.shape[0]):
        for b in range(array_select.shape[1]):
            for c in range(array_select.shape[2]):
                if array_select[a, b, c] == 1:
                    sum_days[a, b%7] = sum_days[a, b%7] + 1
                    sum_dist[a, b%7, c] = sum_dist[a, b%7, c] + 1
        for i in range(7):
            for j in range(array_select.shape[2]):
                sum_dist[a, i, j] = float(sum_dist[a, i, j]/sum_days[a, i])

# select_data_1 format: [[operations of 24 hours in a day], day_info]
    PATH = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/pre-process'    
    foupath = os.path.join(PATH, '%d_dataset.pkl'%house_num)
    with open(foupath, 'wb') as f:
        # print(np.array(selected_data_1).shape)
        pickle.dump(selected_data_1, f)


    PATH = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/pre-process'    
    foupath = os.path.join(PATH, '%d_distribution.pkl'%house_num)
    with open(foupath, 'wb') as f:
        # print(np.array(sum_dist).shape)
        pickle.dump(sum_dist, f)

    print('Distribution: ', sum_dist)
# select_data_2: how many hours per day in operation
    PATH = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/pre-process'    
    foupath = os.path.join(PATH, '%d_day_op_count.pkl'%house_num)
    with open(foupath, 'wb') as f:
        # print(np.array(selected_data_1).shape)
        pickle.dump(selected_data_2, f)

    for i in selected_data_2:
        print(i)

    PATH = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/pre-process'    
    foupath = os.path.join(PATH, '%d_day_info.pkl'%house_num)
    with open(foupath, 'wb') as f:
        # print(np.array(selected_data_1).shape)
        pickle.dump(day_week_lst, f)

    # testing = np.array(selected_data)
    # testing = testing[:, -80:, :]
    # print(testing.shape)
    # PATH = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/'    
    # foupath = os.path.join(PATH, 'testing.pkl')
    # with open(foupath, 'wb') as f:
    #     # print(np.array(sum_dist).shape)
    #     pickle.dump(testing, f)

    return selected_data, day_info, selected_data_1, selected_data_2, sum_dist

def get_selected_power_data(data):
# calculate hourly power total for energy saving calculation

    selected_data = list()
    selected_data_1 = list()
    selected_data_2 = list()
    for device in data:
        dev_data = list()
        day_info = list()
        daily_data = list()
        daily_data_1 = list()
        dev_daily_data = list()
        accumulated_power = 0
        base_day = None
        base_hour = None
        count = 0
        for items in device:
            time, power, months, weeks, days = items
            # if time <= endtime and time >= starttime:
            if base_day == None:
                # base_day = time
                # base_hour = time
                base_day_days = days
                y = time.year
                m = time.month
                d = time.day
                base_day = datetime.datetime(y, m, d, 0, 0)
                base_hour = datetime.datetime(y, m, d, 0, 0)

            if time >= base_day and time < base_day + datetime.timedelta(days = 1):
                if time >= base_hour and time < base_hour + datetime.timedelta(minutes = time_interval):
                    accumulated_power += power
                    continue
                else:
                    daily_data.append(accumulated_power)
                    daily_data_1.append(accumulated_power)
                    base_hour = base_hour + datetime.timedelta(minutes = time_interval)
                    accumulated_power = power
                    continue

            else:
                #print(days)
                daily_data.append(accumulated_power)
                dev_data.append(daily_data)
                daily_data = list()
                base_day = datetime.datetime(time.year, time.month, time.day, 0, 0)
                base_hour = datetime.datetime(time.year, time.month, time.day, 0, 0)
                base_day_days = days
                accumulated_power = power
                continue
            # else:
                # continue
        daily_data.append(accumulated_power)
        dev_data.append(daily_data)
        selected_data.append(dev_data)


    PATH = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/pre-process'    
    foupath = os.path.join(PATH, '%d_power.pkl'%house_num)
    with open(foupath, 'wb') as f:
        # print(np.array(selected_data_1).shape)
        pickle.dump(selected_data, f)


def LSTM_prep(data, days):
    #assume data is an array, days is an array
    days_array = np.zeros((data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        days_array[i] = np.full((1, data.shape[2]), days[i])

    index_array_tmp = np.arange(data.shape[2])
    index_array = np.zeros((data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        index_array[i] = index_array_tmp
    dev_data_prep = np.zeros((3*data.shape[0], data.shape[1], data.shape[2]))
    for i in range(data.shape[0]):
        dev_data_prep[3*i] = data[i]
        dev_data_prep[3*i + 1] = days_array
        dev_data_prep[3*i + 2] = index_array
    
    return dev_data_prep

    # provide output for basic_lstm





def chopt_data(data):
    #data is an array
    #dump data from 00:00 to 06:00, assume chopt rest time to 3,6,9,18 slices

    #chopt every 4,6,8 data as observation, predict 4,6,8 data, window move every 1 hour

    #Generate Pickle File

    for m in range(1, int(data.shape[0]/3)+1):
        for z in range(4, 10, 2): #z for obervation
            for p in range(4, 10, 2):
                obsr_4 = list()
                obsr_6 = list()
                obsr_8 = list()
                #generate observation for training
                PATH = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/Training/Prepare_Input_pkl/dev_%d/%d_min/%d_obsr/%d_pred'%(m,time_interval,z,p)
                foupath = os.path.join(PATH, '%d_obsr.pkl'%z)
                with open(foupath,'wb') as f:
                    for i in range(144):
                        if z == 4:
                            for j in range(6*freq_interval, 24*freq_interval-(z+p), freq_interval):
                                # obsr_4.append([data[3*(m-1):3*m, i, j:j+3]])
                                obsr_4.append([data[3*(m-1):3*m, i, j], data[3*(m-1):3*m, i, j+1], data[3*(m-1):3*m, i, j+2], data[3*(m-1):3*m, i, j+3]])
                        if z == 6:
                            for j in range(6*freq_interval, 24*freq_interval-(z+p), freq_interval):
                                # obsr_6.append([data[3*(m-1):3*m, i, j:j+5]])
                                obsr_6.append([data[3*(m-1):3*m, i, j], data[3*(m-1):3*m, i, j+1], data[3*(m-1):3*m, i, j+2], data[3*(m-1):3*m, i, j+3], data[3*(m-1):3*m, i, j+4], data[3*(m-1):3*m, i, j+5]])
                        if z == 8:
                            for j in range(6*freq_interval, 24*freq_interval-(z+p), freq_interval):
                                # obsr_8.append([data[3*(m-1):3*m, i, j:j+7]])
                                obsr_8.append([data[3*(m-1):3*m, i, j], data[3*(m-1):3*m, i, j+1], data[3*(m-1):3*m, i, j+2], data[3*(m-1):3*m, i, j+3], data[3*(m-1):3*m, i, j+4], data[3*(m-1):3*m, i, j+5], data[3*(m-1):3*m, i, j+6], data[3*(m-1):3*m, i, j+7]])
                    if z == 4:
                        obsr_4 = np.array(obsr_4, dtype = 'f')
                        pickle.dump(obsr_4, f)
                    if z == 6:
                        obsr_6 = np.array(obsr_6, dtype = 'f')
                        pickle.dump(obsr_6, f)
                    if z == 8:
                        obsr_8 = np.array(obsr_8, dtype = 'f')
                        pickle.dump(obsr_8, f)

                #generate observation for testing
                obsr_4 = list()
                obsr_6 = list()
                obsr_8 = list()
                PATH = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/Testing/Prepare_Input_pkl/dev_%d/%d_min/%d_obsr/%d_pred'%(m,time_interval,z,p)
                foupath = os.path.join(PATH, '%d_obsr.pkl'%z)
                with open(foupath,'wb') as f:
                    for i in range(144, data.shape[1]):
                        if z == 4:
                            for j in range(6*freq_interval, 24*freq_interval-(z+p), freq_interval):
                                # obsr_4.append([data[3*(m-1):3*m, i, j:j+3]])
                                obsr_4.append([data[3*(m-1):3*m, i, j], data[3*(m-1):3*m, i, j+1], data[3*(m-1):3*m, i, j+2], data[3*(m-1):3*m, i, j+3]])
                        if z == 6:
                            for j in range(6*freq_interval, 24*freq_interval-(z+p), freq_interval):
                                # obsr_6.append([data[3*(m-1):3*m, i, j:j+5]])
                                obsr_6.append([data[3*(m-1):3*m, i, j], data[3*(m-1):3*m, i, j+1], data[3*(m-1):3*m, i, j+2], data[3*(m-1):3*m, i, j+3], data[3*(m-1):3*m, i, j+4], data[3*(m-1):3*m, i, j+5]])

                        if z == 8:
                            for j in range(6*freq_interval, 24*freq_interval-(z+p), freq_interval):
                                # obsr_8.append([data[3*(m-1):3*m, i, j:j+7]])
                                obsr_8.append([data[3*(m-1):3*m, i, j], data[3*(m-1):3*m, i, j+1], data[3*(m-1):3*m, i, j+2], data[3*(m-1):3*m, i, j+3], data[3*(m-1):3*m, i, j+4], data[3*(m-1):3*m, i, j+5], data[3*(m-1):3*m, i, j+6], data[3*(m-1):3*m, i, j+7]])
                    if z == 4:
                        obsr_4 = np.array(obsr_4, dtype = 'f')
                        pickle.dump(obsr_4, f)
                    if z == 6:
                        obsr_6 = np.array(obsr_6, dtype = 'f')
                        pickle.dump(obsr_6, f)
                    if z == 8:
                        obsr_8 = np.array(obsr_8, dtype = 'f')
                        pickle.dump(obsr_8, f)



                #generate prediction for training
                
                    pred_4 = list()
                    pred_6 = list()
                    pred_8 = list()

                    PATH = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/Training/Prepare_Input_pkl/dev_%d/%d_min/%d_obsr/%d_pred'%(m,time_interval,z,p)
                    foupath = os.path.join(PATH, '%d_pred.pkl'%p)
                    with open(foupath, 'wb') as f:
                        for i in range(144):
                            for j in range(6*freq_interval + z, 24*freq_interval - p, freq_interval):
                                if p == 4:
                                    # pred_4.append([data[3*(m-1):3*m, i, j:j+3]])
                                    pred_4.append([data[3*(m-1):3*m, i, j], data[3*(m-1):3*m, i, j+1], data[3*(m-1):3*m, i, j+2], data[3*(m-1):3*m, i, j+3]])
                                if p == 6:
                                    # pred_6.append([data[3*(m-1):3*m, i, j:j+5]])
                                    pred_6.append([data[3*(m-1):3*m, i, j], data[3*(m-1):3*m, i, j+1], data[3*(m-1):3*m, i, j+2], data[3*(m-1):3*m, i, j+3], data[3*(m-1):3*m, i, j+4], data[3*(m-1):3*m, i, j+5]])
                                if p == 8:
                                    # pred_8.append([data[3*(m-1):3*m, i, j:j+7]])
                                    pred_8.append([data[3*(m-1):3*m, i, j], data[3*(m-1):3*m, i, j+1], data[3*(m-1):3*m, i, j+2], data[3*(m-1):3*m, i, j+3], data[3*(m-1):3*m, i, j+4], data[3*(m-1):3*m, i, j+5], data[3*(m-1):3*m, i, j+6], data[3*(m-1):3*m, i, j+7]])
                        if p == 4:
                            pred_4 = np.array(pred_4, dtype = 'f')
                            pickle.dump(pred_4, f)
                        if p == 6:
                            pred_6 = np.array(pred_6, dtype = 'f')
                            pickle.dump(pred_6, f)
                        if p == 8:
                            pred_8 = np.array(pred_8, dtype = 'f')
                            pickle.dump(pred_8, f)
                    
                    #generate prediction for testing
                    pred_4 = list()
                    pred_6 = list()
                    pred_8 = list()

                    PATH = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/Testing/Prepare_Input_pkl/dev_%d/%d_min/%d_obsr/%d_pred'%(m,time_interval,z,p)
                    foupath = os.path.join(PATH, '%d_pred.pkl'%p)
                    with open(foupath, 'wb') as f:
                        for i in range(144, data.shape[1]):
                            for j in range(6*freq_interval + z, 24*freq_interval - p, freq_interval):
                                if p == 4:
                                    # pred_4.append([data[3*(m-1):3*m, i, j:j+3]])
                                    pred_4.append([data[3*(m-1):3*m, i, j], data[3*(m-1):3*m, i, j+1], data[3*(m-1):3*m, i, j+2], data[3*(m-1):3*m, i, j+3]])
                                if p == 6:
                                    # pred_6.append([data[3*(m-1):3*m, i, j:j+5]])
                                    pred_6.append([data[3*(m-1):3*m, i, j], data[3*(m-1):3*m, i, j+1], data[3*(m-1):3*m, i, j+2], data[3*(m-1):3*m, i, j+3], data[3*(m-1):3*m, i, j+4], data[3*(m-1):3*m, i, j+5]])
                                if p == 8:
                                    # pred_8.append([data[3*(m-1):3*m, i, j:j+7]])
                                    pred_8.append([data[3*(m-1):3*m, i, j], data[3*(m-1):3*m, i, j+1], data[3*(m-1):3*m, i, j+2], data[3*(m-1):3*m, i, j+3], data[3*(m-1):3*m, i, j+4], data[3*(m-1):3*m, i, j+5], data[3*(m-1):3*m, i, j+6], data[3*(m-1):3*m, i, j+7]])
                        if p == 4:
                            pred_4 = np.array(pred_4, dtype = 'f')
                            pickle.dump(pred_4, f)
                        if p == 6:
                            pred_6 = np.array(pred_6, dtype = 'f')
                            pickle.dump(pred_6, f)
                        if p == 8:
                            pred_8 = np.array(pred_8, dtype = 'f')
                            pickle.dump(pred_8, f)
    
    
    #Generate CSV ile

    # for m in range(1, int(data.shape[0]/3)+1):
    #     for z in range(1,4): #z for obervation
    #         #generate observation
    #         PATH = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/Prepare_Input_csv/dev_%d/%d_min/%d_obsr'%(m,time_interval,(2+2*z))
    #         foupath = os.path.join(PATH, '%d_obsr.csv'%(2+2*z))
    #         with open(foupath, 'w') as chopt_1_dev:
    #             writer = csv.writer(chopt_1_dev, delimiter='\t')

    #             for i in range(data.shape[1]):
    #                 if z == 1:
    #                     for j in range(6, 24 - int((4 + 2*z)*time_interval/60), int(4*time_interval/60)):
    #                         writer.writerow([data[3*(m-1):3*m, i, freq_interval*j], data[3*(m-1):3*m, i, freq_interval*j+1], data[3*(m-1):3*m, i, freq_interval*j+2], data[3*(m-1):3*m, i, freq_interval*j+3]])
    #                 if z == 2:
    #                     for j in range(6, 24 - int((6 + 2*z)*time_interval/60), int(4*time_interval/60)):
    #                         writer.writerow([data[3*(m-1):3*m, i, freq_interval*j], data[3*(m-1):3*m, i, freq_interval*j+1], data[3*(m-1):3*m, i, freq_interval*j+2], data[3*(m-1):3*m, i, freq_interval*j+3], data[3*(m-1):3*m, i, freq_interval*j+4], data[3*(m-1):3*m, i, freq_interval*j+5]])
    #                 if z == 3:
    #                     for j in range(6, 24 - int((8 + 2*z)*time_interval/60), int(4*time_interval/60)):
    #                         writer.writerow([data[3*(m-1):3*m, i, freq_interval*j], data[3*(m-1):3*m, i, freq_interval*j+1], data[3*(m-1):3*m, i, freq_interval*j+2], data[3*(m-1):3*m, i, freq_interval*j+3], data[3*(m-1):3*m, i, freq_interval*j+4], data[3*(m-1):3*m, i, freq_interval*j+5], data[3*(m-1):3*m, i, freq_interval*j+6], data[3*(m-1):3*m, i, freq_interval*j+7]])
    #         #generate prediction
    #         for p in range(1, 4):
    #             PATH = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/Prepare_Input_csv/dev_%d/%d_min/%d_obsr'%(m,time_interval,(2+2*z))
    #             foupath = os.path.join(PATH, '%d_pred.csv'%(2+2*p))
    #             with open(foupath, 'w') as chopt_1_dev:
    #                 writer = csv.writer(chopt_1_dev, delimiter='\t')

    #                 for i in range(data.shape[1]):
    #                     for j in range(6+int(4*time_interval/60), 24 - int(2*p*time_interval/60) , int(4*time_interval/60)):
    #                         if p == 1:
    #                             writer.writerow([data[3*(m-1):3*m, i, freq_interval*j], data[3*(m-1):3*m, i, freq_interval*j+1], data[3*(m-1):3*m, i, freq_interval*j+2], data[3*(m-1):3*m, i, freq_interval*j+3]])
    #                         if p == 2:
    #                             writer.writerow([data[3*(m-1):3*m, i, freq_interval*j], data[3*(m-1):3*m, i, freq_interval*j+1], data[3*(m-1):3*m, i, freq_interval*j+2], data[3*(m-1):3*m, i, freq_interval*j+3], data[3*(m-1):3*m, i, freq_interval*j+4], data[3*(m-1):3*m, i, freq_interval*j+5]])
    #                         if p == 3:
    #                             writer.writerow([data[3*(m-1):3*m, i, freq_interval*j], data[3*(m-1):3*m, i, freq_interval*j+1], data[3*(m-1):3*m, i, freq_interval*j+2], data[3*(m-1):3*m, i, freq_interval*j+3], data[3*(m-1):3*m, i, freq_interval*j+4], data[3*(m-1):3*m, i, freq_interval*j+5], data[3*(m-1):3*m, i, freq_interval*j+6], data[3*(m-1):3*m, i, freq_interval*j+7]])




def dev_association(data):
    a = 0
    test_14 = 0
    count = [0, 0, 0, 0, 0, 0]
    itemset = [[] for x in range(data.shape[1]*data.shape[2])]
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            for z in range(data.shape[0]):
                if data[z, i, j] == 1:
                    itemset[a].append(z)
                    count[z] = count[z] + 1
            if data[1, i, j] == data[4, i, j] == 1:
                test_14 += 1
            a += 1;

    # count = [count[z]/data.shape[1]*data.shape[2] for z in range(len(count))]
    print(test_14)
    patterns = fpgrowth.find_frequent_patterns(itemset,50)
    rules = fpgrowth.generate_association_rules(patterns, .2)

    association_rules = apriori(itemset, min_support=0.002, min_confidence=0.4, min_lift=2, min_length=2.5)
    association_results = list(association_rules)

    associ_rules = []
    for item in association_results:
        # associ_rules = []
        pair = []
        for i in item[0]:
            pair.append(i)
        support = item[1]
        for rules in item[2]:
            tmp = []
            for i in rules[0]:
                tmp.append(i)
            for i in rules[1]:
                tmp.append(i)
            associ_rules.append([tmp, rules[2]])

    dev_rules_no_0 = []
    dev_rules= np.array(associ_rules)
    print(dev_rules[0])
    print(dev_rules[0,0])
    # print(dev_rules[0,0,0])
    print(dev_rules[0,0][0])
    for item in dev_rules:
         if item[0][0] == 0:
             continue
         dev_rules_no_0.append([item[0][0:-1], [item[0][-1]], item[1]])

    return patterns, rules, count, association_results, dev_rules_no_0



def training_testing_index(data):
    index = []
    for i in range(data.shape[1]-7):
        tmp = []
        for j in range(i, i + 8):
            tmp.append(j)
        index.append(tmp)
    index = np.array(index)

    PATH = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/pre-process'    
    foupath = os.path.join(PATH, '%d_indexing.pkl'%house_num)
    with open(foupath, 'wb') as f:
        pickle.dump(index, f)
    
    return index

if __name__ == "__main__": 
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
