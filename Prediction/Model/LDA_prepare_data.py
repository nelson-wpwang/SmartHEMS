from hmmlearn import hmm
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
import scipy.io as sio
from tqdm import tqdm
import itertools
from math import ceil
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_random_state

#Set the threshold to mark the active status
power_threshold = 30
time_interval = 60 #min of time interval
chosen_start_date = '2016-10-01'
chosen_end_date = '2016-10-31'

def week_of_month(dt):
    #This will add the first day of the month weekday & present day of month
    first_day = dt.replace(day=1)
    day_of_month = dt.day
    #print day_of_month
    adj_day_of_month = day_of_month + first_day.weekday()
    return int(ceil(adj_day_of_month/7.0))

def get_data(name = None, path = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source'):
    if not os.path.exists(path):
        raise('Directory not exists')

    all_data = list()# L*(N*1440)*5
    file_list = list()

    for file in os.listdir(path):
        if file.startswith('.'):
            continue
        file_list.append(file)
        file_path = os.path.join(path, file)
        if not os.path.isfile(file_path):
            continue
         
        read_data = list()

        prev_time = None
        prev_gt = None
        
        datareader = pd.read_csv(file_path)
        datas = datareader.values.tolist()
        for row in tqdm(datas, desc = "Reading power data"):
            dataid, time, power = row
            time = time[:-3]
            dttime = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
            
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
                for i in range(1, time_diff):
                    miss_time = prev_time + datetime.timedelta(minutes = i)
                    days = datetime.datetime.weekday(miss_time)
                    weeks = week_of_month(miss_time)
                    months = miss_time.month
                    read_data.append([miss_time, prev_gt, months, weeks, days])

            prev_time = dttime
            prev_gt = gt
                
            days = datetime.datetime.weekday(dttime)
            weeks = week_of_month(dttime)
            months = dttime.month
            
            read_data.append([dttime, gt, months, weeks, days])
        
        all_data.append(read_data)
        
    return all_data, file_list

def get_selected_data(start_time, end_time, data):
	starttime = datetime.datetime.strptime(start_time, '%Y-%m-%d')
	endtime = datetime.datetime.strptime(end_time, '%Y-%m-%d')

	starttime = datetime.datetime(starttime.year, starttime.month, starttime.day, 0, 0)
	endtime = datetime.datetime(endtime.year, endtime.month, endtime.day, 23, 59)
	selected_data = list()

	for device in data:
		this_month = None
		#month_list = list()
		this_month_data = list()
		monthly_data = list()
		for items in device:
			time, state, months, weeks, days = items
			if time <= endtime and time >= starttime:
				if this_month == None:
					this_month = months

				if months == this_month:
					this_month_data.append(items)
				else:
					monthly_data.append(this_month_data)
					#month_list.append(this_month)
					this_month_data = list()
					this_month_data.append(items)
					this_month = months
			else:
				continue
		monthly_data.append(this_month_data)
		selected_data.append(monthly_data)

	return selected_data

def prepare_data(data):
	daily_data = list()
	monthly_data = list()
	prepared_data = list()
	accumulated_states = 0
	base_day = None
	base_hour = None
	for device in data:
		for month_year in device:
			for item in month_year:
				time, state, months, weeks, days = item
				if base_day == None:
					base_day = time
					base_hour = time

				if time >= base_day and time < base_day + datetime.timedelta(days = 1):
					if time >= base_hour and time < base_hour + datetime.timedelta(minutes = time_interval):
						if state:
							accumulated_states = 1
							continue
					else:
						daily_data.append(accumulated_states)
						base_hour = base_hour + datetime.timedelta(minutes = time_interval)
						accumulated_states = state
						continue

				else:
					daily_data.append(accumulated_states)
					monthly_data.append(daily_data)
					daily_data = list()
					base_day = time
					base_hour = time
					accumulated_states = state
					continue

			daily_data.append(accumulated_states)
			monthly_data.append(daily_data)
			daily_data = list()
			base_day = time
			base_hour = time
			accumulated_states = state
			continue

		#monthly_data.append(daily_data)
		prepared_data.append(monthly_data)
	return prepared_data


#for item in X_data[0]:
#	print(item)

#change X data frame from L*N*T to N*L*T
def change_frame(data):
	output_data = [[] for i in range(len(data[0]))]
	for dev in data:
		print(len(dev))
		for i in range(len(dev)):
			output_data[i].append(dev[i])
			#for i in range(len(month)):
			#	output_data[i].append(month[i])
	return output_data

#X = np.squeeze(X, axis = -1)
#print(X)
            