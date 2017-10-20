# -*- coding: utf-8 -*-
"""
Created on Fri May 26 23:10:19 2017

@author: basmafatimaanwarhusain
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
import re
from bisect import bisect_left
from scipy import signal


#from operator import itemgetter

filename = '170531Anomark_acc_weardata.csv'
filename2 = '170531Anomark.csv'

def read_accelerometer_data(filename):
    with open (filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)

        timestamps = []
        #data from accelerometer
        x_data = []
        y_data = []
        z_data = []
        # contains data from y axis of gyroscope
        head_data=[]

        for line in reader:

            timestamps.append(line[1])
            x_data.append(line[2])
            y_data.append(line[3])
            z_data.append(line[4])
            head_data.append(line[6])

        return (timestamps, x_data , y_data , z_data, head_data)

acc_timestamps , x_data, y_data , z_data = read_accelerometer_data(filename)

x_data = np.array([float(i) for i in x_data])
y_data = np.array([float(i) for i in y_data])
z_data = np.array([float(i) for i in z_data])
head_data=np.array[float(i) for i in head_data]
acc_timestamps = np.array([float(i.replace(',','.')) for i in acc_timestamps])

acc_timestamps = np.array(acc_timestamps)



def read_event_data(filename2):
    f = open(filename2, 'r')
    time_data = []
    event_data = []
    for line in f:
        t = line.strip().split()
        t1 = re.split(r'[T,+]' , t[1])

        t2 = t1[1].split(':')

        time_sec = int(t2[0]) * 3600 + int(t2[1]) * 60 + float(t2[2]) - 0.033

        event = t[0]

        time_data.append(time_sec)
        event_data.append(event)
    f.close()
    return (time_data , event_data)

time_data , event_data = read_event_data(filename2)

time_data = np.array(time_data)

acc_time_diff = np.zeros(len(acc_timestamps))
time_data_diff = np.zeros(len(time_data))

for x in range (0, len(acc_timestamps)):
    acc_time_diff[x] = acc_timestamps[x] - acc_timestamps[0]

for x in range (0, len(time_data)):
    time_data_diff[x] = time_data[x] - time_data[0]

def findClosest(myList, myNumber):
    pos = bisect_left(myList, myNumber)

    if pos == 0:
        return 0
    if pos == len(myList):
        return -1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return pos
    else:
       return (pos - 1)

indices = []

for x in time_data_diff:
    indices.append(findClosest(acc_time_diff , x))

#t2 = itemgetter(*indices)(t1)

new_x = x_data/7000
new_y = y_data/7000
new_z = z_data/7000
new_head=head_data/7000


def accl_mag(accl_x, accl_y, accl_z):
    ##Calculates the magnitude of the acceleration vector
    mag_accl=np.sqrt(np.square(accl_x)+np.square(accl_y)+np.square(accl_z))

    return mag_accl

new_accl=accl_mag(new_x,new_y,new_z)

plt.figure()
plt.plot(buttered_signal, '#1A8925')


#plt.plot(new_y, '#891A7D')
#plt.plot(new_z, '#02BACF')

#import pickle
#f = open('graph.pickle', 'wb')
#pickle.dump(n, f)
#f.close()
#
#n = pickle.load(open('graph.pickle', 'rb'))
#
#n = np.vstack([new_x,new_y,new_z]).T
#plt.plot(n)



#val = 5000 # this is the value where you want the data to appear on the y-axis.
plt.plot(indices, np.zeros_like(indices), 'o' , color = 'r')
plt.show()
