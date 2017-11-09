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

acc_timestamps , x_data, y_data , z_data, head_data = read_accelerometer_data(filename)

x_data = np.array([float(i) for i in x_data])
y_data = np.array([float(i) for i in y_data])
z_data = np.array([float(i) for i in z_data])
head_data=np.array([float(i) for i in head_data])

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
    mag_accl=np.sqrt(np.square(accl_x)+np.square(accl_y)+np.square(accl_z))/3 #3 to normalize

    return mag_accl

#accl vectoy
new_accl=accl_mag(new_x,new_y,new_z)


i=0
new_array=[]
indices2=np.array(indices)
for x in event_data:

    if x=='Intromission':
       new_array= np.append(new_array,i)
    i=i+1

#find all the intromission indicies

intromission_array=[]
for x in new_array:
    intromission_array=np.append(intromission_array, indices2[x])




#
#fig, ax = plt.subplots(3, 1)
#ax[0].plot(new_accl[(start1-300):(start1+300)])
#ax[1].plot(new_accl[(start2-300):(start2+300)])

def euc_distance(behavior_array, full_data, time_interval):

    #determine size of new Euc dist matrix
    length_behavior_d = len(behavior_array)
    m_sq_dist=np.zeros(length_behavior_d,length_behavior_d)
    p=0

    while p < length_behavior_d:
        start1 = int(intromission_array[p])
        x=0

        while x < length_behavior_d:
            start2 = int(intromission_array[x])
            m_sq_dist[p,x]= np.sqrt(np.sum(np.square(full_data[start2-300:start2+300]-full_data[start1-300:start1+300])))
            x=x+1
    p=p+1

    return m_sq_dist, start1 , start2

euc ,start1, start2=euc_distance(intromission_array, new_accl,300)

#ax[2].plot(sq_diff)
#n=len(new_accl) #size of sample window
#T = n/Fs
#k = np.arange(n)
#
#frq = k/T # two sides frequency range
##frq = frq[:25] # one side frequency range
#
#fft_sig=np.fft.fft(new_accl)/n
##fft_sig=fft_sig[:(n/2)]
#
#fig, ax = plt.subplots(2, 1)
#ax[0].plot(frq,new_accl)
#ax[0].set_xlabel('Time')
#ax[0].set_ylabel('Amplitude')
#ax[1].plot(frq,abs(fft_sig),'r') # plotting the spectrum
#ax[1].set_xlabel('Freq (Hz)')
#ax[1].set_ylabel('|Y(freq)|')
