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
from sklearn.decomposition import PCA



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
    mag_accl=np.sqrt(np.square(accl_x)+np.square(accl_y)+np.square(accl_z)) #3 to normalize

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

start2=int(intromission_array[1])
foo=new_accl[(start2-300):(start2+300)]


def euc_distance(behavior_array, full_data, time_interval):

    #determine size of new Euc dist matrix
    t=int(time_interval)
    length_behavior_d = len(behavior_array)
    m_sq_dist=np.zeros((length_behavior_d,length_behavior_d))
    z=0

    for p in np.nditer(behavior_array):
    #while p < length_behavior_d:
        start1 = int(p)
        y=0
        for x in np.nditer(behavior_array):
        #while x < length_behavior_d:
            #start2 = int(behavior_array[x])
            start2=int(x)
            m_sq_dist[z,y]= np.sqrt(np.sum(np.square(full_data[start2-t:start2+t]-full_data[start1-t:start1+t])))
            y=y+1
        z=z+1
     #return euc matr, start1 and start2 should be equal   
    return m_sq_dist, start1 , start2


############################################################

def coor_btwn_sigs(behavior_array, full_data, time_interval):
    t=int(time_interval)
    length_behavior_d = len(behavior_array)
    m_cross_corr=np.zeros((length_behavior_d,length_behavior_d))
    z=0

    for p in np.nditer(behavior_array):
    #while p < length_behavior_d:
        start1 = int(p)
        y=0
        for x in np.nditer(behavior_array):
        #while x < length_behavior_d:
            #start2 = int(behavior_array[x])
            start2=int(x)
            m_cross_corr[z,y]= np.correlate(full_data[start2-t:start2+t],full_data[start1-t:start1+t])
            y=y+1
        z=z+1
     #return euc matr, start1 and start2 should be equal   
    return m_cross_corr, start1 , start2

euc ,start1, start2=euc_distance(intromission_array, new_accl,30)
cross_corr=coor_btwn_sigs(intromission_array, new_accl,30)




#pca=PCA(n_components=182,svd_solver='full')
#pca.fit(euc)
#foo=pca.explained_variance_
#foo2=pca.transform(euc)
##foo3=pca.singular_values_
#
##print(pca.singular_values_)  
#

plt.clf()
#plt.plot(pca.explained_variance_,)
#plt.axis('tight')
#plt.xlabel('n_components')
#plt.ylabel('explained_variance_')

#ax[2].plot(sq_diff)

fs=100
f, t, Sxx = signal.spectrogram(foo, fs)
fig, ax = plt.subplots(2, 1)
ax[0].plot(foo)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].pcolormesh(t, f, Sxx) # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')


plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

