#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 12:06:48 2021

@author: baylorbrangers
"""
#%% Import

import numpy as np
import csv
from bisect import bisect_left
import matplotlib.pyplot as plt
from cycler import cycler
#%% Filenames
events_filename = '/Users/baylorbrangers/Desktop/timeline.csv'
time_stamp =
#%% FBehavior class creation and read data file
class Behavior:
    def __init__(self,name):
        self.name=name
        self.time_points =[]
        
        
    def add_times(self,time1,time2):
        self.time_points.append([time1,time2])

#make sure that time is an integer data type or integer NumPy array,
def converttime(time):
    #offset = time & 0xFFF
    cycle1 = (time >> 12) & 0x1FFF
    cycle2 = (time >> 25) & 0x7F
    seconds = cycle2 + cycle1 / 8000.
    return seconds

def uncycle(time):
    cycles = np.insert(np.diff(time) < 0, 0, False)
    cycleindex = np.cumsum(cycles)
    return time + cycleindex * 128


events = []
with open (events_filename, 'r') as f:
     reader = csv.reader(f, delimiter = ',')
        
     for line in reader:
         if line[0] == 'T':
             name=Behavior(str(line[1]))
             events.append(name)
         elif line[0] == 'P': 
             name.add_times(int(line[2]), int(line[3]))

#%% Filenames

#data=[i[0] for i in a] list comprehension example

def behavior_element_diff(data_array):
    diff_array=[]
    mean_diff_val=0
    for i in data_array:
        diff_array.append(i[1]-i[0])
    #calculate mean time difference between events
    mean_diff_val=np.mean(diff_array)
    #total number of times animal did behavior
    size=len(diff_array)
    
    return (diff_array,mean_diff_val,size)

time_dif_behavior,mean_difference_behavior,num_events=behavior_element_diff(events[0].time_points)




        