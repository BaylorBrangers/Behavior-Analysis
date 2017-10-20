"""
Created on Fri May 26 23:10:19 2017

@author: Baylor Brangers
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
import re
from bisect import bisect_left
from scipy import signal
from scipy import stats
from sklearn import linear_model, datasets


file_name='cellcounts1.csv'

male_female_data=np.loadtxt(open(file_name, "rb"), delimiter=",",  usecols=(1,2,3,4,5,6,7),skiprows=3)
#def read_data(file_location):
#    file_name=file_location
#    data_matrix=file_lacation([:])
#    return data_matrix



#x is 6 of the 7 females
x=np.transpose(male_female_data)


male_female=linear_model.LogisticRegression()
###
###
target_data=np.array(['female','female' , 'female','female','male','male'])

trainAll=x[[0,1,2,3,4,5],:]
b= male_female.fit(trainAll,target_data)


test=x[6,:]
a=male_female.predict(test[None,:])
