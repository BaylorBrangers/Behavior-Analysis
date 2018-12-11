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
import pandas as pd
from sklearn.manifold import *
from sklearn import *
from sklearn.neighbors import *
from scipy import stats
import matplotlib.animation as animation
import time
file_location='/home/baylor/Desktop/DATA/Behavior_Analysis/Behavior-Analysis/fiber analysis/nidaq2018-03-02T17_06_33.csv'



######################################################

def reshape_data(filename, num_variables):
    data_array = np.fromfile(filename, np.float32)
    size_array=np.size(data_array)
    phot_array=np.reshape(data_array,[(size_array//num_variables),num_variables])
    return phot_array

phot_array=reshape_data(file_location,5)

######################################################
def dfof_traces(trace,binsize,timefzero,signal_offset):
#    fzero = np.median(trace[0:int((timefzero*1000)/binsize)])
    fzero = np.mean(trace)
    dfof_signal=[]
    x=signal_offset
       
    while x < len(trace):
        fzero = np.mean(trace[x-10000:x-1])
        f = trace[x]
        dfof = ((f-fzero)/fzero)
        dfof_signal.append(dfof)
        x=x+1
    return dfof_signal
##############################




def assign_values(array):
    """open session data and reformat"""

    #location of gcamp
    green=array[:,2]
    #location of red channel
    red=array[:,3]
    return green,red

######################################################

def filter_data_gauss(array1, array2,windowWidth):
    """Creates a gaussian filter
    array1 is gCamp6 signal
    array2 is tdT signal
    window is size of filters
    hw is 1/2 halfWidth
    """
    #Construct blurring window.
    green=array1
    red=array2
    windowWidth = int16(5)
    halfWidth = windowWidth / 2
    gaussFilter = gausswin(5);
    gaussFilter = gaussFilter / sum(gaussFilter)
    green = conv(green, gaussFilter)
    red = conv(red, gaussFilter)
    return green,red




def zscore_signal(array1,array2):
    green=stats.zscore(array1)
    red=stats.zscore(array2)
    return green,red

######################################################





######################################################

[green_signal, red_signal]=assign_values(phot_array)

#in Hz
sampling_frequency=1000
#in seconds
offset=300 


#plot_divisions=int((len(green_signal)-(sampling_frequency*offset))/300)

start=offset*sampling_frequency


######################################################
#ransac the night away
#Use a robust fit to determine the relationship between tdt and gcamp
#we will use this to determine the true gcamp6 signal
#######################################################

xg=np.asarray(green_signal[(sampling_frequency*offset):])
xr=np.asarray(red_signal[(sampling_frequency*offset):])

# Fit line using all data
lr = linear_model.LinearRegression()
lr.fit(xr[:,None],xg[:,None])

# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(xr[:,None], xg[:,None])
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
line_X = np.arange(xr.min(), xr.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)

# Compare estimated coefficients
#print("Estimated coefficients (true, linear regression, RANSAC):")
#print(lr.coef_, ransac.estimator_.coef_)

######################################################
#corrected gcamp
ransac_regression_coeff=np.float(ransac.estimator_.coef_)
ransac_regression_intercept=np.float(ransac.estimator_.intercept_)


corrected_gcamp=(ransac_regression_coeff*xr)+ransac.estimator_.intercept_

              
new_gcamp_signal=xg-corrected_gcamp
tree=dfof_traces(green_signal,80,5,40000)

outfile='Filtered_Data_gcamp'
np.savez(outfile,tree=tree)
#new_gcamp_signal_zscore=stats.zscore(new_gcamp_signal)



#new_tdt_signal=np.zeros(np.size(corrected_tdt))                
#new_tdt_signal=np.array(xr-corrected_gcamp)
#new_tdt_signal_zscore=stats.zscore(new_tdt_signal,1)
#
#fig, ax = plt.subplots()
#plt.subplots_adjust(bottom=0.25)
#
#t = np.arange(0.0, 60000, 1)
#s=new_gcamp_signal_zscore[0,120000:180000]
#red=new_tdt_signal_zscore[0,120000:180000]
#l, = plt.plot(t,s,t,red)
#plt.axis([0,60000, -3,4 ])
#
#axcolor = 'lightgoldenrodyellow'
#axpos = plt.axes([0.2, 0.1, 0.65, 0.03], axisbg=axcolor)
#
#spos = Slider(axpos, 'Pos', 1000, 60000)
#
#def update(val):
#    pos = spos.val
#    ax.axis([pos,pos+1000,-2,2])
#    fig.canvas.draw_idle()
#
#spos.on_changed(update)
#
#plt.show()
#

#
############################################################
##Plotting
############################################################
#lw = 2
#
#plt.cla
#
#plt.scatter(xr[inlier_mask], xg[inlier_mask], color='yellowgreen', marker='.',
#            label='Inliers')
#plt.scatter(xr[outlier_mask], xg[outlier_mask], color='gold', marker='.',
#            label='Outliers')
##plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
##plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
##         label='RANSAC regressor')
##plt.legend(loc='lower right')
##plt.xlabel("tdTomato")
##plt.ylabel("Gcamp6f")
##plt.plot(red_signal, green_signal)
#plt.show()
