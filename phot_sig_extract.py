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


file_location='C:/Users/Enrica/Desktop/test0.csv'
data_array = np.fromfile(file_location, np.float32)

def reshape_data(filename, num_variables):
    data_array=np.fromfile(file_location, np.float32)
    size_array=len(data_array)
    phot_array=np.reshape(data_array,[(size_array/num_variables),num_variables])
    return phot_array

phot_array=reshape_data(data_array,4)


def assign_values(array):
    """open session data and reformat"""

    #location of gcamp
    green=array[:,2]
    #location of red channel
    red=array[:,3]
    return green,red

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

def filter_butterworth(array1, array2):
    """Apply a 4th order butterworth filter"""
    #order number and filter construction

    return green,red

def zscore_signal(array1,array2):
    green=stats.zscore(array1)
    red=stats.zscore(array2)
    return green,red



[green_signal, red_signal]=assign_values(phot_array)

#[smooth_green,smooth_red]=filter_data_gauss(green_signal,red_signal)

[g_zscore,r_zscore]=zscore_signal(green_signal,red_signal)

#in Hz
sampling_frequency=1000
offset=300 #in seconds. We are not concerned with the first few minutes of the recordings

plot_divisions=int((len(green_signal)-(sampling_frequency*offset))/300)

start=offset*sampling_frequency

# for i in range(plot_divisions):
#     stop=start+300
#     if (np.amax(g_zscore[start:stop])-np.amin(g_zscore[start:stop]))<1.0:
#         green_baseline=np.median(green_signal[(start-300):start])
#         green_corrected=(green_signal[start:stop]-green_baseline)/green_baseline
#         plt.plot(green_corrected,'r')
#         start=stop
#
#     else:
#          start=stop
#
#
#
plt.plot(red_signal[(len(red_signal)/2):], green_signal[(len(red_signal)/2):]) 
plt.show()
#
#xg=np.asarray(green_signal[(sampling_frequency*offset):])
#
#xr=np.asarray(red_signal[(sampling_frequency*offset):])
## Fit line using all data
#lr = linear_model.LinearRegression()
#lr.fit(xg[:,None],xr[:,None])
#
## Robustly fit linear model with RANSAC algorithm
#ransac = linear_model.RANSACRegressor()
#ransac.fit(red_signal[:,None], green_signal[:,None])
#inlier_mask = ransac.inlier_mask_
#outlier_mask = np.logical_not(inlier_mask)
#
## Predict data of estimated models
#line_X = np.arange(red_signal.min(), red_signal.max())[:, np.newaxis]
#line_y = lr.predict(line_X)
#line_y_ransac = ransac.predict(line_X)
#
## Compare estimated coefficients
#print("Estimated coefficients (true, linear regression, RANSAC):")
#print(lr.coef_, ransac.estimator_.coef_)
#
#lw = 2
#plt.scatter(red_signal[inlier_mask], green_signal[inlier_mask], color='yellowgreen', marker='.',
#            label='Inliers')
#plt.scatter(red_signal[outlier_mask], green_signal[outlier_mask], color='gold', marker='.',
#            label='Outliers')
#plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
#plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
#         label='RANSAC regressor')
#plt.legend(loc='lower right')
#plt.xlabel("Input")
#plt.ylabel("Response")
#plt.plot(green_signal,red_signal)
#plt.show()
