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


file_location='C:/Users/Enrica/Desktop/test5.csv'


def open_and_extract(filename):
    """open session data and reformat"""

    data_array = np.fromfile(filename, np.float32)
    phot_array=np.reshape(data_array,[656000,5])
    #location of gcamp
    green=phot_array[:,3]
    #location of red channel
    red=phot_array[:,4]
    return phot_array, green, red

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

    return green, red

def filter_butterworth(array1, array2):
    """Apply a 4th order butterworth filter"""
    #order number and filter construction

    return green, red

def zscore_signal(array1,array2):
    green=stats.zscore(array1)
    red=stats.zscore(array2)
    return green, red



[all_data,green_signal, red_signal]=open_and_extract(file_location)

#[smooth_green,smooth_red]=filter_data_gauss(green_signal,red_signal)

[g_zscore,r_zscore]=zscore_signal(green_signal,red_signal)


t = np.linspace(-2,2,20)

plot_divisions=int(len(green_signal)/300)
#plot_divisions=1855
start=55000
plot_divisions=2000

              
for i in range(plot_divisions):
    stop=start+300
    if (np.amax(g_zscore[start:stop])-np.amin(g_zscore[start:stop]))>1.0:        
        green_baseline=np.median(green_signal[(start-300):start])
        green_corrected=(green_signal[start:stop]-green_baseline)/green_baseline
        plt.plot(green_corrected,'r') 
        start=stop        
    
    else:
         start=stop

#plt.plot(green_signal[450000:])         
plt.ylim([-2, 2])
plt.show()

#plt.plot(red_signal)
#plt.plot(g_zscore[350000:400000],r_zscore[350000:400000])