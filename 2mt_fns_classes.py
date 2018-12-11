

import numpy as np
import csv
import matplotlib.pyplot as plt
import re
from bisect import bisect_left
from scipy import signal
import tensorflow as tf
import accl_functions as af
import random



class binary_identificatoin_profile(object):
    """Creates a profile for each accelerometer profile"""


    def read_binary_data(self,file_name):
        """ generate acclerometer data vector. Only looking at the
        first three components of the sensor"""

        with open (file_name, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader, None)


            self.x_data = []
            self.y_data = []

            for line in reader:
              self.x_data.append(line[0])
              self.y_data.append(line[1])
       
            #the first row contains column headers
            self.x_data.pop(0)
            self.y_data.pop(0)

            #convert array data to float
            self.x_data = np.array([float(i) for i in self.x_data])
            self.y_data = np.array([float(i) for i in self.y_data])



            #still need to decide how to normalize data
            
            return self.x_data, self.y_data


