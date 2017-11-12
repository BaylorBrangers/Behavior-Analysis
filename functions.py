# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:38:52 2017

@author: 
"""












###########################################################
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

###################################################################
"""tSNE-d"""

###################################################################
def SNE_ME(behavior_array, full_data, time_interval):
"""Performs 

