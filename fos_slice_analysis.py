#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:45:17 2019

@author: baylor
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import re
from bisect import bisect_left
from scipy import signal
from scipy import stats
import pandas as pd
import scipy as sp
from scipy import stats
import matplotlib.animation as animation
import time




class fos_animal(object):
      def __init__(self, name, behavior):  
        self.name = name  
        self.behavior=behavior
        self.dict_animal_info= {}
      def add_cell_count(self,section,count):
          self.dict_animal_info[section]=count

def read_cellcounts_data(filename):
    """ generate acclerometer data vector. Only looking at the
    first three components of the sensor"""
    
    animal_info_array=[]
    animal_class_array=[]
    with open (filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        
        for line in reader:
            animal_id=line[0]
            behavior=line[6]
            #see if animal class instance already exists
            if line[0] not in animal_info_array:
              name=animal_id
              animal_info_array.append(animal_id)
              name=fos_animal(animal_id,behavior)
              animal_class_array.append(name)
              name.add_cell_count(line[2],line[5])
            
            elif line[0] ==animal_id:
              name.add_cell_count(line[2],line[5])
              
             
        return (animal_class_array)


#%%
"""This program runs summary statistics on histological slices obtained with the slide scanner
It is currently not optimized for batch processing but will in the future"""
  
data_file='Sexual Behavior_2MT_MouseTimeline - cfos_counts_python_vlPAG.csv'    
n_groups = 5

def mean_sex(pos):
  return np.mean(pos)

def std_grp(pos):
  return sp.stats.sem(pos)


all_animal_data=read_cellcounts_data(data_file)
#create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8
 
# rects1 = plt.bar(index, means_control, bar_width,
#                  alpha=opacity,
#                  color='c',
#                  label='Control, n=4',yerr=std_control)
 
# rects2 = plt.bar(index + bar_width, means_sex, bar_width,
#                  alpha=opacity,
#                  color='m',
#                  label='Sex, n=5',yerr=std_sex)
 
##
#Need to creat an array of group names
#feed to function below

# group_array=['Sex','Control']
def plt_scatter(index,group_name,group_number):
  color_array=['k','m','g','c']
  color_plot=color_array[group_number]
  plt.scatter(index+group_number*bar_width,group_name,c=str(color_plot),marker='.')
i=0
# for a in group_array:
#   plt_scatter(index,group_array[a],i)
  # i=+1

#90
individual_sex1=[0.0002457951473,0.0003569233555,0.0004113883993,0.000240129026,0.0003903132356]
#91
individual_sex2=[0.0002873743198,0.0003753143257,0.0003675718834,0.0003691410993,0.0002926955677]
#95
individual_sex3=[0.0002257947977, 0.0004547696017, 0.0004284349345, 0.0003398538288, 0.0002656631745]
#96
individual_sex4=[0.000200886686, 0.000216869507, 0.0002385405011, 0.000298744586, 0.0002706121868]
#98
individual_sex5=[0.0002576162401, 0.0002806745953, 0.0005558271789, 0.0001971555143, 0.0003493929298]

#89
control1=[0.0001519688408, 0.0001712164228, 0.0003201039912, 0.0002385431812, 0.00026286672]
#92
control2=[0.0001519688408,0.000207905656, 0.0003611215545,0.0001689082894,0.0002945598707]
#93
control3=[0.0001089254593,0.0002026643372,0.0002908929806,0.0003375235928,0.000234485731]
#97
control4=[0.0002717878203, 0.0004104526284, 0.0003645684509, 0.0002398422481, 0.0003115637882]


scat1=plt.scatter(index+(bar_width),individual_sex1,c='k',marker='.')
scat2=plt.scatter(index+(bar_width),individual_sex2,c='k',marker='.')
scat3=plt.scatter(index+(bar_width),individual_sex3,c='k',marker='.')
scat4=plt.scatter(index+ bar_width,individual_sex4,c='k',marker='.')
scat5=plt.scatter(index+ bar_width,individual_sex5,c='k',marker='.')
scat6=plt.scatter(index,control1,c='b',marker='o')
scat7=plt.scatter(index,control2,c='b',marker='o')
scat8=plt.scatter(index,control3,c='b',marker='o')
scat9=plt.scatter(index,control4,c='b',marker='o')


plt.xlabel('Position from Bregma')
plt.ylabel('Normalized Cell Density (cells/um2)')
plt.title('cFos Expression in vlPAG')
plt.xticks(index + bar_width, ('-4.48','-4.6','-4.72','-4.84','-4.96'))
plt.legend()
 
plt.tight_layout()
plt.show()

# #no sex
# time89=[16,16,16,16,16]
# time92=[40,40,40,40,40]
# time93=[24,24,24,24,24]
# time97=[11,11,11,11,11]

# #sex
# time90=[11,11,11,11,11]
# time91=[40,40,40,40,40]
# time95=[16,16,16,16,16]
# time96=[63,63,63,63,63]
# time98=[24,24,24,24,24]

# plt.figure()
# plt.plot(time90, individual_sex1,'go')  # green dots
# plt.plot(time91, individual_sex2,'b*')  # green dots
# plt.plot(time95,individual_sex3,'yx')  # green dots
# plt.plot(time96,individual_sex4,'rp')  # green dots
# plt.plot(time98,individual_sex5,'mv')  # green dots
# plt.show()
# plt.xlabel('Duration of Sexual Behavior(m)')
# plt.ylabel('Normalized Cell Density(cells/um2)')
# plt.title('Normalized cFos vs Duration of Sexual Behavior')

# plt.figure()
# plt.plot(time89, individual_sex1,'go')  # green dots
# plt.plot(time92, individual_sex2,'b*')  # green dots
# plt.plot(time93,individual_sex3,'yx')  # green dots
# plt.plot(time97,individual_sex4,'rp')  # green dots
# plt.show()
# plt.xlabel('Duration of Sexual Behavior(m)')
# plt.ylabel('Normalized Cell Density (cells/um2)')
# plt.title('Normalized cFos vs Duration of Exposure Male')

#%%
#old code from first fos experiment
#NOTE!!! All data below has been scale by e6 to conver to mm2
pos1_sex=[0.0002457951473,0.0002873743198,0.0002257947977,0.000200886686,0.0002576162401]
pos2_sex=[0.0003569233555,0.0003753143257,0.0004547696017,0.000216869507,0.0002806745953]
pos3_sex=[0.0004113883993,0.0003675718834,0.0004284349345,0.0002385405011,0.0005558271789]
pos4_sex=[0.000240129026,0.0003691410993,0.0003398538288,0.000298744586,0.0001971555143]
pos5_sex=[0.0003903132356,0.0002926955677,0.0002656631745,0.0002706121868,0.0003493929298]


pos1_con=[0.0001519688408,0.0001089254593,0.0002717878203]
pos2_con=[0.0001712164228,0.0002079056563,0.0002026643372,0.0004104526284]
pos3_con=[0.0003201039912,0.0003611215545,0.0002908929806,0.0003645684509]
pos4_con=[0.0002385431812,0.0001689082894,0.0003375235928,0.0002398422481]
pos5_con=[0.00026286672,0.0002945598707,0.000234485731,0.0003115637882]
  

pos1_sex=[x*1000000 for x in pos1_sex]
pos2_sex=[x*1000000 for x in pos2_sex]
pos3_sex=[x*1000000 for x in pos3_sex]
pos4_sex=[x*1000000 for x in pos4_sex]
pos5_sex=[x*1000000 for x in pos5_sex]

pos1_con=[x*1000000 for x in pos1_con]
pos2_con=[x*1000000 for x in pos2_con]
pos3_con=[x*1000000 for x in pos3_con]
pos4_con=[x*1000000 for x in pos4_con]
pos5_con=[x*1000000 for x in pos5_con]


means_sex = (mean_sex(pos1_sex),mean_sex(pos2_sex),mean_sex(pos3_sex),mean_sex(pos4_sex),mean_sex(pos5_sex))
means_control= (mean_sex(pos1_con),mean_sex(pos2_con),mean_sex(pos3_con),mean_sex(pos4_con),mean_sex(pos5_con))
std_sex=(std_grp(pos1_sex),std_grp(pos2_sex),std_grp(pos3_sex),std_grp(pos4_sex),std_grp(pos5_sex))
std_control=(std_grp(pos1_con),std_grp(pos2_con),std_grp(pos3_con),std_grp(pos4_con),std_grp(pos5_con))

######
######
#Data for lateral pag##
pos1_sex_L=[0.0001207673211,0.0001328535843,0.0001039117166,0.0001048816411,0.0000708562107]
pos2_sex_L=[0.0001155236108,0.0001130896975,0.0001104570159,0.000139962306,0.0000757238729]
pos3_sex_L=[0.00007368901633,0.0002623652754,0.0001057884691,0.0001250940045,0.0001085728579]
pos4_sex_L=[0.00005968920449,0.0001628823664,0.0001139946828,0.00008829594522,0.0000651847989]
pos5_sex_L=[0.00009249112789,0.00008584098877,0.00007099397319,0.0001243429605,0.0001046255957]



pos1_nosex_L=[0.00004876190476,0.00003181779615,0.00004689426014]
pos2_nosex_L=[0.00002605022483,0.000010047121,0.0001005537816,0.00009475807361]
pos3_nosex_L=[0.00005166009722,0.00002248528079,0.00007429032977,0.00007879213149]
pos4_nosex_L=[0.00001550238048,0.000005965127863,0.0000710350745,0.0000931380964]
pos5_nosex_L=[0.00002160603555,0.000008530876441,0.00002404472319,0.00001933732737]

pos1_sex_L=[x*1000000 for x in pos1_sex_L]
pos2_sex_L=[x*1000000 for x in pos2_sex_L]
pos3_sex_L=[x*1000000 for x in pos3_sex_L]
pos4_sex_L=[x*1000000 for x in pos4_sex_L]
pos5_sex_L=[x*1000000 for x in pos5_sex_L]

pos1_nosex_L=[x*1000000 for x in pos1_nosex_L]
pos2_nosex_L=[x*1000000 for x in pos2_nosex_L]
pos3_nosex_L=[x*1000000 for x in pos3_nosex_L]
pos4_nosex_L=[x*1000000 for x in pos4_nosex_L]
pos5_nosex_L=[x*1000000 for x in pos5_nosex_L]


means_sex_L = (mean_sex(pos1_sex_L),mean_sex(pos2_sex_L),mean_sex(pos3_sex_L),mean_sex(pos4_sex_L),mean_sex(pos5_sex_L))
means_control_L= (mean_sex(pos1_nosex_L),mean_sex(pos2_nosex_L),mean_sex(pos3_nosex_L),mean_sex(pos4_nosex_L),mean_sex(pos5_nosex_L))
std_sex_L=(std_grp(pos1_sex_L),std_grp(pos2_sex_L),std_grp(pos3_sex_L),std_grp(pos4_sex_L),std_grp(pos5_sex_L))
std_control_L=(std_grp(pos1_nosex_L),std_grp(pos2_nosex_L),std_grp(pos3_nosex_L),std_grp(pos4_nosex_L),std_grp(pos5_nosex_L))


#mean over all positions for each animal
Sex=[328.9098327,
338.4194392,
342.9032674,
245.1306934,
328.1332917,]

No_sex=[228.9398312,
258.1238427,
234.8984202,
319.6429872]


sex_mean=np.mean(Sex)
sex_std=np.std(Sex)

no_sex_mean=np.mean(No_sex)
no_sex_std=np.std(No_sex)

t_stat_vlateral,p_valvlateral=stats.ttest_ind(Sex,No_sex)

# ##below code is for plotting bar graphs of mean of means for each group
# ##plus individual means
# # Create lists for the plot
# materials = ['Sex', 'No_Sex']
# x_pos = np.arange(len(materials))
# CTEs = [sex_mean, no_sex_mean]
# error = [sex_std, no_sex_std]
# w = 0.4   # bar width
# # Build the plot
# fig, ax = plt.subplots()
# ax.bar(x_pos, CTEs, yerr=error, 
#        align='center', 
#        ecolor='black', 
#        color=['green','red'],
#        capsize=10,
#        alpha = 0.6,
#        width=w)

# # distribute scatter randomly across whole width of bar
# ax.scatter(x_pos[0] + np.random.random(np.size(Sex)) * .2 - .1, Sex, color='black')
# ax.scatter(x_pos[1] + np.random.random(np.size(No_sex)) * .2 - .1, No_sex, color='black')

# ax.set_ylabel('Normalized Cell Count (Cells/mm2)')
# ax.set_xticks(x_pos)
# ax.set_xticklabels(materials)
# ax.set_title('Cfos Expression in Posterior vlPAG')

# ax.yaxis.grid(False)


# Save the figure and show
plt.tight_layout()
plt.ylim([0,500])
plt.savefig('bar_plot_with_error_bars.png')
plt.show()


##below code is for plotting means of cells at each position along bregma
fig, ax = plt.subplots()
index = np.arange(5)
bar_width = 0.2
opacity = 0.8
 
scatter_sex_VL=plt.errorbar(index+(bar_width),means_sex,yerr=std_sex,c='k',marker='o',label='Sex-VL')
scatter_nosex_VL=plt.errorbar(index+(bar_width),means_control,yerr=std_control,c='k',marker='^',label='No Sex-VL')

scatter_sex_L=plt.errorbar(index+(bar_width),means_sex_L,yerr=std_sex_L,c='k',marker='s',label='No Sex-L')
scatter_nosex_L=plt.errorbar(index+(bar_width),means_control_L,yerr=std_control_L,c='k',marker='P',label='No Sex-L')
plt.xlabel('Position from Bregma')
plt.ylabel('Normalized Cell Density (cells/mm2)')
plt.title('cFos Expression in PAG')
plt.xticks(index + bar_width, ('-4.48','-4.6','-4.72','-4.84','-4.96'))
plt.plot(index+(bar_width),means_sex,c='b')
plt.plot(index+(bar_width),means_control,c='r')
plt.legend()
plt.ylim([0,500]) 
plt.tight_layout()
plt.show()
#%%
fig, ax = plt.subplots()
index = np.arange(5)
bar_width = 0.2
opacity = 0.8
scatter_sex_L=plt.errorbar(index+(bar_width),means_sex_L,yerr=std_sex_L,c='g',marker='s',label='No Sex-L')
scatter_nosex_L=plt.errorbar(index+(bar_width),means_control_L,yerr=std_control_L,c='y',marker='P',label='No Sex-L')


plt.xlabel('Position from Bregma')
plt.ylabel('Normalized Cell Density (cells/um2)')
plt.title('cFos Expression in lPAG')
plt.xticks(index + bar_width, ('-4.48','-4.6','-4.72','-4.84','-4.96'))
# plt.plot(index+(bar_width),means_sex,c='b')
# plt.plot(index+(bar_width),means_control,c='r')
plt.legend()
plt.ylim([0,0.0002]) 
plt.tight_layout()
plt.show()


#%%
#lateral pag

Sex=[92.43,
151.41,
101.03,
116.52,
84.99]

No_sex=[32.72,
11.76,
60.35,
66.58]

Sex_OVX=[320.23,293.39,434.97,281.94]
sex_mean=np.mean(Sex)
sex_std=np.std(Sex)


no_sex_mean=np.mean(No_sex)
no_sex_std=np.std(No_sex)

sex_OVX_mean=np.mean(Sex_OVX)
sex_OVX_std=np.std(Sex_OVX)

t_stat_lateral,p_val_lateral=stats.ttest_ind(Sex,No_sex)


# Create lists for the plot
materials = ['Sex_NC', 'No_Sex', 'Sex_OVX']
x_pos = np.arange(len(materials))
CTEs = [sex_mean, no_sex_mean, sex_OVX_mean]
error = [sex_std, no_sex_std, sex_OVX_std]
w = 0.4   # bar width
# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, 
       align='center', 
       ecolor='black', 
       color=['green','red','blue'],
       capsize=10,
       alpha = 0.6,
       width=w)

# distribute scatter randomly across whole width of bar
ax.scatter(x_pos[0] + np.random.random(np.size(Sex)) * .2 - .1, Sex, color='black')
ax.scatter(x_pos[1] + np.random.random(np.size(No_sex)) * .2 - .1, No_sex, color='black')
ax.scatter(x_pos[2] + np.random.random(np.size(Sex_OVX)) * .2 - .1, Sex_OVX, color='black')

ax.set_ylabel('Normalized Cell Count (Cells/um2)')
ax.set_xticks(x_pos)
ax.set_xticklabels(materials)
ax.set_title('Cfos Expression in Posterior lateral PAG')

ax.yaxis.grid(False)


# Save the figure and show
plt.tight_layout()
plt.ylim([0,450])
plt.savefig('bar_plot_with_error_bars.png')
plt.show()
#%%
#ventral lateral pag

#mean over all positions for each animal
Sex=[328.9098327,
338.4194392,
342.9032674,
245.1306934,
328.1332917,]

No_sex=[228.9398312,
258.1238427,
234.8984202,
319.6429872]

Sex_OVX=[230.18,242.49,392.73,263.94]

sex_mean=np.mean(Sex)
sex_std=np.std(Sex)


no_sex_mean=np.mean(No_sex)
no_sex_std=np.std(No_sex)

sex_OVX_mean=np.mean(Sex_OVX)
sex_OVX_std=np.std(Sex_OVX)

t_stat_lateral,p_val_lateral=stats.ttest_ind(Sex,No_sex)


# Create lists for the plot
materials = ['Sex_NC', 'No_Sex', 'Sex_OVX']
x_pos = np.arange(len(materials))
CTEs = [sex_mean, no_sex_mean, sex_OVX_mean]
error = [sex_std, no_sex_std, sex_OVX_std]
w = 0.4   # bar width
# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, 
       align='center', 
       ecolor='black', 
       color=['green','red','blue'],
       capsize=10,
       alpha = 0.6,
       width=w)

# distribute scatter randomly across whole width of bar
ax.scatter(x_pos[0] + np.random.random(np.size(Sex)) * .2 - .1, Sex, color='black')
ax.scatter(x_pos[1] + np.random.random(np.size(No_sex)) * .2 - .1, No_sex, color='black')
ax.scatter(x_pos[2] + np.random.random(np.size(Sex_OVX)) * .2 - .1, Sex_OVX, color='black')

ax.set_ylabel('Normalized Cell Count (Cells/um2)')
ax.set_xticks(x_pos)
ax.set_xticklabels(materials)
ax.set_title('Cfos Expression in Posterior Ventral lateral PAG')

ax.yaxis.grid(False)


# Save the figure and show
plt.tight_layout()
plt.ylim([0,450])
plt.savefig('bar_plot_with_error_bars.png')
plt.show()
#%%
#2mt analysis from roatation
pre_stim=(8862,6323,4990,6863,18238,11779,8409,2271)
post_stim=(4902,4213,5810,2928,2155,5902,4449,2551)
wash=(10767,6120,7554,4365,5362,13615,10175,8215)

std_pre=stats.sem(pre_stim)
std_post=stats.sem(post_stim)
std_wash=stats.sem(wash)



n_groups = 1
means_pre = (np.mean(pre_stim))
means_post = (np.mean(post_stim))
means_wash= (np.mean(wash))


conditions = ['PreStim', 'Odor Present', 'Washout']
x_pos = np.arange(len(conditions))
CTEs = [means_pre, means_post, means_wash]
error = [std_pre, std_post, std_wash]


fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10, color='k')
ax.set_ylabel('Mean Distance Traveled (pixels)')
ax.set_xticks(x_pos)
ax.set_xticklabels(conditions)
ax.set_title('2MT Effects on Distance Traveled')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()

#%%

n_groups = 5

#data of cell counts in vlpag at 5 different locations from bregma. Units are cells/mm2

a_vl_496=[60.09230795 ,59.1678109,70.26177545]
a_vl_484=[88.49245167,77.14726556,79.41630278]
a_vl_472=[23.87251574 ,62.82240984, 50.25792787]
a_vl_460=[49.49680215,21.21291521,37.12260161]
a_vl_448=[18.76859264 ,22.52231116 ,13.13801485]

#data of cell counts in lpag at 5 different locations from bregma.Units are cells/mm2

a_l_496=[0,7.914784853,9.497741824]
a_l_484=[43.30608258,63.81949012,36.46828007]
a_l_472=[104.8537814,95.86631442,29.95822326]
a_l_460=[74.35344955,123.9224159,70.81280909]
a_l_448=[129.6483033,125.944066,59.26779577]

means_sex_vl = (mean_sex(a_vl_448),mean_sex(a_vl_460),mean_sex(a_vl_472),mean_sex(a_vl_484),mean_sex(a_vl_496))
std_sex_vl=(std_grp(a_vl_448),std_grp(a_vl_460),std_grp(a_vl_472),std_grp(a_vl_484),std_grp(a_vl_496))

means_sex_l = (mean_sex(a_l_448),mean_sex(a_l_460),mean_sex(a_l_472),mean_sex(a_l_484),mean_sex(a_l_496))
std_sex_l=(std_grp(a_l_448),std_grp(a_l_460),std_grp(a_l_472),std_grp(a_l_484),std_grp(a_l_496))


vl_95=[18.76859264,49.49680215,23.87251574,88.49245167,60.09230795]
vl_97=[22.52231116,21.21291521,62.82240984,77.14726556,59.1678109]
vl_99=[13.13801485,37.12260161,50.25792787,79.41630278,70.26177545]

l_95=[129.6483033,74.35344955,104.8537814,43.30608258,0]
l_97=[125.944066,123.9224159,95.86631442,63.81949012,7.914784853]
l_99=[59.26779577,70.81280909,29.95822326,36.46828007,9.497741824]





# ##below code is for plotting means of cells at each position along bregma
# fig, ax = plt.subplots()
# index = np.arange(5)
# bar_width = 0.2
# opacity = 0.8
 
# scatter_projection_VL=plt.errorbar(index+(bar_width),means_sex,yerr=std_sex,c='k',marker='o',label='vlProjections')
# scatter_projection_L=plt.errorbar(index+(bar_width),means_control_L,yerr=std_control_L,c='k',marker='P',label='lProjections')


# plt.xlabel('Position from Bregma')
# plt.ylabel('Normalized Cell Density (cells/mm2)')
# plt.title('Projections neurons in PAG')
# plt.xticks(index + bar_width, ('-4.48','-4.6','-4.72','-4.84','-4.96'))
# plt.plot(index+(bar_width),means_sex,c='b')
# plt.plot(index+(bar_width),means_control,c='r')
# plt.legend()
# plt.ylim([0,500]) 
# plt.tight_layout()
# plt.show()
index = np.arange(5)
fig, ax = plt.subplots()
index = np.arange(5)
bar_width = 0.2
opacity = 0.8
scat1=plt.scatter(index+(bar_width),vl_95,c='k',marker='.')
scat2=plt.scatter(index+(bar_width),vl_97,c='k',marker='.')
scat3=plt.scatter(index+(bar_width),vl_99,c='k',marker='.')
scatter_nosex_VL=plt.errorbar(index+(bar_width),means_sex_vl,yerr=std_sex_vl,c='k',marker='^')

plt.xlabel('Position from Bregma')
plt.ylabel('Normalized Cell Density (cells/mm2)')
plt.title('Projection neurons in vlPAG')
plt.xticks(index + bar_width, ('-4.48','-4.6','-4.72','-4.84','-4.96'))
plt.legend()
plt.ylim([0,90]) 
plt.tight_layout()
plt.show()

plt.figure()
scat1=plt.scatter(index+(bar_width),l_95,c='k',marker='.')
scat2=plt.scatter(index+(bar_width),l_97,c='k',marker='.')
scat3=plt.scatter(index+(bar_width),l_99,c='k',marker='.')
scatter_nosex_VL=plt.errorbar(index+(bar_width),means_sex_l,yerr=std_sex_l,c='k',marker='^')


plt.xlabel('Position from Bregma')
plt.ylabel('Normalized Cell Density (cells/mm2)')
plt.title('Projection neurons in lPAG')
plt.xticks(index + bar_width, ('-4.48','-4.6','-4.72','-4.84','-4.96'))
plt.legend()
plt.ylim([0,130]) 
plt.tight_layout()
plt.show()
#%%
##The below code plots the overlap from the esr retrograde experiments

vl_overlap=[16.82242991, 19.54887218, 16.25]
l_overlap=[9.316770186, 12.5, 9.090909091]


sex_mean=np.mean(vl_overlap)
sex_std=np.std(vl_overlap)

no_sex_mean=np.mean(l_overlap)
no_sex_std=np.std(l_overlap)


# Create lists for the plot
materials = ['vlPAG', 'lPAG']
x_pos = np.arange(len(materials))
CTEs = [sex_mean, no_sex_mean]
error = [sex_std, no_sex_std]
w = 0.4   # bar width
# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, 
       align='center', 
       ecolor='black', 
       color=['green','red'],
       capsize=10,
       alpha = 0.6,
       width=w)

# distribute scatter randomly across whole width of bar
ax.scatter(x_pos[0] + np.random.random(np.size(vl_overlap)) * .2 - .1, vl_overlap, color='black')
ax.scatter(x_pos[1] + np.random.random(np.size(l_overlap)) * .2 - .1, l_overlap, color='black')

ax.set_ylabel('Percentage of overlap')
ax.set_xticks(x_pos)
ax.set_xticklabels(materials)
ax.set_title('Esr1+ Projection neurons cFos overlap')

ax.yaxis.grid(False)


# Save the figure and show
plt.tight_layout()
plt.ylim([0,30])
plt.savefig('bar_plot_with_error_bars.png')
plt.show()
