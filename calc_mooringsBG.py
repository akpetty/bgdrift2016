############################################################## 
# Date: 01/01/16
# Name: calc_mooringsBG.py
# Author: Alek Petty
# Description: Script to read in the BG ULS moorings and calculate seasonal averages
# Input requirements: Mooring data                 
# Output: Seasonal ULS data

import numpy as np
from pylab import *
import scipy.io
import numpy.ma as ma
from scipy import stats
from glob import glob
import h5py

time_index = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
num_years = 10

def read_moorings(mooring_letter):
#VARIABLES
# dates: date string timeseries
# name: name of the mooring and dataset
# yday: year day timeseries 
# BETA: final beta adjustment timeseries used in ice draft calculations
# BTBETA: initial beta timeseries based on bottom temperature
# ID: number of ice drafts binned daily every 0.1 m from 0.05 to 29.95 m 
# IDS: daily ice draft statistics: number, mean, std, minimum, maximum, median
# OWBETA: beta timeseries determined from open water events
# T: temperature timeseries (C)
# WL: water level timeseries (m)
#number, mean, std, minimum, maximum, median
#NOTE THAT AN EXTRA YEAR IS NEEDED AS 2011 GOES INTO 2012

	total_months = np.zeros((12, num_years+1))
	months_count = np.zeros((12, num_years+1))
	seasons_count = np.zeros((4, num_years+1))
	mean_seasons = ma.masked_all((4, num_years+1))
	total_seasons = ma.masked_all((4, num_years+1))
	for x in xrange(num_years):
		year = '%02d' % (x+3)
		files= glob(datapath+'/DATA/uls'+year+mooring_letter+'*.mat')
		if (size(files)>0):			
			for s in xrange(size(files)):
				print x, size(files), s, year
				if (x<9):
					uls = scipy.io.loadmat(files[s],struct_as_record=False)
					days = uls['dates']
					a = uls['IDS']
					num = a[:, 0]
					mean = a[:, 1]
					std = a[:, 2]
					min = a[:, 3]
					max = a[:, 4]
					med = a[:, 5]

					for d in xrange(size(days)):
						year_s = int(days[d][0:4])
						month_s = int(days[d][5:7])-1
						if (np.isfinite(mean[d])):
							total_months[month_s, year_s-2003]+= mean[d]
							months_count[month_s, year_s-2003]+=1.

				else:
					uls = h5py.File(files[s],'r') 
					ydays = uls.get('yday')[0]
					a = np.array(uls.get('IDS')).T
					num = a[:, 0]
					mean = a[:, 1]
					std = a[:, 2]
					min = a[:, 3]
					max = a[:, 4]
					med = a[:, 5]
					for d in xrange(size(ydays)):
						if ydays[d]<365:
							year_s = 2012
							month_s = next(i for i,v in enumerate(time_index) if v > ydays[d]) - 1
						else:
							year_s = 2013
							yday_m = ydays[d]-365	
							month_s = next(i for i,v in enumerate(time_index) if v > yday_m) - 1
						print month_s
						if (np.isfinite(mean[d])):
							total_months[month_s, year_s-2003]+= mean[d]
							months_count[month_s, year_s-2003]+=1.

		if (size(files)==0):	
			print year, mooring_letter
	mean_months=total_months/months_count
	mean_months = ma.masked_array(mean_months,np.isnan(mean_months))

	for x in xrange(4):
		mean_seasons[x, :] = ma.mean(mean_months[(x*3):(x*3)+3, :], axis=0)
		seasons_count[x, :] = ma.sum(months_count[(x*3):(x*3)+3, :], axis=0)

	mean_seasons=ma.masked_where(seasons_count<45.,  mean_seasons)

	return mean_seasons, months_count

rawdatapath='../../DATA/'
datapath = rawdatapath+'MOORINGS'
outpath = './Data_output/'

mean_seasons_all = ma.masked_all((4, 4, num_years+1))
mooring_letter=['a', 'b', 'c', 'd']
for x in xrange(4):
	mean_seasons_all[x], months_count = read_moorings(mooring_letter[x])

mean_seasons_all.dump(outpath+'/mooringsBG_seasons_all.txt')

print 'Max seasons/moorings:', 4*4*num_years+1
print 'Masked seasons/moorings:', ma.count_masked(mean_seasons_all)
