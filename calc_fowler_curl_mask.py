############################################################## 
# Date: 01/01/16
# Name: calc_fowler_curl.py
# Author: Alek Petty
# Description: Script to read in the daily fowler drift vectors and create monthly fields of the ice drift curl (squared)
# Input requirements: Fowler daily drift data (vectors in x/y direction) and projection lat/lon
#                     Also needs the functions in alek_objects
# Output: Monthly curl (scalars) on the same grid, masked however chosen. 

import BG_functions as BGF
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
from pylab import *
import numpy.ma as ma
from glob import glob
import os
#from scipy.interpolate import griddata

m = Basemap(projection='npstere',boundinglat=66,lon_0=0, resolution='l'  )

datapath='./Data_output/'
xpts = loadtxt(datapath+'/xpts_100km.txt')
ypts = loadtxt(datapath+'/ypts_100km.txt')
lons_t = loadtxt(datapath+'/lons_100km.txt')
lats_t = loadtxt(datapath+'/lats_100km.txt')

init_path = datapath+'/FOWLER/'
dumppath = datapath+'/FOWLER_MA/'

if not os.path.exists(dumppath):
	os.makedirs(dumppath)
start_year = 1980
end_year = 2013
num_years = end_year - start_year + 1
year_str= str(start_year)+'-'+str(end_year)
extra='100km'

curl_months = load(init_path+year_str+'-curl_data_months'+extra+'.txt')
curl_months_ma = ma.masked_all(curl_months.shape)
masked_months_count = ma.masked_all((12, 55, 55))

for x in xrange(12):
	masked_months_count[x] = ma.count_masked(curl_months[:, x], axis=0)

for x in xrange(34):
	curl_months_ma[x]=ma.masked_where(masked_months_count>1,curl_months[x])

curl_months_ma.dump(dumppath+year_str+'-curl_data_months'+extra+'.txt')

plot==0
if (plot==1):
	for x in xrange(34):
		for y in xrange(12):
			apy.plot_var(m, xpts, ypts, curl_months_ma[x, y], out=out_path+str(x+1980)+str(y+1), 
				units_lab=r'm s$^{-1}$', base_mask=1, minval=-5e-8, maxval=5e-8, cbar_type='both', cmap_1=plt.cm.RdBu_r)
