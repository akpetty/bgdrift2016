############################################################## 
# Date: 01/01/16
# Name: calc_ice_conc_int_100km.py
# Author: Alek Petty
# Description: Script to calculate ice concentration (NASA_TEAM or BOOTSTRAP) as monthly averages projected on a 100km grid used in this study (polar sterographic, 66N)
# Input requirements: 100km Polar Stereo projection lat/lon and raw ice concentration
#                     Also needs some functions in BG_functions
# Output: Monthly ice concentration on the 100km grid

import BG_functions as BGF
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
from pylab import *
import numpy.ma as ma
from glob import glob

datapath='./Data_output/'

m = Basemap(projection='npstere',boundinglat=66.,lon_0=0, resolution='l'  )
dx_res = 100000.

team='NASA_TEAM'
start_year=1980
end_year=2013
num_years = end_year-start_year+1
years_str=str(start_year)+'-'+str(end_year)

ice_conc = load(datapath+'/ice_conc_months-'+years_str+team+'.txt')
lats = load(datapath+'/ice_conc_lats'+team+'.txt')
lons = load(datapath+'ice_conc_lons'+team+'.txt')
xpts, ypts = m(lons, lats)

xpts_int = loadtxt(datapath+'/xpts_100km.txt')
ypts_int = loadtxt(datapath+'/ypts_100km.txt')

num_years = ice_conc.shape[0]
num_months = ice_conc.shape[1]

ice_conc = ice_conc.filled(-999)
for x in xrange(num_years):
	for y in xrange(num_months):
		#REPLACE MASKED VALUES LOWER THAN THE POLAR GAP WITH 0 AS THESE ARE JUST 0 FOR INT PURPOSES.
		ice_conc[x, y] = where((ice_conc[x, y] <= -1) & (lats <= 80), 0, ice_conc[x, y])
ice_conc = ma.masked_where(ice_conc<-1, ice_conc)

ice_conc_int = ma.masked_all((num_years, num_months, xpts_int.shape[0], xpts_int.shape[1]))
#conc_int_mask = ma.masked_all((num_years, num_months, xpts_int.shape[0], xpts_int.shape[1]))


for x in xrange(num_years):
	for y in xrange(num_months):
		ice_conc_int[x, y]=BGF.interp_data(ice_conc[x, y], xpts, ypts, xpts_int,ypts_int)
		#INTERPOLATE THE BOLEEN MASKED ARRAY ONTO GRID TOO
		#conc_int_mask[x, y]=interp_data(conc_mask[x, y])

ice_conc_int.dump(datapath+'/ice_conc_months-'+years_str+team+'_100km_int.txt')






