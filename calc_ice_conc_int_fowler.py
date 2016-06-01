############################################################## 
# Date: 01/01/16
# Name: calc_ice_conc_int_fowler.py
# Author: Alek Petty
# Description: Script to calculate ice concentration (NASA_TEAM or BOOTSTRAP) as monthly averages projected on the Fowler grid
# Input requirements: FOWLER projection lat/lon and raw ice concentration
#                     Also needs some functions in BG_functions
# Output: Monthly ice concentration on the folwer grid 

import BG_functions as BGF
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
from pylab import *
from scipy.io import netcdf
import numpy.ma as ma
from glob import glob


figpath='./Figures/'
datapath='./Data_output/'
rawdatapath='../../DATA/'
initpath=rawdatapath+'/ICE_DRIFT/FOWLER/DATA/'

m = Basemap(projection='npstere',boundinglat=66.,lon_0=0, resolution='l'  )
dx_res = 100000.
start_year=1980
end_year=2013
num_years = end_year-start_year+1
years_str=str(start_year)+'-'+str(end_year)

team = 'NASA_TEAM'

ice_conc = load(datapath+'/ice_conc_months-'+years_str+team+'.txt')
lats = load(datapath+'/ice_conc_lats'+team+'.txt')
lons = load(datapath+'ice_conc_lons'+team+'.txt')
xpts, ypts = m(lons, lats)

xptsF, yptsF = BGF.return_xpts_ypts_fowler(initpath, m)
xpts_int = xptsF
ypts_int = yptsF

num_years = ice_conc.shape[0]
num_months = ice_conc.shape[1]

ice_conc = ice_conc.filled(-999)
for x in xrange(num_years):
	for y in xrange(num_months):
		#REPLACE MASKED VALUES LOWER THAN THE POLAR GAP WITH 0 AS THESE ARE JUST 0 FOR INT PURPOSES.
		ice_conc[x, y] = where((ice_conc[x, y] <= -1) & (lats <= 80), 0, ice_conc[x, y])
ice_conc = ma.masked_where(ice_conc<-1, ice_conc)
ice_conc_int = ma.masked_all((num_years, num_months, xpts_int.shape[0], xpts_int.shape[1]))

for x in xrange(num_years):
	for y in xrange(num_months):
		ice_conc_int[x, y]=BGF.interp_data(ice_conc[x, y], xpts, ypts, xpts_int,ypts_int)
		#INTERPOLATE THE BOLEEN MASKED ARRAY ONTO GRID TOO

ice_conc_int.dump(datapath+'/ice_conc_months-'+years_str+team+'_fowler_int.txt')

out_figs=0
if (out_figs==1):
	for year in xrange(num_years):
		for month in xrange(12):
			month_str = '%02d' % (month+1)
			apy.plot_var(m, xptsF , yptsF, ice_conc_int[year, month], out=figpath+'ice_conc_int_'+str(year+1980)+'_'+month_str, 
				units_lab='Ice concentration', string_corner=str(year+1980)+'\n'+month_str, minval=0, maxval=1.0, base_mask=1, 
				cbar_type='neither', cmap_1=plt.cm.Reds)









