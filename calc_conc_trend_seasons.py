############################################################## 
# Date: 01/01/16
# Name: calc_conc_trend_seasons.py
# Author: Alek Petty
# Description: Script to calculate seasonal trends in ice concentration
#			   Used for plot_conc_trend_seasons_ann.py
# Input requirements: monthly ice concentration data, produced using calc_ice_conc_monthly_NT_BT.py
#                    
# Output: Monthly trends (and sig) of ice concentration

import numpy as np
from pylab import *
from scipy.io import netcdf
import numpy.ma as ma
from scipy import stats
from netCDF4 import Dataset
from glob import glob

def BG_box(beau_region):
	lons_beau = np.zeros((40))
	lons_beau[0:10] = np.linspace(beau_region[0], beau_region[1], 10)
	lons_beau[10:20] = np.linspace(beau_region[1], beau_region[1], 10)
	lons_beau[20:30] = np.linspace(beau_region[1], beau_region[0], 10)
	lons_beau[30:40] = np.linspace(beau_region[0], beau_region[0], 10)
	lats_beau = np.zeros((40))
	lats_beau[0:10] = np.linspace(beau_region[2], beau_region[2], 10)
	lats_beau[10:20] = np.linspace(beau_region[2], beau_region[3], 10)
	lats_beau[20:30] = np.linspace(beau_region[3], beau_region[3], 10)
	lats_beau[30:40] = np.linspace(beau_region[3], beau_region[2], 10)

	return lons_beau, lats_beau

def calc_time_regres(var, num_years_req):
	#find the linear regression for each grid point as a function of
	nx = var.shape[1]
	ny = var.shape[2]
	#print nx, ny
	trend = ma.masked_all((nx, ny))
	sig = ma.masked_all((nx, ny))
	r = ma.masked_all((nx, ny))

	years = np.arange(var.shape[0])
	for i in xrange(nx):
		for j in xrange(ny):
			#print i, j
			var_ma = var[:, i, j][~var[:, i, j].mask]
			years_ma = years[~var[:, i, j].mask]
			if len(var_ma>num_years_req):
				trend[i, j], intercept, r[i, j], prob, stderr = stats.linregress(years_ma,var_ma)
				sig[i, j] = 100*(1-prob)
	trend = ma.array(trend,mask=np.isnan(trend))
	r = ma.array(r,mask=np.isnan(r))
	sig = ma.array(sig,mask=np.isnan(sig))

	return trend, sig, r

datapath='./Data_output/'

start_year=1980
end_year=2013

alg = 'NASA_TEAM'
ice_conc = load(datapath+'/ice_conc_months-'+str(start_year)+'-'+str(end_year)+alg+'.txt')


months = [0, 3, 6, 9]
num_years_req = 10

ice_conc_seasons = ma.masked_all((ice_conc.shape[0], 4, ice_conc.shape[2], ice_conc.shape[3]))
conc_seasons_trend = ma.masked_all((4, ice_conc.shape[2], ice_conc.shape[3]))
conc_sig = ma.masked_all((4, ice_conc.shape[2], ice_conc.shape[3]))
conc_r = ma.masked_all((4, ice_conc.shape[2], ice_conc.shape[3]))

for x in xrange(4):
	ice_conc_seasons[:,x] = ma.mean(ice_conc[:,(x*3):(x*3)+3, :], axis=1)
	

for x in xrange(4):
	print x
	conc_seasons_trend[x], conc_sig[x], conc_r[x] = calc_time_regres(ice_conc_seasons[:, x], num_years_req)

conc_seasons_trend.dump(datapath+'/conc_seasons_trend'+alg+'nyrs'+str(num_years_req)+str(start_year)+'-'+str(end_year)+'.txt')
conc_sig.dump(datapath+'/conc_seasons_sig'+alg+'nyrs'+str(num_years_req)+str(start_year)+'-'+str(end_year)+'.txt')
conc_r.dump(datapath+'/conc_seasons_r'+alg+'nyrs'+str(num_years_req)+str(start_year)+'-'+str(end_year)+'.txt')





