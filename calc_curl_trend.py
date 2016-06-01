############################################################## 
# Date: 01/01/16
# Name: calc_curl_trend.py
# Author: Alek Petty
# Description: Script to calculate the trend in the FOWLER/CERSAT curls
# Input requirements:monthly curl data
# Output: trends (and other variabels) of the various curls
# Info: Need to run for the various drift products to produce all the right files


import matplotlib
matplotlib.use("AGG")

from mpl_toolkits.basemap import Basemap, shiftgrid
# Numpy import
import numpy as np
from pylab import *
from scipy.io import netcdf
import numpy.ma as ma
from scipy import stats
from matplotlib import rc
from netCDF4 import Dataset
import os

mpl.rc("ytick",labelsize=10)
mpl.rc("xtick",labelsize=10)
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
#mpl.rc('text', usetex=True)

def calc_time_regres(var, num_years_req):
	#find the linear regression for each grid point as a function of
	nx = var.shape[-2]
	ny = var.shape[-1]

	trend = ma.masked_all((nx, ny))
	sig = ma.masked_all((nx, ny))
	r = ma.masked_all((nx, ny))

	years = np.arange(num_years)
	for i in xrange(nx):
		for j in xrange(ny):
			var_ma = var[:, i, j][~var[:, i, j].mask]
			years_ma = years[~var[:, i, j].mask]
			if len(var_ma>num_years_req):
				trend[i, j], intercept, r[i, j], prob, stderr = stats.linregress(years_ma,var_ma)
				sig[i, j] = 100*(1-prob)
	trend = ma.array(trend,mask=np.isnan(trend))
	r = ma.array(r,mask=np.isnan(r))
	sig = ma.array(sig,mask=np.isnan(sig))

	return trend, sig, r

def BG_box():
	lons_beau = np.zeros((40))
	lons_beau[0:10] = np.linspace(beau_region[3], beau_region[2], 10)
	lons_beau[10:20] = np.linspace(beau_region[2], beau_region[2], 10)
	lons_beau[20:30] = np.linspace(beau_region[2], beau_region[3], 10)
	lons_beau[30:40] = np.linspace(beau_region[3], beau_region[3], 10)
	lats_beau = np.zeros((40))
	lats_beau[0:10] = np.linspace(beau_region[1], beau_region[1], 10)
	lats_beau[10:20] = np.linspace(beau_region[1], beau_region[0], 10)
	lats_beau[20:30] = np.linspace(beau_region[0], beau_region[0], 10)
	lats_beau[30:40] = np.linspace(beau_region[0], beau_region[1], 10)

	return lons_beau, lats_beau

def calc_time_regres_1d(var, num_years_req):
	#find the linear regression for each grid point as a function of
	#years = np.arange(num_years)
	var_ma = var[~var.mask]
	years_ma = years[~var.mask]

	if len(var_ma>num_years_req):
		trend, intercept, r, prob, stderr = stats.linregress(years_ma,var_ma)
		sig = 100*(1-prob)

	return trend, sig, r, intercept


m = Basemap(projection='npstere',boundinglat=66,lon_0=0, resolution='l'  )

month_str = ['JFM', 'AMJ', 'JAS', 'OND']
year_str = '1980-2013'
extra = '100km'
beau_region = [72., 82., -130, -170]

#NEED TO LOOP OVER ALL OF THESE
ascat3=0
ascat6=0
qscat=1
fowler=0
fowler_ma=0
if ascat3==1:
    daylag=3
    mfill=1
    if (mfill==1):
        product = 'ASCAT_3DAYMFILL'
    else:
        product = 'ASCAT_3DAY'
    start_year = 2008
    end_year = 2013
    num_seasons=2
if ascat6==1:
    daylag=6
    product = 'ASCAT_6DAY'
    start_year = 2008
    end_year = 2013
    num_seasons=2
if qscat==1:
    daylag=3
    product = 'QSCAT_3DAYMFILL'
    start_year = 1992
    end_year = 2008
    num_seasons=2
    #extra='50km'
if fowler==1:
    daylag=3
    product = 'FOWLER'
    start_year = 1980
    end_year = 2013
    num_seasons=4
if fowler_ma==1:
    daylag=3
    product = 'FOWLER_MA'
    start_year = 1980
    end_year = 2013
    num_seasons=4

print end_year
num_years = end_year -start_year + 1

datapath = './Data_output/'


drift_curl_months = load(datapath+'/'+product+'/'+str(start_year)+'-'+str(end_year)+'-curl_data_months'+extra+'.txt')

xpts = loadtxt(datapath+'/xpts_'+extra+'.txt')
ypts = loadtxt(datapath+'/ypts_'+extra+'.txt')
lons_t = loadtxt(datapath+'/lons_'+extra+'.txt')
lats_t = loadtxt(datapath+'/lats_'+extra+'.txt')

nx = drift_curl_months.shape[-1]
ny = drift_curl_months.shape[-2]
num_years = drift_curl_months.shape[0]

drift_curl_seasons = ma.masked_all((num_years, num_seasons, nx, ny))
drift_curl_seasons_count = ma.masked_all((num_years, num_seasons, nx, ny))
for x in xrange(num_seasons):
	drift_curl_seasons[:, x] = ma.mean(drift_curl_months[:, (x*3):(x*3)+3], axis=1)
	drift_curl_seasons_count[:, x] = ma.count_masked(drift_curl_months[:, x*3:(x*3)+3], axis=1)
#drift_curl_seasons = ma.masked_where(drift_curl_seasons_count>1.5, drift_curl_seasons)

drift_curl_seasons_trend=ma.masked_all((num_seasons, nx, ny))
drift_curl_seasons_sig=ma.masked_all((num_seasons, nx, ny))
drift_curl_seasons_r=ma.masked_all((num_seasons, nx, ny))

num_years_req=num_years/2

for x in xrange(num_seasons):
	drift_curl_seasons_trend[x], drift_curl_seasons_sig[x], drift_curl_seasons_r[x] = calc_time_regres(drift_curl_seasons[:, x], num_years_req)


#beau_region = [71., 80., -120, -150]


drift_curl_seasons_BG_mean = ma.masked_all((num_seasons, num_years))
for x in xrange(num_seasons):
	for y in xrange(num_years):
		drift_curl_seasons_BG_mean[x, y] = ma.mean(ma.masked_where((lats_t<beau_region[0]) | (lats_t>beau_region[1]) | (lons_t>beau_region[2]) | (lons_t<beau_region[3]), drift_curl_seasons[y, x]))

drift_curl_seasons_BG_trend=ma.masked_all((num_seasons))
drift_curl_seasons_BG_sig=ma.masked_all((num_seasons))
drift_curl_seasons_BG_r=ma.masked_all((num_seasons))
drift_curl_seasons_BG_int=ma.masked_all((num_seasons))
drift_curl_seasons_BG_tline = ma.masked_all((num_seasons, num_years))

years = np.arange(start_year, end_year+1, 1)

for x in xrange(num_seasons):
	print x
	drift_curl_seasons_BG_trend[x], drift_curl_seasons_BG_sig[x], drift_curl_seasons_BG_r[x],drift_curl_seasons_BG_int[x]  = calc_time_regres_1d(drift_curl_seasons_BG_mean[x], num_years_req)
	drift_curl_seasons_BG_tline[x] = (years*drift_curl_seasons_BG_trend[x])+drift_curl_seasons_BG_int[x]


drift_curl_seasons_trend.dump(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_trend'+extra+'.txt')
drift_curl_seasons_sig.dump(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_sig'+extra+'.txt')
drift_curl_seasons_r.dump(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_r'+extra+'.txt')
drift_curl_seasons_BG_mean.dump(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_mean'+extra+'.txt')
drift_curl_seasons_BG_tline.dump(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_tline'+extra+'.txt')
drift_curl_seasons_BG_trend.dump(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_trend'+extra+'.txt')
drift_curl_seasons_BG_sig.dump(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_sig'+extra+'.txt')
drift_curl_seasons_BG_r.dump(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_r'+extra+'.txt')
drift_curl_seasons_BG_int.dump(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_int'+extra+'.txt')




