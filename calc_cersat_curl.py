############################################################## 
# Date: 01/01/16
# Name: calc_cersat_curl.py
# Author: Alek Petty
# Description: Script to read in the daily CERSAT drift vectors and create monthly fields of the ice drift curl (squared)
# Input requirements: CERSAT daily drift data (vectors in x/y direction) and projection lat/lon
#                     Also needs the functions in alek_objects
# Output: Monthly curl (scalars) on the same grid, masked however chosen.
# Info: need to run for the 3 different CERSAT products, also with different data masking options (MFILL)

from netCDF4 import Dataset
import BG_functions as BGF
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
from pylab import *
import numpy.ma as ma
from glob import glob
import os
#from scipy.interpolate import griddata

m = Basemap(projection='npstere',boundinglat=66,lon_0=0, resolution='l'  )

dx_res = 100000.

#NEED TO LOOP OVER ALL OF THESE
ascat3=0
ascat6=0
qscat=1

if ascat3==1:
    daylag=3
    mfill=0
    if (mfill==1):
        product_path = 'DRIFT_ASCAT/3DAY_MFILL'
        product = 'ASCAT_3DAYMFILL'
    else:
        product_path = 'DRIFT_ASCAT/3DAY'
        product = 'ASCAT_3DAY'
    start_year = 2008
    end_year = 2013

if ascat6==1:
    daylag=6
    product_path = 'DRIFT_ASCAT/6DAY'
    product = 'ASCAT_6DAY'
    start_year = 2008
    end_year = 2013
if qscat==1:
    daylag=3
    mfill=0
    if (mfill==1):
        product_path = 'DRIFT_QUICKSCAT/3DAY_MFILL'
        product = 'QSCAT_3DAYMFILL'
    else:
        product_path = 'DRIFT_QUICKSCAT/3DAY'
        product = 'QSCAT_3DAY'    
    start_year = 1992
    end_year = 2008

rawdatapath='../../Data/'
datapath='./Data_output/'
figpath='./Figures/'
initpath = rawdatapath+'/ICE_DRIFT/CERSAT/'+product_path+'/'
outpath = datapath+product+'/'
if not os.path.exists(outpath):
    os.makedirs(outpath)


num_days_lag = daylag


files = glob(initpath+'2007/200701*.nc')
f = Dataset(files[0], 'r')
lon = f.variables['longitude'][:]
lat = f.variables['latitude'][:]
xpts, ypts = m(lon, lat)

arr_res = int(ceil(100000./dx_res))
grid_str=str(int(dx_res/1000))+'km'
nx = int((m.xmax-m.xmin)/dx_res)+1; ny = int((m.ymax-m.ymin)/dx_res)+1
xpts2 = np.linspace(m.xmin, m.xmax, nx)
ypts2 = np.linspace(m.ymin, m.ymax, ny)
xpts2m, ypts2m = np.meshgrid(xpts2, ypts2)

lons_grid, lats_grid = m.makegrid(nx, ny)

output_lonlat=0
if (output_lonlat==1):
    savetxt(datapath+'/lons_'+grid_str+'.txt', lons_grid)
    savetxt(datapath+'/lats_'+grid_str+'.txt', lats_grid)

    savetxt(datapath+'/xpts_'+grid_str+'.txt', xpts2m)
    savetxt(datapath+'/ypts_'+grid_str+'.txt', ypts2m)


num_years = end_year - start_year + 1
curl_months = ma.masked_all((num_years, 6, nx, ny))
drift_month= ma.masked_all((num_years,  6, 2, nx, ny))
drift_month_int= ma.masked_all((num_years,  6, 2, nx, ny))


months = [0, 1, 2, 9, 10, 11]

for y in xrange(num_years):
    for mon in xrange(size(months)):
        month = '%02d' % (months[mon]+1)
        year =  str(start_year + y)
        files = glob(initpath+year+'/'+year+month+'*.nc')
        num_days = size(files) #int(size(files)/int(num_days_lag)+1)
        print num_days
        if size(files)>0:
            drift_days = ma.masked_all((num_days,2,  nx, ny))
            drift_days_int_mask = ma.masked_all((num_days, 2, nx, ny))
            curl_days = ma.masked_all((num_days, nx, ny))
            for x in xrange(0, size(files)):
                print x
                day = '%02d' % x 
                f = Dataset(files[x], 'r')
                u = f.variables['zonal_motion'][0]/(60.*60.*24.*num_days_lag)
                v = f.variables['meridional_motion'][0]/(60.*60.*24.*num_days_lag)
                q = f.variables['quality_flag'][0]

                #ROTATE VECTORS TO X/Y GRID on NSIDC GRID FOR CURL CALC (SO IT IS FLAT ALONG THE BOTTOM)
                u_r,v_r = m.rotate_vector(u,v,lon,lat)
                #grid data onto 100km grid - also matches wind forcing fields
                drift_days[x, 0], drift_days[x, 1], q_int = BGF.interp_uvCSAT(u_r, v_r, q, xpts, ypts, xpts2m, ypts2m)
            
                curl_days[x] = BGF.calc_curl_sq_2d_xy_gradient(drift_days[x, 0], drift_days[x, 1], dx_res)
        drift_days_mask_count = ma.count_masked(drift_days[:, 0], axis=0)
        days_in_month=15
        curl_months[y, mon] = ma.masked_where(drift_days_mask_count>days_in_month, ma.mean(curl_days, axis=0))
        drift_month[y, mon, 0]  = ma.masked_where(drift_days_mask_count>days_in_month, ma.mean(drift_days[:, 0] , axis=0))
        drift_month[y, mon, 1]  = ma.masked_where(drift_days_mask_count>days_in_month, ma.mean(drift_days[:, 1] , axis=0))
  

curl_months.dump(outpath+str(start_year)+'-'+str(end_year)+'-curl_data_months'+grid_str+'.txt')


plot_curl=0
if (plot_curl==1):
    for x in xrange(num_years):
        BGF.plot_var_xy(m, xpts2m, ypts2m, ma.mean(drift_month[x, :, 0], axis=0), ma.mean(drift_month[x, :, 1], axis=0), ma.mean(curl_months[x], axis=0), out=figpath+str(x+start_year)+'_'+str(daylag)+'day', units_lab=r'm s$^{-1}$', units_vec=r'm s$^{-1}$',
                    base_mask=1,res=2, minval=-5e-8, maxval=5e-8, scale_vec=0.5, vector_val=0.2, cbar_type='both', cmap_1=plt.cm.RdBu_r)

    month = 3 #OCTOBER
    for x in xrange(num_years):
        BGF.plot_var_xy(m, xpts2m, ypts2m, drift_month[x, month, 0], drift_month[x, month, 1], curl_months[x, month], 
            out=figpath+str(x+start_year)+'_'+str(month+1)+'_'+str(daylag)+'day', units_lab=r'm s$^{-1}$', units_vec=r'm s$^{-1}$',base_mask=1,res=2, 
            minval=-5e-8, maxval=5e-8, scale_vec=0.5, vector_val=0.2, cbar_type='both', cmap_1=plt.cm.RdBu_r)


