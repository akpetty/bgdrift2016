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

rawdatapath='../../../Data/'
datapath='./Data_output/'
figpath='./Figures/'

outpath = datapath+'/FOWLER/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

initpath = rawdatapath+'ICE_DRIFT/FOWLER/DAILY/'
lon_lat = loadtxt(rawdatapath+'ICE_DRIFT/FOWLER/north_x_y_lat_lon.txt')

lons = np.zeros((361, 361))
lats = np.zeros((361, 361))
lons = np.reshape(lon_lat[:, 3], (361, 361))
lats = np.reshape(lon_lat[:, 2], (361, 361))
xpts, ypts = m(lons, lats)

dx_res = 100000.

arr_res = int(ceil(100000./dx_res))
grid_str=str(int(dx_res/1000))+'km'
nx = int((m.xmax-m.xmin)/dx_res)+1; ny = int((m.ymax-m.ymin)/dx_res)+1
xpts2 = np.linspace(m.xmin, m.xmax, nx)
ypts2 = np.linspace(m.ymin, m.ymax, ny)
xpts2m, ypts2m = np.meshgrid(xpts2, ypts2)

lons_100km, lats_100km = m.makegrid(nx, ny)
savetxt(outpath+'lons_'+grid_str+'.txt', lons_100km)
savetxt(outpath+'lats_'+grid_str+'.txt', lats_100km)
savetxt(outpath+'xpts_'+grid_str+'.txt', xpts2m)
savetxt(outpath+'ypts_'+grid_str+'.txt', ypts2m)


start_year = 1980
end_year = 2013

num_years = end_year-start_year+1
time_index = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

curl_months = ma.masked_all((num_years, 12, nx, ny))
u_month= ma.masked_all((num_years, 12, nx, ny))
v_month= ma.masked_all((num_years, 12, nx, ny))
for y in xrange(num_years):
    print 'y:',y
    files = glob(initpath+str(y+1980)+'/*.bin')
    if (y>32):
        time_index[12]=364
        files = glob(initpath+str(y+1980)+'/*.vec')
    for mon in xrange(12):
        print 'm:',mon
        curl_days = ma.masked_all((time_index[mon+1] - time_index[mon], nx, ny))
        u_int = ma.masked_all((time_index[mon+1] - time_index[mon], nx, ny))
        v_int = ma.masked_all((time_index[mon+1] - time_index[mon], nx, ny))

        for x in xrange(time_index[mon], time_index[mon+1], 1):
            #print 'd:',x

            fd = open(files[x], 'rb')
            motion_dat = fromfile(file=fd, dtype='<i2')
            motion_dat = reshape(motion_dat, [361, 361, 3])

            ut = motion_dat[:, :, 0]/1000.
            vt = motion_dat[:, :, 1]/1000.         
            q = motion_dat[:, :, 2]/1000.

            #grid data onto 100km grid - matches wind forcing fields
            #DONT NEED TO ROTATE AS ALREADY ON AN XY GRID
            #ALSO PUT IT ONTO A GRID THAT IS THE RIGHT WAY AROUND! I>E> BOTTOM TO TOP
            u_int[x-time_index[mon]], v_int[x-time_index[mon]] = BGF.interp_uv(ut, vt, q, xpts, ypts, xpts2m, ypts2m)
            
            curl_days[x-time_index[mon]] = BGF.calc_curl_sq_2d_xy_gradient(u_int[x-time_index[mon]], v_int[x-time_index[mon]], dx_res)
            #apy.plot_var_xy(m, xpts2m, ypts2m, u_int[x-time_index[mon]], v_int[x-time_index[mon]], curl_days[x-time_index[mon]], out=outpath+'/CURL/'+str(y+1980)+'-'+str(mon+1)+'-'+str(x-time_index[mon]), units_lab=r'm s$^{-1}$', units_vec=r'm s$^{-1}$',
            #    base_mask=1,res=2, minval=-5e-8, maxval=5e-8, scale_vec=0.5, vector_val=0.2, cbar_type='both', cmap_1=plt.cm.RdBu_r)

        curl_days_mask_count = ma.count_masked(curl_days, axis=0)
        days_in_month=15
        curl_months[y, mon] = ma.masked_where(curl_days_mask_count>days_in_month, ma.mean(curl_days, axis=0))
        u_month[y, mon]  = ma.masked_where(curl_days_mask_count>days_in_month, ma.mean(u_int, axis=0))
        v_month[y, mon]  = ma.masked_where(curl_days_mask_count>days_in_month, ma.mean(v_int, axis=0))
        
        BGF.plot_var_xy(m, xpts2m, ypts2m, u_month[y, mon], v_month[y, mon], curl_months[y, mon], out=figpath+'/CURL/'+str(y+1980)+'-'+str(mon+1), 
            units_lab=r'm s$^{-1}$', units_vec=r'm s$^{-1}$', base_mask=1,res=2, minval=-5e-8, maxval=5e-8, scale_vec=0.5, vector_val=0.2, cbar_type='both', cmap_1=plt.cm.RdBu_r)

#curl_months.dump(outpath+str(start_year)+'-'+str(end_year)+'-curl_data_months'+grid_str+'.txt')
#u_month.dump(outpath+str(start_year)+'-'+str(end_year)+'-u_drift_months'+grid_str+'.txt')
#v_month.dump(outpath+str(start_year)+'-'+str(end_year)+'-v_drift_months'+grid_str+'.txt')

plot=0

if plot==1:
    for x in xrange(num_years):
        apy.plot_var_xy(m, xpts2m, ypts2m, ma.mean(u_month[x], axis=0), ma.mean(v_month[x], axis=0), ma.mean(curl_months[x], axis=0), out=figpath+str(x+1980), 
            units_lab=r'm s$^{-1}$', units_vec=r'm s$^{-1}$', base_mask=1,res=2, minval=-5e-8, maxval=5e-8, scale_vec=0.5, vector_val=0.2, cbar_type='both', cmap_1=plt.cm.RdBu_r)


    for y in xrange(num_years):
        for mon in xrange(12):
            apy.plot_var_xy(m, xpts2m, ypts2m, u_month[y, mon], v_month[y, mon], curl_months[y, mon], out=figpath+str(y+1980)+'-'+str(mon+1), 
                units_lab=r'm s$^{-1}$', units_vec=r'm s$^{-1}$', base_mask=1,res=2, minval=-5e-8, maxval=5e-8, scale_vec=0.5, vector_val=0.2, cbar_type='both', cmap_1=plt.cm.RdBu_r)

    for x in xrange(num_years):
        apy.plot_var_xy(m, xpts2m, ypts2m, ma.mean(u_month[x], axis=0), ma.mean(v_month[x], axis=0), ma.mean(curl_months[x], axis=0), out=figpath+str(x+1980), 
            units_lab=r'm s$^{-1}$', units_vec=r'm s$^{-1}$', base_mask=1,res=2, minval=-5e-8, maxval=5e-8, scale_vec=0.5, vector_val=0.2, cbar_type='both', cmap_1=plt.cm.RdBu_r)


