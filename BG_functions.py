############################################################## 
# Date: 01/01/16
# Name: Alek Petty
# Description:  Functions/classes used by the BG_drift Python scripts

import matplotlib
matplotlib.use("AGG")

# basemap import
from mpl_toolkits.basemap import Basemap, shiftgrid
# Numpy import
import numpy as np
from pylab import *
from scipy.io import netcdf
import numpy.ma as ma
import string
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid.inset_locator import mark_inset
from mpl_toolkits.axes_grid.anchored_artists import AnchoredSizeBar
from scipy import stats
from math import log10, floor
from netCDF4 import Dataset
import os
from glob import glob
import time
from scipy.interpolate import griddata

def calc_uv(xvel, yvel, lons):
# Script to convert vectors from xy to uv
    u_z = ma.masked_all((xvel.shape[0],xvel.shape[1]))
    v_m = ma.masked_all((xvel.shape[0],xvel.shape[1]))
    mask = ma.getmask(xvel)
    #index = np.where(mask==False)

    for i in xrange(xvel.shape[0]):
        for j in xrange(xvel.shape[1]):
            #TO TRANSPOSE OR NOT?..
            alpha = (lons[i, j])*pi/180.
            if (mask[i, j]==False):
                u_z[i, j] = yvel[i, j]*sin(alpha) + xvel[i, j]*cos(alpha)
                v_m[i, j] = yvel[ i, j]*cos(alpha) - xvel[ i, j]*sin(alpha) 

    return u_z, v_m 

def calc_cersat_drift_UV(m, init_path, save_path, start_year, end_year, years, months, month_str, num_days_lag, plot=1):
# Script to read in CERSAT drift vectors to calculate a monthly drift estimate
    files = glob(init_path+'2007/200701*.nc')

    num_years = end_year - start_year +1
    file_path = files[0]

    f = Dataset(file_path, 'r')

    lon = f.variables['longitude'][:]
    lat = f.variables['latitude'][:]
    xpts, ypts = m(lon, lat)

    #xpts_t = loadtxt(datapath+'xpts_wind_trend'+grid_str+'.txt')
    #ypts_t = loadtxt(datapath+'ypts_wind_trend'+grid_str+'.txt')

    nx = xpts.shape[0]
    ny = xpts.shape[1]
    #months = [9, 10, 11, 12, 1, 2, 3, 4]

    num_months = size(months)

    drift_months = ma.masked_all((num_years, num_months, 2, nx, ny))
    drift_years = ma.masked_all((num_years, 2, nx, ny))

    str_num_days_lag='('+str(num_days_lag)+'day lag)'
    
    for y in xrange (num_years):
        print (y)
        year1 = str(y+start_year)
        year2 = str(y+start_year+years[-1])
        for mon in xrange(size(months)):
            month = '%02d' % (months[mon]+1)
            year =  str(start_year + y + years[mon])

            files = glob(init_path+year+'/'+year+month+'*.nc')
            num_days = size(files) #int(size(files)/int(num_days_lag)+1)


            print size(files), num_days_lag, num_days
            if size(files)>0:
                drift_days = ma.masked_all((num_days,2,  xpts.shape[0], xpts.shape[1]))
                #curl_days = ma.zeros((size(files),  xpts.shape[0], xpts.shape[1]))
                #DO WE WAANT TO PREVENT THE LAG CALCULATIONS FROM OVERLAPPING? IF NOT THEN JUST DO FOR ALL DAILY LAG FIELDS.
                i=0
                #for x in xrange(0, size(files), int(num_days_lag)):
                for x in xrange(0, size(files)):
                    print x
                    day = '%02d' % x 
                    f = Dataset(files[x], 'r')
                    u = f.variables['zonal_motion'][0]/(60.*60.*24.*num_days_lag)
                    v = f.variables['meridional_motion'][0]/(60.*60.*24.*num_days_lag)
                    #less than 0 are flags meaning the drift hasnt passed certain tests. 
                    #q = f.variables['quality_flag'][0]
                    #u = ma.masked_where((q<=0), u)
                    #v = ma.masked_where((q<=0), v)
                    #ROTATE VECOTRS TO X/Y GRID
 
                    drift_days[i, 0] = u #ma.masked_where(sqrt((drift_u**2) + (drift_v**2))<0.001, drift_u)
                    drift_days[i, 1] = v #ma.masked_where(sqrt((drift_u**2) + (drift_v**2))<0.001, drift_v)
                    
                    i+=1
                drift_mask_count = ma.count_masked(drift_days, axis=0)
                drift_months[y, mon, 0] = ma.mean(drift_days[:, 0], axis=0)
                drift_months[y, mon, 1] = ma.mean(drift_days[:, 1], axis=0)
                #MASK WHERE MORE THAN 15 DAYS HAVE NO DATA (I.E. AT LEAST 15 DAYS OF DATA IN THAT MONTH)
                #days_in_month=15
                #drift_months[y, mon] = ma.masked_where(drift_mask_count>days_in_month, drift_months[y, mon])

                #drift_days.dump(init_path+'OUTPUT/'+str(start_year)+'-'+year2+'-'+month_str+'-drift_data_daily.txt')
    
    drift_months.dump(save_path+'/'+str(start_year)+'-'+year2+'-'+month_str+'-drift_data_months_uv.txt')
    
def calc_cersat_drift_XY(m, init_path, save_path, start_year, end_year, years, months, month_str, num_days_lag, plot=1):
#Same as clac_cersat_drift_UV but rotates the vetors back on to a an xy direction, for plotting and comparison with Fowler drifts.
    files = glob(init_path+'2007/*.nc')

    plot==0

    num_years = end_year - start_year +1
    file_path = files[0]

    f = Dataset(file_path, 'r')

    lon = f.variables['longitude'][:]
    lat = f.variables['latitude'][:]
    xpts, ypts = m(lon, lat)

    #xpts_t = loadtxt(datapath+'xpts_wind_trend'+grid_str+'.txt')
    #ypts_t = loadtxt(datapath+'ypts_wind_trend'+grid_str+'.txt')

    nx = xpts.shape[0]
    ny = xpts.shape[1]
    #months = [9, 10, 11, 12, 1, 2, 3, 4]

    num_months = size(months)

    drift_months = ma.masked_all((num_years, num_months, 2, nx, ny))
    drift_years = ma.masked_all((num_years, 2, nx, ny))

    str_num_days_lag='('+str(num_days_lag)+'day lag)'
    
    for y in xrange (num_years):
        print (y)
        year1 = str(y+start_year)
        year2 = str(y+start_year+years[-1])
        for mon in xrange(size(months)):
            month = '%02d' % (months[mon]+1)
            year =  str(start_year + y + years[mon])

            files = glob(init_path+year+'/'+year+month+'*.nc')
            num_days = size(files) #int(size(files)/int(num_days_lag)+1)


            print size(files), num_days_lag, num_days
            if size(files)>0:
                drift_days = ma.masked_all((num_days,2,  xpts.shape[0], xpts.shape[1]))
                #curl_days = ma.zeros((size(files),  xpts.shape[0], xpts.shape[1]))
                #DO WE WAANT TO PREVENT THE LAG CALCULATIONS FROM OVERLAPPING? IF NOT THEN JUST DO FOR ALL DAILY LAG FIELDS.
                i=0
                #for x in xrange(0, size(files), int(num_days_lag)):
                for x in xrange(0, size(files)):
                    print x
                    day = '%02d' % x 
                    f = Dataset(files[x], 'r')
                    u = f.variables['zonal_motion'][0]/(60.*60.*24.*num_days_lag)
                    v = f.variables['meridional_motion'][0]/(60.*60.*24.*num_days_lag)
                    #less than 0 are flags meaning the drift hasnt passed certain tests. 
                    #q = f.variables['quality_flag'][0]
                    #u = ma.masked_where((q<=0), u)
                    #v = ma.masked_where((q<=0), v)

                    #ROTATE VECOTRS TO X/Y GRID
                    drift_u,drift_v = m.rotate_vector(u,v,lon,lat)
                    #drift_u = interp(drift_u, xpts, ypts, xpt_t, ypts_t)
                    #drift_v = interp(drift_u, xpts, ypts, xpt_t, ypts_t)

                    drift_days[i, 0] = drift_u #ma.masked_where(sqrt((drift_u**2) + (drift_v**2))<0.001, drift_u)
                    drift_days[i, 1] = drift_v #ma.masked_where(sqrt((drift_u**2) + (drift_v**2))<0.001, drift_v)
                    date = year+'-'+month+'-'+day
                    i+=1

                    #if (plot==1):
                    #    plot_var_xy(m, xpts , ypts, drift_days[x, 0], drift_days[x, 1], sqrt((drift_days[x, 0]**2) + (drift_days[x, 1]**2)), out=init_path+'/FIGS/days/'+date, units_lab=r'm s$^{-1}$', units_vec=r'm s$^{-1}$',
                    #        minval=0., maxval=0.2, base_mask=1,res=1, scale_vec=1, vector_val=0.2, year_string=year, month_string=month, cbar_type='max', cmap_1=plt.cm.jet)
                drift_mask_count = ma.count_masked(drift_days, axis=0)
                drift_months[y, mon, 0] = ma.mean(drift_days[:, 0], axis=0)
                drift_months[y, mon, 1] = ma.mean(drift_days[:, 1], axis=0)
                #MASK WHERE MORE THAN 15 DAYS HAVE NO DATA (I.E. AT LEAST 15 DAYS OF DATA IN THAT MONTH)
                days_in_month=15
                drift_months[y, mon] = ma.masked_where(drift_mask_count>days_in_month, drift_months[y, mon])

            #drift_months[y, mon, 0] = ma.masked_where(sqrt((drift_u_months_temp**2) + (drift_v_months_temp**2))<0.05, drift_u_months_temp)
            #drift_months[y, mon, 1] = ma.masked_where(sqrt((drift_u_months_temp**2) + (drift_v_months_temp**2))<0.05, drift_v_months_temp)

            if (plot==1):
                plot_var_xy(m, xpts , ypts, drift_months[y, mon, 0], drift_months[y, mon, 1], sqrt((drift_months[y, mon, 0]**2) + (drift_months[y, mon, 1]**2)), out=init_path+'/FIGS/months/'+year+'-'+month, units_lab=r'm s$^{-1}$', units_vec=r'm s$^{-1}$',
                    minval=0., maxval=0.3, base_mask=1,res=2, scale_vec=0.5, vector_val=0.2, year_string=year, month_string=month,extra=str_num_days_lag, cbar_type='max', cmap_1=plt.cm.YlOrRd)
        drift_years[y, 0] = ma.mean(drift_months[y, :, 0], axis=0)
        drift_years[y, 1] = ma.mean(drift_months[y, :, 1], axis=0)
        if (plot==1):
            plot_var_xy(m, xpts , ypts, drift_years[y, 0], drift_years[y, 1], sqrt((drift_years[y, 0]**2) + (drift_years[y, 1]**2)), out=init_path+'/FIGS/years/'+year1+'-'+year2+'-'+month_str, units_lab=r'm s$^{-1}$', units_vec=r'm s$^{-1}$',
                minval=0., maxval=0.2, base_mask=1,res=2, scale_vec=0.5, vector_val=0.2, year_string=year1+'-'+year2, month_string=month_str,extra=str_num_days_lag, cbar_type='max', cmap_1=plt.cm.YlOrRd)

    drift_months.dump(save_path+str(start_year)+'-'+year2+'-'+month_str+'-drift_data_months.txt')
    xpts.dump(save_path+'/xpts.txt')
    ypts.dump(save_path+'ypts.txt')

def plot_var_xy(m, xpts , ypts, var_x, var_y, var_mag, out='./figure', units_lab='units', units_vec=r'm s$^{-1}$',
 minval=1., maxval=1., base_mask=1,res=1, scale_vec=1, vector_val=1, year_string='year', month_string='months', extra='',cbar_type='both', cmap_1=plt.cm.RdBu_r):

        #PLOT SCALAR FIELD WITH OVERLYING VECTORS. 
        #VAR MAG MAY NOT NECCESARRILY BE THE MAGNITUDE OF THE VECTORS (E.G. IN THE CASE OF WIND CURL)

        fig = figure(figsize=(4.25,5))
        ax1 = fig.add_axes([0.0, 0.15, 1.0, 0.85])
        #if etopo==1:
         #       im_etopo = m.pcolormesh(xpts_etopo, ypts_etopo , etopo_var, cmap=plt.cm.Greens_r, vmin=0, vmax=1000, zorder=1)
        if (maxval-minval)<1e-10:
            minval = -round_to_1(np.amax(var_mag))
            maxval = round_to_1(np.amax(var_mag))

        #var_mag=ma.masked_where(var_mag<1e8, var_mag)
        im1 = m.pcolormesh(xpts , ypts, var_mag, cmap=cmap_1,vmin=minval, vmax=maxval,shading='flat', zorder=4)
        # LOWER THE SCALE THE LARGER THE ARROW
        Q = m.quiver(xpts[::res, ::res], ypts[::res, ::res], var_x[::res, ::res], var_y[::res, ::res], units='inches',scale=scale_vec, zorder=5)
        #ASSIGN A LEGEND OF THE VECTOR 
        #m.plot(xpts[191, 100], ypts[191, 100], 'x', zorder=10)
        m.drawparallels(np.arange(90,-90,-10), linewidth = 0.25, zorder=10)
        m.drawmeridians(np.arange(-180.,180.,30.), linewidth = 0.25, zorder=10)
        #m.drawmapboundary(fill_color='0.3')
        #m.drawmapboundary(fill_color='0.4' , zorder=1)
        if base_mask==1:
        #m.drawmapboundary(fill_color='0.4' , zorder=1)
            m.fillcontinents(color='grey',lake_color='grey', zorder=6)
        m.drawcoastlines(linewidth=0.5)

        cax = fig.add_axes([0.1, 0.1, 0.8, 0.04])
        cbar = colorbar(im1,cax=cax, orientation='horizontal', extend=cbar_type, use_gridspec=True)
        cbar.set_label(units_lab)
        xticks = np.linspace(minval, maxval, 5)
        cbar.set_ticks(xticks)
        cbar.formatter.set_powerlimits((-3, 4))
        cbar.formatter.set_scientific(True)
        cbar.update_ticks() 

        xS, yS = m(120, 55)
        qk = quiverkey(Q, xS, yS, vector_val, str(vector_val)+' '+units_vec, fontproperties={'size': 'medium'}, coordinates='data', zorder = 11)   
        
        xS, yS = m(235, 48)
        ax1.text(xS, yS, year_string+'\n'+month_string+'\n'+extra,fontsize=12, zorder = 11)

        subplots_adjust(bottom=0.0, left=0.0, top = 1.0, right=1.0)

        savefig(out+'.png', dpi=300)
        close(fig)
#plot vector map (with vectors in x/y directions)

def interp_uv(ut, vt, q, xpts, ypts, xpts2m, ypts2m):

    #TURN UT/VT INTO MASKED ARRAYS
    #IF Q NEGATIVE THEN REMOVE AS IT IS BY COAST
    #IF Q GREATER THAN 0.5 THEN REMOVE AS THIS MEANS NO POINTS NEARBY IN MAKING THE VECTOR
    #IF 0 THEN MASK AS NO DATA BUT THINK THAT"S ALREADY THE CASE

    #negative q is coastal, 0 is no data, greater than 1 means no data closeby
    #create mask where 0 is mask and 1 is data
    mask = where((q<=0) | (q>1), 0, 1)

    ut = ma.masked_where(mask<0.5, ut)
    vt = ma.masked_where(mask<0.5, vt)

    u_int=interp_data(ut, xpts, ypts, xpts2m, ypts2m)
    v_int=interp_data(vt, xpts, ypts, xpts2m, ypts2m)

    mask_int=interp_data_nomask(mask, xpts, ypts, xpts2m, ypts2m)
    #DO 0.5 NOT 1 AS THIS MEANS MIGHT BE INFECTED WITH OTHER DATA ON THE INTERP GRID?
    u_int = ma.masked_where(mask_int<0.25, u_int)
    v_int = ma.masked_where(mask_int<0.25, v_int)


    return u_int, v_int
#Script to interpolate vectors onto a new grid
#needs interp_data and interp_data_nomask

def interp_uvCSAT(ut, vt, q, xpts, ypts, xpts2m, ypts2m):

    q = q.filled(0)
    mask = where(q==0, 0, 1)

    u_int=interp_data(ut, xpts, ypts, xpts2m, ypts2m)
    v_int=interp_data(vt, xpts, ypts, xpts2m, ypts2m)

    #q_ma = ma.masked_where(q<0, q)
    mask_int=interp_data_nomask(mask, xpts, ypts, xpts2m, ypts2m)

    #LESS THAN A QUARTER OF DATA
    u_int = ma.masked_where(mask_int<0.25, u_int)
    v_int = ma.masked_where(mask_int<0.25, v_int)

    return u_int, v_int, mask_int
#Script to interpolate vectors onto a new grid for the CSAT drift vectors
#needs interp_data and interp_data_nomask

def interp_data(var, xpts, ypts, xpts2m, ypts2m, int_meth='cubic'):
#interpoalte data onto a regular 2d grid. Used by interp_uv
    data = ma.getdata(var)
    mask = ma.getmask(var)
    index = np.where(mask==False)
    data_masked = data[index]
    xpoints = xpts[index]
    ypoints = ypts[index]
    points = (xpoints, ypoints)
    var_int = griddata(points,data_masked,(xpts2m,ypts2m),method=int_meth)
    var_int = ma.masked_array(var_int,np.isnan(var_int))

    return var_int

def interp_data_nomask(var, xpts, ypts, xpts2m, ypts2m, int_meth='cubic'):
#interpoalte mask of data onto a regular 2d grid. Used by interp_uv
    index = np.where(var>-1)
    xpoints = xpts[index]
    ypoints = ypts[index]
    points = (xpoints, ypoints)
    data_ma = var[index]
    #tck = interpolate.bisplrep(points, data_masked, , s=0)
    #var_int = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)
    var_int = griddata(points,data_ma,(xpts2m,ypts2m),method=int_meth)
    return var_int

def calc_curl_sq_2d_xy_gradient(x_vel, y_vel, dx_res):
# calculate the curl (squared) of a field given some input vector grid
    #CALCULATE THE CURL OF AN X/Y VEXTOR FIELD. DX_RES IF THE GRID RESOLUTION OF THIS REGULARLY SPACED GRID.
    #MULTIPLY BY MAGNITUDE TO GET THIS IN SQUARED UNITS ANALAGOUS TO THE WIND STRESS CURL.
    #USE GRADIENT FUNCTION WHICH USES CENTRAL DIFFERENCES IN MIDDLE CELLS AND FIRST DIFFERENCES AT THE BOUNDARIES (GIVES SAME SHAPE AS INPUT)

    mag = sqrt((x_vel**2) + (y_vel**2))

    x_vel_mag = x_vel*mag
    y_vel_mag = y_vel*mag

    #gradient [0] returns row divs (y direction) then [1] gives column divs (x direction)
    dvelydx = np.gradient(y_vel_mag, dx_res)[1]
    dvelxdy = np.gradient(x_vel_mag, dx_res)[0]

    zeta = dvelydx - dvelxdy
    #MASK ARRAY WHERE VALUES ARE NAN
    zeta = ma.array(zeta,mask=np.isnan(zeta))

    return zeta

def plot_drift(drift1, zi, x, y, out, xpts2, ypts2):
    fig = figure(figsize=(4.25,5))
    ax1 = fig.add_axes([0.0, 0.15, 1.0, 0.85])
    #if etopo==1:
     #       im_etopo = m.pcolormesh(xpts_etopo, ypts_etopo , etopo_var, cmap=plt.cm.Greens_r, vmin=0, vmax=1000, zorder=1)
    minval = -0.2
    maxval = 0.1

    im1 = m.pcolormesh(xpts , ypts, drift1, cmap=cm.jet,shading='flat', vmin=minval, vmax=maxval, zorder=4)
    im2 = m.scatter(xpts2 , ypts2, c=zi, edgecolor='none', vmin=minval, vmax=maxval, zorder=5)
    #m2 = m.hexbin(xpts2 , ypts2, c=zi, edgecolor='k', vmin=minval, vmax=maxval, zorder=5)


    #bar = colorbar(im1,orientation='horizontal', extend='both')
    m.drawparallels(np.arange(90,-90,-10), linewidth = 0.25, zorder=10)
    m.drawmeridians(np.arange(-180.,180.,30.), linewidth = 0.25, zorder=10)
    #m.drawmapboundary(fill_color='0.3')
    base_mask=1
    if base_mask==1:
    #m.drawmapboundary(fill_color='0.4' , zorder=1)
        m.fillcontinents(color='grey',lake_color='grey', zorder=5)
        m.drawcoastlines(linewidth=0.5)

    cax = fig.add_axes([0.1, 0.1, 0.8, 0.04])
    cbar = colorbar(im1,cax=cax, orientation='horizontal', extend='both', use_gridspec=True)
    cbar.set_label('drift')
    xticks = np.linspace(minval, maxval, 5)
    cbar.set_ticks(xticks)


    subplots_adjust(bottom=0.0, left=0.0, top = 1.0, right=1.0)

    month = y+10
    if (month>12):
        month=month-12
        x=x+1

    savefig(init_path+'/AREA_FLUX/area_flux'+str(2008+x)+'_'+str(month)+out+'.png', dpi=300)
    close(fig)
#plot drift vector and interpolated scatter of drift, to check the flux gate interpoaltion worked

def plot_gates_map(lon1, lon2, lon3, lat1, lat2, lat3):
#plot a map of flux gates used in flux gate analysis
    fig = figure(figsize=(4,4))

    gate1_lons = np.linspace(lon1, lon2, 10)
    gate1_lats = np.linspace(lat3, lat3, 10)

    gate2_lons = np.linspace(lon2, lon3, 10)
    gate2_lats = np.linspace(lat3, lat3, 10)

    gate3_lons = np.linspace(lon3, lon3, 10)
    gate3_lats = np.linspace(lat1, lat2, 10)

    gate4_lons = np.linspace(lon3, lon3, 10)
    gate4_lats = np.linspace(lat2, lat3, 10)

    xpts_g1, ypts_g1 = m(gate1_lons, gate1_lats)
    xpts_g2, ypts_g2 = m(gate2_lons, gate2_lats)
    xpts_g3, ypts_g3 = m(gate3_lons, gate3_lats)
    xpts_g4, ypts_g4 = m(gate4_lons, gate4_lats)

    m.plot(xpts_g1, ypts_g1, '-', color='r')
    m.plot(xpts_g2, ypts_g2, '-', color='b')
    m.plot(xpts_g3, ypts_g3, '-', color='g')
    m.plot(xpts_g4, ypts_g4, '-', color='m')

    m.drawparallels(np.arange(90,-90,-10), linewidth = 0.25, zorder=10)
    m.drawmeridians(np.arange(-180.,180.,30.), linewidth = 0.25, zorder=10)

    m.fillcontinents(color='grey',lake_color='grey', zorder=5)
    m.drawcoastlines(linewidth=0.5)
    savefig(outpath+'/area_flux_gates_quads_map.pdf', dpi=300)
    close(fig)

def return_xpts_ypts_fowler(path, m):
    lon_lat = loadtxt(path+'north_x_y_lat_lon.txt')
    lons = np.zeros((361, 361))
    lats = np.zeros((361, 361))
    lons = np.reshape(lon_lat[:, 3], (361, 361))
    lats = np.reshape(lon_lat[:, 2], (361, 361))
    xpts, ypts = m(lons, lats)
    return xpts, ypts

def return_xpts_ypts(path, m):
#return cersat xpts/ypts on given projection
    files = glob(path+'2008/*.nc')
    f = Dataset(files[0], 'r')
    lon = f.variables['longitude'][:]
    lat = f.variables['latitude'][:]
    xpts, ypts = m(lon, lat)

    return xpts, ypts

def calc_era_wind_curl_NEW(m, start_year, end_year, years, months, months_str, date_str, grid_str, box_str, dx_res, arr_res, nx, ny, figpath, datapath, outpath, plot=0):
# ERA WIND CURL CALC
    num_years = end_year-start_year+1 #INITIAL YEAR
    years_str= str(start_year)+'-'+str(end_year)

    time_index = [0, 31*4, 59*4, 90*4, 120*4, 151*4, 181*4, 212*4, 243*4, 273*4, 304*4, 334*4, 365*4]
    #time_index = [0, 31*4, 59*4, 90*4, 120*4, 151*4, 181*4, 212*4, 243*4, 263*4, 294*4, 334*4, 365*4]

    ave_x_vel_years = ma.masked_all((num_years, nx, ny))
    ave_y_vel_years = ma.masked_all((num_years, nx, ny))
    ave_wind_curl_years = ma.masked_all((num_years, nx, ny))
    wind_curl_month_ave = ma.masked_all((num_years, size(months), nx, ny))
    x_vel_month_ave = ma.masked_all((num_years, size(months), nx, ny))
    y_vel_month_ave = ma.masked_all((num_years, size(months), nx, ny))

    #CALC LONS/LATS ETC ONCE
    filepath = datapath+'/ERAI_WINDS_6HOURLY_'+str(2000)+'.nc'
    read_winds_obj = read_era_vector(filepath, 'u10', 'v10')
    xpts, ypts, lons1d, lons2d, lats1d, lats2d = read_winds_obj.x_y_lons_lats_era(m)
    u10_temp, v10_temp = read_winds_obj.vars(0)
    x_vel_temp, y_vel_temp, xpts_t, ypts_t = m.transform_vector(u10_temp,v10_temp, lons1d,lats1d,nx,ny,returnxy=True)

    maskpath = datapath+'/ERA_MASK.nc'

    read_mask_obj = read_era_1var(maskpath, 'lsm')
    lsm = read_mask_obj.var()
    lsm_t = m.transform_scalar(lsm, lons1d,lats1d,nx,ny,returnxy=False)
    lons2d_t = m.transform_scalar(lons2d, lons1d,lats1d,nx,ny,returnxy=False)
    lats2d_t = m.transform_scalar(lats2d, lons1d,lats1d,nx,ny,returnxy=False)
    savetxt(outpath+'lsm_t'+grid_str+'.txt', lsm_t)
    savetxt(outpath+'lons2d_t'+grid_str+'.txt', lons2d_t)
    savetxt(outpath+'lats2d_t'+grid_str+'.txt', lats2d_t)
    savetxt(outpath+'xpts_wind_trend'+grid_str+'.txt', xpts_t)
    savetxt(outpath+'ypts_wind_trend'+grid_str+'.txt', ypts_t)

    for x in xrange(num_years):
        print 'Year:', x
        #x_vel_sum = np.zeros((nx, ny))
        #y_vel_sum = np.zeros((nx, ny))
        #wind_curl_sum = np.zeros((nx, ny))
        #num_periods=1
        year1 = str(x+start_year)
        year2 = str(x+start_year+years[-1])
        for i in xrange(size(months)):
            print i

            #ADD YEAR IF STRADDLE OVER TWO YEARS
            year =  start_year + x + years[i]
            filepath = datapath+'/ERAI_WINDS_6HOURLY_'+str(year)+'.nc'
            read_winds_obj = read_era_vector(filepath, 'u10', 'v10')

            wind_curl_month = ma.masked_all((time_index[months[i]+1] - time_index[months[i]], nx, ny))
            x_vel_month = ma.masked_all((time_index[months[i]+1] - time_index[months[i]], nx, ny))
            y_vel_month = ma.masked_all((time_index[months[i]+1] - time_index[months[i]], nx, ny))
            #start = time.time()
            #print start

            for j in xrange(time_index[months[i]], time_index[months[i]+1], 1):

                u10_temp, v10_temp = read_winds_obj.vars(j)
                x_vel_month[j-time_index[months[i]]], y_vel_month[j-time_index[months[i]]] = m.transform_vector(u10_temp,v10_temp, lons1d,lats1d,nx,ny,returnxy=False)
                wind_curl_month[j-time_index[months[i]]] = calc_curl_sq_2d_xy_gradient(x_vel_month[j-time_index[months[i]]], y_vel_month[j-time_index[months[i]]], dx_res)

                #plot_var_xy(m, xpts_t, ypts_t,x_vel_month[j-time_index[months[i]]], y_vel_month[j-time_index[months[i]]], wind_curl_month[j-time_index[months[i]]],out=figpath+'wind_curl_'+grid_str+str(j)+'new',\
            #units_lab=r'm s$^{-2}$',minval=-2e-4, maxval=2e-4, vector_val=10, base_mask=1, res=arr_res, scale_vec=50, year_string=year1+'-'+year2, month_string=str(j))
            #end = time.time()
            #print end - start

            wind_curl_month_ave[x, i]  = ma.mean(wind_curl_month, axis=0)
            x_vel_month_ave[x, i]  = ma.mean(x_vel_month, axis=0)
            y_vel_month_ave[x, i]  = ma.mean(y_vel_month, axis=0)

        ave_wind_curl_years[x] = ma.mean(wind_curl_month_ave[x], axis=0)
        ave_x_vel_years[x] = ma.mean(x_vel_month_ave[x], axis=0)
        ave_y_vel_years[x] = ma.mean(y_vel_month_ave[x], axis=0)
        
        date = year1+'-'+year2+months_str

    #REMOVE BOX_STR AS NOT ACT CALCULATING WIHTIN THAT REGION
    wind_curl_month_ave.dump(outpath+date_str+'/wind_curl_month_ave'+date_str+box_str+'.txt')
    x_vel_month_ave.dump(outpath+date_str+'/x_vel_month_ave'+date_str+box_str+'.txt')
    y_vel_month_ave.dump(outpath+date_str+'/y_vel_month_ave'+date_str+box_str+'.txt')

    ave_wind_curl_years.dump(outpath+date_str+'/ave_wind_curl_years'+date_str+box_str+'.txt')
    ave_x_vel_years.dump(outpath+date_str+'/ave_xvel_years'+date_str+box_str+'.txt')
    ave_y_vel_years.dump(outpath+date_str+'/ave_yvel_years'+date_str+box_str+'.txt')

def calc_ncep_wind_curl_NEW(m, start_year, end_year, years, months, months_str, date_str, grid_str, dx_res, arr_res, nx, ny, figpath, datapath, plot=0):
# READ IN REANLYSIS 1 WIND DATA (995 SIGMA) AND CALCULATE THE CURL
    num_years = end_year-start_year+1 #INITIAL YEAR
    years_str= str(start_year)+'-'+str(end_year)
    
    time_index = [0, 31*4, 59*4, 90*4, 120*4, 151*4, 181*4, 212*4, 243*4, 273*4, 304*4, 334*4, 365*4]
    #time_index = [0, 31*4, 59*4, 90*4, 120*4, 151*4, 181*4, 212*4, 243*4, 263*4, 294*4, 334*4, 365*4]

    ave_x_vel_years = np.zeros((num_years, nx, ny))
    ave_y_vel_years = np.zeros((num_years, nx, ny))
    ave_wind_curl_years = np.zeros((num_years, nx, ny))

    #CALC LONS/LATS ETC ONCE
    read_winds_obj = read_ncep_vector(2000)
    xpts, ypts, lons1d, lons2d, lats1d, lats2d = read_winds_obj.x_y_lons_lats(m)
    u10_temp, v10_temp = read_winds_obj.vars(0)
    x_vel_temp, y_vel_temp, xpts_t, ypts_t = m.transform_vector(u10_temp,v10_temp, lons1d,lats1d,nx,ny,returnxy=True)
    lsm = read_ncep_mask()
    lsm_t = m.transform_scalar(lsm, lons1d,lats1d,nx,ny,returnxy=False)
    lons2d_t = m.transform_scalar(lons2d, lons1d,lats1d,nx,ny,returnxy=False)
    lats2d_t = m.transform_scalar(lats2d, lons1d,lats1d,nx,ny,returnxy=False)
    savetxt(datapath+'lsm_t'+grid_str+'.txt', lsm_t)
    savetxt(datapath+'lons2d_t'+grid_str+'.txt', lons2d_t)
    savetxt(datapath+'lats2d_t'+grid_str+'.txt', lats2d_t)
    savetxt(datapath+'xpts_wind_trend'+grid_str+'.txt', xpts_t)
    savetxt(datapath+'ypts_wind_trend'+grid_str+'.txt', ypts_t)

    for x in xrange(num_years):
        print 'Year:', x
        x_vel_sum = np.zeros((nx, ny))
        y_vel_sum = np.zeros((nx, ny))
        wind_curl_sum = np.zeros((nx, ny))
        num_periods=0
        year1 = str(x+start_year)
        year2 = str(x+start_year+years[-1])
        for i in xrange(size(months)):
            #ADD YEAR IF STRADDLE OVER TWO YEARS
            year =  start_year + x + years[i]
            read_winds_obj = read_ncep_vector(year)

            for j in xrange(time_index[months[i]-1], time_index[months[i]], 1):
                num_periods+=1
            #print 'Time period:', i

                u10_temp, v10_temp = read_winds_obj.vars(j)
                x_vel_temp, y_vel_temp = m.transform_vector(u10_temp,v10_temp, lons1d,lats1d,nx,ny,returnxy=False)
                wind_curl_temp = calc_curl_sq_2d_xy_gradient(x_vel_temp, y_vel_temp, dx_res)

                x_vel_sum = x_vel_sum + x_vel_temp
                y_vel_sum = y_vel_sum + y_vel_temp  
                wind_curl_sum = wind_curl_sum + wind_curl_temp

        ave_x_vel_years[x]= ma.masked_where(x_vel_sum==0, x_vel_sum)/num_periods
        ave_y_vel_years[x]= ma.masked_where(y_vel_sum==0, y_vel_sum)/num_periods
        ave_wind_curl_years[x] = ma.masked_where(wind_curl_sum==0, wind_curl_sum)/num_periods

        #ave_x_vel_years[x]= ma.masked_where(x_vel_sum==0, x_vel_sum)/num_periods
        date = year1+'-'+year2+months_str

    ave_wind_curl_years.dump(datapath+date_str+'/ave_wind_curl_years'+date_str+'.txt')
    ave_x_vel_years.dump(datapath+date_str+'/ave_xvel_years'+date_str+'.txt')
    ave_y_vel_years.dump(datapath+date_str+'/ave_yvel_years'+date_str+'.txt')
    #xpts_t.dump('/Users/apetty/NOAA/DATA/WINDS/PCA/xpts'+grid_str+'.txt')
    #ypts_t.dump('/Users/apetty/NOAA/DATA/WINDS/PCA/ypts'+grid_str+'.txt')

def calc_ncep2_wind_curl_NEW(m, start_year, end_year, years, months, months_str, date_str, grid_str, box_str, dx_res, arr_res, nx, ny, figpath, datapath, outpath, plot=0):
# READ IN REANLYSIS 2 WIND DATA AND CALCULATE THE CURL
    num_years = end_year-start_year+1 #INITIAL YEAR
    years_str= str(start_year)+'-'+str(end_year)
    
    time_index = [0, 31*4, 59*4, 90*4, 120*4, 151*4, 181*4, 212*4, 243*4, 273*4, 304*4, 334*4, 365*4]
    #time_index = [0, 31*4, 59*4, 90*4, 120*4, 151*4, 181*4, 212*4, 243*4, 263*4, 294*4, 334*4, 365*4]

    ave_x_vel_years = ma.masked_all((num_years, nx, ny))
    ave_y_vel_years = ma.masked_all((num_years, nx, ny))
    ave_wind_curl_years = ma.masked_all((num_years, nx, ny))
    wind_curl_month_ave = ma.masked_all((num_years, size(months), nx, ny))
    x_vel_month_ave = ma.masked_all((num_years, size(months), nx, ny))
    y_vel_month_ave = ma.masked_all((num_years, size(months), nx, ny))

    #CALC LONS/LATS ETC ONCE FOR A RANDOM YEAR
    filepath = datapath+'/REANAL2'
    read_winds_obj = read_ncep2_vector(2000, filepath)
    xpts, ypts, lons1d, lons2d, lats1d, lats2d = read_winds_obj.x_y_lons_lats(m)
    u10_temp, v10_temp = read_winds_obj.vars(0)
    x_vel_temp, y_vel_temp, xpts_t, ypts_t = m.transform_vector(u10_temp,v10_temp, lons1d,lats1d,nx,ny,returnxy=True)
    lsm = read_ncep2_mask(filepath)
    lsm_t = m.transform_scalar(lsm, lons1d,lats1d,nx,ny,returnxy=False)
    lons2d_t = m.transform_scalar(lons2d, lons1d,lats1d,nx,ny,returnxy=False)
    lats2d_t = m.transform_scalar(lats2d, lons1d,lats1d,nx,ny,returnxy=False)
    savetxt(outpath+'lsm_t'+grid_str+'.txt', lsm_t)
    savetxt(outpath+'lons2d_t'+grid_str+'.txt', lons2d_t)
    savetxt(outpath+'lats2d_t'+grid_str+'.txt', lats2d_t)
    savetxt(outpath+'xpts_wind_trend'+grid_str+'.txt', xpts_t)
    savetxt(outpath+'ypts_wind_trend'+grid_str+'.txt', ypts_t)

    for x in xrange(num_years):

        year1 = str(x+start_year)
        year2 = str(x+start_year+years[-1])
        for i in xrange(size(months)):
            #print i
            #ADD YEAR IF STRADDLE OVER TWO YEARS
            year =  start_year + x + years[i]
            print year
            filepath = datapath+'/REANAL2'
            read_winds_obj = read_ncep2_vector(year, filepath)

            wind_curl_month = ma.masked_all((time_index[months[i]+1] - time_index[months[i]], nx, ny))
            x_vel_month = ma.masked_all((time_index[months[i]+1] - time_index[months[i]], nx, ny))
            y_vel_month = ma.masked_all((time_index[months[i]+1] - time_index[months[i]], nx, ny))

            for j in xrange(time_index[months[i]], time_index[months[i]+1], 1):

                u10_temp, v10_temp = read_winds_obj.vars(j)
                x_vel_month[j-time_index[months[i]]], y_vel_month[j-time_index[months[i]]] = m.transform_vector(u10_temp,v10_temp, lons1d,lats1d,nx,ny,returnxy=False)
                wind_curl_month[j-time_index[months[i]]] = calc_curl_sq_2d_xy_gradient(x_vel_month[j-time_index[months[i]]], y_vel_month[j-time_index[months[i]]], dx_res)

            wind_curl_month_ave[x, i]  = ma.mean(wind_curl_month, axis=0)
            x_vel_month_ave[x, i]  = ma.mean(x_vel_month, axis=0)
            y_vel_month_ave[x, i]  = ma.mean(y_vel_month, axis=0)

        ave_wind_curl_years[x] = ma.mean(wind_curl_month_ave[x], axis=0)
        ave_x_vel_years[x] = ma.mean(x_vel_month_ave[x], axis=0)
        ave_y_vel_years[x] = ma.mean(y_vel_month_ave[x], axis=0)
        
        #ave_x_vel_years[x]= ma.masked_where(x_vel_sum==0, x_vel_sum)/num_periods
        date = year1+'-'+year2+months_str

    wind_curl_month_ave.dump(outpath+date_str+'/wind_curl_month_ave'+date_str+box_str+'.txt')
    x_vel_month_ave.dump(outpath+date_str+'/x_vel_month_ave'+date_str+box_str+'.txt')
    y_vel_month_ave.dump(outpath+date_str+'/y_vel_month_ave'+date_str+box_str+'.txt')

    ave_wind_curl_years.dump(outpath+date_str+'/ave_wind_curl_years'+date_str+box_str+'.txt')
    ave_x_vel_years.dump(outpath+date_str+'/ave_xvel_years'+date_str+box_str+'.txt')
    ave_y_vel_years.dump(outpath+date_str+'/ave_yvel_years'+date_str+box_str+'.txt')
    #xpts_t.dump('/Users/apetty/NOAA/DATA/WINDS/PCA/xpts'+grid_str+'.txt')
    #ypts_t.dump('/Users/apetty/NOAA/DATA/WINDS/PCA/ypts'+grid_str+'.txt')

def calc_jra_wind_curl_NEW(m, start_year, end_year, years, months, months_str, date_str, grid_str, box_str, dx_res, arr_res, nx, ny, figpath, datapath, outpath, plot=0):
# READ IN REANLYSIS 2 WIND DATA AND CALCULATE THE CURL
    num_years = end_year-start_year+1 #INITIAL YEAR
    years_str= str(start_year)+'-'+str(end_year)

    time_index = [0, 31*4, 59*4, 90*4, 120*4, 151*4, 181*4, 212*4, 243*4, 273*4, 304*4, 334*4, 365*4]
    #time_index = [0, 31*4, 59*4, 90*4, 120*4, 151*4, 181*4, 212*4, 243*4, 263*4, 294*4, 334*4, 365*4]

    ave_x_vel_years = ma.masked_all((num_years, nx, ny))
    ave_y_vel_years = ma.masked_all((num_years, nx, ny))
    ave_wind_curl_years = ma.masked_all((num_years, nx, ny))
    wind_curl_month_ave = ma.masked_all((num_years, size(months), nx, ny))
    x_vel_month_ave = ma.masked_all((num_years, size(months), nx, ny))
    y_vel_month_ave = ma.masked_all((num_years, size(months), nx, ny))

    #CALC LONS/LATS ETC ONCE
    read_winds_obj = read_jra_vector(2000, datapath)
    xpts, ypts, lons1d, lons2d, lats1d, lats2d = read_winds_obj.x_y_lons_lats(m)
    u10_temp, v10_temp = read_winds_obj.vars(0)
    x_vel_temp, y_vel_temp, xpts_t, ypts_t = m.transform_vector(u10_temp,v10_temp, lons1d,lats1d,nx,ny,returnxy=True)
    #lsm = read_era_mask()
    #lsm_t = m.transform_scalar(lsm, lons1d,lats1d,nx,ny,returnxy=False)
    lons2d_t = m.transform_scalar(lons2d, lons1d,lats1d,nx,ny,returnxy=False)
    lats2d_t = m.transform_scalar(lats2d, lons1d,lats1d,nx,ny,returnxy=False)
    #savetxt(datapath+'lsm_t'+grid_str+'.txt', lsm_t)
    savetxt(outpath+'lons2d_t'+grid_str+'.txt', lons2d_t)
    savetxt(outpath+'lats2d_t'+grid_str+'.txt', lats2d_t)
    savetxt(outpath+'xpts_wind_trend'+grid_str+'.txt', xpts_t)
    savetxt(outpath+'ypts_wind_trend'+grid_str+'.txt', ypts_t)

    
    for x in xrange(num_years):
        #print 'Year:', x

        year1 = str(x+start_year)
        year2 = str(x+start_year+years[-1])
        for i in xrange(size(months)):
            #print i
            #ADD YEAR IF STRADDLE OVER TWO YEARS
            year =  start_year + x + years[i]
            read_winds_obj = read_jra_vector(year, datapath)

            wind_curl_month = ma.masked_all((time_index[months[i]+1] - time_index[months[i]], nx, ny))
            x_vel_month = ma.masked_all((time_index[months[i]+1] - time_index[months[i]], nx, ny))
            y_vel_month = ma.masked_all((time_index[months[i]+1] - time_index[months[i]], nx, ny))

            for j in xrange(time_index[months[i]], time_index[months[i]+1], 1):

                u10_temp, v10_temp = read_winds_obj.vars(j)
                x_vel_month[j-time_index[months[i]]], y_vel_month[j-time_index[months[i]]] = m.transform_vector(u10_temp,v10_temp, lons1d,lats1d,nx,ny,returnxy=False)
                wind_curl_month[j-time_index[months[i]]] = calc_curl_sq_2d_xy_gradient(x_vel_month[j-time_index[months[i]]], y_vel_month[j-time_index[months[i]]], dx_res)


            wind_curl_month_ave[x, i]  = ma.mean(wind_curl_month, axis=0)
            x_vel_month_ave[x, i]  = ma.mean(x_vel_month, axis=0)
            y_vel_month_ave[x, i]  = ma.mean(y_vel_month, axis=0)

        ave_wind_curl_years[x] = ma.mean(wind_curl_month_ave[x], axis=0)
        ave_x_vel_years[x] = ma.mean(x_vel_month_ave[x], axis=0)
        ave_y_vel_years[x] = ma.mean(y_vel_month_ave[x], axis=0)
        
        date = year1+'-'+year2+months_str

    wind_curl_month_ave.dump(outpath+date_str+'/wind_curl_month_ave'+date_str+box_str+'.txt')
    x_vel_month_ave.dump(outpath+date_str+'/x_vel_month_ave'+date_str+box_str+'.txt')
    y_vel_month_ave.dump(outpath+date_str+'/y_vel_month_ave'+date_str+box_str+'.txt')

    ave_wind_curl_years.dump(outpath+date_str+'/ave_wind_curl_years'+date_str+box_str+'.txt')
    ave_x_vel_years.dump(outpath+date_str+'/ave_xvel_years'+date_str+box_str+'.txt')
    ave_y_vel_years.dump(outpath+date_str+'/ave_yvel_years'+date_str+box_str+'.txt')

class read_era_vector:
    #READ IN U/V VECTOR COMPONENETS FORM ERA-I
    #SHIFT LATS FROM SOUTH TO NORTH
    #SHIFT LONS FROM 0-360 to -180-180 DEG.

    def __init__(self, filepath, varname1, varname2):
        
        self.f = Dataset(filepath, 'r')
        self.varname1 = varname1
        self.varname2 = varname2
        
        self.lats = self.f.variables['latitude'][::-1]
        self.lons0 = self.f.variables['longitude'][:] 
        var_array_t = self.f.variables[varname1][0, ::-1, ::]

        var_temp, self.lons_s = shiftgrid(180.,var_array_t,self.lons0,start=False)
        

    def vars(self, index):

        var_array1 = self.f.variables[self.varname1][index, ::-1, ::]
        var_array2 = self.f.variables[self.varname2][index, ::-1, ::]
        var_temp1, lons_temp = shiftgrid(180.,var_array1,self.lons0,start=False)
        var_temp2, lons_temp = shiftgrid(180.,var_array2,self.lons0,start=False)

        return var_temp1, var_temp2

    def x_y_lons_lats_era(self, m):

        lons1d = self.lons_s
        lats1d = self.lats
        
        lons2d, lats2d = np.meshgrid(self.lons_s, self.lats)
        xpts, ypts = m(lons2d, lats2d)
        return xpts, ypts, lons1d, lons2d, lats1d, lats2d

class read_era_1var:
    #CLASS TO READ IN ERA-I VARIABLE
    def __init__(self, filepath, varname):
        
        self.f = Dataset(filepath, 'r')
        self.varname = varname
        self.var_array = self.f.variables[varname][0,::-1, ::]
        self.lats = self.f.variables['latitude'][::-1]
        self.lons0 = self.f.variables['longitude'][:] 
        var_temp, self.lons_s = shiftgrid(180.,self.var_array,self.lons0,start=False)
        

    def var(self):
        #SHIFT GRID FROM 0-360 to -180-180 DEG.

        var_temp, lons_temp = shiftgrid(180.,self.var_array,self.lons0,start=False)

        return var_temp

    def x_y_lons_lats_era(self, m):

        lons1d = self.lons_s
        lats1d = self.lats
        
        lons2d, lats2d = np.meshgrid(self.lons_s, self.lats)
        xpts, ypts = m(lons2d, lats2d)

        return xpts, ypts, lons1d, lons2d, lats1d, lats2d

class read_ncep_vector:
    #SAME AS ABOVE BUT FOR NCEP DATA
    def __init__(self, year):

        filepath_u = '/Users/apetty/NOAA/DATA/WINDS/NCEP/DATA/REANAL1/uwnd.sig995.'+str(year)+'.nc'
        filepath_v = '/Users/apetty/NOAA/DATA/WINDS/NCEP/DATA/REANAL1/vwnd.sig995.'+str(year)+'.nc'
        self.f_u = Dataset(filepath_u, 'r')
        self.f_v = Dataset(filepath_v, 'r')
        self.varname1 = 'uwnd'
        self.varname2 = 'vwnd'
        
        self.lats = self.f_u.variables['lat'][::-1]
        self.lons0 = self.f_u.variables['lon'][:] 
        var_array_t = self.f_u.variables['uwnd'][0, ::-1, ::]

        var_temp, self.lons_s = shiftgrid(180.,var_array_t,self.lons0,start=False)
        

    def vars(self, index):

        var_array1 = self.f_u.variables[self.varname1][index, ::-1, ::]
        var_array2 = self.f_v.variables[self.varname2][index, ::-1, ::]
        var_temp1, lons_temp = shiftgrid(180.,var_array1,self.lons0,start=False)
        var_temp2, lons_temp = shiftgrid(180.,var_array2,self.lons0,start=False)

        return var_temp1, var_temp2


    def x_y_lons_lats(self, m):

        lons1d = self.lons_s
        lats1d = self.lats
        
        lons2d, lats2d = np.meshgrid(self.lons_s, self.lats)
        xpts, ypts = m(lons2d, lats2d)
        return xpts, ypts, lons1d, lons2d, lats1d, lats2d

class read_ncep2_vector:
    #SAME AS ABOVE BUT FOR NCEP REANALYSIS 2 DATA
    def __init__(self, year, filepath):

        filepath_u = filepath+'/uwnd.10m.gauss.'+str(year)+'.nc'
        filepath_v = filepath+'/vwnd.10m.gauss.'+str(year)+'.nc'
        self.f_u = Dataset(filepath_u, 'r')
        self.f_v = Dataset(filepath_v, 'r')
        self.varname1 = 'uwnd'
        self.varname2 = 'vwnd'
        #AS ARRAYS ARE QUITE BIG - ONLY USE DATA FROM AROUND 25 DEG NORTH
        self.lats  = self.f_u.variables['lat'][::-1][60:-1]
        self.lons0 = self.f_u.variables['lon'][:] 
        var_array_t = self.f_u.variables['uwnd'][0,0, ::-1, ::][60:-1, :]

        var_temp, self.lons_s = shiftgrid(180.,var_array_t,self.lons0,start=False)
        

    def vars(self, index):

        var_array1 = self.f_u.variables[self.varname1][index,0, ::-1, ::][60:-1, :]
        var_array2 = self.f_v.variables[self.varname2][index,0, ::-1, ::][60:-1, :]
        var_temp1, lons_temp = shiftgrid(180.,var_array1,self.lons0,start=False)
        var_temp2, lons_temp = shiftgrid(180.,var_array2,self.lons0,start=False)

        return var_temp1, var_temp2


    def x_y_lons_lats(self, m):

        lons1d = self.lons_s
        lats1d = self.lats
        
        lons2d, lats2d = np.meshgrid(self.lons_s, self.lats)
        xpts, ypts = m(lons2d, lats2d)
        return xpts, ypts, lons1d, lons2d, lats1d, lats2d

class read_jra_vector:
    #SAME AS ABOVE BUT FOR NCEP REANALYSIS 2 DATA
    def __init__(self, year, filepath):

        filepath_u = glob(filepath+'/anl_surf125.033_ugrd.'+str(year)+'*.nc')
        #print filepath_u
        filepath_v = glob(filepath+'/anl_surf125.034_vgrd.'+str(year)+'*.nc')
        self.f_u = Dataset(filepath_u[0], 'r')
        self.f_v = Dataset(filepath_v[0], 'r')
        self.varname1 = 'UGRD_GDS0_HTGL'
        self.varname2 = 'VGRD_GDS0_HTGL'
        #AS ARRAYS ARE QUITE BIG - ONLY USE DATA FROM AROUND 25 DEG NORTH
        self.lats  = self.f_u.variables['g0_lat_1'][::-1]
        self.lons0 = self.f_u.variables['g0_lon_2'][:] 
        var_array_t = self.f_u.variables['UGRD_GDS0_HTGL'][0, ::-1, ::]

        var_temp, self.lons_s = shiftgrid(180.,var_array_t,self.lons0,start=False)
        

    def vars(self, index):

        var_array1 = self.f_u.variables[self.varname1][index,::-1, ::]
        var_array2 = self.f_v.variables[self.varname2][index,::-1, ::]
        var_temp1, lons_temp = shiftgrid(180.,var_array1,self.lons0,start=False)
        var_temp2, lons_temp = shiftgrid(180.,var_array2,self.lons0,start=False)

        return var_temp1, var_temp2


    def x_y_lons_lats(self, m):

        lons1d = self.lons_s
        lats1d = self.lats
        
        lons2d, lats2d = np.meshgrid(self.lons_s, self.lats)
        xpts, ypts = m(lons2d, lats2d)
        return xpts, ypts, lons1d, lons2d, lats1d, lats2d

def read_ncep2_mask(filepath):
    #READ IN NCEP 2 LAND MASK
    #SHIFT GRID FROM 0-360 to -180-180

    filepath_m = filepath+'/land_sfc_gauss.nc'
    f = Dataset(filepath_m, 'r')

    mask_array = f.variables['land'][0,::-1, ::][60:-1, :]
    lats = f.variables['lat'][::-1][60:-1]
    lons0 = f.variables['lon'][:] 
    mask_shift, lons_s = shiftgrid(180.,mask_array,lons0,start=False)
        
    return mask_shift

def calc_wind_curl_trend_bc(REANAL, grid_str, date_str, figpath, datapath, years_str, months_str, box_str, arr_res, start_year, end_year, nx, ny,dx_res, m, beau_lonlat, cent_lonlat):

    ave_wind_curl_years= load(datapath+date_str+'/ave_wind_curl_years'+date_str+box_str+'.txt')
    ave_x_vel_years= load(datapath+date_str+'/ave_xvel_years'+date_str+box_str+'.txt')
    ave_y_vel_years= load(datapath+date_str+'/ave_yvel_years'+date_str+box_str+'.txt')

    #NOT SURE WHAT THE JRA ONE IS, SO USE NCEP/ERA lsm_t100KM!
    lsm_t = loadtxt(datapath+'/lsm_t'+grid_str+'.txt')
    lons2d_t = loadtxt(datapath+'lons2d_t'+grid_str+'.txt')
    lats2d_t = loadtxt(datapath+'lats2d_t'+grid_str+'.txt')
    nx = ave_x_vel_years.shape[1]
    ny = ave_x_vel_years.shape[1]
    num_years = ave_x_vel_years.shape[0]

    wind_x_trend, wind_x_sig, r_x, int_x = var_trend(ave_x_vel_years, num_years)
    wind_y_trend, wind_y_sig, r_y, int_y = var_trend(ave_y_vel_years, num_years)

    wind_curl_trend, wind_curl_sig, r_a, int_a = var_trend(ave_wind_curl_years, num_years)
    beau_region = np.ones((nx, ny))
    central_region = np.ones((nx, ny))

    beau_region = where((lsm_t>0.5) | (lons2d_t<beau_lonlat[0]) | (lons2d_t>beau_lonlat[1]) | (lats2d_t>beau_lonlat[2]) | (lats2d_t<beau_lonlat[3]), 0, beau_region)
    central_region = where((lsm_t>0.5) | (lons2d_t<cent_lonlat[0]) | (lons2d_t>cent_lonlat[1]) | (lats2d_t>cent_lonlat[2]) | (lats2d_t<cent_lonlat[3]), 0, central_region)


    wind_curl_trend_central = ma.masked_where(central_region<0.5, wind_curl_trend)
    wind_curl_trend_beau = ma.masked_where(beau_region<0.5, wind_curl_trend)

    ave_wind_curl_years_b = np.zeros((num_years))
    ave_wind_curl_years_c = np.zeros((num_years))
    for x in xrange(num_years):
      ave_wind_curl_years_b[x] =  np.mean(ma.masked_where(beau_region<0.5, ave_wind_curl_years[x]))
      ave_wind_curl_years_c[x] =  np.mean(ma.masked_where(central_region<0.5, ave_wind_curl_years[x]))

    #plot_wc_trend(m, xpts_t, ypts_t,wind_x_trend,wind_y_trend, wind_curl_trend, \
    #    out=outpath+'WINDCURL/wind_curl_trend_'+str(dx_res/1000)+'km_'+years_str+months_str,units_lab=r'm s$^{-2}$yr$^{-1}$',\
    #    units_vec=r'm s$^{-1}$yr$^{-1}$',minval=-1e-6, maxval=1e-6, vector_val=0.1, base_mask=1, res=arr_res, scale_vec=1, year_string=years_str, month_string=months_str)
    
    savetxt(datapath+date_str+'/wind_curl_trend'+box_str+'.txt', wind_curl_trend)
    savetxt(datapath+date_str+'/wind_curl_sig'+box_str+'.txt', wind_curl_sig)
    savetxt(datapath+date_str+'/wind_x_trend'+box_str+'.txt', wind_x_trend)
    savetxt(datapath+date_str+'/wind_x_sig'+box_str+'.txt', wind_x_sig)
    savetxt(datapath+date_str+'/wind_y_trend'+box_str+'.txt', wind_y_trend)
    savetxt(datapath+date_str+'/wind_y_sig'+box_str+'.txt', wind_y_sig)

    savetxt(datapath+date_str+'/ave_wind_curl_years_b'+box_str+'.txt', ave_wind_curl_years_b)
    savetxt(datapath+date_str+'/ave_wind_curl_years_c'+box_str+'.txt', ave_wind_curl_years_c)

    #PLOT LINEPLOT
    #plot_wind_curl_lines(REANAL, grid_str, date_str, figpath, datapath, years_str, months_str, arr_res, start_year, end_year, nx, ny,dx_res, m, beau_lonlat, cent_lonlat)

def var_trend(var, num_years):
#LINEAR REGRESSION OF VAR AGAINST NUMBER OF YEARS FOR EACH GRID CELL (ASSUMED TO BE 2D)
    #SIG IS SIGNIFICANCE OF TREND AS A PERCENTAGE

    num_x = var.shape[1]
    num_y = var.shape[2]

    years = np.arange(num_years)
    trend = ma.masked_all((num_x, num_y))
    sig_a = ma.masked_all((num_x, num_y))
    r_a = ma.masked_all((num_x, num_y))
    int_a = ma.masked_all((num_x, num_y))


    for i in xrange(num_x):
            for j in xrange(num_y):
                if (var[0, :, :] is not ma.masked):
                    slope, intercept, r, prob, stderr = stats.linregress(years,var[:, i, j])
                    trend[i, j] = slope
                    sig_a[i, j] = 100*(1-prob)
                    r_a[i, j] = r
                    int_a[i, j] = intercept

    return trend, sig_a, r_a, int_a

def var_trend_1D(var):
#LINEAR REGRESSION OF VAR WHICH IS ASSUMED TO BE A 1D COLUMN
    trend, intercept, r, prob, stderr = stats.linregress(np.arange(var.shape[0]),var[:])   
    sig = 100*(1-prob)

    return trend, sig, r, intercept

def  perceptual_colormap(name, PATH, reverse=0):
    #cubeYF
    #cube1
    LinL = np.loadtxt(PATH+'{name}_0-1.csv'.format(name=name),delimiter=",")
    if (reverse==1):
        LinL=np.flipud(LinL)
    #blue 
    b3=LinL[:,2] # value of blue at sample n
    b2=LinL[:,2] # value of blue at sample n
    b1=np.linspace(0,1,len(b2)) # position of sample n - ranges from 0 to 1
    #green
    g3=LinL[:,1]
    g2=LinL[:,1]
    g1=np.linspace(0,1,len(g2))
    #red
    r3=LinL[:,0]
    r2=LinL[:,0]
    r1=np.linspace(0,1,len(r2))
    # creating list
    R=zip(r1,r2,r3)
    G=zip(g1,g2,g3)
    B=zip(b1,b2,b3)
    # transposing list
    RGB=zip(R,G,B)
    rgb=zip(*RGB)
    # creating dictionary
    k=['red', 'green', 'blue']
    data=dict(zip(k,rgb)) # makes a dictionary from 2 lists
    my_cmap = mpl.colors.LinearSegmentedColormap(name,data)
    return my_cmap

def correlate(var1, var2):
#correlate two variables
    trend, intercept, r_a, prob, stderr = stats.linregress(var1, var2)
    sig = 100*(1-prob)
    return trend, sig, r_a, intercept 

def calc_beau_lonlat(beau_lonlat):
    lats_beau = np.zeros((40))
    lats_beau[0:10] = np.linspace(beau_lonlat[3], beau_lonlat[2], 10)
    lats_beau[10:20] = np.linspace(beau_lonlat[2], beau_lonlat[2], 10)
    lats_beau[20:30] = np.linspace(beau_lonlat[2], beau_lonlat[3], 10)
    lats_beau[30:40] = np.linspace(beau_lonlat[3], beau_lonlat[3], 10)
    lons_beau = np.zeros((40))
    lons_beau[0:10] = np.linspace(beau_lonlat[1], beau_lonlat[1], 10)
    lons_beau[10:20] = np.linspace(beau_lonlat[1], beau_lonlat[0], 10)
    lons_beau[20:30] = np.linspace(beau_lonlat[0], beau_lonlat[0], 10)
    lons_beau[30:40] = np.linspace(beau_lonlat[0], beau_lonlat[1], 10)

    return lons_beau, lats_beau


