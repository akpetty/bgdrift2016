############################################################## 
# Date: 01/01/16
# Name: Alek Petty
# Description: Script to read in the daily drift vectors and create monthly drift file to be used in the monthly ice area flux estimates.
# Input requirements: Fowler daily drift data (vectors in x/y direction) and projection lat/lon
#					  Also needs some functions in BGF
# Output: Monthly drift vectors on the same grid, masked however chosen. Also rotates vectors (uv file) so the components are in lat/lon.

import BG_functions as BGF
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
from pylab import *
from scipy.io import netcdf
import numpy.ma as ma
import matplotlib
matplotlib.use("AGG")
from netCDF4 import Dataset
from glob import glob
from scipy.interpolate import griddata
#import string
#from matplotlib.patches import Polygon
#from mpl_toolkits.axes_grid.inset_locator import zoomed_inset_axes
#from mpl_toolkits.axes_grid.inset_locator import mark_inset
#from mpl_toolkits.axes_grid.anchored_artists import AnchoredSizeBar
#from scipy import stats
#from matplotlib import rc

m = Basemap(projection='npstere',boundinglat=48.52,lon_0=0, resolution='l'  )


rawdatapath='../../Data/'
datapath='./Data_output/'
figpath='./Figures/'

outpath = '../OUT/FOWLER/'
initpath = rawdatapath+'ICE_DRIFT/FOWLER/DATA/DAILY/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

lon_lat = loadtxt(initpath+'/north_x_y_lat_lon.txt')
lons = np.reshape(lon_lat[:, 3], (361, 361))
lats = np.reshape(lon_lat[:, 2], (361, 361))

start_year = 1980
end_year = 2013
mon_end=12
num_years = end_year-start_year+1
time_index = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
drift_uv = ma.masked_all((num_years, 12,2, 361, 361))
drift_xy = ma.masked_all((num_years, 12,2, 361, 361))

for y in xrange(num_years):
	year = start_year + y
	print 'y:',y
	files = glob(initpath+str(y+start_year)+'/*.bin')
	if (year>=2013):
		time_index[12]=364
		files = glob(initpath+str(y+start_year)+'/*.vec')
	if (year==2014):
		#ONLY HAVE HALF A YEAR OF DATA FOR 2014 CURRENTLY SO ADD CATCH!
		mon_end = 6
	for mon in xrange(mon_end):
		print 'm:',mon
		uv_xy = ma.masked_all((2,time_index[mon+1] - time_index[mon], 361, 361))
		uv_lonlat = ma.masked_all((2,time_index[mon+1] - time_index[mon], 361, 361))
		v = ma.masked_all((time_index[mon+1] - time_index[mon], 361, 361))
		for x in xrange(time_index[mon], time_index[mon+1], 1):
			print 'd:',x
			fd = open(files[x], 'rb')
			motion_dat = fromfile(file=fd, dtype='<i2')
			#FLIP ROWS FROM TOP TO BOTTOM TO BOTTOM TO TOP
			motion_dat = reshape(motion_dat, [361, 361, 3])

			ut = motion_dat[:, :, 0]/1000.
			vt = motion_dat[:, :, 1]/1000.
			#BGF.plot_var_xy(m, xpts , ypts, ut, vt, sqrt((ut**2) + (vt**2)), out=out_path+str(y+1980)+'-'+str(mon+1)+'_'+str(x), units_lab=r'm s$^{-1}$', units_vec=r'm s$^{-1}$',
            #        minval=0., maxval=0.2, base_mask=1,res=4, scale_vec=0.5, vector_val=0.2, cbar_type='max', cmap_1=plt.cm.YlOrRd)
						
			q = motion_dat[:, :, 2]/1000.
			#negative q is coastal, 0 is no data, greater than 1 means no data closeby
			uv_xy[0,x-time_index[mon]] = ma.masked_where((q<=0) | (q>1), ut)
			uv_xy[1,x-time_index[mon]] = ma.masked_where((q<=0) | (q>1), vt)
		
		drift_xy[y, mon, 0] = ma.mean(uv_xy[0], axis=0)
		drift_xy[y, mon, 1] = ma.mean(uv_xy[1], axis=0)
		uv_xy_mask_count = ma.count_masked(uv_xy, axis=1)
		days_in_month=15
		#MASK WHERE MORE THAN 15 DAYS HAVE NO DATA (I.E. AT LEAST 15 DAYS OF DATA IN THAT MONTH)
		drift_xy[y, mon] = ma.masked_where(uv_xy_mask_count>days_in_month, drift_xy[y, mon])
		drift_uv[y, mon, 0], drift_uv[y, mon, 1] = BGF.calc_uv(drift_xy[y, mon, 0], drift_xy[y, mon, 1], lons)

		#BGF.plot_var_xy(m, xpts , ypts, drift_xy[y, mon, 0], drift_xy[y, mon, 1], sqrt((drift_xy[y, mon, 0]**2) + (drift_xy[y, mon, 1]**2)), out=out_path+str(y+1980)+'-'+str(mon+1), units_lab=r'm s$^{-1}$', units_vec=r'm s$^{-1}$',
        #            minval=0., maxval=0.2, base_mask=1,res=4, scale_vec=0.5, vector_val=0.2, cbar_type='max', cmap_1=plt.cm.YlOrRd)

drift_xy.dump(outpath+str(start_year)+'-'+str(end_year)+'-drift_data_months_xy.txt')
drift_uv.dump(outpath+str(start_year)+'-'+str(end_year)+'-drift_data_months_uv.txt')

output = 0
if output==1:
	drift_uv_OA = ma.masked_all((num_years, 7, 2, 361, 361))
	for x in xrange(num_years):
		drift_uv_OA[x, 0:3] = drift_uv[x, 9:12]
		drift_uv_OA[x, 3:7] = drift_uv[x+1, 0:4]
	drift_uv_OA.dump(figpath+str(start_year)+'1980-'+str(end_year)+'-O-A-drift_data_months_uv.txt')




