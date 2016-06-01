############################################################## 
# Date: 01/01/16
# Name: plot_3rean_4seasons_wind_trend.py
# Author: Alek Petty
# Description: Script to plot seasonal wind curl trend over the Arctic/BG
# Input requirements: Wind curl trend data, produced by calc_wind_curl_lineplot_ERA_NCEP_JRA.py     
# Output: Maps and lineplots of wind curl trends from 3 reanalyses


import BG_functions as BGF
import matplotlib
matplotlib.use("AGG")
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
from pylab import *
import numpy.ma as ma
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid.inset_locator import mark_inset
from mpl_toolkits.axes_grid.anchored_artists import AnchoredSizeBar
from matplotlib import rc

mpl.rc("ytick",labelsize=10)
mpl.rc("xtick",labelsize=10)
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
majorLocator   = MultipleLocator(8)
majorFormatter = FormatStrFormatter('%d')
minorLocator   = MultipleLocator(1)

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

m = Basemap(projection='npstere',boundinglat=66,lon_0=0, resolution='l'  )
dx_res = 100000.
grid_str=str(int(dx_res/1000))+'km'
arr_res = int(ceil(100000/dx_res))

box=0
if (box==0):
    box_str='wbox'
    beau_lonlat = [-175., -125., 85., 70.]
if (box==1):
    box_str='ibox'
    beau_lonlat = [-170., -130., 82., 72.]


#beau_lonlat = [-170., -120., 85., 73.]
cent_lonlat = [-150., 10., 90., 81.]
lons_beau, lats_beau = calc_beau_lonlat(beau_lonlat)

start_year = 1980
end_year = 2013
num_years = end_year - start_year + 1
year_str= str(start_year)+'-'+str(end_year)

datapath='./Data_output/WINDS/'
figpath='./Figures/'

xpts = loadtxt(datapath+'/NCEP2/WINDCURL/'+grid_str+'/xpts_wind_trend'+grid_str+'.txt')
ypts = loadtxt(datapath+'/NCEP2/WINDCURL/'+grid_str+'/ypts_wind_trend'+grid_str+'.txt')

wind_x_trend = ma.masked_all((3,4, 55, 55))
wind_y_trend = ma.masked_all((3,4, 55, 55))
wind_x_sig = ma.masked_all((3,4, 55, 55))
wind_y_sig = ma.masked_all((3,4, 55, 55))
wind_curl_trend = ma.masked_all((3,4, 55, 55))
wind_curl_sig = ma.masked_all((3,4, 55, 55))
ave_wind_curl_years_b = ma.masked_all((3,4, num_years))

wind_curl_b_trend=ma.masked_all((3, 4))
wind_curl_b_sig=ma.masked_all((3, 4))
wind_curl_b_r=ma.masked_all((3, 4))
wind_curl_b_int=ma.masked_all((3, 4))
#trend_b_str = []
#sig_b_str = []
b_trendline = ma.masked_all((3, 4, num_years))


reanals= ['NCEP2', 'ERA', 'JRA']
month_strs = ['J-M', 'A-J', 'J-S', 'O-D']

for r in xrange(3):
    for x in xrange(4):
        date_str = str(start_year)+'-'+str(end_year)+month_strs[x]
        path = datapath+reanals[r]+'/WINDCURL/'+grid_str+'/'+date_str
        wind_x_trend[r, x] = loadtxt(path+'/wind_x_trend'+box_str+'.txt')
        wind_y_trend[r, x] = loadtxt(path+'/wind_y_trend'+box_str+'.txt')
        wind_x_sig[r, x] = loadtxt(path+'/wind_x_sig'+box_str+'.txt')
        wind_y_sig[r, x] = loadtxt(path+'/wind_y_sig'+box_str+'.txt')
        wind_curl_trend[r, x] = loadtxt(path+'/wind_curl_trend'+box_str+'.txt')
        wind_curl_sig[r, x] = loadtxt(path+'/wind_curl_sig'+box_str+'.txt')
        ave_wind_curl_years_b[r, x] = loadtxt(path+'/ave_wind_curl_years_b'+box_str+'.txt')
        wind_curl_b_trend[r, x], wind_curl_b_sig[r, x], wind_curl_b_r[r, x], wind_curl_b_int[r, x] = BGF.var_trend_1D(ave_wind_curl_years_b[r, x])
        b_trendline[r, x] = wind_curl_b_trend[r, x]*np.arange(num_years) + wind_curl_b_int[r, x]


minval=-3
maxval=3
base_mask=1
res=2
vector_val=0.1 
scale_vec=0.5
years = np.arange(start_year, start_year+num_years, 1)

reanal_labels = ['NCEP-R2', 'ERA-I', 'JRA-55']
axesname = ['ax1', 'ax2', 'ax3', 'ax4', 'ax5', 'ax6', 'ax7', 'ax8', 'ax9', 'ax10', 'ax11', 'ax12', 'ax13', 'ax14', 'ax15', 'ax16']

fig = figure(figsize=(7,7))
axes=[]
axes.append([0.01, 0.53, 0.46, 0.46])
axes.append([0.53, 0.53, 0.46, 0.46])
axes.append([0.01, 0.01, 0.46, 0.46])
axes.append([0.55, 0.055, 0.43, 0.41])

for x in xrange(4):
    for r in xrange(3):
        vars()[axesname[(x*4)+r]] = subplot(4,4,(x*4)+r+1) #fig.add_axes(axes[plotnum])

        xb, yb = m(lons_beau, lats_beau) # forgot this line
        m.plot(xb, yb, '-', linewidth=1.5, color='k', zorder=5)

        im1 = m.pcolormesh(xpts , ypts, wind_curl_trend[r, x]*1e6, cmap=plt.cm.RdBu_r,vmin=minval, vmax=maxval,shading='gouraud', zorder=4, rasterized=True)
        im2 = m.contour(xpts , ypts, wind_curl_sig[r, x], levels=[95], colors='y', zorder=5)
        # LOWER THE SCALE THE LARGER THE ARROW
        Q = m.quiver(xpts[::res, ::res], ypts[::res, ::res], wind_x_trend[r, x, ::res, ::res], wind_y_trend[r, x,::res, ::res], units='inches',scale=scale_vec , width = 0.01, zorder=7)
        #ASSIGN A LEGEND OF THE VECTOR    

        m.drawparallels(np.arange(90,-90,-10), linewidth = 0.25, zorder=10)
        m.drawmeridians(np.arange(-180.,180.,30.), linewidth = 0.25, zorder=10)
        m.fillcontinents(color='0.7',lake_color='grey', zorder=7)
        #m.drawcoastlines(linewidth=0.5, zorder=7)
        if ((x==0) & (r==0)):
            xS, yS = m(155, 65.5)
            qk = vars()[axesname[0]].quiverkey(Q, xS, yS, vector_val, str(vector_val)+' '+r'm s$^{-1}$yr$^{-1}$', fontproperties={'size': 'small'}, labelsep = 0.01, coordinates='data', zorder = 8) 



        xS, yS = m(228, 63)
        text(xS, yS,reanal_labels[r],fontsize=10, zorder = 11)

        xa,ya = m(-66,67) # we define the corner 1
        x2a,y2a = m(150,66) # then corner 2
        vars()[axesname[(x*4)+r]].set_xlim(xa,x2a) # and we apply the limits of the zoom plot to the inset axes
        vars()[axesname[(x*4)+r]].set_ylim(ya,y2a) # idem


    vars()[axesname[(x*4)+3]] = subplot(4,4,(x*4)+3+1)

    pl3 = plot(years,ave_wind_curl_years_b[2, x]/1e-5, linestyle='-',linewidth=1, color='b')
    pl32 = plot(years,b_trendline[2, x] /1e-5, linestyle='--',linewidth=1, color='b')

    pl1 = plot(years,ave_wind_curl_years_b[0, x]/1e-5, linestyle='-',linewidth=1, color='m')
    pl12 = plot(years,b_trendline[0, x]/1e-5 , linestyle='--',linewidth=1, color='m')

    pl2 = plot(years,ave_wind_curl_years_b[1, x]/1e-5, linestyle='-',linewidth=1, color='g')
    pl22 = plot(years,b_trendline[1, x]/1e-5 , linestyle='--',linewidth=1, color='g')

    vars()[axesname[(x*4)+3]].yaxis.tick_right()
    vars()[axesname[(x*4)+3]].set_xticklabels( [])
    vars()[axesname[(x*4)+3]].xaxis.set_major_locator(majorLocator)
    vars()[axesname[(x*4)+3]].xaxis.set_minor_locator(minorLocator)
    vars()[axesname[(x*4)+3]].yaxis.grid(True)
    vars()[axesname[(x*4)+3]].xaxis.grid(True, which='major')
    vars()[axesname[(x*4)+3]].set_xlim(years[0], years[-1])
    vars()[axesname[(x*4)+3]].set_ylim(-8, 4)

vars()[axesname[(x*4)+3]].xaxis.set_major_formatter(majorFormatter)

ax = fig.add_axes( [0., 0., 1, 1] )
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(.98, 0.5, 'Wind curl '+r'(10$^{-5}$m s$^{-2}$)', rotation=-90,horizontalalignment='center', size='small', verticalalignment='center')
ax.text(.015, 0.82, 'JFM' ,rotation=90, horizontalalignment='center', size='small', verticalalignment='center')
ax.text(.015, 0.61, 'AMJ',rotation=90,horizontalalignment='center', size='small', verticalalignment='center')
ax.text(.015, 0.4, 'JAS',rotation=90,horizontalalignment='center', size='small', verticalalignment='center')
ax.text(.015, 0.19, 'OND',rotation=90,horizontalalignment='center', size='small', verticalalignment='center')

ax16.set_xlabel( 'Years',fontsize='small')

plts_net = pl1+pl2+pl3
leg = ax4.legend(plts_net, reanal_labels, loc=1, ncol=3,columnspacing=0.1, handletextpad=0.0001, bbox_to_anchor=(0.99, 1.33), frameon=False)
llines = leg.get_lines()
setp(llines, linewidth=2.0)
ltext  = leg.get_texts()
setp(ltext, fontsize='small')

cax = fig.add_axes([0.2, 0.07, 0.4, 0.02])
cbar = colorbar(im1,cax=cax, orientation='horizontal', extend='both', use_gridspec=True)
cbar.set_label(r'Wind curl trend (10$^{-6}$m s$^{-2}$yr$^{-1}$)', fontsize=11)
xticks = np.linspace(minval, maxval, 5)
cbar.set_ticks(xticks)
cbar.formatter.set_powerlimits((-3, 4))
cbar.formatter.set_scientific(True)
cbar.update_ticks() 
cbar.solids.set_rasterized(True)


subplots_adjust(bottom=0.1, left=0.025, right=0.94, hspace=0.055, wspace = 0.02)

savefig(figpath+'/12comp_wind_curl'+year_str+box_str+'.png', dpi=150)
close(fig)



