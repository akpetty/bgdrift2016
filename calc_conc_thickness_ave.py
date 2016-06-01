############################################################## 
# Date: 01/01/16
# Name: calc_conc_thickness_ave.py
# Author: Alek Petty
# Description: Script to calculate BG concentration/thickness estimates
# Output: datasets of seasonal BG ice thickness and concentration

import numpy as np
from pylab import *
import scipy.io
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata


rawdatapath='../../DATA/'
pmasdatapath=rawdatapath+'/PIOMAS/heff_txt/'
datapath='./Data_output/'
# CHOSSE BG BOX
beau_lonlat = [-175., -125., 85., 70.]
#beau_lonlat = [-170., -130., 82., 72.]

m = Basemap(projection='npstere',boundinglat=66,lon_0=0, resolution='l'  )

alg = 'BOOTSTRAP'
start_year=1980
end_year=2013
ice_conc = load(rawdatapath+'/ICE_CONC/'+alg+'/ice_conc_months-'+str(start_year)+'-'+str(end_year)+'.txt')
latsI = load(rawdatapath+'/ICE_CONC/'+alg+'/ice_conc_lats.txt')
lonsI = load(rawdatapath+'/ICE_CONC/'+alg+'/ice_conc_lons.txt')
xptsI,yptsI = m(lonsI, latsI)

thickness_year=np.zeros((2013-1980+1, 12, 448, 304))

Thickness_season=np.zeros((2013-1980+1, 4, 448, 304))
Thickness_BG=np.zeros((2013-1980+1, 4))

Ice_conc_season=np.zeros((2013-1980+1, 4, 448, 304))
Ice_conc_BG=np.zeros((2013-1980+1, 4))

for year in xrange(2013-1980+1):
	year_str = str(1980+year)
	hiT = loadtxt(pmasdatapath+'heff.txt'+year_str)
	#thickness_BG2D = griddata((xpts, ypts),hiT, (xptsI, yptsI), method='linear')
	lats = hiT[:, 1]
	#lonsT = hiT[:, 2]
	lonsT=np.copy(hiT[:, 2])
	lonsT[where(lonsT>180.)] = -(360-lonsT[where(lonsT>180.)])
	xpts,ypts = m(lonsT, lats)
	for x in xrange(12):
		thickness_monthT = hiT[:, 3+x].T
		thickness_year[year, x] = griddata((xpts, ypts),thickness_monthT, (xptsI, yptsI), method='linear')

for year in xrange(2013-1980+1):
	for x in xrange(4):

		Thickness_season[year, x] = ma.mean(thickness_year[year, x*3:(x*3)+3], axis=0)
		Thickness_BG[year, x] = ma.mean(ma.masked_where((lonsI<beau_lonlat[0]) | (lonsI>beau_lonlat[1]) | (latsI>beau_lonlat[2])| (latsI<beau_lonlat[3]), Thickness_season[year, x]))

		Ice_conc_season[year, x] = ma.mean(ice_conc[year, x*3:(x*3)+3], axis=0)
		Ice_conc_BG[year, x] = ma.mean(ma.masked_where((lonsI<beau_lonlat[0]) | (lonsI>beau_lonlat[1]) | (latsI>beau_lonlat[2])| (latsI<beau_lonlat[3]), Ice_conc_season[year, x]))

Thickness_BG.dump(datapath+'/ThicknessBG_PMAS.txt')
Ice_conc_BG.dump(datapath+'/IceconcBG_'+alg+'.txt')


