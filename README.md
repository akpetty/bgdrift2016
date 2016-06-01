## Beaufort Gyre sea ice circulation scripts

Python scripts used to assess the circulation of ice around the Beaufort Gyre. The plots produced are all contained in the recent publication:

Petty, A. A., J. K. Hutchings, J. A. Richter-Menge, M. A. Tschudi (2016), Sea ice circulation around the Beaufort Gyre: The changing role of wind forcing and the sea ice state, J. Geophys. Res., doi:10.1002/2015JC010903.

Note that individual descriptions should be included at the top of each script.

The 'calc' scripts are all used to process raw wind/drift/concentration data into specific datasets used by the 'plot' plotting scripts. 

The processed data can be obtained directly from http://dx.doi.org/10.5281/zenodo.48464. 
Data should be placed into the 'Data_output' folder.

If you want to run the raw procesing scripts, the follwing datasets are needed:

The NCEP-R2 data: http://www.esrl.noaa. gov/psd/data/gridded/data.ncep.rean- alysis2.html
The ERA-I data: http://apps.ecmwf.int/datasets/data/interim_full_ daily/
The JRA-55 data: http://rda.ucar.edu/ datasets/ds628.0. 
The Polar Pathfinder sea ice drift data: http://nsidc.org/data/nsidc- 0116. 
All CERSAT/IFREMER data sets: ftp://ftp.ifremer.fr/ifremer/cersat/products/gridded/psi-drift/. 
The passive microwave NASA Team and Bootstrap concentration data: http://nsidc.org/data/nsidc-0051 and http://nsidc.org/data/nsidc-0079, respectively. 
The ice draft mooring data: http://www.whoi.edu/page.do?pid566559. 
PIOMAS ice thickness data: http://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/. 

The run_plotting_scripts.sh shell script can be used to run all the python plotting scripts automatically.

Note also that Python 2.7 was used for all processing. I have not tested these scripts in Python 3.


