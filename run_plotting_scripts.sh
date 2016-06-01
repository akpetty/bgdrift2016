#!/bin/bash
# Date: 01/01/16
# Name: run_plotting_scripts.py
# Author: Alek Petty
# Description: Shell script to run all the plotting scripts contained in this repo.


echo plot BG map
#python plot_map_BG_gates_SMDT.py

echo plotting concentration seasonal
python plot_season_BG_conc_BTNT4.py

echo plot summer ice extent
python plot_summer_extent_NTBT.py

echo plot concentration trends
python plot_conc_trend_seasons_ann.py

echo plot ice thickness
python plot_thickness_IB_ULS_PMAS.py

echo plot fixed axes area flux
python plot_area_flux_3_seasons_12_fixedy.py

echo plot wind-ice drift corrrelation seasonal
python plot_wind_drift_curl_3_decorr.py

echo plot wind curl trend
python plot_3rean_4seasons_wind_trend.py

echo plot ice drift curl trends
python plot_drift_curl_trend_seasons_top.py

echo plot ice strength
python plot_pressure_concthickFILL.py

echo plot variable axes area flux
python plot_area_flux_3_seasons_12_variabley.py

echo plot regression scatter plot
python plot_wind_drift_scatter.py

echo plot wind-ice drift corrrelation annual
python plot_wind_drift_curl_ann_dcorr.py




