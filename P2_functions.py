import pandas as pd
import numpy as np
from scipy import stats

from coastal_data import CD_geometry, CD_statistics

import pdb

# ===================================================================================
# Functions for notebooks 12 and 13 (Terschelling)
# ===================================================================================

def get_all_volumes_Ters(jarkus_red, kf_interp, rts_interp, transect_polys, epsg_local):
    volumes_jarkus = pd.DataFrame(index=[int(_) for _ in jarkus_red.drop(columns='geometry').columns])
    volumes_kf = volumes_jarkus.copy()
    volumes_rts = volumes_jarkus.copy()    
    
    for idx_transect in transect_polys.keys():
        volumes_jarkus[idx_transect] = CD_geometry.compute_volume_changes(jarkus_red, transect_polys[idx_transect], epsg_out=epsg_local)
        volumes_kf[idx_transect] = CD_geometry.compute_volume_changes(kf_interp, transect_polys[idx_transect], epsg_out=epsg_local)
        volumes_rts[idx_transect] = CD_geometry.compute_volume_changes(rts_interp, transect_polys[idx_transect], epsg_out=epsg_local)
        
    return volumes_jarkus, volumes_kf, volumes_rts

def split_Ters(volumes_jarkus, volumes_kf, volumes_rts):
    # West Terschelling erosive section
    idx_west = [25, 57]
    # Middle accretive section
    idx_middle = [58, 111]
    # East Terschelling erosive section
    idx_east = [112, 144]

    volumes_jarkus_west = volumes_jarkus.iloc[:, idx_west[0]:idx_west[1]]
    volumes_jarkus_middle = volumes_jarkus.iloc[:, idx_middle[0]:idx_middle[1]]
    volumes_jarkus_east = volumes_jarkus.iloc[:, idx_east[0]:idx_east[1]]

    volumes_kf_west = volumes_kf.iloc[:, idx_west[0]:idx_west[1]]
    volumes_kf_middle = volumes_kf.iloc[:, idx_middle[0]:idx_middle[1]]
    volumes_kf_east = volumes_kf.iloc[:, idx_east[0]:idx_east[1]]

    volumes_rts_west = volumes_rts.iloc[:, idx_west[0]:idx_west[1]]
    volumes_rts_middle = volumes_rts.iloc[:, idx_middle[0]:idx_middle[1]]
    volumes_rts_east = volumes_rts.iloc[:, idx_east[0]:idx_east[1]]
    
    return volumes_jarkus_west, volumes_jarkus_middle, volumes_jarkus_east, volumes_kf_west, volumes_kf_middle, volumes_kf_east, volumes_rts_west, volumes_rts_middle, volumes_rts_east

def volume_stats_Ters(volumes_jarkus, volumes_kf, volumes_rts, volumes_jarkus_west, volumes_jarkus_middle, volumes_jarkus_east, volumes_kf_west, volumes_kf_middle, volumes_kf_east, volumes_rts_west, volumes_rts_middle, volumes_rts_east, print_output=False):
       
    rmse = {}
    corr = {}
    pvalues = {}
    trends = {}
    
    # RMSE
    rmse['all'] = CD_statistics.RMSE_timeseries(volumes_rts.mean(axis=1), volumes_jarkus.mean(axis=1))
    rmse['west'] = CD_statistics.RMSE_timeseries(volumes_rts_west.mean(axis=1), volumes_jarkus_west.mean(axis=1))
    rmse['center'] = CD_statistics.RMSE_timeseries(volumes_rts_middle.mean(axis=1), volumes_jarkus_middle.mean(axis=1))
    rmse['east'] = CD_statistics.RMSE_timeseries(volumes_rts_east.mean(axis=1), volumes_jarkus_east.mean(axis=1))
    
    # Correlation
    corr['all'], pvalues['all'] = stats.pearsonr(volumes_rts.mean(axis=1), volumes_jarkus.mean(axis=1))
    corr['west'], pvalues['west'] = stats.pearsonr(volumes_rts_west.mean(axis=1), volumes_jarkus_west.mean(axis=1))
    corr['center'], pvalues['center'] = stats.pearsonr(volumes_rts_middle.mean(axis=1), volumes_jarkus_middle.mean(axis=1))
    corr['east'], pvalues['east'] = stats.pearsonr(volumes_rts_east.mean(axis=1), volumes_jarkus_east.mean(axis=1))
    
    # Trends
    years = volumes_jarkus.index.values

    trends['jarkus_all'], _, _ = CD_statistics.compute_trend_with_error(years, volumes_jarkus.mean(axis=1))
    trends['jarkus_west'], _, _ = CD_statistics.compute_trend_with_error(years, volumes_jarkus_west.mean(axis=1))
    trends['jarkus_center'], _, _ = CD_statistics.compute_trend_with_error(years, volumes_jarkus_middle.mean(axis=1))
    trends['jarkus_east'], _, _ = CD_statistics.compute_trend_with_error(years, volumes_jarkus_east.mean(axis=1))

    trends['kf_all'], _, _ = CD_statistics.compute_trend_with_error(years, volumes_kf.mean(axis=1))
    trends['kf_west'], _, _ = CD_statistics.compute_trend_with_error(years, volumes_kf_west.mean(axis=1))
    trends['kf_center'], _, _ = CD_statistics.compute_trend_with_error(years, volumes_kf_middle.mean(axis=1))
    trends['kf_east'], _, _ = CD_statistics.compute_trend_with_error(years, volumes_kf_east.mean(axis=1))

    trends['rts_all'], _, _ = CD_statistics.compute_trend_with_error(years, volumes_rts.mean(axis=1))
    trends['rts_west'], _, _ = CD_statistics.compute_trend_with_error(years, volumes_rts_west.mean(axis=1))
    trends['rts_center'], _, _ = CD_statistics.compute_trend_with_error(years, volumes_rts_middle.mean(axis=1))
    trends['rts_east'], _, _ = CD_statistics.compute_trend_with_error(years, volumes_rts_east.mean(axis=1))
    
    trend_diff = {}
    trend_diff_perc = {}
    for area in ['all', 'west', 'center', 'east']:
        trend_diff[area] = trends[f'rts_{area}'] - trends[f'jarkus_{area}']

        if np.sign(trends[f'rts_{area}']) == np.sign(trends[f'jarkus_{area}']):
            # both trends are neg/pos (median, hope that sign is same for each parameter combination): perc as quotient of absolute trend diff
            trend_diff_perc[area] = (np.abs(trend_diff[area]) / np.abs(trends[f'jarkus_{area}'])) * 100
        else:
            # trends have different signs: perc should be negative
            trend_diff_perc[area] = (trend_diff[area] / trends[f'jarkus_{area}']) * 100
    
    # Print output
    if print_output:
        print('Trend differences in % of total trend:')
        for key, value in trend_diff_perc.items():
            print(key, round(value,2))

        print('RMSE relative to total mean volume:')
        for key, value in rmse.items():
            total_mean_vol = np.nanmean(volumes_jarkus.mean(axis=1))
            perc = (np.abs(value) / np.abs(total_mean_vol)) * 100
            print(key, round(perc,1))

        print('Correlation:')
        for key, value in corr.items():
            print(key, round(value,2))

        print('\np-values:')
        for key, value in pvalues.items():
            print(key, round(value,2))
        
    return trends, corr, pvalues, rmse