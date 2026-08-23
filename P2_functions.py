import pandas as pd
import numpy as np
from scipy import stats

from matplotlib.lines import Line2D

from coastal_data import CD_geometry, CD_statistics

import pdb

def get_all_volumes(vali_red, kf_interp, rts_interp, transect_polys, epsg_local):
    volumes_vali = pd.DataFrame(index=[int(_) for _ in vali_red.drop(columns='geometry').columns])
    volumes_kf = volumes_vali.copy()
    volumes_rts = volumes_vali.copy()    
    
    for idx_transect in transect_polys.keys():
        volumes_vali[idx_transect] = CD_geometry.compute_volume_changes(vali_red, transect_polys[idx_transect], epsg_out=epsg_local)
        volumes_kf[idx_transect] = CD_geometry.compute_volume_changes(kf_interp, transect_polys[idx_transect], epsg_out=epsg_local)
        volumes_rts[idx_transect] = CD_geometry.compute_volume_changes(rts_interp, transect_polys[idx_transect], epsg_out=epsg_local)
        
    return volumes_vali, volumes_kf, volumes_rts

# ===================================================================================
# Functions for notebooks 12 and 13 (Terschelling)
# ===================================================================================

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
    rmse_perc = {}
    for area in ['all', 'west', 'center', 'east']:
        trend_diff[area] = trends[f'rts_{area}'] - trends[f'jarkus_{area}']

        if np.sign(trends[f'rts_{area}']) == np.sign(trends[f'jarkus_{area}']):
            # both trends are neg/pos (median, hope that sign is same for each parameter combination): perc as quotient of absolute trend diff
            trend_diff_perc[area] = (np.abs(trend_diff[area]) / np.abs(trends[f'jarkus_{area}'])) * 100
        else:
            # trends have different signs: perc should be negative
            trend_diff_perc[area] = (trend_diff[area] / trends[f'jarkus_{area}']) * 100

        total_mean_vol = np.nanmean(volumes_jarkus.mean(axis=1))
        rmse_perc[area] = (np.abs(rmse[area]) / np.abs(total_mean_vol)) * 100
    
    # Print output
    if print_output:
        print('Trend differences in % of total trend:')
        for key, value in trend_diff_perc.items():
            print(key, round(value,2))

        print('\nRMSE relative to total mean volume:')
        for key, value in rmse_perc.items():
            print(key, round(value,1))

        print('\nCorrelation:')
        for key, value in corr.items():
            print(key, round(value,2))

        print('\np-values:')
        for key, value in pvalues.items():
            print(key, round(value,2))
        
    return trends, trend_diff_perc, corr, pvalues, rmse, rmse_perc

# ===================================================================================
# Functions for notebooks 22 and 23 (Duck) and 32 and 33 (Narrabeen)
# ===================================================================================

def volume_stats_averaged(volumes_vali, volumes_kf, volumes_rts, print_output=False):
    rmse = CD_statistics.RMSE_timeseries(volumes_rts.mean(axis=1), volumes_vali.mean(axis=1))
    rmse_perc = (np.abs(rmse)/np.abs(volumes_vali.mean().mean())) * 100
    
    corr, pvalues = stats.pearsonr(volumes_rts.mean(axis=1), volumes_vali.mean(axis=1))

    trends = {}
    years = volumes_vali.index.values
    trends['vali'], _, _ = CD_statistics.compute_trend_with_error(years, volumes_vali.mean(axis=1))
    trends['kf'], _, _ = CD_statistics.compute_trend_with_error(years, volumes_kf.mean(axis=1))
    trends['rts'], _, _ = CD_statistics.compute_trend_with_error(years, volumes_rts.mean(axis=1))

    trend_diff = trends['rts'] - trends['vali']
    
    # both trends are neg/pos: perc as quotient of absolute trend diff
    if np.sign(trends['rts']) == np.sign(trends['vali']):
        trend_diff_perc = np.abs(trend_diff) / np.abs(trends['vali']) * 100
    # trends have different signs: perc should be negative
    else:
        trend_diff_perc = trend_diff / trends['vali'] * 100
        
    # Print output
    if print_output:        
        print('Trend differences in % of total trend:', round(trend_diff_perc,2))        
        print('RMSE relative to total mean volume:', round(rmse_perc,1))
        print(f'Correlation:{round(corr,2)}, p: {round(pvalues,2)}')
        
    return trends, trend_diff_perc, corr, pvalues, rmse, rmse_perc

# ===================================================================================
# Functions for notebooks 32 and 33 (Narrabeen)
# ===================================================================================

def volume_stats_Narrabeen(volumes_vali, volumes_kf, volumes_rts, profile_polys_red, print_output=True):
    rmse = {}
    rmse_perc = {}
    corr = {}
    pvalues = {}
    trends = {}
    trend_diff_perc = {}
    
    for area in profile_polys_red.keys():
        rmse[area] = CD_statistics.RMSE_timeseries(volumes_rts[str(area)], volumes_vali[str(area)])
        rmse_perc[area] = (np.abs(rmse[area])/np.abs(volumes_vali[str(area)].mean())) * 100
        corr[area], pvalues[area] = stats.pearsonr(volumes_rts[str(area)], volumes_vali[str(area)])

        trends[f'vali_{area}'], _, _ = CD_statistics.compute_trend_with_error(volumes_vali.index.values.astype(int), volumes_vali[str(area)].values)
        trends[f'kf_{area}'], _, _ = CD_statistics.compute_trend_with_error(volumes_kf.index.values.astype(int), volumes_kf[str(area)].values)
        trends[f'rts_{area}'], _, _ = CD_statistics.compute_trend_with_error(volumes_rts.index.values.astype(int), volumes_rts[str(area)].values)

        trend_diff = trends[f'rts_{area}'] - trends[f'vali_{area}']
        # both trends are neg/pos: perc as quotient of absolute trend diff
        if np.sign(trends[f'rts_{area}']) == np.sign(trends[f'vali_{area}']):
            trend_diff_perc[area] = np.abs(trend_diff) / np.abs(trends[f'vali_{area}']) * 100
        # trends have different signs: perc should be negative
        else:
            trend_diff_perc[area] = trend_diff / trends[f'vali_{area}'] * 100

    if print_output:
        for area in profile_polys_red.keys():
            print(area, 'Trend differences in % of total trend:', round(trend_diff_perc[area],2))
            print(area, 'RMSE relative to total mean volume:', round(rmse_perc[area],1))
            print(area, f'Correlation:{round(corr[area],2)}, p: {round(pvalues[area],2)}', '\n')

    return trends, trend_diff_perc, corr, pvalues, rmse, rmse_perc

# ===================================================================================
# Plot functions for notebooks 12, 22 and 32 (parameter tuning)
# ===================================================================================

def plot_legend(ax, c_list, ls_list, factors, std_init_values):
    legend_elements = [
        Line2D([0], [0], color='none', label='$\\bf{Color: \\sigma_q}$', lw=0, ms=0),  # Subtitle
        
        Line2D([0], [0], color=c_list[0], label=f'$\\sigma_{{q}}$ = {factors[0]}$\\cdot \\sigma_{{obs}}$ m'),
        Line2D([0], [0], color=c_list[1], label=f'$\\sigma_{{q}}$ = {factors[1]}$\\cdot \\sigma_{{obs}}$ m'),
        Line2D([0], [0], color=c_list[2], label=f'$\\sigma_{{q}}$ = {factors[2]}$\\cdot \\sigma_{{obs}}$ m'),

        Line2D([0], [0], color='none', label='', lw=0, ms=0), # dummy for white space
        Line2D([0], [0], color='none', label='$\\bf{Linestyle: \\sigma_{{init}}}$', lw=0, ms=0),  # Subtitle

        Line2D([0], [0], color='grey', ls=ls_list[0], label=f'$\\sigma_{{x_0}}$ = {std_init_values[0]} m'),
        Line2D([0], [0], color='grey', ls=ls_list[1], label=f'$\\sigma_{{x_0}}$ = {std_init_values[1]} m'),
        Line2D([0], [0], color='grey', ls=ls_list[2], label=f'$\\sigma_{{x_0}}$ = {std_init_values[2]} m'),

        Line2D([0], [0], color='none', label='', lw=0, ms=0), # dummy for white space

        # Line2D([0], [0], marker='o', lw=0, ms=10, color='grey', label='p-value < 0.05')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1,1.1))

def plot_vol_stats(ax, data, pvalues_df, factors, std_init_values, param_df, ls_list, c_list, ylabel, title, corr=False, plot_zero=False, ylim=None):
    for f, factor in enumerate(factors):
        for g, std_init in enumerate(std_init_values):
            idx, = np.where((param_df.factor == factor) & (param_df.std_init == std_init))
            ax.plot(param_df.loc[idx, 'sigma_l'], data.loc[idx], ls=ls_list[g], c=c_list[f], marker='.')
            
            if corr:
                pvalues_box = np.where(pvalues_df.T.loc[idx] < 0.05)[0]
                ax.plot(param_df.loc[idx[pvalues_box], 'sigma_l'], data.loc[idx[pvalues_box]].values,
                        marker='o', lw=0, ms=10, color='grey', label='p-value < 0.05')
                ax.legend(handles=[Line2D([0], [0], marker='o', lw=0, ms=10, color='grey', label='p-value < 0.05')]) #, loc='lower right')

            if plot_zero:
                ax.axhline(y=0, color='grey')            

    if ylim != None:
        ax.set_ylim(ylim[0], ylim[1])

    # if not corr:
    #     ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # ax.ticklabel_format(useOffset=False, style='plain')
    
    ax.set_xlabel(r'$\sigma_{l}$ [m]')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=22)
    ax.grid()

def mark_selected_param(ax, idx_selected, data, param_df):
    params_selected = param_df.iloc[idx_selected]
    ax.plot(params_selected['sigma_l'], data.iloc[idx_selected], 'o', c='None', markeredgecolor='red', ms=10, markeredgewidth=2)

























    