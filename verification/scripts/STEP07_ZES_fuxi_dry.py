'''
Zonal energy spectrum

Yingkai Sha
ksha@ucar.edu
'''

import os
import sys
import yaml
import argparse
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu
import score_utils as su

config_name = os.path.realpath('../verif_config.yml')

with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('verif_ind_start', help='verif_ind_start')
parser.add_argument('verif_ind_end', help='verif_ind_end')
args = vars(parser.parse_args())

verif_ind_start = int(args['verif_ind_start'])
verif_ind_end = int(args['verif_ind_end'])

# ====================== #
model_name = 'fuxi_dry'
lead_range = conf[model_name]['lead_range']
verif_lead_range = conf[model_name]['verif_lead_range']

leads_exist = list(np.arange(lead_range[0], lead_range[-1]+lead_range[0], lead_range[0]))
leads_verif = [24, 120, 240]
#list(np.arange(verif_lead_range[0], verif_lead_range[-1]+verif_lead_range[0], verif_lead_range[0]))
ind_lead = vu.lead_to_index(leads_exist, leads_verif)

print('Verifying lead times: {}'.format(leads_verif))
print('Verifying lead indices: {}'.format(ind_lead))
# ====================== #

path_verif = conf[model_name]['save_loc_verif']+'combined_zes_{}_{}_{}'.format(
    verif_ind_start, verif_ind_end, model_name)

# ---------------------------------------------------------------------------------------- #
# forecast
filename_OURS = sorted(glob(conf[model_name]['save_loc_gather']+'*.nc'))

# pick years
year_range = conf[model_name]['year_range']
years_pick = np.arange(year_range[0], year_range[1]+1, 1).astype(str)
filename_OURS = [fn for fn in filename_OURS if any(year in fn for year in years_pick)]

L_max = len(filename_OURS)
assert verif_ind_end <= L_max, 'verified indices (days) exceeds the max index available'

filename_OURS = filename_OURS[verif_ind_start:verif_ind_end]

# ---------------------------------------------------------------------------------------- #
variables_levels = {
    'T': [500,],
    'U': [500],
    'V': [500]
}

levels = np.array([   1.,   50.,  150.,  200.,  250.,  
                    300.,  400.,  500.,  600.,  700., 
                     850.,  925., 1000.])

# ---------------------------------------------------------------------------------------- #
# loop over lead time, init time, variables to compute zes
for i, ind_pick in enumerate(ind_lead):
    # allocate result for the current lead time
    verif_results = []
    
    for fn_ours in filename_OURS:
        ds_ours = xr.open_dataset(fn_ours)
        ds_ours['level'] = levels
        ds_ours = vu.ds_subset_everything(ds_ours, variables_levels)
        ds_ours = ds_ours.isel(time=ind_pick)
        ds_ours = ds_ours.compute()
        
        # -------------------------------------------------------------- #
        # potential temperature
        ds_ours['theta'] = ds_ours['T'] * (1000/500)**(287.0/1004)

        # -------------------------------------------------------------- #
        zes_temp = []
        for var in ['U', 'V', 'theta']:
            zes = su.zonal_energy_spectrum_sph(ds_ours.isel(latitude=slice(1, None)), var)
            zes_temp.append(zes)
            
        verif_results.append(xr.merge(zes_temp))
    
    ds_verif = xr.concat(verif_results, dim='time')
    save_name = path_verif+'_lead{}.nc'.format(leads_verif[i])
    ds_verif.to_netcdf(save_name)
