'''
SEEPS thresholds from climatology

Yingkai Sha
ksha@ucar.edu
'''

import os
import sys
import yaml
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import pandas as pd

sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu
import seeps_utils as seeps

config_name = os.path.realpath('../verif_config.yml')

with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

# ---------------------------------------------------------------------------------------- #
# ERA5 verif target
filename_ERA5 = sorted(glob(conf['ERA5_ours']['save_loc']))

# pick years
year_range = [1990, 2020]
years_pick = np.arange(year_range[0], year_range[1]+1, 1).astype(str)
filename_ERA5 = [fn for fn in filename_ERA5 if any(year in fn for year in years_pick)]

# merge yearly ERA5 as one
ds_ERA5 = [vu.get_forward_data(fn) for fn in filename_ERA5]
ds_ERA5_merge = xr.concat(ds_ERA5, dim='time')

variable_levels = {
    'total_precipitation': None,
}

ds_target = ds_ERA5_merge
ds_target = vu.ds_subset_everything(ds_target, variable_levels)
ds_target_24h = ds_target.resample(time='24h').sum()

thres_calc = seeps.SEEPSThreshold(dry_threshold_mm=0.25, var='total_precipitation')
ds_clim = thres_calc.compute(ds_target_24h, dim='time')

save_name = conf['ERA5_weatherbench']['save_loc_clim'] + 'ERA5_clim_1990_2019_SEEPS.nc'
ds_clim.to_netcdf(save_name, mode='w')


