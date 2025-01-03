'''
This script merges all gathered ERA5 zarr files as one (not used / replaced by interpolation + merge)

Yingkai Sha
ksha@ucar.edu
'''

import os
import sys
import yaml
import dask
import zarr
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
from dask.utils import SerializableLock

import calendar
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

# ==================================================================================== #
# get year from input
year = int(args['year'])

# import variable name and save location form yaml
config_name = os.path.realpath('../data_config_6h.yml')

with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

# ==================================================================================== #
base_dir = conf['zarr_opt']['save_loc']

zarr_name_surf = base_dir+'surf/ERA5_plevel_6h_surf_{}.zarr'
zarr_name_surf_extra = base_dir+'surf/ERA5_plevel_6h_surf_extend_{}.zarr'
zarr_name_accum = base_dir+'accum/ERA5_plevel_6h_accum_{}.zarr'
zarr_name_forcing = base_dir+'forcing/ERA5_plevel_6h_forcing_{}.zarr'
zarr_name_upper = base_dir+'upper_subset/ERA5_subset_6h_upper_air_{}.zarr'
zarr_name_upper_full = base_dir+'upper_air/ERA5_plevel_6h_upper_air_{}.zarr'
zarr_name_upper_Q = base_dir+'upper_air/ERA5_plevel_6h_Q_{}.zarr'
zarr_name_Q = base_dir+'upper_subset/ERA5_subset_6h_Q_{}.zarr'

ds_surf = xr.open_zarr(zarr_name_surf.format(year))
ds_surf_extra = xr.open_zarr(zarr_name_surf_extra.format(year))
ds_accum = xr.open_zarr(zarr_name_accum.format(year))
ds_forcing = xr.open_zarr(zarr_name_forcing.format(year))
ds_upper = xr.open_zarr(zarr_name_upper.format(year))
ds_upper_full = xr.open_zarr(zarr_name_upper_full.format(year))
ds_upper_Q = xr.open_zarr(zarr_name_upper_Q.format(year))
ds_Q = xr.open_zarr(zarr_name_Q.format(year))

ds_500hPa = xr.merge([ds_upper_full.isel(level=21), ds_upper_Q.isel(level=21)])
ds_500hPa = ds_500hPa.rename({'T': 'T500', 
                              'U': 'U500', 
                              'V': 'V500', 
                              'Z': 'Z500', 
                              'Q': 'Q500', 
                              'specific_total_water': 'specific_total_water_500'})

# ==================================================================================== #
# combining land-sea mask and sea-ice cover

ds_static = xr.open_zarr(base_dir+'static/ERA5_plevel_6h_static.zarr')
land_sea_mask = ds_static['land_sea_mask']
sea_ice_cover = ds_surf_extra['CI']

land_sea_mask_expanded = land_sea_mask.broadcast_like(sea_ice_cover)
land_sea_CI_mask = xr.where(
    (land_sea_mask_expanded == 0) & (sea_ice_cover > 0),
    -sea_ice_cover,
    land_sea_mask_expanded
)

land_sea_CI_mask.name = 'land_sea_CI_mask'
ds_mask = land_sea_CI_mask.to_dataset()

# ==================================================================================== #
# overall merge and chunking
ds_merge = xr.merge([ds_surf, ds_accum, ds_forcing, ds_upper, ds_Q, ds_mask, ds_500hPa])
ds_merge = ds_merge.drop_vars('SSTK')

varnames = list(ds_merge.keys())
varname_4D = ['U', 'V', 'T', 'Z', 'Q', 'specific_total_water']

for i_var, var in enumerate(varnames):
    if var in varname_4D:
        ds_merge[var] = ds_merge[var].chunk(conf['zarr_opt']['chunk_size_4d'])
    else:
        ds_merge[var] = ds_merge[var].chunk(conf['zarr_opt']['chunk_size_3d'])
        
# ========================================================================== #
# zarr encodings
dict_encoding = {}

chunk_size_3d = dict(chunks=(conf['zarr_opt']['chunk_size_3d']['time'],
                             conf['zarr_opt']['chunk_size_3d']['latitude'],
                             conf['zarr_opt']['chunk_size_3d']['longitude']))

chunk_size_4d = dict(chunks=(conf['zarr_opt']['chunk_size_4d']['time'],
                             conf['zarr_opt']['chunk_size_4d']['level'],
                             conf['zarr_opt']['chunk_size_4d']['latitude'],
                             conf['zarr_opt']['chunk_size_4d']['longitude']))

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    if var in varname_4D:
        dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
    else:
        dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

# save
save_name = base_dir+'all_in_one/ERA5_plevel_6h_{}.zarr'.format(year)
ds_merge.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)

