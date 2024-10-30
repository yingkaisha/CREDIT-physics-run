'''
This script is the same as PP_ERA5_plevel_accum.py but for forcing variables.

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
# the sub-folder to store data
base_dir = conf['ARCO']['save_loc'] + 'forcing/' 
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# ==================================================================================== #
# encoding options
compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

chunk_size_3d = dict(chunks=(conf['ARCO']['chunk_size_3d']['time'],
                             conf['ARCO']['chunk_size_3d']['latitude'],
                             conf['ARCO']['chunk_size_3d']['longitude']))

dict_encoding = {}

for i_var, var in enumerate(conf['ARCO']['varname_forcing']):
    dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

# ==================================================================================== #
# main

ERA5_1h = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks=None,
    storage_options=dict(token='anon'),)

time_start = '{}-12-31T00'.format(year-1) # hourly accum var needs one extra day to accum on 6 hrs
time_start_save = '{}-01-01T00'.format(year)
time_end = '{}-12-31T23'.format(year)

# ----------------------------------------------------------------------- #
# get hourly variables
ERA5_1h_yearly = ERA5_1h.sel(time=slice(time_start, time_end))

variables_levels = {}
for varname in conf['ARCO']['varname_forcing']:
    variables_levels[varname] = None

ERA5_1h_save = vu.ds_subset_everything(ERA5_1h_yearly, variables_levels)

# ----------------------------------------------------------------------- #
# accum to 6 hourly variables
ERA5_1h_shifted = ERA5_1h_save.shift(time=-1)
ERA5_6h = ERA5_1h_shifted.resample(time='6h').sum()
ERA5_6h['time'] = ERA5_6h['time'] + pd.Timedelta(hours=6)

ERA5_6h_save = ERA5_6h.sel(time=slice(time_start_save, time_end))

# ----------------------------------------------------------------------- #
# chunking and save
ERA5_6h_save = ERA5_6h_save.chunk(conf['ARCO']['chunk_size_3d'])
save_name = base_dir + conf['ARCO']['prefix'] + '_forcing_{}.zarr'.format(year)
ERA5_6h_save.to_zarr(save_name, mode="w", consolidated=True, compute=True, encoding=dict_encoding)
