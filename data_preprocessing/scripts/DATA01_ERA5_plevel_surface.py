'''
This script collects ERA5 single-level analysis from NCAR/RDA.
Internal access through the glade file system is required. 
Yearly zarr files are produced after gathering

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
N_months = 12

# ==================================================================================== #
# import variable name and save location form yaml
config_name = os.path.realpath('../data_config_6h.yml')

with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

# ==================================================================================== #
# the sub-folder to store data
base_dir_save = conf['RDA']['save_loc'] + 'surf/' 
if not os.path.exists(base_dir_save):
    os.makedirs(base_dir_save)

# ==================================================================================== #
# encoding options
compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

chunk_size_3d = dict(chunks=(conf['RDA']['chunk_size_3d']['time'],
                             conf['RDA']['chunk_size_3d']['latitude'],
                             conf['RDA']['chunk_size_3d']['longitude']))
dict_encoding = {}
for i_var, var in enumerate(conf['RDA']['varname_single']):
    dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

# ==================================================================================== #
# main

xr.set_options(file_cache_maxsize=500) # increase the file cache size
netcdf_lock = SerializableLock() # lock for safe parallel access

# all days within a year
start_time = datetime(year, 1, 1, 0, 0)
dt_list = [start_time + relativedelta(months=i) for i in range(N_months)]

# var names
varnames = list(conf['RDA']['varname_single'].values())

ds_list = []

for i_mon, dt in enumerate(dt_list):
    # file source info
    base_dir = dt.strftime(conf['RDA']['source']['ansfc_format'])

    first_day = datetime(year, dt.month, 1)
    last_day = datetime(year, dt.month, calendar.monthrange(year, dt.month)[1])
    
    dt_pattern = dt.strftime(conf['RDA']['source']['ansfc_dt_pattern_format'])
    dt_pattern = dt_pattern.format(first_day.day, last_day.day)
    
    # get upper-air vars
    filename_collection = [glob(base_dir + f'*{var}*{dt_pattern}*')[0] for var in varnames]
    
    if len(filename_collection) != len(varnames):
        raise ValueError(f'Year {year}, day {day_idx} has incomplete files')
    
    # Open with a lock to avoid race conditions when accessing files
    ds = xr.open_mfdataset(filename_collection, combine='by_coords', parallel=True, lock=netcdf_lock)

    # drop useless var
    ds = ds.drop_vars('utc_date', errors='ignore')

    # hourly --> 6 hourly
    ds = ds.isel(time=slice(0, -1, 6))
    
    #  chunking
    ds = ds.chunk(conf['RDA']['chunk_size_3d'])
    ds_list.append(ds)
    
# concatenate
ds_yearly = xr.concat(ds_list, dim='time')



# save
save_name = base_dir_save + conf['RDA']['prefix'] + '_surf_{}.zarr'.format(year)
ds_yearly.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)

print('...all done...')
