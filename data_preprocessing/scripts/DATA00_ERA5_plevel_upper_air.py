'''
This script collects ERA5 pressure-level analysis from NCAR/RDA.
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
from glob import glob
from datetime import datetime, timedelta
from dask.utils import SerializableLock

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
N_days = 366 if year % 4 == 0 else 365

# ==================================================================================== #
# import variable name and save location form yaml
config_name = os.path.realpath('../data_config_6h.yml')

with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

# ==================================================================================== #
# the sub-folder to store data
base_dir_save = conf['RDA']['save_loc'] + 'upper_air/' 
if not os.path.exists(base_dir_save):
    os.makedirs(base_dir_save)

# ==================================================================================== #
# encoding options
compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

chunk_size_4d = dict(chunks=(conf['RDA']['chunk_size_4d']['time'],
                             conf['RDA']['chunk_size_4d']['level'],
                             conf['RDA']['chunk_size_4d']['latitude'],
                             conf['RDA']['chunk_size_4d']['longitude']))

dict_encoding = {}

for i_var, var in enumerate(conf['RDA']['varname_upper_air']):
    dict_encoding[var] = {'compressor': compress, **chunk_size_4d}

# ==================================================================================== #
# main
xr.set_options(file_cache_maxsize=500) # increase the file cache size
netcdf_lock = SerializableLock() # lock for safe parallel access

# all days within a year
start_time = datetime(year, 1, 1, 0, 0)
dt_list = [start_time + timedelta(days=i) for i in range(N_days)]

# upper-air var names
varnames = list(conf['RDA']['varname_upper_air'].values())

# collect 6 hourly ds on each day
ds_list = []

for i_day, dt in enumerate(dt_list):
    # file source info
    base_dir = dt.strftime(conf['RDA']['source']['anpl_format'])
    dt_pattern = dt.strftime(conf['RDA']['source']['anpl_dt_pattern_format'])

    # get upper-air vars
    filename_collection = [glob(base_dir + f'*{var}*{dt_pattern}*')[0] for var in varnames]
    
    if len(filename_collection) != len(varnames):
        raise ValueError(f'Year {year}, day {day_idx} has incomplete files')
    
    # open mf with lock
    ds = xr.open_mfdataset(filename_collection, combine='by_coords', parallel=True, lock=netcdf_lock)

    # drop useless var
    ds = ds.drop_vars('utc_date', errors='ignore')

    # hourly --> 6 hourly
    ds = ds.isel(time=slice(0, -1, 6))
    
    #  chunking
    ds = ds.chunk(conf['RDA']['chunk_size_4d'])
    ds_list.append(ds)
    
# concatenate
ds_yearly = xr.concat(ds_list, dim='time')

# save
save_name = base_dir_save + conf['RDA']['prefix'] + '_upper_air_{}.zarr'.format(year)
ds_yearly.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)

print('...all done...')



