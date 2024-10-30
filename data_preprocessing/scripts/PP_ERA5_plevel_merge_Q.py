'''
This script combines all Q components

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

chunk_size_4d = dict(chunks=(conf['zarr_opt']['chunk_size_4d']['time'],
                             conf['zarr_opt']['chunk_size_4d']['level'],
                             conf['zarr_opt']['chunk_size_4d']['latitude'],
                             conf['zarr_opt']['chunk_size_4d']['longitude']))

base_dir = conf['zarr_opt']['save_loc']
zarr_name_upper = sorted(glob(base_dir+'upper_air/ERA5_plevel_6h_upper_air_*.zarr'))
zarr_name_cloud = sorted(glob(base_dir+'cloud/*.zarr'))

fn_upper = [fn for fn in zarr_name_upper if str(year) in fn][0]
fn_cloud = [fn for fn in zarr_name_cloud if str(year) in fn][0]

variables_levels = {}
variables_levels['Q'] = None

ds_upper = xr.open_zarr(fn_upper).chunk(conf['zarr_opt']['chunk_size_4d'])
ds_upper = vu.ds_subset_everything(ds_upper, variables_levels)

ds_cloud = xr.open_zarr(fn_cloud).chunk(conf['zarr_opt']['chunk_size_4d'])

ds_upper['Q'] = ds_upper['Q'] + ds_cloud['CLWC'] + ds_cloud['CRWC']
ds_upper = ds_upper.rename({'Q': 'specific_total_water'})

dict_encoding = {}
compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)
dict_encoding['specific_total_water'] = {'compressor': compress, **chunk_size_4d}

save_name = base_dir+'upper_air/ERA5_plevel_6h_Q_{}.zarr'.format(year)

ds_upper.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)





