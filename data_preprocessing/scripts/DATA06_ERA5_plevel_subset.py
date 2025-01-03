'''
This script subsets 1.0 degree ERA5 to 13 levels.

Yingkai Sha
ksha@ucar.edu
'''

import os
import sys
import yaml
import dask
import zarr
import xesmf as xe
import numpy as np
import xarray as xr
from glob import glob

sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu
import interp_utils as iu

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

level_sub = [1, 50, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
conf['zarr_opt']['chunk_size_4d_1deg']['level'] = len(level_sub)

base_dir_1deg = conf['zarr_opt']['save_loc_1deg']

name_full_level = base_dir_1deg + 'all_in_one/ERA5_plevel_1deg_6h_{}_conserve.zarr'.format(year)

ds_full = xr.open_zarr(name_full_level)
subset_inds = [np.where(ds_full['level'].values == val)[0][0] for val in level_sub]
ds_sub = ds_full.isel(level=subset_inds)

# ========================================================================== #
# chunking
varnames = list(ds_sub.keys())
varname_4D = ['U', 'V', 'T', 'Z', 'Q', 'specific_total_water']

for i_var, var in enumerate(varnames):
    if var in varname_4D:
        ds_sub[var] = ds_sub[var].chunk(conf['zarr_opt']['chunk_size_4d_1deg'])
    else:
        ds_sub[var] = ds_sub[var].chunk(conf['zarr_opt']['chunk_size_3d_1deg'])

# zarr encodings
dict_encoding = {}

chunk_size_3d = dict(chunks=(conf['zarr_opt']['chunk_size_3d_1deg']['time'],
                             conf['zarr_opt']['chunk_size_3d_1deg']['latitude'],
                             conf['zarr_opt']['chunk_size_3d_1deg']['longitude']))

chunk_size_4d = dict(chunks=(conf['zarr_opt']['chunk_size_4d_1deg']['time'],
                             conf['zarr_opt']['chunk_size_4d_1deg']['level'],
                             conf['zarr_opt']['chunk_size_4d_1deg']['latitude'],
                             conf['zarr_opt']['chunk_size_4d_1deg']['longitude']))

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    if var in varname_4D:
        dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
    else:
        dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

# ========================================================================== #
save_name = base_dir_1deg + 'upper_subset/ERA5_subset_1deg_6h_{}_conserve.zarr'.format(year)
ds_sub.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)
