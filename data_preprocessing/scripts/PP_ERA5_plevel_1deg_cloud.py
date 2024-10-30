'''
1-deg interpolation

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
    
# ==================================================================================== #

base_dir = conf['zarr_opt']['save_loc']
base_dir_1deg = conf['zarr_opt']['save_loc_1deg']

# load all 0.25 deg ERA5 data
zarr_name_cloud = base_dir+'cloud/ERA5_plevel_6h_cloud_{}.zarr'
ds_cloud = xr.open_zarr(zarr_name_cloud.format(year))

# ======================================================================================= #
# 0.25 deg to 1 deg interpolation using conservative approach

# Define the target 1-degree grid
lon_1deg = np.arange(0, 360, 1)
lat_1deg = np.arange(-90, 91, 1)
target_grid = iu.Grid.from_degrees(lon_1deg, lat_1deg)

lon_025deg = ds_cloud['longitude'].values
lat_025deg = ds_cloud['latitude'].values[::-1]
source_grid = iu.Grid.from_degrees(lon_025deg, lat_025deg)

regridder = iu.ConservativeRegridder(source=source_grid, target=target_grid)

ds_cloud = ds_cloud.chunk({'longitude': -1, 'latitude': -1})
ds_cloud_1deg = regridder.regrid_dataset(ds_cloud)

# Reorder the dimensions for all variables
for var in ds_cloud_1deg.data_vars:
    # Get the current dimensions of the variable
    current_dims = ds_cloud_1deg[var].dims
    
    # If both 'latitude' and 'longitude' are present, reorder them
    if 'latitude' in current_dims and 'longitude' in current_dims:
        # New order: move 'latitude' and 'longitude' to the first two positions, preserve other dimensions
        new_order = [dim for dim in current_dims if dim not in ['latitude', 'longitude']] + ['latitude', 'longitude']
        
        # Transpose the variable to the new order
        ds_cloud_1deg[var] = ds_cloud_1deg[var].transpose(*new_order)

lon_1deg = np.arange(0, 360, 1)
lat_1deg = np.arange(-90, 91, 1)

# Add latitude and longitude as coordinates to ds_cloud_1deg
ds_cloud_1deg = ds_cloud_1deg.assign_coords({
    'latitude': lat_1deg,
    'longitude': lon_1deg
})

# flip latitude from -90 --> 90 to 90 --> -90
ds_cloud_1deg = ds_cloud_1deg.isel(latitude=slice(None, None, -1))

# float64 --> float32
ds_cloud_1deg = ds_cloud_1deg.astype(
    {var: np.float32 for var in ds_cloud_1deg if ds_cloud_1deg[var].dtype == np.float64})

# Convert latitude, longitude, and level coordinates to float32
ds_cloud_1deg = ds_cloud_1deg.assign_coords({
    'latitude': ds_cloud_1deg['latitude'].astype(np.float32),
    'longitude': ds_cloud_1deg['longitude'].astype(np.float32),
    'level': ds_cloud_1deg['level'].astype(np.float32)
})

# ========================================================================== #
# chunking
varnames = list(ds_cloud_1deg.keys())

for i_var, var in enumerate(varnames):
    ds_cloud_1deg[var] = ds_cloud_1deg[var].chunk(conf['zarr_opt']['chunk_size_4d_1deg'])

# zarr encodings
dict_encoding = {}

chunk_size_4d = dict(chunks=(conf['zarr_opt']['chunk_size_4d_1deg']['time'],
                             conf['zarr_opt']['chunk_size_4d_1deg']['level'],
                             conf['zarr_opt']['chunk_size_4d_1deg']['latitude'],
                             conf['zarr_opt']['chunk_size_4d_1deg']['longitude']))

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    dict_encoding[var] = {'compressor': compress, **chunk_size_4d}

# ========================================================================== #
# save
save_name = base_dir_1deg + 'cloud/ERA5_plevel_1deg_6h_cloud_{}_conserve.zarr'.format(year)
ds_cloud_1deg.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)
