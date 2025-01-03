'''
1-deg interpolation using conservertive interpolation.

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
zarr_name_surf = base_dir+'surf/ERA5_plevel_6h_surf_{}.zarr'
zarr_name_surf_extra = base_dir+'surf/ERA5_plevel_6h_surf_extend_{}.zarr'
zarr_name_accum = base_dir+'accum/ERA5_plevel_6h_accum_{}.zarr'
zarr_name_forcing = base_dir+'forcing/ERA5_plevel_6h_forcing_{}.zarr'
zarr_name_upper = base_dir+'upper_air/ERA5_plevel_6h_upper_air_{}.zarr'
zarr_name_upper_Q = base_dir+'upper_air/ERA5_plevel_6h_Q_{}.zarr'
zarr_name_static = base_dir+'static/ERA5_plevel_6h_static.zarr'

ds_surf = xr.open_zarr(zarr_name_surf.format(year))
ds_surf_extra = xr.open_zarr(zarr_name_surf_extra.format(year))
ds_accum = xr.open_zarr(zarr_name_accum.format(year))
ds_forcing = xr.open_zarr(zarr_name_forcing.format(year))
ds_upper = xr.open_zarr(zarr_name_upper.format(year))
ds_upper_Q = xr.open_zarr(zarr_name_upper_Q.format(year))
ds_static = xr.open_zarr(zarr_name_static)

# merge all and drop SST (this var is not needed)
ds_merge = xr.merge([ds_surf, ds_accum, ds_forcing, ds_upper, ds_upper_Q, ds_surf_extra])
ds_merge = ds_merge.drop_vars('SSTK')

# ======================================================================================= #
# 0.25 deg to 1 deg interpolation using conservative approach

# Define the target 1-degree grid
lon_1deg = np.arange(0, 360, 1)
lat_1deg = np.arange(-90, 91, 1)
target_grid = iu.Grid.from_degrees(lon_1deg, lat_1deg)

lon_025deg = ds_merge['longitude'].values
lat_025deg = ds_merge['latitude'].values[::-1]
source_grid = iu.Grid.from_degrees(lon_025deg, lat_025deg)

regridder = iu.ConservativeRegridder(source=source_grid, target=target_grid)

ds_merge = ds_merge.chunk({'longitude': -1, 'latitude': -1})
ds_merge_1deg = regridder.regrid_dataset(ds_merge)

# Reorder the dimensions for all variables in ds_merge_1deg
for var in ds_merge_1deg.data_vars:
    # Get the current dimensions of the variable
    current_dims = ds_merge_1deg[var].dims
    
    # If both 'latitude' and 'longitude' are present, reorder them
    if 'latitude' in current_dims and 'longitude' in current_dims:
        # New order: move 'latitude' and 'longitude' to the first two positions, preserve other dimensions
        new_order = [dim for dim in current_dims if dim not in ['latitude', 'longitude']] + ['latitude', 'longitude']
        
        # Transpose the variable to the new order
        ds_merge_1deg[var] = ds_merge_1deg[var].transpose(*new_order)

lon_1deg = np.arange(0, 360, 1)
lat_1deg = np.arange(-90, 91, 1)

# Add latitude and longitude as coordinates to ds_merge_1deg
ds_merge_1deg = ds_merge_1deg.assign_coords({
    'latitude': lat_1deg,
    'longitude': lon_1deg
})

# flip latitude from -90 --> 90 to 90 --> -90
ds_merge_1deg = ds_merge_1deg.isel(latitude=slice(None, None, -1))

# float64 --> float32
ds_merge_1deg = ds_merge_1deg.astype({var: np.float32 for var in ds_merge_1deg if ds_merge_1deg[var].dtype == np.float64})

# ======================================================================================= #
# process land-sea mask and sea ice
land_sea_mask = ds_static['land_sea_mask']

# Define the target 1-degree grid
lon_1deg = np.arange(0, 360, 1)
lat_1deg = np.arange(-90, 91, 1)[::-1]

# Create target grid as an xarray Dataset
ds_out = xr.Dataset(
    {
        'latitude': (['latitude'], lat_1deg),
        'longitude': (['longitude'], lon_1deg)
    }
)

# Create the regridder object for bilinear interpolation
regridder = xe.Regridder(ds_merge, ds_out, 'nearest_s2d')

# Apply the regridding to interpolate all variables
land_sea_mask_1deg = regridder(land_sea_mask)

# combine CI and land-sea mask
sea_ice_cover = ds_merge_1deg['CI']
land_sea_mask_expanded = land_sea_mask_1deg.broadcast_like(sea_ice_cover)

land_sea_CI_mask = xr.where(
    (land_sea_mask_expanded == 0) & (sea_ice_cover > 0),
    -sea_ice_cover,
    land_sea_mask_expanded
)

ds_merge_1deg['land_sea_CI_mask'] = land_sea_CI_mask
ds_merge_1deg = ds_merge_1deg.drop_vars('CI')

# Convert latitude, longitude, and level coordinates to float32
ds_merge_1deg = ds_merge_1deg.assign_coords({
    'latitude': ds_merge_1deg['latitude'].astype(np.float32),
    'longitude': ds_merge_1deg['longitude'].astype(np.float32),
    'level': ds_merge_1deg['level'].astype(np.float32)
})

# ========================================================================== #
# chunking
varnames = list(ds_merge_1deg.keys())
varname_4D = ['U', 'V', 'T', 'Z', 'Q', 'specific_total_water']

for i_var, var in enumerate(varnames):
    if var in varname_4D:
        ds_merge_1deg[var] = ds_merge_1deg[var].chunk(conf['zarr_opt']['chunk_size_4d_1deg'])
    else:
        ds_merge_1deg[var] = ds_merge_1deg[var].chunk(conf['zarr_opt']['chunk_size_3d_1deg'])

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
# save
save_name = base_dir_1deg + 'all_in_one/ERA5_plevel_1deg_6h_{}_conserve.zarr'.format(year)
ds_merge_1deg.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)

