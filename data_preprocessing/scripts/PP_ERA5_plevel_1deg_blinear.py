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

ds_surf = xr.open_zarr(zarr_name_surf.format(year))
ds_surf_extra = xr.open_zarr(zarr_name_surf_extra.format(year))
ds_accum = xr.open_zarr(zarr_name_accum.format(year))
ds_forcing = xr.open_zarr(zarr_name_forcing.format(year))
ds_upper = xr.open_zarr(zarr_name_upper.format(year))
ds_upper_Q = xr.open_zarr(zarr_name_upper_Q.format(year))

# --------------------------------------------------------------------- #
# combining land-sea mask and sea-ice cover
ds_static = xr.open_zarr(base_dir+'static/ERA5_plevel_6h_static.zarr')
land_sea_mask = ds_static['land_sea_mask']
sea_ice_cover = ds_surf_extra['CI']

# land = 1, ocean = 0, sea-ice = -1~0 (depends on the ice cover)
land_sea_mask_expanded = land_sea_mask.broadcast_like(sea_ice_cover)
land_sea_CI_mask = xr.where(
    (land_sea_mask_expanded == 0) & (sea_ice_cover > 0),
    -sea_ice_cover,
    land_sea_mask_expanded
)

land_sea_CI_mask.name = 'land_sea_CI_mask'
ds_mask = land_sea_CI_mask.to_dataset()
# --------------------------------------------------------------------- #

# merge all and drop SST (this var is not needed)
ds_merge = xr.merge([ds_surf, ds_accum, ds_forcing, ds_upper, ds_upper_Q, ds_mask,])
ds_merge = ds_merge.drop_vars('SSTK')

# ========================================================================== #
# define chunk sizes
varnames = list(ds_merge.keys())
varname_4D = ['U', 'V', 'T', 'Z', 'Q', 'specific_total_water']

for i_var, var in enumerate(varnames):
    if var in varname_4D:
        ds_merge[var] = ds_merge[var].chunk(conf['zarr_opt']['chunk_size_4d_1deg'])
    else:
        ds_merge[var] = ds_merge[var].chunk(conf['zarr_opt']['chunk_size_3d_1deg'])

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
# Interpolate to 1-degree resolution using xESMF

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
regridder = xe.Regridder(ds_merge, ds_out, 'bilinear')

# Apply the regridding to interpolate all variables
ds_merge_1deg = regridder(ds_merge)

# Post-process 'land_sea_CI_mask' to ensure ocean=0, land=1, sea-ice=-1~0 after interpolation
land_sea_CI_mask_interp = ds_merge_1deg['land_sea_CI_mask']

# Apply the following logic:
# - If value >= 0.5, set to 1 (land)
# - If value <= -0.01, keep as is (sea-ice)
# - Else, set to 0 (ocean)
land_sea_CI_mask_interp = xr.where(
    land_sea_CI_mask_interp >= 0.5,
    1,
    xr.where(
        land_sea_CI_mask_interp <= -0.01,
        land_sea_CI_mask_interp,
        0
    )
)

# Update the dataset
ds_merge_1deg['land_sea_CI_mask'] = land_sea_CI_mask_interp

save_name = base_dir_1deg + 'all_in_one/ERA5_plevel_1deg_6h_{}_bilinear.zarr'.format(year)
ds_merge_1deg.to_zarr(
    save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding
)