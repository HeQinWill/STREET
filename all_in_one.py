import xarray as xr
import skgstat as skg
import numpy as np
import proplot as pplt

slave_nc_file='./slave_tropomi_LA.nc'
master_nc_file = './master_tropomi_LA.nc'

slave_ds = xr.open_dataset(slave_nc_file)
master_ds = xr.open_dataset(master_nc_file)

model = 'gaussian'  # 'gaussian', 'spherical', 'exponential', 'stable'
maxlag = 3  # maxlag (float): the maximum lag distance (default to 5 degree ~ 550 km)
n_bins = 200  # n_bins (int): number of bins in semivariogram
deg2km = 110.0  # degree to km conversion (default = 110 km)

#%% semivariogram calculation
def cal_sem(x, y, z, ds, random_selection_n=1000):
    # prepocess the data
    df = ds[[x, y, z]].to_dataframe().reset_index()
    df.dropna(inplace=True)
    if random_selection_n is not None:
        df = df.sample(random_selection_n)
    
    # running the skg (this can be memory intensive if the field is large)
    vario_obj = skg.Variogram(np.array(df[[x, y]]), np.array(df[z]),
                                n_lags=n_bins,
                                estimator='matheron',
                                model=model,
                                maxlag=maxlag)
    # fitted model
    fitted_model = vario_obj.fitted_model
    # returning the goods
    return vario_obj, fitted_model

print("Building and modeling semivariogram for the slave")
vario_obj_slave,fitted_model_slave = cal_sem('lon', 'lat', 'values', slave_ds)
print("Building and modeling semivariogram for the master")
vario_obj_master,fitted_model_master = cal_sem('lon', 'lat', 'values', master_ds)

#%% Plot the semivariogram
fig, axs = pplt.subplots(nrows=1, ncols=2)
vario_obj_slave.plot(axes=axs[0])
vario_obj_master.plot(axes=axs[1])
# fig.savefig('semivariogram.png', dpi=300)

#%% error estimator
# the range of x for error estimation    
x_value = np.arange(0, maxlag+0.1, 0.1)

# get their gamma values (i.e., variance)
var_slave = fitted_model_slave(x_value)
var_master = fitted_model_master(x_value)

# the error
spatial_rep_err = 1 - var_slave/var_master

# Plot the error
fig, axs = pplt.subplots(nrows=1, ncols=1)
axs.plot(x_value*deg2km,
        spatial_rep_err*100.0,
        linewidth=3,
        color='purple')
axs.format(title='Spatial Representation Error',
            xlabel='Length Scale [km]',
            ylabel='Loss of Spatial Information [%]',
            xlim=(0, maxlag*deg2km))

#%% estimating the error for a given length scale
length_scale = 55  # unit: km, the user can get a value for a given length scale instead of estimating the error for a wide range
spatial_rep_err_spc = 100.0*(1 - fitted_model_slave(length_scale/deg2km)\
       /fitted_model_master(length_scale/deg2km))
print(spatial_rep_err_spc)
