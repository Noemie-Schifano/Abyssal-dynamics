'''
NS 2025/11/19: compute the error between w and ugradh over the domain, using different time-mean   
'''

import matplotlib
matplotlib.use('Agg') 
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors   as colors
import matplotlib.ticker   as ticker
from netCDF4 import Dataset
import sys
sys.path.append('/home/datawork-lops-rrex/nschifan/Python_Modules_p3/')
import R_tools as tools
import R_tools_fort as toolsF
import time as time
import calendar as cal
import datetime as datetime
from croco_simulations_jonathan_ncra_longrun import Croco_longrun
from croco_simulations_jonathan import Croco
from croco_simulations_jonathan_hist import Croco as Croco_hist
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
matplotlib.rcParams.update({'font.size': 20})
import gc

# ------------ parameters ------------ 
file_w_wu   = 'pathway2data/rrexnumsb200_w_notracer_time_'
name_exp    = 'rrexnum200'
name_exp_path ='rrexnums200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_NOFILT_T'
name_exp_grd  = ''
var_list      = ['zeta']
ptw_savefig   = '/home/datawork-lops-rrex/nschifan/Figures/WMT/error_w_wu_domain_subdomains_time.png'


Time     = ['1 day','7 days','30 days','60 days','90 days','120 days','150 days','180 days','220 days']
Timeline = [2,       2*7,     2*30,     2*2*30,   3*2*30,    4*2*30,     5*2*30, 6*2*30,  7*2*30]

choice = 1 
if choice == 0:
    time     = Time[0:3]
    timeline = Timeline[0:3]
    time_ncw = '1_30d'
elif choice == 1:
    time     = Time[3:6]
    timeline = Timeline[3:6]
    time_ncw = '2_4m'
else:
    time     = Time[6:9]
    timeline = Timeline[6:9]
    time_ncw = '5_7m'
nbin_t    = len(time)

# --- subdomains ---
#     imin, imax, jmin,jmax
sub0 = [0,502,0,402]
sub1 = [502,1002,0,402]
sub2 = [0,502,402,802]
sub3 = [502,1002,402,802]

subdomains = [ sub0,sub1,sub2,sub3 ]


# --- save netcdf ---
Test = False
if Test == True:
    file_out    = 'pathway2data/rrexnumsb200_err_w_wu_7months_test.nc'
    n_dom       = 2
else:
    file_out    = 'pathway2data/rrexnumsb200_err_w_wu_7months_'+time_ncw+'.nc'
    n_dom       = len(subdomains)

# --- plot options --- 
fs      = 20      # fontsize 
lon_0,lat_0= -32,57.5 # centre of the map a
extent     = [-37.5,-21.2,53,62.5]
jsec        = [150,450,650]
cjsec       = ['k','m','r']
jsec0       = jsec[1]
# ------------ bin for hab ------
h_e = np.array([ 0,   3,   6,   9,  12,  15,  18,  21,  26,36, 66,  96, 126, 156, 186, 216, 246, 276, 306, 336])
h_c = 0.5*(h_e[1:]+h_e[:-1]) # hab centres
nbin_z      = h_c.shape[0]


# ------------- function ------------
def moy(err, window_size):
    # Create a convolution kernel for a moving average
    kernel = np.ones(window_size) / window_size
    #Apply the convolution along the time axis (axis=2) using 'valid' mode to avoid edge effects
    err_moy = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='valid'), axis=2, arr=err)
    return err_moy


def define_var3D_at_area(var,points):
    # select a variable at the area define between points
    # points containes (x,y) coordinates of the limits of the square 
    return var[points[0]:points[1],points[2]:points[3],:]

# -- output error binning
err_bin = np.zeros((nbin_t, nbin_z), dtype=np.float32)

# --- get hab for binning 
# --------------- read CROCO ---------------
data = Croco(name_exp,'200',str(int(24)),name_exp_grd,name_pathdata)
data.get_grid()
data.get_outputs(0,var_list,get_date=False)
# ------------ read data ------------ 
[z_r,z_w]      = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
h_tile = np.transpose(np.tile(data.h,(200,1,1)),(1,2,0))
hab    = (h_tile + z_r)
del z_w, z_r, h_tile
gc.collect()

for idx in range(nbin_t):
    print("timeline", idx)
    # one container per vertical bin (exact equivalent to stats.binned_statistic)
    bin_err_values = [ [] for _ in range(nbin_z) ]
    t0  = timeline[idx]
    for dom in range(n_dom):
        sub = subdomains[dom]
        nc = Dataset(file_w_wu+str(dom)+'_7months.nc')
        w_var  = nc.variables["w"]
        wu_var = nc.variables["wu"]
        hab_var= define_var3D_at_area(hab,sub)[:,:,:w_var.shape[2]]    # profondeur (z), petit OK
        if Test == True:
            nk = 10
        else:
            nk = w_var.shape[2]
        for k in range(nk):
            w_k  = moy(w_var[:, :, k, :],  t0)
            wu_k = moy(wu_var[:, :, k, :], t0)

            err_k = 100.0 * np.abs(wu_k - w_k) / np.abs(w_k)

            # aplatissement
            err_r     = err_k.ravel()
            hab_var_k = hab_var[:,:,k]
            hab_r = np.resize(hab_var_k.ravel(), err_r.size).astype(np.float32) #np.full(err_r.size, hab_var_k.ravel(), dtype=np.float32)

            # bin attribution (IDENTICAL to scipy)
            bin_idx = np.digitize(hab_r, h_e) - 1

            for b in range(nbin_z):
                mask = bin_idx == b
                if np.any(mask):
                    bin_err_values[b].append(err_r[mask])
            del w_k, wu_k, err_k, err_r, hab_r
            gc.collect()

        nc.close()

    # ----- final median per bin (exact) -----
    for b in range(nbin_z):
        if bin_err_values[b]:
            err_bin[idx, b] = np.median(np.concatenate(bin_err_values[b]))
        else:
            err_bin[idx, b] = np.nan

    del bin_err_values
    gc.collect()


# --- create nc
nc_out = Dataset(file_out, "w", format="NETCDF4")
nc_out.createDimension("nbin_t", nbin_t)
nc_out.createDimension("nbin_z", nbin_z)
err_var = nc_out.createVariable(
    "err_var",
    "f4",
    ("nbin_t", "nbin_z"),
    zlib=True,
    complevel=4,
    chunksizes=(1, nbin_z)  # À ADAPTER
)
err_var[:, :] = err_bin
nc_out.close()

