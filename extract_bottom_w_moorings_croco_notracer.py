''''
NS 18/09/2025: interpolate w at morrings
               --> depth_cv is the "correct" depth
               --> new definition of hab 
'''

import matplotlib
matplotlib.use('Agg') #Choose the backend (needed for plotting inside subprocess)
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors   as colors
import matplotlib.ticker   as ticker
from netCDF4 import Dataset
import sys
sys.path.append('Python_Modules_p3/')
import R_tools as tools
import R_tools_fort as toolsF
import time as time
import calendar as cal
import datetime as datetime
from croco_simulations_noemie_hist_notracer import Croco
import gsw as gsw
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
matplotlib.rcParams.update({'font.size': 14})

# ------------ parameters in-situ ------------ 
file_data_d     = 'rrex_dD_10km.nc'
file_out        = 'rrex_bottom_w_mooring_notracer.nc'
choice_mooring  = ['IRW','IRM','IRE','RRT','ICW','ICM','ICE']
# --> collocato CROCO and moorings, from 'plot_data_Rene.py'
idx_lon, idx_lat= [191, 266, 309, 373, 438, 503, 599],[363, 387, 402, 416, 400, 385, 361]
depth_croco     = [-2107.22785688, -1604.09945298, -1522.02389565, -1451.25163171, -2109.34025253, -2213.65804133, -2381.34074426] # seafloor croco 
depth_cv        = [-2160, -1570, -1580, -1528, -2162, -2151, -2379]   # new depth from clement
#depth           = [-2059.787376721023, -1477.0246327724094, -1463.8085350144408, -1446.5276736037133, -2104.9496352372353, -2096.795333466324, -2299.77015273472] # deeper measure 
hab_moorings    = [100,93,116,81,57,46,282]
print(np.array(depth_croco) + np.array(hab_moorings))

# ------------ parameters croco ------------ 
name_exp    = 'rrexnum200' 
name_exp_path ='rrexnum200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_T'
nbr_levels  = '200'
name_exp_grd= ''
ndfiles     = 24  # number of days per netcdf
var_list    = ['zeta','u','v']
time        = np.arange(24,1430,24)
nt= len(time)*ndfiles

# --- plot options --- 
jsec    = 400
fs      = 20       # fontsize 
lw      = 2     # linewidth
ms      = 20      # markersize 
lw_c    = 0.2   # linewidth coast 
my_bbox = dict(fc='w',ec='k',pad=2,lw=0.,alpha=0.5)
res   = 'h' # resolution of the coastline: c (coarse), l (low), i (intermediate), h (high)
proj  = 'lcc'  # map projection. lcc = lambert conformal 
paral = np.arange(0,80,5)
merid = np.arange(0,360,5)
lon_0,lat_0   = -31.5,58.6 # centre of the map 
Lx,Ly         = 1400e3,1500e3 # [km] zonal and meridional extent of the map 
scale_km      = 400 # [km] map scale 
cf0           = colors.to_rgba('teal')

# --- w ---
label_w     = r'w [m day$^{-1}$]'

# --- create variables ---
w_irw = np.zeros(nt)
w_irm = np.zeros(nt)
w_ire = np.zeros(nt)
w_rrt = np.zeros(nt)
w_icw = np.zeros(nt)
w_icm = np.zeros(nt)
w_ice = np.zeros(nt)

wu_irw = np.zeros(nt)
wu_irm = np.zeros(nt)
wu_ire = np.zeros(nt)
wu_rrt = np.zeros(nt)
wu_icw = np.zeros(nt)
wu_icm = np.zeros(nt)
wu_ice = np.zeros(nt)

# ----------- read in-situ D ---------
ncd     = Dataset(file_data_d,'r')
D       = -ncd.variables['D'][:]
ncd.close()

# --- select depth to interpolate on CROCO
Dd = np.array(depth_croco) + np.array(hab_moorings)

# --------------- read CROCO ---------------
tt = 0
for t_nc in range(len(time)):
    count_NL=0
    for t in range(ndfiles):
        data = Croco(name_exp,nbr_levels,str(int(time[t_nc])),name_exp_grd,name_pathdata)
        data.get_grid()
        # ------------ make plot ------------ 
        print(tt)
        # -- valeurs min et max longitude et latitude
        # --- [ lon min, lon max, lat min, lat max]               
        extent     = [-37.5,-21.2,53,62.5] # °E °N
        data.get_outputs(t,var_list,get_date=False)
        print(data.h[idx_lon,idx_lat])
        print(depth_croco)
        # ------------ read data ------------ 
        [z_r,z_w]   = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
        dhdx   = tools.u2rho(data.h[1:,:]-data.h[:-1,:])*data.pm
        dhdy   = tools.v2rho(data.h[:,1:]-data.h[:,:-1])*data.pn
        dhdx3d = np.transpose(np.tile(dhdx,(200,1,1)),(1,2,0))
        dhdy3d = np.transpose(np.tile(dhdy,(200,1,1)),(1,2,0))
        w           = 3600*24*toolsF.get_wvlcty(data.var['u'],data.var['v'],z_r,z_w,data.pm,data.pn)
        wu          = -3600*24*(tools.u2rho(data.var['u'])*dhdx3d+tools.v2rho(data.var['v'])*dhdy3d)
        w_irw[tt] = tools.vinterp(w,[Dd[0]],z_r,z_w)[idx_lon[0],idx_lat[0]]
        w_irm[tt] = tools.vinterp(w,[Dd[1]],z_r,z_w)[idx_lon[1],idx_lat[1]]
        w_ire[tt] = tools.vinterp(w,[Dd[2]],z_r,z_w)[idx_lon[2],idx_lat[2]]
        w_rrt[tt] = tools.vinterp(w,[Dd[3]],z_r,z_w)[idx_lon[3],idx_lat[3]]
        w_icw[tt] = tools.vinterp(w,[Dd[4]],z_r,z_w)[idx_lon[4],idx_lat[4]]
        w_icm[tt] = tools.vinterp(w,[Dd[5]],z_r,z_w)[idx_lon[5],idx_lat[5]]
        w_ice[tt] = tools.vinterp(w,[Dd[6]],z_r,z_w)[idx_lon[6],idx_lat[6]]

        wu_irw[tt] = tools.vinterp(wu,[Dd[0]],z_r,z_w)[idx_lon[0],idx_lat[0]]
        print('irw',wu_irw[tt])
        wu_irm[tt] = tools.vinterp(wu,[Dd[1]],z_r,z_w)[idx_lon[1],idx_lat[1]]
        wu_ire[tt] = tools.vinterp(wu,[Dd[2]],z_r,z_w)[idx_lon[2],idx_lat[2]]
        wu_rrt[tt] = tools.vinterp(wu,[Dd[3]],z_r,z_w)[idx_lon[3],idx_lat[3]]
        wu_icw[tt] = tools.vinterp(wu,[Dd[4]],z_r,z_w)[idx_lon[4],idx_lat[4]]
        wu_icm[tt] = tools.vinterp(wu,[Dd[5]],z_r,z_w)[idx_lon[5],idx_lat[5]]
        wu_ice[tt] = tools.vinterp(wu,[Dd[6]],z_r,z_w)[idx_lon[6],idx_lat[6]]

        tt+=1

print('===================================================')
print(' ... save in netcdf file ... ')
print('===================================================')
nc = Dataset(file_out,'w')
nc.createDimension('nt',int(nt))
nc.createVariable('wb0','f',('nt'))
nc.createVariable('wb1','f',('nt'))
nc.createVariable('wb2','f',('nt'))
nc.createVariable('wb3','f',('nt'))
nc.createVariable('wb4','f',('nt'))
nc.createVariable('wb5','f',('nt'))
nc.createVariable('wb6','f',('nt'))

nc.createVariable('wub0','f',('nt'))
nc.createVariable('wub1','f',('nt'))
nc.createVariable('wub2','f',('nt'))
nc.createVariable('wub3','f',('nt'))
nc.createVariable('wub4','f',('nt'))
nc.createVariable('wub5','f',('nt'))
nc.createVariable('wub6','f',('nt'))


nc.variables['wb0'][:]     = w_irw
nc.variables['wb1'][:]     = w_irm
nc.variables['wb2'][:]     = w_ire
nc.variables['wb3'][:]     = w_rrt
nc.variables['wb4'][:]     = w_icw
nc.variables['wb5'][:]     = w_icm
nc.variables['wb6'][:]     = w_ice

nc.variables['wub0'][:]     = wu_irw
nc.variables['wub1'][:]     = wu_irm
nc.variables['wub2'][:]     = wu_ire
nc.variables['wub3'][:]     = wu_rrt
nc.variables['wub4'][:]     = wu_icw
nc.variables['wub5'][:]     = wu_icm
nc.variables['wub6'][:]     = wu_ice


nc.close()





















