''''
NS 2024/04/12: extract time serie of bottom vertical velocity with snapshot   
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
sys.path.append('/home/datawork-lops-rrex/nschifan/Python_Modules_p3/')
import R_tools as tools
import R_tools_fort as toolsF
import time as time
import calendar as cal
import datetime as datetime
from croco_simulations_jonathan_hist import Croco
import gsw as gsw
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
matplotlib.rcParams.update({'font.size': 14})

# ------------ parameters in-situ ------------ 
choice_mooring  = ['IRW','IRM','IRE','RRT','ICW','ICM','ICE']
file_out        = '/home/datawork-lops-rrex/nschifan/Data_in_situ_Rene/rrexnumsb200_bottom_w.nc'
# --> collocato CROCO and moorings, from 'plot_data_Rene.py'
idx_lon, idx_lat= [191, 266, 309, 373, 438, 503, 599],[363, 387, 402, 416, 400, 385, 361]
depth_croco     = [-2107.22785688, -1604.09945298, -1522.02389565, -1451.25163171, -2109.34025253, -2213.65804133, -2381.34074426] # seafloor croco 
depth           = [-2059.787376721023, -1477.0246327724094, -1463.8085350144408, -1446.5276736037133, -2104.9496352372353, -2096.795333466324, -2299.77015273472] # deeper measure 
hab_moorings    = [3,1,3,2,5,2,2] # hab of deeper measure from moorings 

# ------------ parameters croco ------------ 
name_exp    = 'rrexnum200' 
name_exp_path ='rrexnum200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_NOFILT_T'
nbr_levels  = '200'
name_exp_grd= ''
ndfiles     = 2  # number of days per netcdf
var_list    = ['zeta','u','v']
time1       = np.arange(22,60,2)
time2       = np.arange(62,471,2)
time        =np.zeros(len(np.arange(22,471,2))-1)
time[0:len(time1)] = time1
time[len(time1):] = time2
nt= len(time)*ndfiles

# --- create variables ---
w      = np.zeros((1002,802,nt))
wu     = np.zeros((1002,802,nt))

# --------------- read CROCO ---------------
tt = 0
for t_nc in range(len(time)):
    count_NL=0
    for t in range(ndfiles):
        data = Croco(name_exp,nbr_levels,str(int(time[t_nc])),name_exp_grd,name_pathdata)
        data.get_grid()
        # ------------ make plot ------------ 
        extent     = [-37.5,-21.2,53,62.5] # °E °N
        data.get_outputs(t,var_list,get_date=False)
        # ------------ read data ------------ 
        [z_r,z_w]      = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
        dhdx   = tools.u2rho(data.h[1:,:]-data.h[:-1,:])*data.pm
        dhdy   = tools.v2rho(data.h[:,1:]-data.h[:,:-1])*data.pn
        dhdx3d = np.transpose(np.tile(dhdx,(200,1,1)),(1,2,0))
        dhdy3d = np.transpose(np.tile(dhdy,(200,1,1)),(1,2,0))
        # --> compute w and w from u 
        wu[:,:,tt]     = -3600*24*(tools.u2rho(data.var['u'])*dhdx3d + tools.v2rho(data.var['v'])*dhdy3d)[:,:,0]
        w[:,:,tt]      = 3600*24*toolsF.get_wvlcty(data.var['u'],data.var['v'],z_r,z_w,data.pm,data.pn)[:,:,0]
        tt+=1

print('===================================================')
print(' ... save in netcdf file ... ')
print('===================================================')
nc = Dataset(file_out,'w')
nc.createDimension('nlon',1002)
nc.createDimension('nlat',802)
nc.createDimension('nt',nt)
nc.createVariable('w','f',('nlon','nlat','nt'))
nc.createVariable('wu','f',('nlon','nlat','nt'))
nc.variables['w'][:]     = w
nc.variables['wu'][:]     = w
nc.close()








