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
sys.path.append('Python_Modules_p3/')
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
int100 = True
if int100 == True:
    file_out  = 'rrexnumsb200_w100m_point_time_variability.nc' 
else:
    file_out  = 'rrexnumsb200_w_point_time_variability.nc'

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


# points for time variation
xrn, yrn = 314,656 # tracer 1
xrs, yrs = 278,134 # tracer 6
xap, yap = 800,406  
xlist    = [xrn,xrs,xap]
ylist    = [yrn,yrs,yap]


# --- create variables ---
w      = np.zeros((len(xlist),nt))

# --------------- read CROCO ---------------
tt = 0
for t_nc in range(len(time)):
    count_NL=0
    for t in range(ndfiles):
        data = Croco(name_exp,nbr_levels,str(int(time[t_nc])),name_exp_grd,name_pathdata)
        data.get_grid()
        data.get_outputs(t,var_list,get_date=False)
        data.get_zlevs()
        h_tile = np.transpose(np.tile(data.h,(200,1,1)),(1,2,0))
        hab    = h_tile + data.z_r
        u_rho = tools.u2rho(data.var['u']) #np.zeros([1002,802,200])
        v_rho = tools.v2rho(data.var['v'])
        for point in range(len(xlist)):
            if int100==False:
                wint         = 3600*24*toolsF.get_wvlcty(data.var['u'],data.var['v'],data.z_r,data.z_w,data.pm,data.pn)[xlist[point],ylist[point],:]
                wint[hab[xlist[point],ylist[point],:]>100]    = np.nan
                w[point,tt]  = np.nanmean(wint,axis=-1)
            else:
                # interpolate at hab = 100m
                depth = -data.h[xlist[point],ylist[point]]+100
                wint  = 3600*24*toolsF.get_wvlcty(data.var['u'],data.var['v'],data.z_r,data.z_w,data.pm,data.pn)
                w[point,tt] = tools.vinterp(wint,[depth],data.z_r,data.z_w)[xlist[point],ylist[point]]
        tt+=1

print('===================================================')
print(' ... save in netcdf file ... ')
print('===================================================')
nc = Dataset(file_out,'w')
nc.createDimension('nt',nt)
nc.createDimension('npoint',len(xlist))
nc.createVariable('w','f',('npoint','nt'))
nc.variables['w'][:]     = w
nc.close()








