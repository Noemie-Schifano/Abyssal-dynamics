'''
NS 2025/11/19: plot the error between w and ugradh over the domain, using different time-mean   
                --> need 'precompute_appendixB.py' to be run first to generate the data
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


# ------------ parameters ------------ 
name_exp    = 'rrexnum200' 
name_exp_grd  = ''
name_exp_path ='rrexnums200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_NOFILT_T'
ptw_savefig   = 'pathway2fig/appendixB.png'

time     = ['1 day','7 days','30 days','60 days','90 days','120 days','150 days','180 days','220 days']
timeline = [2,       2*7,     2*30,     2*2*30,   3*2*30,    4*2*30,     5*2*30, 6*2*30,  7*2*30]
pthw_err       = 'pathway2data/
file1 = pthw_err + 'rrexnumsb200_err_w_wu_7months_2_4m.nc'
file2 = pthw_err + 'rrexnumsb200_err_w_wu_7months_5_7m.nc'
file0 = pthw_err + 'rrexnumsb200_err_w_wu_7months_1_30d.nc'
files = [file0,file1,file2]

ntime    = len(time)

# --- plot options --- 
fs      = 22      
lw      = 2.8
ls      = ['solid','solid','solid','dashdot','dashdot','dashdot','dashed','dashed','dashed' ]
ctf     = ['#000000','#2E8B57','#FF8C00',
           '#000000','#2E8B57','#FF8C00',
           '#000000','#2E8B57','#FF8C00']

# ------------ bin for hab ------
h_e = np.array([ 0,   3,   6,   9,  12,  15,  18,  21,  26,36, 66,  96, 126, 156, 186, 216, 246, 276, 306, 336])
h_c = 0.5*(h_e[1:]+h_e[:-1]) # hab centres
nbin_z      = h_c.shape[0]


# ------------ read error in subdomains ---
err = np.zeros((ntime,nbin_z))

# err [time, nbin_hab]
nc   = Dataset(file0,'r')
err[0:3,:]  = nc.variables['err_var'][:,:]
nc.close()
nc1   = Dataset(file1,'r')
err[3:6,:]  = nc1.variables['err_var'][:,:]
nc1.close()
nc2   = Dataset(file2,'r')
err[6:9,:]  = nc2.variables['err_var'][:,:]
nc2.close()


# --- plot ---
print('------------- MAKE PLOT ------------')
plt.figure(figsize=(15,15))
plt.title('rrexnumsb200',fontsize=fs)
for u in range(ntime):
    plt.plot(err[u,:],h_c,linewidth=lw,linestyle=ls[u],c=ctf[u],label=time[u])
plt.axhline(y=10,c='k',linewidth=0.2,alpha=0.7)
for iw in range(0,110,10):
    plt.axvline(x=iw,c='k',linewidth=0.2,alpha=0.7)
for iw in range(50,150,50):
    plt.axvline(x=iw,c='k',linewidth=1.,alpha=0.7)
for jw in range(100,400,100):
    plt.axhline(y=jw,c='k',linewidth=0.2,alpha=0.7)
plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.15),prop={'size':fs},ncol=5)
#plt.yscale('log')
plt.xticks(rotation=30)
plt.ylabel(r'hab [m]',fontsize=fs)
plt.xlabel(r'Err [$\%$]',fontsize=fs)

plt.savefig(ptw_savefig,dpi=200,bbox_inches='tight')
plt.close()
