'''
NS 2022/10/10: is negative concentration ??   
'''

import matplotlib
matplotlib.use('Agg') #Choose the backend (needed for plotting inside subprocess)
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['text.usetex'] = True
import matplotlib.gridspec as gridspec
import matplotlib.colors   as colors
import matplotlib.ticker   as ticker
from netCDF4 import Dataset
import sys
sys.path.append('/home/datawork-lops-rrex/nschifan/Python_Modules_p3-master/')

from Modules import *
from Modules_gula import *
import R_tools as tools
import R_vars_gula as toolsvarg
import R_tools_fort as toolsF
import R_tools_fort_gula as toolsF_g
import time as time
import calendar as cal
import datetime as datetime
from croco_simulations_jonathan_ncra_longrun import Croco_longrun
#from croco_simulations_hist import Croco_hist
import cartopy.crs as ccrs
import gsw as gsw
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
matplotlib.rcParams.update({'font.size': 14})



# ------------ parameters ------------ 
file_out        = '/home/datawork-lops-rrex/nschifan/Data_in_situ_Rene/BBL_height_N2.nc'
name_exp    = 'rrexnum200' #['rrex100-up3','rrex100-up5','rrex100-weno5','rrex200-up3','rrex200-up5','rrex200-weno5','rrex300-up3','rrex300-500cpu-up5','rrex300-500cpu-weno5']
name_exp_path ='rrexnums200_rsup5'#-rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_NOFILT_T'#RSUP5_NOFILT_T'
name_exp_grd= ''
nbr_levels  = '200'
name_exp_saveup = name_exp_path
var_list      = ['zeta','bvf','u','v','w']
time          =  ['40']
ndfiles     = 1  # number of days per netcdf
nt          = ndfiles*len(time)

# --- plot options --- 
fs          = 14      # fontsize 
lwh         = 2
lon_0,lat_0 = -32,57.5 # centre of the map a
extent      = [-37.5,-21.2,53,62.5]
jsec        = [150,450,650]
cjsec       = ['k','m','r']
jsec0       = jsec[1]
cjsec0      = cjsec[1]

# --- N2 ---
pmin,pmax,pint = -1e-5,1e-5,1e-8
cmap_N2        = plt.cm.bwr
norm_N2        = colors.SymLogNorm(linthresh=pint, linscale=1, vmin=pmin, vmax=pmax,base=10) #colors.Normalize(vmin=pmin,vmax=pmax)
cbticks_N2     = [pmin,-1e-8,0,1e-8,pmax]
cblabel_N2     = r'$N^2_{BBL}$ [s$^{-2}$]'


# --- hab ---
pmin,pmax,pint = 0,150,1
cmap_hab        = plt.cm.ocean_r
norm_hab        = colors.Normalize(vmin=pmin,vmax=pmax)
cbticks_hab     = [pmin,50,100,150]
cblabel_hab     = r'$hab_{BBL}$ [m]'


# ------------ read data longrun ------------ 
data = Croco_longrun(name_exp,nbr_levels,['0'],name_exp_grd,name_pathdata)
data.get_grid()
# ------------ make plot ------------ 
print(' ... read avg file data + make plot ... ')
# ------------ read data ------------ 
data.get_outputs(0,var_list,get_date=False)
data.get_grid()
[z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
h_tile = np.transpose(np.tile(data.h,(200,1,1)),(1,2,0))
habr   = h_tile + z_r
hab    = tools.rho2u(habr)
# --> hab w 
hw_tile = np.transpose(np.tile(data.h,(201,1,1)),(1,2,0))
habw    = hw_tile + z_w
N2u     = tools.rho2u(tools.w2rho(data.var['bvf']))
N2      = tools.w2rho(data.var['bvf'])
u      = tools.u2rho(data.var['u'])
v      = tools.v2rho(data.var['v'])
w      = 3600*24*tools.w2rho(data.var['w'])
h       = data.h

# ---> find height of the BBL
h_bbl       = np.zeros((np.shape(N2)[0],np.shape(N2)[1]))
hab_bbl     = np.zeros((np.shape(N2)[0],np.shape(N2)[1]))
s_bbl       = np.zeros((np.shape(N2)[0],np.shape(N2)[1]))
criteria_N2 = 5e-9 #0.5e-6
eps         = 1e-8

hab_max     = 500

s_min          = 5
N2_slope       = np.zeros((np.shape(N2)[0],np.shape(N2)[1],(int(nbr_levels)//4-s_min)))
slope_N2_slope = np.zeros((np.shape(N2)[0],np.shape(N2)[1],(int(nbr_levels)//4-s_min)))
for s in range(s_min,int(nbr_levels)//4):
    N2_slope[:,:,s-s_min]     = N2[:,:,s+1]-N2[:,:,s]
for s in range(np.shape(N2_slope)[-1]-1):
    slope_N2_slope[:,:,s]   = N2_slope[:,:,s+1]-N2_slope[:,:,s]    


for i in range(np.shape(N2)[0]):
    for j in range(np.shape(N2)[1]):
        a            = slope_N2_slope[i,j,:]
        sf           = np.unravel_index(np.argmax(a, axis=-1), a.shape)[0]+( s_min-1)
        h_bbl[i,j]   = z_r[i,j,sf]
        hab_bbl[i,j] = habr[i,j,sf]
        s_bbl[i,j]   = sf

# --> averaged N2 over the BBL
N2_bbl       = np.zeros((np.shape(N2)[0],np.shape(N2)[1]))
u_bbl        = np.zeros((np.shape(N2)[0],np.shape(N2)[1]))
v_bbl        = np.zeros((np.shape(N2)[0],np.shape(N2)[1]))
w_bbl        = np.zeros((np.shape(N2)[0],np.shape(N2)[1]))
for i in range(np.shape(N2)[0]):
    for j in range(np.shape(N2)[1]):
        N2_bbl[i,j] = np.nanmean(N2[i,j,0:int(s_bbl[i,j])])
        u_bbl[i,j] = np.nanmean(u[i,j,0:int(s_bbl[i,j])])
        v_bbl[i,j] = np.nanmean(v[i,j,0:int(s_bbl[i,j])])
        w_bbl[i,j] = np.nanmean(w[i,j,0:int(s_bbl[i,j])])
N2_bbl[s==0]  = np.nan
u_bbl[s==0]  = np.nan
v_bbl[s==0]  = np.nan
w_bbl[s==0]  = np.nan
hab_bbl[s==0] = np.nan

# ----------- make plot ------
print('------------- MAKE PLOT ------------')
plt.figure(figsize=(10,10))
gs = gridspec.GridSpec(3,2,height_ratios=[2,0.1,1],hspace=0.7,wspace=0.2)

# --> horizontal map of hab_bbl
ax = plt.subplot(gs[0,0])
plt.title('a) ',fontsize=fs)
ctf01 = ax.pcolormesh(hab_bbl.T,cmap=cmap_hab,zorder=1,norm=norm_hab)
bathy = ax.contour(h.T,levels =[1000,1500,2000,2500,3000,3500,4000] , colors='k',linewidths= 1,zorder=3)
ax.tick_params(labelsize=fs)        
plt.axhline(y=jsec0,color=cjsec0,lw=2,linestyle='dashed',label='y='+str(jsec0))


cax = plt.subplot(gs[1,0])  # ----------------------------------------------------------------------------- colorbar hab_bbl
cb     = plt.colorbar(ctf01,cax,orientation='horizontal',extend='both',ticks=cbticks_hab)
cb.set_label(cblabel_hab,fontsize=fs,labelpad=-70)
cb.ax.tick_params(labelsize=fs)


# --> horizontal map of N2_bbl
ax = plt.subplot(gs[0,1])
plt.title('b) ',fontsize=fs)
ctf01 = ax.pcolormesh(N2_bbl.T,cmap=cmap_N2,zorder=1,norm=norm_N2)
bathy = ax.contour(h.T,levels =[1000,1500,2000,2500,3000,3500,4000] , colors='k',linewidths= 1,zorder=3)
ax.tick_params(labelsize=fs)
plt.axhline(y=jsec0,color=cjsec0,lw=2,linestyle='dashed',label='y='+str(jsec0))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

cax = plt.subplot(gs[1,1])  # ----------------------------------------------------------------------------- colorbar N2_bbl
cb     = plt.colorbar(ctf01,cax,orientation='horizontal',extend='both',ticks=cbticks_N2)
cb.set_label(cblabel_N2,fontsize=fs,labelpad=-70)
cb.ax.tick_params(labelsize=fs)


ax = plt.subplot(gs[2,:]) # ------------------------ N2
plt.title('c)',fontsize=fs)
lonsec = 0.5*(data.lonr[1:,jsec0]+data.lonr[:-1,jsec0])
lonsec = np.tile(lonsec,(z_r.shape[-1],1)).T
lon_tile = np.tile(data.lonr[:,jsec0],(z_r.shape[-1],1)).T
ctf = plt.pcolormesh(lonsec,hab[:,jsec0,:],N2u[:,jsec0,:],norm=norm_N2,cmap=cmap_N2,zorder=1)
plt.plot(data.lonr[:,jsec0],hab_bbl[:,jsec0],'k',linestyle='dashed',linewidth=1.5)
plt.fill_between(data.lonr[:,jsec0],habr[:,jsec0,199],1e4,fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.ylim(1,1e4)
ax.set_yticks([1,10,100,1000])
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.ylabel('hab [m]',fontsize=fs)
ax.tick_params(labelsize=fs)
plt.yscale('log')
plt.plot(data.lonr[:,jsec0],habr[:,jsec0,199],'k',linewidth=lwh)


plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/WMT/height_BBL_based_on_N2.png',dpi=200,bbox_inches='tight')
plt.close()


print('===================================================')
print(' ... save in netcdf file ... ')
print('===================================================')
nc = Dataset(file_out,'w')
nc.createDimension('nlon',1002)
nc.createDimension('nlat',802)
nc.createVariable('hab_bbl','f',('nlon','nlat'))
nc.createVariable('h_bbl','f',('nlon','nlat'))
nc.createVariable('s','f',('nlon','nlat'))
nc.createVariable('N2_bbl','f',('nlon','nlat'))
nc.createVariable('u_bbl','f',('nlon','nlat'))
nc.createVariable('v_bbl','f',('nlon','nlat'))
nc.createVariable('w_bbl','f',('nlon','nlat'))
nc.variables['hab_bbl'][:] = hab_bbl
nc.variables['h_bbl'][:]   = h_bbl
nc.variables['s'][:]       = s
nc.variables['N2_bbl'][:]  = N2_bbl
nc.variables['u_bbl'][:]   = u_bbl
nc.variables['v_bbl'][:]   = v_bbl
nc.variables['w_bbl'][:]   = w_bbl
nc.close()












