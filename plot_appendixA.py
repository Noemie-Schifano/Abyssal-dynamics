'''
NS: Vertical grid resolution at the deepest point of the domain 
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
from croco_simulations_jonathan import Croco
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
matplotlib.rcParams.update({'font.size': 14})

# ------------ parameters ------------ 
name_exp    = 'rrexnum200' 
name_exp_path ='rrexnums200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_NOFILT_T'
name_exp_grd= ''
nbr_levels  = '200'
name_exp_saveup = name_exp_path
var_list = ['zeta']
time        =  ['10']
ndfiles     = 1  # number of days per netcdf
nt          = ndfiles*len(time)
# points, same as for time variation
xrn, yrn = 314,656 # tracer 1
xrs, yrs = 278,134 # tracer 6
xap, yap = 800,406

# --- plot options --- 
fs      = 14          # fontsize 
lon_0,lat_0= -32,57.5 # centre of the map a
extent     = [-37.5,-21.2,53,62.5]

# --> colorbar bathymetry 
cmap_h         = plt.cm.jet_r 
norm_h         = colors.Normalize(vmin=0,vmax=4000)
levels_h       = np.arange(0,4200,200)
levels_hplot   = np.arange(0,4000,500)
cbticks_h      = [0,1000,2000,3000,4000]  
cblabel_h      = 'h [m]'

# --> colorbar dz
cmap_dz    = plt.cm.rainbow
levels_dz  = np.arange(0,41,1) 
norm_dz    = colors.BoundaryNorm(levels_dz,ncolors=cmap_dz.N,clip=True)
cbticks_dz = [0,10,20,30,40]
cblabel_dz = 'dz [m]' 


# ------------ read data ------------ 
data = Croco(name_exp,nbr_levels,time[0],name_exp_grd,name_pathdata)
data.get_outputs(0,var_list)
data.get_grid()
data.get_zlevs()
data.dz = abs(np.diff(data.z_w,axis=-1))
print(np.shape(data.dz),np.shape(data.z_w),np.shape(data.z_r))

# ---> find location of deepest point (h is positive)
ideep = int(np.where(data.h == data.h.max())[0])
jdeep = int(np.where(data.h == data.h.max())[1])
print(ideep,jdeep,data.h.max())

# --- plot ---
print('------------- MAKE PLOT ------------')
jsec = jdeep
plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(2,2,height_ratios=[1,0.05],width_ratios=[1,0.35],hspace=0.6,wspace=0.3)# including colorbars  

ax = plt.subplot(gs[0,0]) # --------- horizontal map of bathymetry
plt.title('a)',fontsize=fs)
lonsec = 0.5*(data.lonr[1:,jsec]+data.lonr[:-1,jsec])
lonsec = np.tile(lonsec,(data.z_w.shape[-1],1)).T
zsec = 0.5*(data.z_w[1:,jsec,:]+data.z_w[:-1,jsec,:])
ctf = plt.pcolormesh(lonsec,zsec,data.dz[1:-1,jsec,:],norm=norm_dz,cmap=cmap_dz)
for k in np.arange(0,data.z_w.shape[-1],5):
    plt.plot(data.lonr[:,jsec],data.z_w[:,jsec,k],'k',lw=0.3)
plt.fill_between(data.lonr[:,jsec],-4000,-data.h[:,jsec],fc='lightgray',ec='k',alpha=0.5)
plt.axvline(x=data.lonr[ideep,jdeep],linewidth=3,color='b',linestyle='dotted',zorder=4)
plt.ylim(-4000,0)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.ylabel('z [m]',fontsize=fs)
plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)
ax.tick_params(labelsize=fs)

ax = plt.subplot(gs[1,0]) # ------------------------ colorbar vertical resolution
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',ticks=cbticks_dz)
cb.set_label(cblabel_dz,fontsize=fs,labelpad=-87)  
cb.ax.tick_params(labelsize=fs)

ax = plt.subplot(gs[:,1]) # ------------------------ deepest point 
plt.title('b)',fontsize=fs)
plt.plot(data.dz[ideep,jdeep,:],data.z_r[ideep,jdeep,:],color='b',linewidth=2)
plt.plot(data.dz[xrn,yrn,:],data.z_r[xrn,yrn,:],color='r',linewidth=2)
plt.plot(data.dz[xrs,yrs,:],data.z_r[xrs,yrs,:],color='k',linewidth=2)
plt.plot(data.dz[xap,yap,:],data.z_r[xap,yap,:],color='m',linewidth=2)

plt.axhline(y=-3000,color='k',alpha=0.3)
plt.axhline(y=-2000,color='k',alpha=0.3)
plt.axhline(y=-1000,color='k',alpha=0.3)
plt.axhline(y=-3500,color='k',alpha=0.1)
plt.axhline(y=-2500,color='k',alpha=0.1)
plt.axhline(y=-1500,color='k',alpha=0.1)
plt.axhline(y=-500,color='k',alpha=0.1)
plt.axhline(y=0,color='k',alpha=0.3)
for i in range(39):
    plt.axvline(x=i,color='k',alpha=0.1)
plt.axvline(x=5,color='k',alpha=0.3)
plt.axvline(x=10,color='k',alpha=0.3)
plt.axvline(x=15,color='k',alpha=0.3)
plt.axvline(x=20,color='k',alpha=0.3)
plt.axvline(x=25,color='k',alpha=0.3)
plt.axvline(x=30,color='k',alpha=0.3)
plt.axvline(x=35,color='k',alpha=0.3)
plt.ylabel('z [m]',fontsize=fs)
plt.xlabel('dz [m]',fontsize=fs)
ax.set_xticks([0,10,20,30],fontsize=fs)

plt.savefig('appendixA.png',dpi=200,bbox_inches='tight')
plt.close()

