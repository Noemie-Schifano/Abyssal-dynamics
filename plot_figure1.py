'''
NS: Plot horizontal map to introduce the studied domain
'''
###
import matplotlib
matplotlib.use('Agg') #Choose the backend (needed for plottingteDimension('time',None) inside subprocess)
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors   as colors
import matplotlib.ticker   as ticker
from  matplotlib.cm import ScalarMappable
from  matplotlib.colors import ListedColormap, BoundaryNorm
from netCDF4 import Dataset
import sys
sys.path.append('/home2/datahome/nschifan/Python_Modules_p3/')
import R_tools as tools
import R_tools_fort as toolsF
import gsw as gsw
import time as time
import calendar as cal
import datetime as datetime
import cartopy.crs as ccrs
from distance_sphere_matproof import dist_sphere
import cartopy.feature as cfeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
from croco_simulations_jonathan_hist import Croco
from croco_simulations_jonathan_ncra_longrun import Croco_longrun
matplotlib.rcParams.update({'font.size': 28})


# ------------ parameters ------------ 
name_exp    = 'rrexnum200' 
name_exp_path ='rrexnums200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_NOFILT_T'
name_exp_grd= ''
nbr_levels  = '200'
name_exp_saveup = name_exp_path
var_list = ['zeta','u','v']
choice_listp = ['IRW','IRM','IRE','RRT','ICW','ICM','ICE']
file_topo  = 'topo15_NorthAtl.nc'
file_nc_croco_where_mooring = 'rrexnum200-rsup5_Keff_AKt_rrex-middleline.nc'
file_data_full              = 'rrex_twd_full.nc'
time        = ['40']
ndfiles     = 1 # number of days per netcdf
nt = len(time)*ndfiles
jsec       = 250
fs         = 28
lw         = 2 #1
lw_c       = 0.5
rho0       = 1027.4
extent     = [-40,-19,52.5,64.5]
lon_0,lat_0= -32,57.5 # centre of the map 
cf       = colors.to_rgba('teal')

# ------------ bin for concentration ------
minc      = 1e-4
maxc      = 0.2
nce       = 1e-4       # 0.0085
nbin      = 50          # 100
h_e    = np.arange(minc,maxc,nce) # np.linspace(minc,maxc,num=nbin)
h_c    = 0.5*(h_e[1:]+h_e[:-1])  # bin center  

# ------------ norm bathymetry -------------
cmap_h         = plt.cm.terrain_r 
norm_h         = colors.Normalize(vmin=0,vmax=5000)
levels_h       = np.arange(0,5500,500)
levels_hplot   = np.arange(0,4000,500)
cbticks_h      = [0,1000,2000,3000,4000] 
cblabel_h      = 'Bathymetry [m]'
# ------------ norm gradh ----
cmap_gradh    = plt.cm.copper_r
norm_gradh    = colors.Normalize(vmin=0,vmax=0.12)
levels_gradh  = np.arange(0,0.121,0.01)
cbticks_gradh = [0,0.06,0.12] 
cblabel_gradh = r'slope'

# ---> points 
xrn, yrn    = 314,656 # tracer 1
xrs, yrs    = 278,134 # tracer 6
xap, yap    = 800,406
xdeep,ydeep = 951, 3 
hdeep       = '3663.29'


def make_line(lonctd,latctd,lonr,latr):
    # lonctd, latctd : from in situ, etremities of line 
    # lonr, latr     : from CROCO
    # ------------ processing ------------ 
    nctd   = len(lonctd)
    ddeg   = abs(lonr[0,1]-lonr[0,0]) # model resolution 
    npts   = np.zeros(nctd-1) # number of points per segment between stations 
    for i in range(nctd-1):
        npts[i] = np.ceil(np.max((abs(lonctd[i+1]-lonctd[i])/ddeg,
                                  abs(latctd[i+1]-latctd[i])/ddeg)))
    print(np.shape(npts))
    print(npts)
    lonitp = np.concatenate([np.linspace(lonctd[i],lonctd[i+1],int(npts[i]))
                                for i in range(nctd-1)])
    latitp = np.concatenate([np.linspace(latctd[i],latctd[i+1],int(npts[i]))
                                for i in range(nctd-1)])

    ij = []
    for s in range(len(lonitp)):
        dist = dist_sphere(latitp[s],lonitp[s],latr,lonr)
        idist = np.nanargmin(dist)
        if idist not in ij: ij.append(idist)
    npts_line = len(ij)
    print('dim ',npts_line)
    ii,jj = np.unravel_index(ij,latr.shape)
    return [ii,jj]


# --- read variables ---
data = Croco(name_exp,nbr_levels,time[0],name_exp_grd,name_pathdata)
data.get_grid()
data.get_outputs(0,var_list,get_date=False)
[z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
h         = data.h
# ------------- Read bathymetry of the North Atlantic ----------
nc = Dataset(file_topo,'r')
lon15 = nc.variables['lon'][:]
lat15 = nc.variables['lat'][:]
z15   = -nc.variables['z'][:]
nc.close()

# ----------- read in-situ ---------
nc     = Dataset(file_data_full,'r')
lat_full    = nc.variables['lat'][:]
lon_full    = nc.variables['lon'][:]
nc.close()

# --> create CROCO section at in-situ location 
# --- first segment
lon1,lat1 = [lon_full[0],lon_full[2]],[lat_full[0],lat_full[2]]
ii1, jj1  = make_line(lon1,lat1,data.lonr,data.latr)
# --- second segment
lon2,lat2 = [lon_full[3],lon_full[6]],[lat_full[3],lat_full[6]]
ii2, jj2  = make_line(lon2,lat2,data.lonr,data.latr)

############################################################################################################################################## PLOT
# --- make plot ---
line   = 0
column = 0
figure = plt.figure(figsize=(20,20))
gs     = gridspec.GridSpec(2,1,height_ratios=[1,0.05])#,hspace=0.6)
##############################################################################################################################################        h
ax = plt.subplot(gs[0,0],projection=ccrs.Orthographic(lon_0,lat_0)) # ---> bathy h
ax.set_extent(extent,ccrs.PlateCarree())                                                                            #'moccasin'
gl = ax.gridlines(draw_labels=True,crs=ccrs.PlateCarree(),
                            x_inline=False, y_inline=False, linewidth=0.8, color='k', alpha=0.7, linestyle='--',zorder=0)
ax.coastlines('110m', alpha=0.1)
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='k', facecolor='gray',zorder=4))
ctf1 = ax.contourf(data.lonr,data.latr,h,levels=levels_h,cmap=cmap_h,norm=norm_h,transform=ccrs.PlateCarree(),zorder=1)
# --> bathymetry CROCO grid
ct  = ax.contour(data.lonr,data.latr,h,levels=levels_h,colors='k',linewidths=lw_c,transform=ccrs.PlateCarree(),zorder=2)
# --> bathymetry outside CROCO grid 
bathym = ax.contour(lon15,lat15,z15,levels=levels_h,colors='k',linewidths=lw_c,zorder=1,transform=ccrs.PlateCarree())

# ---> area above ridge north
index = [250,250,450,450,250]
index2 = [600,700,700,600,600]
x = [data.lonr[index[0],index2[0]],data.lonr[index[1],index2[1]],data.lonr[index[2],index2[2]],data.lonr[index[3],index2[3]],data.lonr[index[4],index2[4]]]
y = [data.latr[index[0],index2[0]],data.latr[index[1],index2[1]],data.latr[index[2],index2[2]],data.latr[index[3],index2[3]],data.latr[index[4],index2[4]]]
ax.fill(x, y, transform=ccrs.PlateCarree(), color='r',linewidth=2,fill=False,zorder=3)


# ---> area above ridge
index = [225,225,425,425,225]
index2 = [100,200,200,100,100]
x = [data.lonr[index[0],index2[0]],data.lonr[index[1],index2[1]],data.lonr[index[2],index2[2]],data.lonr[index[3],index2[3]],data.lonr[index[4],index2[4]]]
y = [data.latr[index[0],index2[0]],data.latr[index[1],index2[1]],data.latr[index[2],index2[2]],data.latr[index[3],index2[3]],data.latr[index[4],index2[4]]]
ax.fill(x, y, transform=ccrs.PlateCarree(), color='k',linewidth=2,fill=False,zorder=3)

# --> area above abyssal plain
index = [700,700,900,900,700]
index2 = [400,500,500,400,400]
x = [data.lonr[index[0],index2[0]],data.lonr[index[1],index2[1]],data.lonr[index[2],index2[2]],data.lonr[index[3],index2[3]],data.lonr[index[4],index2[4]]]
y = [data.latr[index[0],index2[0]],data.latr[index[1],index2[1]],data.latr[index[2],index2[2]],data.latr[index[3],index2[3]],data.latr[index[4],index2[4]]]
ax.fill(x, y, transform=ccrs.PlateCarree(), color='m',linewidth=2,fill=False,zorder=3)

# - plot points in situ measurements from CROCO -
plt.scatter(lon_full,lat_full,marker='^',s=30,linewidth=3,color='m',zorder=4,transform=ccrs.PlateCarree(),label='Moorings')
lat_listp    = [-0.35,-0.5,-0.5,0.4,0.3,0.1,0.1]
lon_listp    = [-1,-0.8,-0.3,0.3,0.3,0.3,0.3]
for k in range(len(choice_listp)):
    props = dict(boxstyle='round', facecolor='w', alpha=0.6)
    plt.text(lon_full[k]+lon_listp[k],lat_full[k]+lat_listp[k], choice_listp[k],color='k',fontsize=14,bbox=props,transform=ccrs.PlateCarree(),zorder=5)

# - plot points for vertical resolution & w variation in time
plt.scatter(data.lonr[xrn,yrn],data.latr[xrn,yrn],marker='X',s=80,linewidth=0.5,color='r',zorder=4,transform=ccrs.PlateCarree())
plt.scatter(data.lonr[xrs,yrs],data.latr[xrs,yrs],marker='X',s=80,linewidth=0.5,color='k',zorder=4,transform=ccrs.PlateCarree())
plt.scatter(data.lonr[xap,yap],data.latr[xap,yap],marker='X',s=80,linewidth=0.5,color='m',zorder=4,transform=ccrs.PlateCarree())
plt.scatter(data.lonr[xdeep,ydeep],data.latr[xdeep,ydeep],marker='X',s=80,linewidth=0.5,color='b',zorder=4,transform=ccrs.PlateCarree())

# - plot section deepest
plt.plot(data.lonr[:,ydeep],data.latr[:,ydeep],color='b',lw=2*lw,linestyle='dotted',zorder=3,transform=ccrs.PlateCarree())

# - plot section variation of w over time
plt.plot(data.lonr[:,450],data.latr[:,450],color='b',lw=2*lw,linestyle='dashed',zorder=3,transform=ccrs.PlateCarree())


# --- legend axes ---
gl.top_labels = False
gl.right_labels = False
gl.xlocator = LongitudeLocator()
gl.ylocator = LatitudeLocator()
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': fs, 'color': 'k'}
gl.ylabel_style = {'size': fs, 'color': 'k'}

cax = plt.subplot(gs[1,0])                                       # ---> colorbar h
cb     = plt.colorbar(ctf1,cax,orientation='horizontal',ticks=cbticks_h)
cb.set_label(cblabel_h,fontsize=fs,labelpad=-120)
cb.ax.tick_params(labelsize=fs)


plt.savefig('figure1.png',dpi=200,bbox_inches='tight')
plt.close()










