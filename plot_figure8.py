'''
'''

# ------------ parameters ------------ 

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
import gsw as gsw
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
matplotlib.rcParams.update({'font.size': 18})
from matplotlib.ticker import PercentFormatter
from croco_simulations_noemie_hist import Croco


# ------------ parameters ------------ 
name_exp    = 'rrexnum200' 
name_exp_path ='rrexnum200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_T'
nbr_levels  = '200'
name_exp_grd= ''
time        = ['0','24'] # 0h, 24h and 48h
nt          = 3 
pathway_save_fig = '/home/datawork-lops-rrex/nschifan/Figures/WMT/Deep_tracer/horizontal_slice_'+name_exp+'16T_0_3_6_days.png'
tpas_list   = ['tpas0'+str(i) for i in range(1,10)]+['tpas'+str(i) for i in range(10,17)]
text_list   = [str(i) for i in range(1,17)]
var_list    = ['zeta','temp','salt']
var_list   += tpas_list



# - select tracers to plot - 
sig_min,sig_max = 27.9,27.55 # axis limits 

# --- plot options --- 

jsec    = 400
fs      = 24       # fontsize 
lw      = 1        # linewidth
ms      = 20      # markersize 
lw_c    = 0.8   # linewidth coast 
my_bbox = dict(fc='w',ec='k',pad=2,lw=0.,alpha=0.5)
res   = 'h' # resolution of the coastline: c (coarse), l (low), i (intermediate), h (high)
proj  = 'lcc'  # map projection. lcc = lambert conformal 
paral = np.arange(0,80,5)
merid = np.arange(0,360,5)
lon_0,lat_0   = -31.5,58.6 # centre of the map 
Lx,Ly         = 1400e3,1500e3 # [km] zonal and meridional extent of the map 
scale_km      = 400 # [km] map scale 
extent     = [-37.5,-21.2,53,62.5] # °E °N
xmin,xmax  = [300,200,250,150], [500,400,450,350]
ymin,ymax  = [450,500,0,100], [650,700,200,300]

# - log - 
pmin,pmax,pint = -3,0,0.1 #-3,0, 0.1
cmap_tpas00    = plt.cm.autumn_r #plt.cm.Reds
cmap_tpas03    = plt.cm.winter_r #plt.cm.Reds
cmap_tpas06    = plt.cm.magma_r    #Greys
levels_tpas    = np.power(10,np.arange(pmin,pmax+pint,pint))
norm_tpas00    = colors.BoundaryNorm(np.logspace(pmin,pmax,int((pmax-pmin)/pint+1)),
                        ncolors=cmap_tpas00.N,clip=True)
norm_tpas03    = colors.BoundaryNorm(np.logspace(pmin,pmax,int((pmax-pmin)/pint+1)),
                        ncolors=cmap_tpas03.N,clip=True)
norm_tpas06    = colors.BoundaryNorm(np.logspace(pmin,pmax,int((pmax-pmin)/pint+1)),
                        ncolors=cmap_tpas06.N,clip=True)


cblabel_tpas00  = '[tracer], release'
cblabel_tpas03  = '[tracer], 24 hours'
cblabel_tpas06  = '[tracer], 48 hours '

cbticks_tpas   = np.logspace(pmin,pmax,pmax-pmin+1)
logfmt         = ticker.LogFormatterMathtext(10,labelOnlyBase=False) # nice writting in colorbar


# ------------ norm bathymetry -------------
cmap_h         = plt.cm.terrain_r #jet_r #gray
norm_h         = colors.Normalize(vmin=0,vmax=5000)
levels_h       = np.arange(0,5100,100)
cbticks_h      = [0,1000,2000,3000,4000] #[-3000,-2000,-1000] 
cblabel_h      = 'Bathymetry [m]'

#levels_h = np.arange(0,5000,100) #200

# --- functions ---
def find_j_tracer(tpas):
    # find center of gravity of tpas which is 3D
    #tracer = np.nansum(np.nansum(tpas,axis=0),axis=-1)
    return int(np.argmax(np.nansum(np.nansum(tpas,axis=0),axis=-1)))

def find_i_tracer(tpas):
    # find center of gravity of tpas which is 3D
    #tracer = np.nansum(np.nansum(tpas,axis=0),axis=-1)
    return int(np.argmax(np.nansum(np.nansum(tpas,axis=1),axis=-1)))


# ------------ read data ------------ 
tt      = 0
tpas    = np.zeros((nt,1002,802,len(tpas_list)))
jtracer = np.zeros(len(tpas_list))
itracer = np.zeros(len(tpas_list))

for t_nc in range(len(time)):
    if t_nc==0:
        data = Croco(name_exp,nbr_levels,time[t_nc],name_exp_grd,name_pathdata)
        data.get_grid()
        # t = 0h
        data.get_outputs(0,var_list)
        for itpas in range(len(tpas_list)): #data.nt
            # ------------ read data ------------ 
            tpas[tt,:,:,itpas] = np.nansum(data.var[tpas_list[itpas]],axis=2)
            eps   = 1e-6
            tpas[tt,:,:,itpas] =  np.ma.masked_array(tpas[tt,:,:,itpas],tpas[tt,:,:,itpas]<eps)
            print(' ... get vertical levels ... ')
            if itpas ==0:
                latr = data.latr
                lonr = data.lonr
                h    = data.h
            if tt==0:
                # ------ find y-axis index of the center of gravity of the tracer at the release 
                jtracer[itpas]   = find_j_tracer(data.var[tpas_list[itpas]])
                itracer[itpas]   = find_i_tracer(data.var[tpas_list[itpas]])

        tt+=1
        # t = 24h
        data.get_outputs(23,var_list)
        for itpas in range(len(tpas_list)): #data.nt
            # ------------ read data ------------ 
            tpas[tt,:,:,itpas] = np.nansum(data.var[tpas_list[itpas]],axis=2)
            eps   = 1e-6
            tpas[tt,:,:,itpas] =  np.ma.masked_array(tpas[tt,:,:,itpas],tpas[tt,:,:,itpas]<eps)
            print(' ... get vertical levels ... ')
            if itpas ==0:
                latr = data.latr
                lonr = data.lonr
                h    = data.h
            if tt==0:
                # ------ find y-axis index of the center of gravity of the tracer at the release 
                jtracer[itpas]   = find_j_tracer(data.var[tpas_list[itpas]])
        tt+=1
    # t = 48h
    else:
        data = Croco(name_exp,nbr_levels,time[t_nc],name_exp_grd,name_pathdata)
        data.get_grid()
        data.get_outputs(23,var_list)
        for itpas in range(len(tpas_list)): #data.nt
            # ------------ read data ------------ 
            tpas[tt,:,:,itpas] = np.nansum(data.var[tpas_list[itpas]],axis=2)
            eps   = 1e-6
            tpas[tt,:,:,itpas] =  np.ma.masked_array(tpas[tt,:,:,itpas],tpas[tt,:,:,itpas]<eps)
            print(' ... get vertical levels ... ')
            if itpas ==0:
                latr = data.latr
                lonr = data.lonr
                h    = data.h
            if tt==0:
                # ------ find y-axis index of the center of gravity of the tracer at the release 
                jtracer[itpas]   = find_j_tracer(data.var[tpas_list[itpas]])
        tt+=1


# ------------ make plot coord ij  ------------
count_x=0 #  line
count_y=0 #  column
plt.figure(figsize=(20,20))
gs = gridspec.GridSpec(3,3,height_ratios=[1,0.05,0.05],hspace=0.3) 
ax = plt.subplot(gs[0,:])
plt.gca().set_aspect('equal', adjustable='box')
ctf1 = ax.contourf(h.T,levels=levels_h,cmap=cmap_h,norm=norm_h,zorder=1)
ax.contour(h.T,levels=levels_h,colors='k',linewidths=lw_c,alpha=0.6)
for itpas in range(len(tpas_list)):
    # -------------------------------------- section with tracer concentration 
    for t in range(nt):
        if t==0:   # --> 0 days
            ctf00 = ax.contourf(tpas[t,:,:,itpas].T,levels=levels_tpas,cmap=cmap_tpas00,extend='max',zorder=6,norm=norm_tpas00)
            xx, yy = itracer[itpas],jtracer[itpas]+50
            print(xx,yy)
            props = dict(boxstyle='round', facecolor='w', alpha=0.85)
            ax.text(xx,yy,text_list[itpas],color='k',zorder=2,bbox=props)
        elif t==1: # --> 3 days
            ctf03 = ax.contourf(tpas[t,:,:,itpas].T,levels=levels_tpas,cmap=cmap_tpas03,extend='max',zorder=5,norm=norm_tpas03) 
        else:      # --> 6 days                                                                                      # norm = colors.LogNorm()
            ctf06 = ax.contourf(tpas[t,:,:,itpas].T,levels=levels_tpas,cmap=cmap_tpas06,extend='max',zorder=4,norm=norm_tpas06)
#plt.ylabel('grid-cells in j-direction',fontsize=fs)
#plt.xlabel('grid-cells in i-direction',fontsize=fs)
ax.set_xticks([0,250,500,750,1000],['0','200','400','600','800'])
ax.set_yticks([0,250,500,750],['0','200','400','600'])
plt.xlabel(r'km in $\xi$-direction',fontsize=fs)
plt.ylabel(r'km in $\eta$-direction',fontsize=fs)

# - actual colorbar - 
ax     = plt.subplot(gs[1,0])
cb     = plt.colorbar(ctf00,ax,orientation='horizontal',ticks=cbticks_tpas)
cb.set_label(cblabel_tpas00,fontsize=fs,labelpad=20)
cb.ax.set_xticklabels([r'10$^{-3}$',r'10$^{-2}$',r'10$^{-1}$','1'])

ax     = plt.subplot(gs[1,1])
cb     = plt.colorbar(ctf03,ax,orientation='horizontal',ticks=cbticks_tpas)
cb.set_label(cblabel_tpas03,fontsize=fs,labelpad=20)
cb.ax.set_xticklabels([r'10$^{-3}$',r'10$^{-2}$',r'10$^{-1}$','1'])

ax     = plt.subplot(gs[1,2])
cb     = plt.colorbar(ctf06,ax,orientation='horizontal',ticks=cbticks_tpas)
cb.set_label(cblabel_tpas06,fontsize=fs,labelpad=20)
cb.ax.set_xticklabels([r'10$^{-3}$',r'10$^{-2}$',r'10$^{-1}$','1'])


ax     = plt.subplot(gs[2,:])
cb     = plt.colorbar(ctf1,ax,orientation='horizontal',ticks=cbticks_h)
cb.set_label(cblabel_h,fontsize=fs,labelpad=20)
#cb.ax.set_xticklabels([r'10$^{-3}$',r'10$^{-2}$',r'10$^{-1}$','1'])

print(pathway_save_fig)
plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/WMT/Deep_tracer/horizontal_slice_'+name_exp+'16T_0_3_6_days.png',dpi=180,bbox_inches='tight')
plt.close()



"""
# ------------ make plot  ------------
count_x=0 #  line
count_y=0 #  column
last_line=0
plt.figure(figsize=(20,20))
gs = gridspec.GridSpec(4,6,height_ratios=[1,1,1,0.05]) #,width_ratios=[1,1],height_ratios=[1,1])
for itpas in range(len(tname)):
    print(count_x,count_y)
    ax = plt.subplot(gs[count_x,count_y:count_y+3],projection=ccrs.Orthographic(lon_0,lat_0))
    # -------------------------------------- section with tracer concentration <--- COLORBARS ISSUE
    for t in range(nt):
        if t==0:   # --> 0 days
            ctf00 = ax.contourf(lonr,latr,tpas[t,:,:,itpas],levels=levels_tpas,cmap=cmap_tpas00,extend='max',zorder=6,norm=norm_tpas00,transform=ccrs.PlateCarree())
        elif t==1: # --> 3 days
            ctf03 = ax.contourf(lonr,latr,tpas[t,:,:,itpas],levels=levels_tpas,cmap=cmap_tpas03,extend='max',zorder=5,norm=norm_tpas03,transform=ccrs.PlateCarree()) # colourful filled contours
        else:      # --> 6 days                                                                                      # norm = colors.LogNorm()
            ctf06 = ax.contourf(lonr,latr,tpas[t,:,:,itpas],levels=levels_tpas,cmap=cmap_tpas06,extend='max',zorder=4,norm=norm_tpas06,transform=ccrs.PlateCarree())
    ax.set_extent(extent)
    gl = ax.gridlines(draw_labels=True,dms=True, x_inline=False, y_inline=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    ax.coastlines('110m', alpha=0.1)
    bathy = ax.contour(lonr,latr,h,levels =[1000,2000,3000,4000] , colors='gray',linewidths= lw_c,transform=ccrs.PlateCarree())
    if itpas>0:
        ax.set_yticklabels([])
    gl.top_labels    = False
    gl.bottom_labels = False
    gl.right_labels  = False
    if count_y>0:
        # --- legend axes ---
        gl.ylocator = LatitudeLocator()
        gl.yformatter = LatitudeFormatter()
    else:
        gl.ylocator = LatitudeLocator()
        gl.yformatter = LatitudeFormatter()
        plt.ylabel('Latitude [$^{\circ}$N]',fontsize=fs)
    ax.set_extent(extent)
    ax.coastlines('110m', alpha=0.1)
    bathy = ax.contour(data.lonr,data.latr,data.h,levels =[1000,2000,3000,4000] , colors='gray',linewidths= lw_c,transform=ccrs.PlateCarree())
    plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)
    ax.tick_params(labelsize=fs-2)        
    plt.title(tname[itpas],fontsize=fs)
    last_line+=1
    count_y+=3
    if (count_y%6)==0:
        count_x += 1
        count_y = 0
    plt.plot(lonr[:,0],latr[:,0],color='gray',lw=0.5, transform=ccrs.PlateCarree())
    plt.plot(lonr[:,-1],latr[:,-1],color='gray',lw=0.5, transform=ccrs.PlateCarree())
    plt.plot(lonr[0,:],latr[0,:],color='gray',lw=0.5, transform=ccrs.PlateCarree())
    plt.plot(lonr[-1,:],latr[-1,:],color='gray',lw=0.5, transform=ccrs.PlateCarree())

# - actual colorbar - 
ax     = plt.subplot(gs[3,0:1])
cb     = plt.colorbar(ctf00,ax,orientation='horizontal',ticks=cbticks_tpas)
cb.set_label(cblabel_tpas00,fontsize=fs,labelpad=30)
cb.ax.set_xticklabels([r'10$^{-3}$',r'10$^{-2}$',r'10$^{-1}$','1'])

ax     = plt.subplot(gs[3,2:3])
cb     = plt.colorbar(ctf03,ax,orientation='horizontal',ticks=cbticks_tpas)
cb.set_label(cblabel_tpas03,fontsize=fs,labelpad=30)
cb.ax.set_xticklabels([r'10$^{-3}$',r'10$^{-2}$',r'10$^{-1}$','1'])

ax     = plt.subplot(gs[3,4:5])
cb     = plt.colorbar(ctf06,ax,orientation='horizontal',ticks=cbticks_tpas)
cb.set_label(cblabel_tpas06,fontsize=fs,labelpad=30)
cb.ax.tick_params(labelsize=fs)
cb.ax.set_xticklabels([r'10$^{-3}$',r'10$^{-2}$',r'10$^{-1}$','1'])



plt.savefig('Figures/Deep_tracer/horizontal_slice_'+name_exp+'_0_3_6_days.png',dpi=180,bbox_inches='tight')
plt.close()
    
"""
