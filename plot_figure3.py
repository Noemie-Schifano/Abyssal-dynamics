'''
NS: Need "extract_w_points_time_variability.py" to run before   
    Analysis of vertical velocity time variability at selected points for the standard deviation
    and vertical sections of w averaged over different time scales    
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
from croco_simulations_jonathan_ncra_longrun import Croco_longrun
from croco_simulations_jonathan import Croco
from croco_simulations_jonathan_hist import Croco as Croco_hist
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
matplotlib.rcParams.update({'font.size': 20})


# ------------ parameters ------------ 
file_w_var  = 'rrexnumsb200_w100m_point_time_variability.nc'
name_exp    = 'rrexnum200' 
name_exp_path ='rrexnums200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_NOFILT_T'
name_exp_grd= ''
nbr_levels  = '200'
name_exp_saveup = name_exp_path
var_list      = ['zeta','w']
var_list_his  = ['zeta','u','v']
time          =  ['70']
time_his      = ['140']
time_week     = np.arange(10,18,1)
time_month    = np.arange(10,41,1)
time_3month   = np.arange(10,101,1)
# only for selected points
time_2month   = np.arange(41,71,1)
time_4month   = np.arange(101,131,1)
time_5month   = np.arange(131,161,1)
time_6month   = np.arange(161,191,1)
# -- choose if time serie point is bottom or averaged over the whole column
choice_list = ['bottom','100m']
choice      = choice_list[1]

# points for time variation
xrn, yrn = 314,656 # tracer 1
xrs, yrs = 278,134 # tracer 6
xap, yap = 800,406  
xlist    = [xrn,xrs,xap]
ylist    = [yrn,yrs,yap]

# time definition
ndfiles        = 1  # number of days per netcdf
ndfiles_his    = 2  
nt             = ndfiles*len(time)
nt_week        = ndfiles*len(time_week)
nt_month       = ndfiles*len(time_month)
nt_2month      = ndfiles*(len(time_month)+len(time_2month))
nt_3month      = ndfiles*len(time_3month)
nt_4month      = ndfiles*(len(time_3month)+len(time_4month))
nt_5month      = nt_4month + ndfiles*len(time_5month)
nt_6month      = nt_5month + ndfiles*len(time_6month)

# --- plot options --- 
fs      = 20      # fontsize 
lon_0,lat_0= -32,57.5 # centre of the map a
extent     = [-37.5,-21.2,53,62.5]
jsec        = [150,450,650]
cjsec       = ['k','m','r']
jsec0       = jsec[1]
cjsec0      = cjsec[1]

# -- save figure
if choice == 'bottom':
    ptw_savefig = 'vertical_slice_jsec_'+str(jsec0)+'mean_month_week_avg_his_w_bottom.png'
else:
    ptw_savefig = 'vertical_slice_jsec_'+str(jsec0)+'mean_month_week_avg_his_w_column.png'

# --- colorbar w ---
cmap_w     =  plt.cm.RdBu_r 
pmin,pmax,pint   = -600, 600,1
norm_w        = colors.SymLogNorm(linthresh=10, linscale=1, vmin=pmin, vmax=pmax,base=10)
cbticks_w     =  [-1000,-100,-10,0,10,100,1000]
cblabel_w     = r'w [m day$^{-1}$]'

# --- create variables ---
# --> y-axis section
w_week      = np.zeros((nt_week,1002,200))
w_month     = np.zeros((nt_month,1002,200))
w_3month    = np.zeros((nt_3month,1002,200))
# --> points
w_weekp       = np.zeros((nt_week,len(xlist)))
w_monthp      = np.zeros((nt_month,len(xlist)))
w_2monthp     = np.zeros((nt_2month,len(xlist)))
w_3monthp     = np.zeros((nt_3month,len(xlist)))
w_4monthp     = np.zeros((nt_4month,len(xlist)))
w_5monthp     = np.zeros((nt_5month,len(xlist)))
w_6monthp     = np.zeros((nt_6month,len(xlist)))


# ------------- function ------------
def moy(wp, window_size):

    # Create a convolution kernel for the moving average
    kernel = np.ones(window_size) / window_size

    # Apply the convolution along the time axis (axis=0)
    wp_moy = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='valid'), axis=0, arr=wp)

    return wp_moy

# ------------ read file points for std(w) ---
# wp (points,time]: w averaged over 100 meters above the bottom 
nc = Dataset(file_w_var,'r')
wp = nc.variables['w'][:,:]
nc.close()

# ------------ read data longrun ------------ 
data = Croco_longrun(name_exp,nbr_levels,time[0],name_exp_grd,name_pathdata)
data.get_outputs(0,var_list)
data.get_grid()
data.get_zlevs()
h_tile = np.transpose(np.tile(data.h,(200,1,1)),(1,2,0))
hab    = h_tile + data.z_r
w      = 3600*24*tools.rho2u(data.var['w'])
h      = data.h

# ------------ read data avg ------------ 
data_avg = Croco(name_exp,nbr_levels,time[0],name_exp_grd,name_pathdata)
data_avg.get_outputs(0,var_list)
data_avg.get_grid()
data_avg.get_zlevs()
hab_avg = h_tile + data_avg.z_r
w_avg   = 3600*24*tools.rho2u(data_avg.var['w'])

# ------------ read data hist ------------ 
data_his = Croco_hist(name_exp,nbr_levels,time_his[0],name_exp_grd,name_pathdata)
data_his.get_outputs(0,var_list_his)
data_his.get_grid()
data_his.get_zlevs()
hab_his = h_tile + data_his.z_r
w_his   = 3600*24*tools.rho2u(toolsF.get_wvlcty(data_his.var['u'],data_his.var['v'],data_his.z_r,data_his.z_w,data_his.pm,data_his.pn))

# ---------- create weekly average --------
print(' ... weekly average ... ')
ttw = 0
for t_nc in range(len(time_week)):
    for t in range(ndfiles):
        data_w = Croco(name_exp,nbr_levels,str(int(time_week[t_nc])),name_exp_grd,name_pathdata)
        print(' ... read avg file data + make plot ... ')
        data_w.get_outputs(t,var_list,get_date=False)
        data_w.get_grid()
        data_w.get_zlevs()
        h_tile = np.transpose(np.tile(data.h,(200,1,1)),(1,2,0))
        hab_w  = h_tile + data_w.z_r
        # ------------ read data ------------ 
        w_week[ttw,:,:]  = 3600*24*data_w.var['w'][:,jsec0,:]
        for idx in range(len(xlist)):
            if choice=='bottom':
                w_weekp[ttw,idx]  = 3600*24*data_w.var['w'][xlist[idx],ylist[idx],0]
            else:
                 wint              = 3600*24*data_w.var['w'][xlist[idx],ylist[idx],:]
                 wint[hab_w[xlist[idx],ylist[idx],:]>100]   = np.nan
                 w_weekp[ttw,idx]  = np.nanmean(wint,axis=-1)
        ttw+=1

# ---------- create monthly average --------
print(' ... monthy average ... ')
ttm = 0
for t_nc in range(len(time_month)):
    for t in range(ndfiles):
        data_m = Croco(name_exp,nbr_levels,str(int(time_month[t_nc])),name_exp_grd,name_pathdata)
        print(' ... read avg file data + make plot ... ')
        data_m.get_outputs(t,var_list,get_date=False)
        data_m.get_grid()
        data_m.get_zlevs()
        hab_m  = h_tile + data_m.z_r
        # ------------ read data ------------ 
        w_month[ttm,:,:]  = 3600*24*data_m.var['w'][:,jsec0,:]
        for idx in range(len(xlist)):
            if choice=='bottom':
                w_monthp[ttm,idx]  = 3600*24*data_w.var['w'][xlist[idx],ylist[idx],0]
            else:
                wint               = 3600*24*data_w.var['w'][xlist[idx],ylist[idx],:]
                wint[hab_m[xlist[idx],ylist[idx],:]>100]    = np.nan
                w_monthp[ttm,idx]  = np.nanmean(wint,axis=-1)
        ttm+=1

# ---------- create 3-months average --------
print(' ... 3-months average ... ')
tt3m = 0
for t_nc in range(len(time_3month)):
    for t in range(ndfiles):
        data_3m = Croco(name_exp,nbr_levels,str(int(time_3month[t_nc])),name_exp_grd,name_pathdata)
        print(' ... read avg file data + make plot ... ')
        data_3m.get_outputs(t,var_list,get_date=False)
        data_3m.get_grid()
        data_3m.get_zlevs()
        hab_3m  = h_tile + data_3m.z_r
        # ------------ read data ------------ 
        w_3month[tt3m,:,:]  = 3600*24*data_3m.var['w'][:,jsec0,:]
        for idx in range(len(xlist)):
            if choice=='bottom':
                w_3monthp[tt3m,idx]  = 3600*24*data_3m.var['w'][xlist[idx],ylist[idx],0]
            else:
                wint               = 3600*24*data_3m.var['w'][xlist[idx],ylist[idx],:]
                wint[hab_3m[xlist[idx],ylist[idx],:]>100]    = np.nan
                w_3monthp[tt3m,idx]  = np.nanmean(wint,axis=-1)
        tt3m+=1



# --- time-averaged ---
w_weekly  = tools.rho2u(np.nanmean(w_week,axis=0))
w_monthly = tools.rho2u(np.nanmean(w_month,axis=0))
w_3months = tools.rho2u(np.nanmean(w_3month,axis=0))


# --- nice sorting of data for points to plot ---

habu_his  = tools.rho2u(hab_his)
habu_avg  = tools.rho2u(hab_avg)
w_his100  = np.copy(w_his)
w_avg100  = np.copy(w_avg)
w_his100[habu_his>100]=np.nan
w_avg100[habu_avg>100]=np.nan
idx     = 0

listwrn = [ np.std(moy(wp[idx,:],2)),
            np.std(moy(wp[idx,:],2*7)), np.std(moy(wp[idx,:],2*30)),np.std(moy(wp[idx,:],2*2*30)),np.std(moy(wp[idx,:],2*3*30)),
            np.std(moy(wp[idx,:],2*4*30)),np.std(moy(wp[idx,:],2*5*30)),np.std(moy(wp[idx,:],2*6*30)),np.std(moy(wp[idx,:],2*7*30))]
idx = 1
listwrs = [ np.std(moy(wp[idx,:],2)),
            np.std(moy(wp[idx,:],2*7)), np.std(moy(wp[idx,:],2*30)),np.std(moy(wp[idx,:],2*2*30)),np.std(moy(wp[idx,:],2*3*30)),
            np.std(moy(wp[idx,:],2*4*30)),np.std(moy(wp[idx,:],2*5*30)),np.std(moy(wp[idx,:],2*6*30)),np.std(moy(wp[idx,:],2*7*30))]

idx = 2
listwap = [ np.std(moy(wp[idx,:],2)),
            np.std(moy(wp[idx,:],2*7)), np.std(moy(wp[idx,:],2*30)),np.std(moy(wp[idx,:],2*2*30)),np.std(moy(wp[idx,:],2*3*30)),
            np.std(moy(wp[idx,:],2*4*30)),np.std(moy(wp[idx,:],2*5*30)),np.std(moy(wp[idx,:],2*6*30)),np.std(moy(wp[idx,:],2*7*30))]


# --- plot ---
print('------------- MAKE PLOT ------------')
plt.figure(figsize=(20,20))
gs = gridspec.GridSpec(4,3,height_ratios=[1,1,1,0.1],hspace=0.5,wspace=0.2)

################################################################################## time variation, ridge versus ap
ax = plt.subplot(gs[0,:]) 
plt.title('a)',fontsize=fs)
xtime=np.arange(len(listwrn))
plt.plot(xtime,listwrn,linestyle='--',c='r')
plt.plot(xtime,listwrs,linestyle='--',c='k')
plt.plot(xtime,listwap,linestyle='--',c='m')
plt.scatter(xtime,listwrn,marker='x',c='r')
plt.scatter(xtime,listwrs,marker='x',c='k')
plt.scatter(xtime,listwap,marker='x',c='m')
plt.axhline(y=1,c='k',linewidth=0.2,alpha=0.7)
plt.axhline(y=10,c='k',linewidth=0.2,alpha=0.7)
plt.axhline(y=100,c='k',linewidth=0.2,alpha=0.7)
for iw in range(2,10):
    plt.axhline(y=iw,c='k',linewidth=0.2,alpha=0.5)
for iw in range(20,100,10):
    plt.axhline(y=iw,c='k',linewidth=0.2,alpha=0.5)
for iw in range(200,400,100):
    plt.axhline(y=iw,c='k',linewidth=0.2,alpha=0.5)
plt.xticks([0,1,2,3,4,5,6,7,8],['1 day','7 days','30 days','60 days',
                                  '90 days','120 days','150 days','180 days','219 days'])
plt.yscale('log')
plt.ylim(0,350)
plt.xticks(rotation=30)
plt.ylabel(r'std($w_{hab=100m}$) [m.$day^{-1}$]')
################################################################################### snapshot
ax = plt.subplot(gs[1,0]) # ------------------------ w
plt.title('b) Snapshot',fontsize=fs)
lonsec = 0.5*(data.lonr[1:,jsec0]+data.lonr[:-1,jsec0])
lonsec = np.tile(lonsec,(data.z_r.shape[-1],1)).T
zsec =  0.5*(data.z_r[1:,jsec0,:]+data.z_r[:-1,jsec0,:])
ctf = plt.pcolormesh(lonsec,zsec,w_his[:,jsec0,:],norm=norm_w,cmap=cmap_w,zorder=1)
plt.fill_between(data.lonr[:,jsec0],-4000,-h[:,jsec0],fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.ylim(-4000,0)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)
plt.ylabel('Depth [m]',fontsize=fs)
ax.tick_params(labelsize=fs)


################################################################################### daily average
ax = plt.subplot(gs[1,1]) # ------------------------ w
plt.title('c) Daily average',fontsize=fs)
ctf = plt.pcolormesh(lonsec,zsec,w_avg[:,jsec0,:],norm=norm_w,cmap=cmap_w,zorder=1)
plt.fill_between(data.lonr[:,jsec0],-4000,-h[:,jsec0],fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.ylim(-4000,0)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)
ax.set_yticks([])
ax.tick_params(labelsize=fs)

################################################################################### weekly average
ax = plt.subplot(gs[1,2]) # ------------------------ w
plt.title('d) 7-day average',fontsize=fs)
ctf = plt.pcolormesh(lonsec,zsec,w_weekly[:,:],norm=norm_w,cmap=cmap_w,zorder=1)
plt.fill_between(data.lonr[:,jsec0],-4000,-h[:,jsec0],fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.ylim(-4000,0)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)
ax.set_yticks([])
ax.tick_params(labelsize=fs)


################################################################################### monthly average
ax = plt.subplot(gs[2,0]) # ------------------------ w
plt.title('e) 30-day average',fontsize=fs)
ctf = plt.pcolormesh(lonsec,zsec,w_monthly[:,:],norm=norm_w,cmap=cmap_w,zorder=1)
plt.fill_between(data.lonr[:,jsec0],-4000,-h[:,jsec0],fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.ylim(-4000,0)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)
plt.ylabel('Depth [m]',fontsize=fs)
ax.tick_params(labelsize=fs)


################################################################################### 3-months average
ax = plt.subplot(gs[2,1]) # ------------------------ w
plt.title('f) 90-day average',fontsize=fs)
ctf = plt.pcolormesh(lonsec,zsec,w_3months[:,:],norm=norm_w,cmap=cmap_w,zorder=1)
plt.fill_between(data.lonr[:,jsec0],-4000,-h[:,jsec0],fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.ylim(-4000,0)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)
ax.set_yticks([])
ax.tick_params(labelsize=fs)


################################################################################### longrun
ax = plt.subplot(gs[2,2]) # ------------------------ w
plt.title('g) 219-day average',fontsize=fs)
ctf = plt.pcolormesh(lonsec,zsec,w[:,jsec0,:],norm=norm_w,cmap=cmap_w,zorder=1)
plt.fill_between(data.lonr[:,jsec0],-4000,-h[:,jsec0],fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.ylim(-4000,0)
plt.xlim(lonsec[0,0],lonsec[1000,0])
ax.set_yticks([])
plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)

ax = plt.subplot(gs[3,1]) # ------------------------ colorbar w  
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_w)
cb.set_label(cblabel_w,fontsize=fs,labelpad=0)

plt.savefig(ptw_savefig,dpi=200,bbox_inches='tight')
plt.close()

