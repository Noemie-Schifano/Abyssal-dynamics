''''
NS: plot horizontal map of bottom topostrophy and vertical velocity, both time-averaged on 219 days. 
     Statistics on bottom topostrophy and vertical vleocity considering 3 subdomains and the entire domain     
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
sys.path.append('Python_Modules_p3/')
import R_tools as tools
import R_tools_fort as toolsF
import time as time
import calendar as cal
import datetime as datetime
from croco_simulations_jonathan_ncra import Croco
from croco_simulations_jonathan_ncra_longrun import Croco_longrun
import gsw as gsw
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
matplotlib.rcParams.update({'font.size': 18}) 



# ------------ parameters ------------ 
name_exp    = 'rrexnum200' 
name_exp_path ='rrexnum200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_NOFILT_T'
nbr_levels  = '200'
name_exp_grd= 'rrex200-up5'
nbr_levels  = '200'
ndfiles     = 1  # number of days per netcdf
time= ['20']
var_list = ['w','zeta','u','v']
omega_T = 7.2921e-5 # --> rotation of the Earth

# --- plot options --- 
jsec    = 400
fs      = 18    # fontsize 
lw      = 2     # linewidth
ms      = 20    # markersize 
lw_c    = 0.4   # 0.2 linewidth coast 
my_bbox = dict(fc='w',ec='k',pad=2,lw=0.,alpha=0.5)
res   = 'h' # resolution of the coastline: c (coarse), l (low), i (intermediate), h (high)
proj  = 'lcc'  # map projection. lcc = lambert conformal 
paral = np.arange(0,80,5)
merid = np.arange(0,360,5)
lon_0,lat_0   = -31.5,58.6 # centre of the map 
Lx,Ly         = 1400e3,1500e3 # [km] zonal and meridional extent of the map 
scale_km      = 400 # [km] map scale 
# --> for Quiver horizontal currents
xlon        = np.arange(0,1003)
ylat        = np.arange(0,803)
q_x, q_y    = np.meshgrid(xlon,ylat, indexing='ij')
cfq         = 'k' 
# --W depth to interpolate w 
depth = [-1000,-2000]


# --- bin topostrophy [cm/s]
toe = np.arange(0,0.2,0.002)
toc = 0.5*(toe[1:]+toe[:-1])
nbin = 100

# bin w [m/day] 
we = np.arange(-25,5,0.3)
wc = 0.5*(we[1:]+we[:-1])


# --- w ---
cmap_w     =  plt.cm.RdBu_r 
pmin,pmax,pint   = -50, 50,1
levels_w      = np.arange(pmin,pmax+pint,pint)
norm_w        = colors.Normalize(vmin=pmin,vmax=pmax)
cbticks_w     =  [pmin,0,pmax]
cblabel_w     = r'w [m day$^{-1}$]'

sigmin,sigmax = -0.006,0.003

# --- to ---
cmap_to         =  cmap_w#plt.cm.Spectral_r 
pmin,pmax,pint  = -1,1,0.01
norm_to         = colors.SymLogNorm(linthresh=0.003, linscale=0.003,
                                              vmin=pmin, vmax=pmax)
cbticks_to      =  [-1,-0.1,0,0.1,1]
cblabel_to      = r'$\tau$ [cm s$^{-1}$]'

levels_hplot   = np.arange(0,4000,500)


# --- define areas ---
# -->  [imin,imax,jmin,jmax]
points_rs = [225,425,100,200]
points_rn = [250,450,600,700]
points_ap = [700,900,400,500]

# --- color for bar plot 
cfg  = 'b'
# ridge north
cfrn  = 'r'
# ridge south
cfrs  = colors.to_rgba('dimgray')
# ridge north
cfap  = 'm'
# all together
cf = [cfg, cfrn, cfrs, cfap]
cfw = [cfg, cfg, cfrn, cfrn, cfrs, cfrs, cfap, cfap]
 
# --- functions ---
def define_var_at_area(var,points):
    # select a variable at the area define between points
    # points containes (x,y) coordinates of the limits of the square 
    return var[points[0]:points[1],points[2]:points[3]]

def plot_k_oneline(ax,K,name_x_jon,fs,cf,cfw,sigmin,sigmax):    
    for expk in range(np.shape(K)[0]):
        print(expk)
    data = [K[expk][:] for expk in range(np.shape(K)[0])]
    # --- deal with Nan issues when using boxplot
    mask = ~np.isnan(data)
    filtered_data = [d[m] for d, m in zip(data, mask)]
    flierprop = dict( markersize=2)
    bp = ax.boxplot(filtered_data,medianprops=dict(color='k'),flierprops=flierprop,patch_artist = True, showfliers=False)
    for patch, colors in zip(bp['boxes'], cf):
        patch.set_facecolor(colors)
        patch.set(color = colors)
    for moustache, colors in zip(bp['whiskers'], cfw):
        moustache.set(color = colors)
    for cap, colors in zip(bp['caps'], cfw):
        cap.set(color = colors)
    plt.axhline(y=0)
    plt.xticks(ticks=[1,2,3,4],labels=name_x_jon,fontsize=fs)
    plt.xticks(rotation=30)
    return

# --- create variables ---
w_stat = np.zeros((3))
to_stat = np.zeros((3))
wrs_stat = np.zeros((3))
tors_stat = np.zeros((3))
wrn_stat = np.zeros((3))
torn_stat = np.zeros((3))
wap_stat = np.zeros((3))
toap_stat = np.zeros((3))
choice_list = ['all grid','ridge north','ridge south','abyssal plain']

# ------------ read data ------------ 

# --- read time-mean data from CROCO outputs (averaged each day) ---
# --> time-mean over the first 39 days
# --> for the quiver 
data         = Croco(name_exp,nbr_levels,time[0],name_exp_grd,name_pathdata)
# --- read longrun data from CROCO ---
# --> time-mean on 219 days
data_longrun = Croco_longrun(name_exp,nbr_levels,time[0],name_exp_grd,name_pathdata) 
data.get_grid()
# ------------ make plot ------------ 
print(' ... read avg file data + make plot ... ')
# -- valeurs min et max longitude et latitude
# --- [ lon min, lon max, lat min, lat max]               
extent     = [-37.5,-21.2,53,62.5] # °E °N
data.get_outputs(t,var_list,get_date=False)
data_longrun.get_outputs(t,var_list,get_date=False)
# ------------ read data ------------ 
[z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
# --- volume cell ---
dsurf   = 1./np.transpose(np.tile(data.pm*data.pn,(int(nbr_levels),1,1)),(1,2,0))
dvol   = np.diff(z_w,axis=-1)*dsurf
# -- w [m/day] ---
w   = 3600*24*data.var['w']
u_rho = tools.u2rho(data.var['u']) 
v_rho = tools.v2rho(data.var['v']) 


# ---> compute topostrophy [cm/s]
dhdx = tools.u2rho((data.h[1:,:]-data.h[:-1,:]))*data.pm
dhdy = tools.v2rho((data.h[:,1:]-data.h[:,:-1]))*data.pn
dhdx3d = np.transpose(np.tile(dhdx,(200,1,1)),(1,2,0))
dhdy3d = np.transpose(np.tile(dhdy,(200,1,1)),(1,2,0))
to   = 100*(tools.u2rho(data.var['u'])*dhdy3d - tools.v2rho(data.var['v'])*dhdx3d)


# --> read longrun mean 
w_longrun        = data_longrun.var['w']
to_longrun       = (tools.u2rho(data_longrun.var['u'])*dhdy3d - tools.v2rho(data_longrun.var['v'])*dhdx3d)
w0               = 3600*24*w_longrun[:,:,0]
to0              = 100*to_longrun[:,:,0]

# --> statistics 
w_stat = np.nanpercentile(np.ravel(w0),[10,50,90])
to_stat = np.nanpercentile(np.ravel(to0),[10,50,90])
print(' ----------------------------------------------------------------------------')
print('--> statistics over the domain, w  = ',w_stat[1])
print('--> statistics over the domain, to = ',to_stat[1])

wrn_stat = np.nanpercentile(np.ravel(define_var_at_area(w0,points_rn)),[10,50,90])
torn_stat = np.nanpercentile(np.ravel(define_var_at_area(to0,points_rn)),[10,50,90])

wrs_stat = np.nanpercentile(np.ravel(define_var_at_area(w0,points_rs)),[10,50,90])
tors_stat = np.nanpercentile(np.ravel(define_var_at_area(to0,points_rs)),[10,50,90])

wap_stat = np.nanpercentile(np.ravel(define_var_at_area(w0,points_ap)),[10,50,90])
toap_stat = np.nanpercentile(np.ravel(define_var_at_area(to0,points_ap)),[10,50,90])

# --> interpolate w
u_z1 = tools.vinterp(u_rho[:,:,:],[depth[0]],z_r,z_w)
v_z1 = tools.vinterp(v_rho[:,:,:],[depth[0]],z_r,z_w)
u_z2 = tools.vinterp(u_rho[:,:,:],[depth[1]],z_r,z_w)
v_z2 = tools.vinterp(v_rho[:,:,:],[depth[1]],z_r,z_w)


print(' ... get vertical levels ... ')
latr = data.latr
lonr = data.lonr
h    = data.h


print( '--- MAKE PLOT ---')
# ------------ make plot  ------------ 
plt.figure(figsize=(15,15))                                  
gs = gridspec.GridSpec(3,2,height_ratios=[1,0.05,1],hspace=0.45,wspace=0.3) 

# ---> topostrophy(s0)
ax = plt.subplot(gs[0,0])
plt.gca().set_aspect('equal', adjustable='box')
ctf02 = ax.pcolormesh(to0.T,cmap=cmap_to,zorder=1,norm=norm_to)
bathy = ax.contour(h.T,levels =[1000,2000,3000,4000] , colors='k',linewidths= lw_c,zorder=2)
eps=25
Q =  plt.quiver(q_x[::eps,::eps].T, q_y[::eps,::eps].T, 3*u_rho[::eps,::eps,0].T, 3*v_rho[::eps,::eps,0].T, units='width',scale=7,color=cfq,zorder=3)
ax.quiverkey(Q, X=0.9, Y=1.05, U=0.2,label=r'$20 \frac{cm}{s}$', labelpos='E')
# ---> area above ridge north
index  = [250,250,450,450,250]
index2 = [600,700,700,600,600]
x,y    = index, index2
ax.fill(x, y, color='r',linewidth=2,fill=False,zorder=3)

# ---> area above ridge
index  = [225,225,425,425,225]
index2 = [100,200,200,100,100]
x,y    = index, index2
ax.fill(x, y, color='k',linewidth=2,fill=False,zorder=3)

# --> area above abyssal plain
index  = [700,700,900,900,700]
index2 = [400,500,500,400,400]
x,y    = index, index2
ax.fill(x, y, color='m',linewidth=2,fill=False,zorder=3)

ax.set_xticks([0,250,500,750,1000],['0','200','400','600','800'])
ax.set_yticks([0,250,500,750],['0','200','400','600'])
plt.xlabel(r'km in $\xi$-direction',fontsize=fs)
plt.ylabel(r'km in $\eta$-direction',fontsize=fs)

plt.title(r'a) $\tau$($s_0$)',fontsize=fs)

# --> w(s0)
ax = plt.subplot(gs[0,1]) # --> w(s0)
plt.gca().set_aspect('equal', adjustable='box')
ctf01 = ax.pcolormesh(w0.T,cmap=cmap_w,zorder=1,norm=norm_w)
bathy = ax.contour(h.T,levels =[1000,2000,3000,4000] , colors='k',linewidths= lw_c,zorder=2)
# ---> area above ridge north
index  = [250,250,450,450,250]
index2 = [600,700,700,600,600]
x,y    = index, index2
ax.fill(x, y, color='r',linewidth=2,fill=False,zorder=3)

# ---> area above ridge
index  = [225,225,425,425,225]
index2 = [100,200,200,100,100]
x,y    = index, index2
ax.fill(x, y, color='k',linewidth=2,fill=False,zorder=3)

# --> area above abyssal plain
index  = [700,700,900,900,700]
index2 = [400,500,500,400,400]
x,y    = index, index2
ax.fill(x, y, color='m',linewidth=2,fill=False,zorder=3)
ax.set_xticks([0,250,500,750,1000],['0','200','400','600','800'])
ax.set_yticks([0,250,500,750],['0','200','400','600'])
plt.xlabel(r'km in $\xi$-direction',fontsize=fs)
plt.ylabel(r'km in $\eta$-direction',fontsize=fs)

plt.title(r'b) w($s_0$)',fontsize=fs)


cax = plt.subplot(gs[1,1]) # --> w(s0)
cb     = plt.colorbar(ctf01,cax,orientation='horizontal',extend='both',ticks=cbticks_w)
cb.set_label(cblabel_w,fontsize=fs,labelpad=-65)
cb.ax.tick_params(labelsize=fs)


# - bounding box 'aext' -  # --> to(so)
cax = plt.subplot(gs[1,0])
cb     = plt.colorbar(ctf02,cax,orientation='horizontal',extend='both',ticks=cbticks_to)
cb.set_label(cblabel_to,fontsize=fs,labelpad=-65)
cb.ax.tick_params(labelsize=fs)


# --> statistics to(s0)
ax = plt.subplot(gs[2,0])
W  = [to_stat,torn_stat,tors_stat,toap_stat]
plot_k_oneline(ax,W,choice_list,fs,cf,cfw,sigmin,sigmax)
plt.title(r'c)',fontsize=fs)
plt.ylabel(r'$\tau$($s_0$) [cm s$^{-1}$]')


# --> statistics w(s0)
ax = plt.subplot(gs[2,1])
W  = [w_stat,wrn_stat,wrs_stat,wap_stat]
plot_k_oneline(ax,W,choice_list,fs,cf,cfw,sigmin,sigmax)
plt.title('d)',fontsize=fs)
plt.ylabel(r'w($s_0$) [m day$^{-1}$]')

plt.savefig('figure2.png',dpi=180,bbox_inches='tight')
plt.close()


