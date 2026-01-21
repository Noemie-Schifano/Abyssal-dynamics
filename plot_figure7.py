'''
NS: Vertical slice of the stratification, momentum and topostrophy as a function 
         of height above the bottom
         --> Using 219 time-averaged
         --> Spatially averaged on few vertical, i.e. y-axis, sections  
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
sys.path.append('Python_Modules_p3-master/')

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
import cartopy.crs as ccrs
import gsw as gsw
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
matplotlib.rcParams.update({'font.size': 28})


# ------------ file bbl -------------
file_bbl        = '/home/datawork-lops-rrex/nschifan/Data_in_situ_Rene/BBL_height_N2.nc'

# ------------ parameters ------------ 
name_exp    = 'rrexnum200' 
name_exp_path ='rrexnums200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_NOFILT_T'
name_exp_grd= ''
nbr_levels  = '200'
name_exp_saveup = name_exp_path
var_list      = ['zeta','w','AKt','bvf','salt','temp','u','v']
time          =  ['40']
ndfiles     = 1  # number of days per netcdf
nt          = ndfiles*len(time)

# --- plot options --- 
fs          = 28      # fontsize 
lwh         = 2
lwr         = 0.3 # rho-contours
lon_0,lat_0 = -32,57.5 # centre of the map a
extent      = [-37.5,-21.2,53,62.5]
jsec        = [700]  #[150,450,650]
cjsec       = ['b']
jsec0       = jsec[0]
cjsec0      = cjsec[0]
epsj        = 25 # to do an avg on 10 points
levels_rho_contour = np.arange(31.5,33.0,0.05)
lbp = -95



# --- kkpp & keff ---
pmin,pmax,pint = -5,0,0.1
cmap_k        = plt.cm.Reds
levels_k      = np.power(10,np.arange(pmin,pmax+pint,pint))
norm_k        = colors.LogNorm(vmin=1e-5,vmax=1)
cbticks_k     = [1e-5,1e-4,1e-3,1e-2,1e-1,1] 
cblabel_kkpp  = r'$\kappa_{KPP}$ [m$^2$ s$^{-1}$]'
cblabel_keff  = r'$\kappa_{eff}$ [m$^2$ s$^{-1}$]'


# --- N2 ---
pmin,pmax,pint = -1e-5,1e-5,1e-8
cmap_N2        = plt.cm.bwr
norm_N2        = colors.SymLogNorm(linthresh=pint, linscale=1, vmin=pmin, vmax=pmax,base=10) #colors.Normalize(vmin=pmin,vmax=pmax)
cbticks_N2     = [pmin,-1e-6,-1e-7,-1e-8,0,1e-8,1e-7,1e-6,pmax]
cbticks_N2     = [-1e-6,-1e-8,0,1e-8,1e-6]
cblabel_N2     = r'$N^2$ [s$^{-2}$]'
cblabel_N20    = r'$N^2(s_0)$ [s$^{-2}$]'
cblabel_N2bbl  = r'$N^2_{BBL}$ [s$^{-2}$]'


# --- wN2 ---
cmap_wN2        = plt.cm.PuOr_r 
pmin,pmax,pint  =  -5e-10, 5e-10,5e-12 
norm_wN2        = colors.Normalize(vmin=pmin,vmax=pmax) 
cbticks_wN2     = [pmin,0,pmax]
cblabel_wN2     = r'-w $\cdot$ $N^2$ [m $s^{-3}$]'
cblabel_wN2bbl  = r'-w $\cdot$ $N^2_{BBL}$ [m $s^{-3}$]'

# --- w ---
cmap_w     =  plt.cm.RdBu_r 
pmin,pmax,pint   = -100, 100,1
norm_w        = colors.Normalize(vmin=pmin,vmax=pmax)
cbticks_w     =  [pmin,-50,0,50,pmax]
cblabel_w     = r'w [m day$^{-1}$]'

# --- u ---
cmap_u          = cmap_w
pmin,pmax,pint  = -2e-1, 2e-1,1e-3
norm_u          = colors.Normalize(vmin=pmin,vmax=pmax)
cbticks_u       =  [pmin,0,pmax]
cblabel_u       = r'u [m s$^{-1}$]'
cblabel_v       = r'v [m s$^{-1}$]'


# --- to ---
cmap_to        =  cmap_w  
pmin,pmax,pint = -0.5, 0.5,0.005
levels_to      =  np.arange(pmin,pmax+pint,pint)
norm_to        =  colors.Normalize(vmin=pmin,vmax=pmax)
cbticks_to     =  [pmin,0,pmax]
cblabel_to     = r'$\tau$ [cm s$^{-1}$]'

# --- buoyancy ---
cmap_b        =  plt.cm.PuOr_r 
pmin,pmax,pint = -1e-10, 1e-10,5e-12
levels_b       =  np.arange(pmin,pmax+pint,pint)
norm_b         =  colors.Normalize(vmin=pmin,vmax=pmax)
cbticks_b      =  [pmin,0,pmax]
# --- buoyancy decomposed ---
pmin,pmax,pint = -5e-10, 5e-10,5e-12
levels_bd       =  np.arange(pmin,pmax+pint,pint)
norm_bd         =  colors.Normalize(vmin=pmin,vmax=pmax)
cbticks_bd      =  [pmin,0,pmax]

cblabel_b_rhs        = r'$b_{rhs}$ [$m^2$ $s^{-3}$]'
cblabel_b_adv        = r'$b_{adv}$ [$m^2$ $s^{-3}$]'
cblabel_b_adv_mean   = r'$-\overline{\mathbf{u}} \cdot \overline{\mathbf{\nabla} b}}$ [$m^2$ $s^{-3}$]'
cblabel_b_adv_eddy   = r'$-\overline{\mathbf{u^\prime} \cdot \mathbf{\nabla} b^\prime}$ [$m^2$ $s^{-3}$]'


# ---------------- function --------------------------
def compute_tsadv(T,u,v,w,z_r,z_w,f,g,rho0,pm,pn,newadv):
    '''
    check alternate formulation for r.h.s. of tracer equation
    '''
    if newadv:
        advx = tools.u2rho(u * tools.diffx(T,pm))
        advy = tools.v2rho(v * tools.diffy(T,pn))
        omega = w
        print(np.shape(omega),np.shape(T),np.shape(z_r))
        advz = omega[:,:,1:-1] * (T[:,:,1:]-T[:,:,:-1])/(z_r[:,:,1:]-z_r[:,:,:-1])
        advz = tools.vinterp(advz,z_r,z_w[:,:,1:-1],z_r)
    else:
        advx = tools.u2rho(u * tools.diffxi(T,pm,z_r,z_w))
        advy = tools.v2rho(v * tools.diffeta(T,pn,z_r,z_w))
        omega =w
        advz = omega[:,:,:] * tools.diffz_sig(T,z_r,z_w)
    return advx + advy, advz


def get_brhs(simul,T,S, t_rhs, s_rhs,z_r,z_w, nlbrhs = False):
    if 'NONLIN_EOS' in simul.cpp:
        [alpha,beta] = toolsF.alfabeta_3d(T,S,z_r,z_w,simul.rho0);
    else:
        print('using LIN_EOS')
        [alpha,beta] = [simul.Tcoef,simul.Scoef]/simul.rho0
    if nlbrhs:
        b_rhs = nonlin_brhs(T,S, t_rhs, s_rhs, simul.rho0, simul.g)
    else:
        b_rhs = simul.g * (alpha * t_rhs - beta * s_rhs)  # rho-rho grid
    return b_rhs


# ----------- read BBL -------------
nc      = Dataset(file_bbl,'r')
N2_bbl  = nc.variables['N2_bbl'][:,:]
hab_bbl  = nc.variables['hab_bbl'][:,:]
nc.close()


# ---------------- load simulation as Jonathan to compute buoyancy from advection -----------
mean = 'mean.00010-000229'
year=0; time_his=20; time_diag=time_his; time_avg=int(time_his/2); section_lat = False
mysimul = 'rrexnumsb200_RSUP5_NOFILT_T_avg_' + mean; 

simul0 = load(mysimul,time=time_his,output=True)

if mysimul == 'rrexnum100_UP3':
    title = 'UP3'
elif mysimul == 'rrexnum100':
    title = 'RSUP3'
else:
    title = mysimul

fifig = './Figures/' 
zoom = [simul0.x.min(), simul0.x.max(), simul0.y.min(), simul0.y.max()]; domain = 'all';

lon1,lat1 = zoom[0], zoom[2]
lon2,lat2 = zoom[1], zoom[3]
# Only load area around the section
ix1,iy1 = tools.find_nearest_points(simul0.x,simul0.y,lon1,lat1)
ix2,iy2 = tools.find_nearest_points(simul0.x,simul0.y,lon1,lat2)
ix3,iy3 = tools.find_nearest_points(simul0.x,simul0.y,lon2,lat1)
ix4,iy4 = tools.find_nearest_points(simul0.x,simul0.y,lon2,lat2)
ixs = [ix1,ix2,ix3,ix4]
iys = [iy1,iy2,iy3,iy4]

i_sec = 500 
dx_sec = 400;
j_sec = 350 
dy_sec = 200; 

coordinates= ' [' + format(np.max([j_sec-dy_sec,0])) + ','\
              + format(j_sec+dy_sec) + ','\
              + format(np.max([i_sec-dx_sec,0])) + ','\
              + format(i_sec+dx_sec) + ',[1,300,1]] '
simul = simul0
# --- load variables  ---
[z_r,z_w] = tools.get_depths(simul)
t=var('temp',simul,depths=[0])
[TXadv,TYadv,TVadv] = t.get_tracer_advection(simul); del t
T = var('temp',simul).data; S = var('salt',simul).data
b_adv_mean                    = tools.rho2u( get_brhs(simul,T[:,:,:],S[:,:,:],\
                                                 TXadv[:,:,:,0]+TYadv[:,:,:,0]+TVadv[:,:,:,0],\
                                                 TXadv[:,:,:,1]+TYadv[:,:,:,1]+TVadv[:,:,:,1],z_r,z_w) )

# ------------ read data longrun ------------ 
data = Croco_longrun(name_exp,nbr_levels,['0'],name_exp_grd,name_pathdata)
data.get_grid()
# ------------ make plot ------------ 
print(' ... read avg file data + make plot ... ')
# ------------ read data ------------ 
data.get_outputs(0,var_list,get_date=False)
data.get_grid()
data.get_zlevs()
[z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
h_tile = np.transpose(np.tile(data.h,(200,1,1)),(1,2,0))
habr   = h_tile + z_r
hab    = tools.rho2u(habr)
# --> hab w 
hw_tile = np.transpose(np.tile(data.h,(201,1,1)),(1,2,0))
habw    = hw_tile + z_w
kkpp = tools.w2rho(tools.rho2u(np.asfortranarray(data.var['AKt'])))
print( ' ############ -------------------------------------------------------------------- ################')
keff = tools.rho2u(data.get_diffusivity(0,'diffusivity'))
N2   = tools.rho2u(tools.w2rho(data.var['bvf']))
wr   = 3600*24*data.var['w']
w    = tools.rho2u(wr)
u    = data.var['u']
v    = tools.rho2u(tools.v2rho(data.var['v']))

print(' ... compute potential density referenced at 1000 meters... ')
p       = gsw.p_from_z(z_r,np.nanmean(data.latr))
SA      = gsw.SA_from_SP(data.var['salt'],p,np.nanmean(data.lonr),np.nanmean(data.latr))
CT      = gsw.CT_from_pt(SA,data.var['temp'])
rho_po  = gsw.sigma1(SA,CT)
print('-------------------------> rho_po')
print(rho_po[300,700,:])

# ---> buoyancy balance 
[b_rhsr,b_advr] = data.get_buoyancy_balance(0)
b_rhs, b_adv    = tools.rho2u(b_rhsr), tools.rho2u(b_advr)
b_adv_eddy    = b_adv - b_adv_mean
# ---> compute topostrophy
h    = data.h
dhdx = (data.h[1:,:]-data.h[:-1,:])*tools.rho2u(data.pm)
dhdy = tools.rho2u(tools.v2rho((data.h[:,1:]-data.h[:,:-1]))*data.pn)
gradhi =np.sqrt(dhdx**2 + dhdy**2)
gradh  = np.transpose(np.tile(gradhi,(200,1,1)),(1,2,0))
print(' --- compute topostrophy ---')
to      = 100*(data.var['u']*np.transpose(np.tile(dhdy,(200,1,1)),(1,2,0)) - tools.rho2u(tools.v2rho(data.var['v']))*np.transpose(np.tile(dhdx,(200,1,1)),(1,2,0)))


# ----------- make plot ------
print('------------- MAKE PLOT ------------')
plt.figure(figsize=(30,30))
gs0 = gridspec.GridSpec(6,3,height_ratios=[2,0.1,1,0.1,1,0.1],hspace=0.55,wspace=0.35)
gs  = gridspec.GridSpec(6,3,height_ratios=[2,0.1,1,0.1,1,0.1],hspace=0.55,wspace=0.35) 

# --> vertical slice bathymetry
ax = plt.subplot(gs0[0,0:1])
plt.title(' a) ',fontsize=fs)
lonsec = 0.5*(data.lonr[1:,jsec0]+data.lonr[:-1,jsec0])
lonsec = np.tile(lonsec,(data.z_r.shape[-1],1)).T
lon_tile = np.tile(data.lonr[:,jsec0],(z_r.shape[-1],1)).T
plt.fill_between(data.lonr[:,jsec0],-4000,-np.mean(data.h[:,jsec0-epsj:jsec0+epsj],axis=1),fc='lightgray',ec='k',alpha=0.5)
plt.contour(lon_tile,np.nanmean(z_r[:,jsec0-epsj:jsec0+epsj,:],axis=1),np.nanmean(rho_po[:,jsec0-epsj:jsec0+epsj,:],axis=1),levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.ylim(-4000,0)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.ylabel('Depth [m]',fontsize=fs)
plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)



# --> horizontal map N2_BBL 
ax = plt.subplot(gs0[0,1])
plt.title(' b) ',fontsize=fs)
ctf01 = ax.pcolormesh(N2_bbl.T,cmap=cmap_N2,zorder=1,norm=norm_N2)
bathy = ax.contour(h.T,levels =[1000,1500,2000,2500,3000,3500,4000] , colors='k',linewidths= 1,zorder=3)
ax.tick_params(labelsize=fs)
plt.axhline(y=jsec0-epsj,color=cjsec0,lw=4,linestyle='dashed',label='y='+str(jsec0))
plt.axhline(y=jsec0+epsj,color=cjsec0,lw=4,linestyle='dashed',label='y='+str(jsec0))
ax.set_xticks([0,250,500,750,1000],['0','200','400','600','800'])
ax.set_yticks([0,250,500,750],['0','200','400','600'])
plt.xlabel(r'km in $\xi$-direction',fontsize=fs)
plt.ylabel(r'km in $\eta$-direction',fontsize=fs)

cax = plt.subplot(gs0[1,1])  # ----------------------------------------------------------------------------- colorbar N2_bbl
cb     = plt.colorbar(ctf01,cax,orientation='horizontal',extend='both',ticks=cbticks_N2)
cb.set_label(cblabel_N2bbl,fontsize=fs,labelpad=lbp)
cb.ax.tick_params(labelsize=fs)

# -----> vertical slice, first column
ax = plt.subplot(gs[2,0]) # ------------------------ N2
plt.title(' c)',fontsize=fs)
lonsec = 0.5*(data.lonr[1:,jsec0]+data.lonr[:-1,jsec0])
lonsec = np.tile(lonsec,(data.z_r.shape[-1],1)).T
lon_tile = np.tile(data.lonr[:,jsec0],(z_r.shape[-1],1)).T
ctf = plt.pcolormesh(lonsec,np.nanmean(hab[:,jsec0-epsj:jsec0+epsj,:],axis=1),np.nanmean(N2[:,jsec0-epsj:jsec0+epsj,:],axis=1),norm=norm_N2,cmap=cmap_N2,zorder=1)
plt.fill_between(data.lonr[:,jsec0],habr[:,jsec0,199],1e4,fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.contour(lon_tile,np.nanmean(habr[:,jsec0-epsj:jsec0+epsj,:],axis=1),np.nanmean(rho_po[:,jsec0-epsj:jsec0+epsj,:],axis=1),levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.plot(data.lonr[:,jsec0],np.nanmean(hab_bbl[:,jsec0-epsj:jsec0+epsj],axis=1),'k',linestyle='dotted',linewidth=1.5)
plt.ylim(1,1e4)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.ylabel('hab [m]',fontsize=fs)
plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)
ax.tick_params(labelsize=fs)
plt.plot(data.lonr[:,jsec0],habr[:,jsec0,199],'k',linewidth=lwh)
plt.axvline(x=-28.7,color='k',lw=2,linestyle='dashed',zorder=3)
plt.axvline(x=-28.1,color='k',lw=2,linestyle='dashed',zorder=3)
plt.ylim(0,300)

ax = plt.subplot(gs[3,0]) # ------------------------ colorbar N2
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_N2)
cb.set_label(cblabel_N2,fontsize=fs,labelpad=lbp)  
cb.ax.tick_params(labelsize=fs)


ax = plt.subplot(gs[2,1]) # ------------------------ u
plt.title(' d)',fontsize=fs)#loc='left')
lonsec = 0.5*(data.lonr[1:,jsec0]+data.lonr[:-1,jsec0])
lonsec = np.tile(lonsec,(data.z_r.shape[-1],1)).T
lon_tile = np.tile(data.lonr[:,jsec0],(z_r.shape[-1],1)).T
ctf = plt.pcolormesh(lonsec,np.nanmean(hab[:,jsec0-epsj:jsec0+epsj,:],axis=1),np.nanmean(u[:,jsec0-epsj:jsec0+epsj,:],axis=1),norm=norm_u,cmap=cmap_u,zorder=1)
plt.contour(lon_tile,np.nanmean(habr[:,jsec0-epsj:jsec0+epsj,:],axis=1),np.nanmean(rho_po[:,jsec0-epsj:jsec0+epsj,:],axis=1),levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.ylabel('hab [m]',fontsize=fs)
plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)
ax.tick_params(labelsize=fs)
plt.axvline(x=-28.7,color='k',lw=4,linestyle='dashed',zorder=3)
plt.axvline(x=-28.1,color='k',lw=4,linestyle='dashed',zorder=3)
plt.ylim(0,300)

ax = plt.subplot(gs[3,1]) # ------------------------ colorbar u
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_u)
cb.set_label(cblabel_u,fontsize=fs,labelpad=lbp)  
cb.ax.tick_params(labelsize=fs)

x = plt.subplot(gs[2,2]) # ------------------------ v
plt.title(' e)',fontsize=fs)#loc='left')
ctf = plt.pcolormesh(lonsec,np.nanmean(hab[:,jsec0-epsj:jsec0+epsj,:],axis=1),np.nanmean(v[:,jsec0-epsj:jsec0+epsj,:],axis=1),norm=norm_u,cmap=cmap_u,zorder=1)
plt.contour(lon_tile,np.nanmean(habr[:,jsec0-epsj:jsec0+epsj,:],axis=1),np.nanmean(rho_po[:,jsec0-epsj:jsec0+epsj,:],axis=1),levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.xlim(lonsec[0,0],lonsec[1000,0])
ax.tick_params(labelsize=fs)
plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)
plt.ylabel('hab [m]',fontsize=fs)
plt.axvline(x=-28.7,color='k',lw=4,linestyle='dashed',zorder=3)
plt.axvline(x=-28.1,color='k',lw=4,linestyle='dashed',zorder=3)
plt.ylim(0,300)

ax = plt.subplot(gs[3,2]) # ------------------------ colorbar v
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_u)
cb.set_label(cblabel_v,fontsize=fs,labelpad=lbp)
cb.ax.tick_params(labelsize=fs)

ax = plt.subplot(gs[4,0]) # ------------------------ w
plt.title(' f)',fontsize=fs)
ctf = plt.pcolormesh(lonsec,np.nanmean(hab[:,jsec0-epsj:jsec0+epsj,:],axis=1),np.nanmean(w[:,jsec0-epsj:jsec0+epsj,:],axis=1),norm=norm_w,cmap=cmap_w,zorder=1)
plt.fill_between(data.lonr[:,jsec0],habr[:,jsec0,199],1e4,fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.contour(lon_tile,np.nanmean(habr[:,jsec0-epsj:jsec0+epsj,:],axis=1),np.nanmean(rho_po[:,jsec0-epsj:jsec0+epsj,:],axis=1),levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.ylim(1,1e4)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.ylabel('hab [m]',fontsize=fs)
plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)
ax.tick_params(labelsize=fs)
plt.plot(data.lonr[:,jsec0],habr[:,jsec0,199],'k',linewidth=lwh)
plt.axvline(x=-28.7,color='k',lw=4,linestyle='dashed',zorder=3)
plt.axvline(x=-28.1,color='k',lw=4,linestyle='dashed',zorder=3)
plt.ylim(0,300)

ax = plt.subplot(gs[5,0]) # ------------------------ colorbar w
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_w)
cb.set_label(cblabel_w,fontsize=fs,labelpad=lbp)  
cb.ax.tick_params(labelsize=fs)

ax = plt.subplot(gs[4,1]) # ------------------------ to
plt.title(' g)',fontsize=fs)
ctf = plt.pcolormesh(lonsec,np.nanmean(hab[:,jsec0-epsj:jsec0+epsj,:],axis=1),np.nanmean(to[:,jsec0-epsj:jsec0+epsj,:],axis=1),norm=norm_to,cmap=cmap_to,zorder=1)
plt.fill_between(data.lonr[:,jsec0],habr[:,jsec0,199],1e4,fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.contour(lon_tile,np.nanmean(habr[:,jsec0-epsj:jsec0+epsj,:],axis=1),np.nanmean(rho_po[:,jsec0-epsj:jsec0+epsj,:],axis=1),levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.ylim(1,1e4)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.ylabel('hab [m]',fontsize=fs)
plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)
ax.tick_params(labelsize=fs)
plt.plot(data.lonr[:,jsec0],habr[:,jsec0,199],'k',linewidth=lwh)
plt.axvline(x=-28.7,color='k',lw=4,linestyle='dashed',zorder=3)
plt.axvline(x=-28.1,color='k',lw=4,linestyle='dashed',zorder=3)
plt.ylim(0,300)

ax = plt.subplot(gs[5,1]) # ------------------------ colorbar to
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_to)
cb.set_label(cblabel_to,fontsize=fs,labelpad=lbp) 
cb.ax.tick_params


plt.savefig('figure7.png',dpi=200,bbox_inches='tight')
plt.close()















"""
# ----------- make plot ------
print('------------- MAKE PLOT ------------')
plt.figure(figsize=(40,40))
gs = gridspec.GridSpec(10,3,height_ratios=[2,0.1,1,0.1,1,0.1,1,0.1,1,0.1],hspace=0.8,wspace=0.2)

# --> vertical slice bathymetry
ax = plt.subplot(gs[0,0:1])
plt.title('a) ',fontsize=fs)
lonsec = 0.5*(data.lonr[1:,jsec0]+data.lonr[:-1,jsec0])
lonsec = np.tile(lonsec,(data.z_r.shape[-1],1)).T
lon_tile = np.tile(data.lonr[:,jsec0],(z_r.shape[-1],1)).T
plt.fill_between(data.lonr[:,jsec0],-4000,-data.h[:,jsec0],fc='lightgray',ec='k',alpha=0.5)
plt.contour(lon_tile,z_r[:,jsec0,:],rho_po[:,jsec0,:],levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.ylim(-4000,0)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.ylabel('Depth [m]',fontsize=fs)
plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)


# --> horizontal map N2_BBL 
ax = plt.subplot(gs[0,1])
plt.title('b) ',fontsize=fs)
ctf01 = ax.pcolormesh(N2_bbl.T,cmap=cmap_N2,zorder=1,norm=norm_N2)
bathy = ax.contour(h.T,levels =[1000,1500,2000,2500,3000,3500,4000] , colors='k',linewidths= 1,zorder=3)
ax.tick_params(labelsize=fs)
plt.axhline(y=jsec0,color=cjsec0,lw=4,linestyle='dashed',label='y='+str(jsec0))
plt.xlabel('grid-cells in i-direction',fontsize=fs)
plt.ylabel('grid-cells in j-direction',fontsize=fs)

cax = plt.subplot(gs[1,1])  # ----------------------------------------------------------------------------- colorbar N2_bbl
cb     = plt.colorbar(ctf01,cax,orientation='horizontal',extend='both',ticks=cbticks_N2)
cb.set_label(cblabel_N2bbl,fontsize=fs,labelpad=0)
cb.ax.tick_params(labelsize=fs)


# --> horizontal map N2_BBL 
ax = plt.subplot(gs[0,2])
plt.title('c) ',fontsize=fs)
ctf01 = ax.pcolormesh((-wr[:,:,0]*N2_bbl).T,cmap=cmap_wN2,zorder=1,norm=norm_wN2)
bathy = ax.contour(h.T,levels =[1000,1500,2000,2500,3000,3500,4000] , colors='k',linewidths= 1,zorder=3)
ax.tick_params(labelsize=fs)
plt.axhline(y=jsec0,color=cjsec0,lw=4,linestyle='dashed',label='y='+str(jsec0))
plt.xlabel('grid-cells in i-direction',fontsize=fs)
plt.ylabel('grid-cells in j-direction',fontsize=fs)

cax = plt.subplot(gs[1,2])  # ----------------------------------------------------------------------------- colorbar N2_bbl
cb     = plt.colorbar(ctf01,cax,orientation='horizontal',extend='both',ticks=cbticks_wN2)
cb.set_label(cblabel_wN2bbl,fontsize=fs,labelpad=0)
cb.ax.tick_params(labelsize=fs)


# -----> vertical slice, first column
ax = plt.subplot(gs[2,0]) # ------------------------ N2
plt.title('d)',fontsize=fs)
lonsec = 0.5*(data.lonr[1:,jsec0]+data.lonr[:-1,jsec0])
lonsec = np.tile(lonsec,(data.z_r.shape[-1],1)).T
lon_tile = np.tile(data.lonr[:,jsec0],(z_r.shape[-1],1)).T
ctf = plt.pcolormesh(lonsec,hab[:,jsec0,:],N2[:,jsec0,:],norm=norm_N2,cmap=cmap_N2,zorder=1)
plt.fill_between(data.lonr[:,jsec0],habr[:,jsec0,199],1e4,fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.contour(lon_tile,habr[:,jsec0,:],rho_po[:,jsec0,:],levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.plot(data.lonr[:,jsec0],hab_bbl[:,jsec0],'k',linestyle='dashed',linewidth=4.)
plt.ylim(1,1e4)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.ylabel('hab [m]',fontsize=fs)
ax.tick_params(labelsize=fs)
plt.plot(data.lonr[:,jsec0],habr[:,jsec0,199],'k',linewidth=lwh)
plt.axvline(x=-28.7,color='m',lw=2,linestyle='dashed',zorder=3)
plt.axvline(x=-28.1,color='m',lw=2,linestyle='dashed',zorder=3)
plt.ylim(0,300)

ax = plt.subplot(gs[3,0]) # ------------------------ colorbar N2
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_N2)
cb.set_label(cblabel_N2,fontsize=fs,labelpad=-0)  #-90
cb.ax.tick_params(labelsize=fs)


ax = plt.subplot(gs[4,0]) # ------------------------ u
plt.title('e)',fontsize=fs)
lonsec = 0.5*(data.lonr[1:,jsec0]+data.lonr[:-1,jsec0])
lonsec = np.tile(lonsec,(data.z_r.shape[-1],1)).T
lon_tile = np.tile(data.lonr[:,jsec0],(z_r.shape[-1],1)).T
ctf = plt.pcolormesh(lonsec,hab[:,jsec0,:],u[:,jsec0,:],norm=norm_u,cmap=cmap_u,zorder=1)
plt.contour(lon_tile,habr[:,jsec0,:],rho_po[:,jsec0,:],levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.ylabel('hab [m]',fontsize=fs)
ax.tick_params(labelsize=fs)
plt.axvline(x=-28.7,color='m',lw=2,linestyle='dashed',zorder=3)
plt.axvline(x=-28.1,color='m',lw=2,linestyle='dashed',zorder=3)
#ct  = plt.contour(lon_tile,habr[:,jsec0,:],s_por[:,jsec0,:],levels=levels_rho,colors='k',linewidths=0.2,zorder=3)
plt.ylim(0,300)

ax = plt.subplot(gs[5,0]) # ------------------------ colorbar u
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_u)
cb.set_label(cblabel_u,fontsize=fs,labelpad=0)  #-90
cb.ax.tick_params(labelsize=fs)

x = plt.subplot(gs[6,0]) # ------------------------ v
plt.title('f)',fontsize=fs)
ctf = plt.pcolormesh(lonsec,hab[:,jsec0,:],v[:,jsec0,:],norm=norm_u,cmap=cmap_u,zorder=1)
plt.contour(lon_tile,habr[:,jsec0,:],rho_po[:,jsec0,:],levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.xlim(lonsec[0,0],lonsec[1000,0])
ax.tick_params(labelsize=fs)
plt.ylabel('hab [m]',fontsize=fs)
plt.axvline(x=-28.7,color='m',lw=2,linestyle='dashed',zorder=3)
plt.axvline(x=-28.1,color='m',lw=2,linestyle='dashed',zorder=3)
#ct  = plt.contour(lon_tile,habr[:,jsec0,:],s_por[:,jsec0,:],levels=levels_rho,colors='k',linewidths=0.2,zorder=3)
plt.ylim(0,300)

ax = plt.subplot(gs[7,0]) # ------------------------ colorbar v
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_u)
cb.set_label(cblabel_v,fontsize=fs,labelpad=0)
cb.ax.tick_params(labelsize=fs)

ax = plt.subplot(gs[8,0]) # ------------------------ w
plt.title('g)',fontsize=fs)
ctf = plt.pcolormesh(lonsec,hab[:,jsec0,:],w[:,jsec0,:],norm=norm_w,cmap=cmap_w,zorder=1)
plt.fill_between(data.lonr[:,jsec0],habr[:,jsec0,199],1e4,fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.contour(lon_tile,habr[:,jsec0,:],rho_po[:,jsec0,:],levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.ylim(1,1e4)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.ylabel('hab [m]',fontsize=fs)
ax.tick_params(labelsize=fs)
plt.plot(data.lonr[:,jsec0],habr[:,jsec0,199],'k',linewidth=lwh)
plt.axvline(x=-28.7,color='m',lw=2,linestyle='dashed',zorder=3)
plt.axvline(x=-28.1,color='m',lw=2,linestyle='dashed',zorder=3)
plt.ylim(0,300)

ax = plt.subplot(gs[9,0]) # ------------------------ colorbar w
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_w)
cb.set_label(cblabel_w,fontsize=fs,labelpad=0)  #-90
cb.ax.tick_params(labelsize=fs)


# -----> vertical slice, second column
ax = plt.subplot(gs[2,1]) # ------------------------ kkpp
plt.title('h) ',fontsize=fs)
lonsec = 0.5*(data.lonr[1:,jsec0]+data.lonr[:-1,jsec0])
lonsec = np.tile(lonsec,(data.z_r.shape[-1],1)).T
print(np.shape(lonsec),np.shape(hab),np.shape(kkpp))
ctf = plt.pcolormesh(lonsec,hab[:,jsec0,:],kkpp[:,jsec0,:],norm=norm_k,cmap=cmap_k,zorder=1)
plt.fill_between(data.lonr[:,jsec0],habr[:,jsec0,199],1e4,fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.contour(lon_tile,habr[:,jsec0,:],rho_po[:,jsec0,:],levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.ylim(1,1e4)
# ax.set_yticklabels(())
plt.xlim(lonsec[0,0],lonsec[1000,0])
ax.tick_params(labelsize=fs)
plt.plot(data.lonr[:,jsec0],habr[:,jsec0,199],'k',linewidth=lwh)
plt.ylim(0,300)


ax = plt.subplot(gs[3,1]) # ------------------------ colorbar kkpp
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_k)
cb.set_label(cblabel_kkpp,fontsize=fs,labelpad=0)  #-90
cb.ax.tick_params(labelsize=fs)

ax = plt.subplot(gs[4,1]) # ------------------------ keff
plt.title('i)',fontsize=fs)
ctf = plt.pcolormesh(lonsec,hab[:,jsec0,:],keff[:,jsec0,:],norm=norm_k,cmap=cmap_k,zorder=1)
plt.fill_between(data.lonr[:,jsec0],habr[:,jsec0,199],1e4,fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.contour(lon_tile,habr[:,jsec0,:],rho_po[:,jsec0,:],levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.ylim(1,1e4)
plt.xlim(lonsec[0,0],lonsec[1000,0])
ax.tick_params(labelsize=fs)
plt.plot(data.lonr[:,jsec0],habr[:,jsec0,199],'k',linewidth=lwh)
plt.ylim(0,300)

ax = plt.subplot(gs[5,1]) # ------------------------ colorbar keff
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_k)
cb.set_label(cblabel_keff,fontsize=fs,labelpad=0)  #-90
cb.ax.tick_params(labelsize=fs)


ax = plt.subplot(gs[6,1]) # ------------------------ to
plt.title('j)',fontsize=fs)
ctf = plt.pcolormesh(lonsec,hab[:,jsec0,:],to[:,jsec0,:],norm=norm_to,cmap=cmap_to,zorder=1)
plt.fill_between(data.lonr[:,jsec0],habr[:,jsec0,199],1e4,fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.contour(lon_tile,habr[:,jsec0,:],rho_po[:,jsec0,:],levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.ylim(1,1e4)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)
ax.tick_params(labelsize=fs)
plt.plot(data.lonr[:,jsec0],habr[:,jsec0,199],'k',linewidth=lwh)
plt.ylim(0,300)

ax = plt.subplot(gs[7,1]) # ------------------------ colorbar to
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_to)
cb.set_label(cblabel_to,fontsize=fs,labelpad=0)  #-90
cb.ax.tick_params(labelsize=fs)

ax = plt.subplot(gs[8,1]) # ------------------------ b_rhs
plt.title('k)',fontsize=fs)
ctf = plt.pcolormesh(lonsec,hab[:,jsec0,:],b_rhs[:,jsec0,:],norm=norm_b,cmap=cmap_b,zorder=1)
plt.fill_between(data.lonr[:,jsec0],habr[:,jsec0,199],1e4,fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.ylim(1,1e4)
plt.xlim(lonsec[0,0],lonsec[1000,0])
ax.tick_params(labelsize=fs)
plt.plot(data.lonr[:,jsec0],habr[:,jsec0,199],'k',linewidth=lwh)
plt.ylim(0,300)

ax = plt.subplot(gs[9,1]) # ------------------------ colorbar b_rhs
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_b)
cb.set_label(cblabel_b_rhs,fontsize=fs,labelpad=0)  #-90
cb.ax.tick_params(labelsize=fs)


# -----> vertical slice, third column

ax = plt.subplot(gs[2,2]) # ------------------------ b_adv
plt.title('l)',fontsize=fs)
ctf = plt.pcolormesh(lonsec,hab[:,jsec0,:],b_adv[:,jsec0,:],norm=norm_b,cmap=cmap_b,zorder=1)
plt.fill_between(data.lonr[:,jsec0],habr[:,jsec0,199],1e4,fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.contour(lon_tile,habr[:,jsec0,:],rho_po[:,jsec0,:],levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.ylim(1,1e4)
plt.xlim(lonsec[0,0],lonsec[1000,0])
ax.tick_params(labelsize=fs)
plt.plot(data.lonr[:,jsec0],habr[:,jsec0,199],'k',linewidth=lwh)
plt.ylim(0,300)

ax = plt.subplot(gs[3,2]) # ------------------------ colorbar b_adv
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_b)
cb.set_label(cblabel_b_adv,fontsize=fs,labelpad=0)  #-90
cb.ax.tick_params(labelsize=fs)


ax = plt.subplot(gs[4,2]) # ------------------------ b_adv_eddy
plt.title('m)',fontsize=fs)
ctf = plt.pcolormesh(lonsec,hab[:,jsec0,:],b_adv_eddy[:,jsec0,:],norm=norm_bd,cmap=cmap_b,zorder=1)
plt.fill_between(data.lonr[:,jsec0],habr[:,jsec0,199],1e4,fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.contour(lon_tile,habr[:,jsec0,:],rho_po[:,jsec0,:],levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.ylim(1,1e4)
plt.xlim(lonsec[0,0],lonsec[1000,0])
ax.tick_params(labelsize=fs)
plt.plot(data.lonr[:,jsec0],habr[:,jsec0,199],'k',linewidth=lwh)
plt.ylim(0,300)

ax = plt.subplot(gs[5,2]) # ------------------------ colorbar b_adv_eddy
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_bd)
cb.set_label(cblabel_b_adv_eddy,fontsize=fs,labelpad=0)  #-90
cb.ax.tick_params(labelsize=fs)

ax = plt.subplot(gs[6,2]) # ------------------------ b_adv_mean
plt.title('n)',fontsize=fs)
ctf = plt.pcolormesh(lonsec,hab[:,jsec0,:],b_adv_mean[:,jsec0,:],norm=norm_bd,cmap=cmap_b,zorder=1)
plt.fill_between(data.lonr[:,jsec0],habr[:,jsec0,199],1e4,fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.contour(lon_tile,habr[:,jsec0,:],rho_po[:,jsec0,:],levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.ylim(1,1e4)
plt.xlim(lonsec[0,0],lonsec[1000,0])
ax.tick_params(labelsize=fs)
plt.plot(data.lonr[:,jsec0],habr[:,jsec0,199],'k',linewidth=lwh)
plt.ylim(0,300)

ax = plt.subplot(gs[7,2]) # ------------------------ colorbar b_adv_mean
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_bd)
cb.set_label(cblabel_b_adv_mean,fontsize=fs,labelpad=0)  #-90
cb.ax.tick_params(labelsize=fs)


ax = plt.subplot(gs[8,2]) # ------------------------ w*N2
plt.title('o)',fontsize=fs)
ctf = plt.pcolormesh(lonsec,hab[:,jsec0,:],-(w[:,jsec0,:]/(3600*24))*N2[:,jsec0,:],norm=norm_wN2,cmap=cmap_wN2,zorder=1)
plt.fill_between(data.lonr[:,jsec0],habr[:,jsec0,199],1e4,fc='lightgray',ec='k',alpha=0.5,zorder=2)
plt.contour(lon_tile,habr[:,jsec0,:],rho_po[:,jsec0,:],levels=levels_rho_contour,colors='k',linewidths=lwr,zorder=3)
plt.ylim(1,1e4)
plt.xlim(lonsec[0,0],lonsec[1000,0])
plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs)
ax.tick_params(labelsize=fs)
plt.plot(data.lonr[:,jsec0],habr[:,jsec0,199],'k',linewidth=lwh)
plt.ylim(0,300)

ax = plt.subplot(gs[9,2]) # ------------------------ colorbar w*N2
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',extend='both',ticks=cbticks_wN2)
cb.set_label(cblabel_wN2,fontsize=fs,labelpad=0)  #-90
cb.ax.tick_params(labelsize=fs)

plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/WMT/vertical_slice_jsec_'+str(jsec0)+'_buoyancy_balance.png',dpi=200,bbox_inches='tight')
plt.close()
"""
