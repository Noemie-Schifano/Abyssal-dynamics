'''
NS 23/07/2024: plot diags_tracer
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
from croco_simulations_jon_hist_last6h import Croco_6h
#from croco_simulations_hist import Croco_hist
import cartopy.crs as ccrs
import gsw as gsw
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
matplotlib.rcParams.update({'font.size': 18})

# ------------ file BBL -------------
file_bbl        = '/home/datawork-lops-rrex/nschifan/Data_in_situ_Rene/BBL_height_N2.nc'
file_n2_bbl     = '/home/datawork-lops-rrex/nschifan/Data_in_situ_Rene/rrexnumsb200_bottom_bvf.nc'
file_buoyb      = '/home/datawork-lops-rrex/nschifan/DIAGS/rrexnumsb200-rsup5_16tracers_k_buoy_rhop_tracer_balance.nc'
file_buoyb6h    = '/home/datawork-lops-rrex/nschifan/DIAGS/rrexnumsb200-rsup5_16tracers_6h_k_buoy_rhop_tracer_balance.nc'
file_zcrhs      = '/home/datawork-lops-rrex/nschifan/DIAGS/rrexnumsb200-rsup5_allTREs_zc_rhs.nc'

# ------------ parameters ------------ 
name_exp      = 'rrexnum200' 
name_exp_path ='rrexnums200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_T'
name_exp_grd= ''
nbr_levels  = '200'
name_exp_saveup = name_exp_path
# - select tracers for analysis - 
dt         = 3600 #1h between CROCO outputs
ndfiles    = 24
time       = np.arange(0,25,24)
nt         = len(time)*ndfiles
var_list   = ['zeta','u','v','bvf','tpas01','tpas02','tpas03','tpas04']
tpas_list  = ['tpas0'+str(i) for i in range(1,10)]+['tpas'+str(i) for i in range(10,17)]
tname      = ['Tracer '+str(i) for i in range(1,17)]
ntpas      = len(tpas_list)
ntimes     = np.arange(0,nt)

# --------- plot options -------
plot_buoy_balance      = True
plot_is_brhs_eq_dbdt   = False
plot_tracer_b          = True
brhs_crhs              = True
ncol_tpas = 3
lw        = 2
lw0       = 1
fs        = 10
z_min,z_max = -85,150 
alphab    = ['a) ','b) ','c) ','d) ','e) ','f) ','g) ','h) ','i) ','j) ','k) ','l) ','m) ','n) ','o) ','p) ']
tname_plot= [alphab[i]+tname[i] for i in range(0,16)]
cf1 = colors.to_rgba('mediumpurple')
cf2,cf3,cf4=colors.to_rgba('teal'),colors.to_rgba('crimson'),colors.to_rgba('chocolate')

b_lim = [ [-0.1e-5,3.e-5] ,[-1e-5,2.5e-5] ,[-0.1e-6,5e-6]  ,[-2.6e-5,0.5e-5],
          [-7.5e-6,0.2e-6],[-1.3e-5,4e-6],[-0.5e-6,1.5e-5],[-1e-6,1e-5],
          [-6e-6,1.1e-5]  ,[-0.1e-5,5e-5],[-3.5e-6,7e-6]    ,[-0.1e-5,3e-5],
          [-0.5e-6,2e-6]  ,[-0.1e-5,7e-5],[-1.1e-5,0.1e-5],[-0.25e-6,8.3e-6] ]

if plot_buoy_balance == True:
    ls  = ['solid','solid','solid','solid',
           'solid','solid','solid','solid',
           'solid','solid','solid','solid',
           'solid','solid','solid','solid']
           #'dashed','dashed','dashed','dashed',
           #'dashed','dashed','dashed','dashed']
        #    'dotted','dotted','dotted','dotted',
        #    'dashed','dashed','dashed','dashed',
        #    'dashdot','dashdot','dashdot','dashdot']
    cmap_data     = plt.cm.Paired
    cfmap         = cmap_data(np.linspace(0.,1.,5))
    cfmap         = ['r','b','c','m']
    #cfmap         = [ colors.to_rgba('blue'),'k',colors.to_rgba('blueviolet'),colors.to_rgba('red')]
    cfc           = colors.to_rgba('darkgrey')
    #cfmap         = cmap_data(np.linspace(0.2,0.75,8))
    #cfmap[0]      = cmap_data([0.])
    #cfmap[1]      = cmap_data([0.15])
    #cfbb          = [cfmap[i] for i in range(len(cfmap))] + [cfmap[i] for i in range(len(cfmap))]

    tpas2c      = [1,3,5,6] # --> tracers to color
    tpasn2c     = [0,2,4,7,8,9,10,11,12,13,14,15] # --> tracers not 2 colored

    #tpas2c      = [0,2,4,7]
    #tpasn2c     = [1,3,5,6,8,9,10,11,12,13,14,15]

    cfbb = [ cfc,cfc,cfc,cfc,
             cfc,cfc,cfc,cfc,
             cfc,cfc,cfc,cfc,
             cfc,cfc,cfc,cfc]
    zordert =[0,0,0,0,
              0,0,0,0,
              0,0,0,0,
              0,0,0,0]
    lwt     = [ lw0,lw0,lw0,lw0,
                lw0,lw0,lw0,lw0,
                lw0,lw0,lw0,lw0,
                lw0,lw0,lw0,lw0]

    for i in range(len(tpas2c)):       
        cfbb[tpas2c[i]]    = cfmap[i]
        zordert[tpas2c[i]] = 1
        lwt[tpas2c[i]]     = lw0+0.5
# ------------ read TRE_ref ------------
nc2         = Dataset('/home/datawork-lops-rrex/nschifan/DIAGS/rrexnumsb200-rsup5_16T_porcentage_in_BBL_noemie.nc','r')
z_r_tpas    = nc2.variables['z_r_tpas'][:].T
nc2.close()

nc = Dataset(file_buoyb,'r')
w        = (nc.variables['w_avg'][:].T)/(3600*24)
nc.close()


# ------------ read TRE_6h ------------
nc26h         = Dataset('/home/datawork-lops-rrex/nschifan/DIAGS/rrexnumsb200-rsup5_16T_6h_porcentage_in_BBL.nc','r')
z_r_tpas6h    = nc26h.variables['z_r_tpas'][:].T
nc26h.close()

nc6 = Dataset(file_buoyb6h,'r')
w6h        = (nc6.variables['w_avg'][:].T)/(3600*24)
nc6.close()

# ------------ read zc_rhs all TRE ---------
ncz         = Dataset(file_zcrhs,'r')
zc_rhs      = ncz.variables['zc_rhs_avg'][:].T
zc_rhs6h    = ncz.variables['zc_rhs_avg6h'][:].T
ncz.close()


if plot_tracer_b == True:
    # --> TREs originels
    int_w      = np.zeros((np.shape(w)))
    int_w[0,:] = z_r_tpas[0,:]
    int_z_crhs = np.zeros((np.shape(w)))
    int_z_crhs[0,:] = z_r_tpas[0,:]
    for i in range(1,nt):
        #int_brhs[i-1,:]   = 0.5*(buoy[i-1,:]+buoy[i,:])+ 2*np.sum(b_rhs[:(i+1),:],axis=0)*((i+1)*dt)
        int_w[i,:]           = z_r_tpas[0,:]+ np.sum(w[:(i+1),:],axis=0)*dt
        int_z_crhs[i,:]      = z_r_tpas[0,:]+ np.sum(w[:(i+1),:],axis=0)*dt+np.sum(zc_rhs[:(i+1),:],axis=0)*dt
    # --> TREs 6h
    int_w6h      = np.zeros((np.shape(w6h)))
    int_w6h[0,:] = z_r_tpas6h[0,:]
    int_z_crhs6h = np.zeros((np.shape(w6h)))
    int_z_crhs6h[0,:] = z_r_tpas6h[0,:]
    for i in range(1,nt):
        #int_brhs[i-1,:]   = 0.5*(buoy[i-1,:]+buoy[i,:])+ 2*np.sum(b_rhs[:(i+1),:],axis=0)*((i+1)*dt)
        int_w6h[i,:]           = z_r_tpas6h[0,:]+ np.sum(w6h[:(i+1),:],axis=0)*dt
        int_z_crhs6h[i,:]      = z_r_tpas6h[0,:]+ np.sum(w6h[:(i+1),:],axis=0)*dt+np.sum(zc_rhs6h[:(i+1),:],axis=0)*dt
    ntimes      = np.arange(nt)
    ntimessbbl  = (ntimes[1:]+ntimes[:-1])/2
    print('--------- make plot ---------')
    count_x, count_y = 0,0
    plt.figure(figsize=(20,20))
    gs     = gridspec.GridSpec(int(ntpas/4),int(ntpas/4),hspace=0.15,wspace=0.25)
    for itpas in range(ntpas):
        ax = plt.subplot(gs[count_x,count_y])
        # --> TRE originel
        ax.plot(ntimes,z_r_tpas[:,itpas]-z_r_tpas[0,itpas],color='b',linewidth=lw,label=r'$\langle$z$\rangle$-$\langle$z$\rangle$$_{t=0}$, TREs$_{ref}$')
        ax.plot(ntimes,int_w[:,itpas]-int_w[0,itpas],color='b',linestyle='dotted',linewidth=lw,label=r'$\int$ $\langle$w$\rangle$dt, TREs$_{ref}$')
        ax.plot(ntimes,int_z_crhs[:,itpas]-int_z_crhs[0,itpas],color='b',linestyle='dashed',linewidth=lw,label=r'$\int$ $\langle$w$\rangle$+ $\langle z*$c$_{rhs}$$\rangle$/$\langle$c$\rangle$  dt, TREs$_{ref}$')

        # --> TRE, 6h
        ax.plot(ntimes,z_r_tpas6h[:,itpas]-z_r_tpas6h[0,itpas],color='r',linewidth=lw,label=r'$\langle$z$\rangle$-$\langle$z$\rangle$$_{t=0}$, TREs$_{6h}$')
        ax.plot(ntimes,int_w6h[:,itpas]-int_w6h[0,itpas],color='r',linestyle='dotted',linewidth=lw,label=r'$\int$ $\langle$w$\rangle$dt, TREs$_{6h}$')
        ax.plot(ntimes,int_z_crhs6h[:,itpas]-int_z_crhs6h[0,itpas],color='r',linestyle='dashed',linewidth=lw,label=r'$\int$ $\langle$w$\rangle$+ $\langle z*$c$_{rhs}$$\rangle$/$\langle$c$\rangle$  dt, TREs$_{6h}$')
        if itpas==0:
            plt.legend(bbox_to_anchor=[4.0, 1.6],ncol=2) #4.35
        ax.axhline(y=0,c='k',alpha=0.5)
        #ax.set_ylim(b_lim[itpas][0],b_lim[itpas][1])
        if count_x==int(ntpas/4)-1:
            ax.set_xlabel('Time since release [h]')
        else:
            ax.set_xticklabels([])
        if count_y==0:
            ax.set_ylabel(r' [m]')
        plt.title(tname_plot[itpas])
        if count_y==int(ntpas/4)-1:
            count_x+=1
            count_y=0
        else:
            count_y+=1
        #ax.tick_params(axis='y', colors='b')
    plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/WMT/Deep_tracer_6h/16tracer_z_w_crhs_allTREs.png',dpi=180,bbox_inches='tight')
    plt.close()


