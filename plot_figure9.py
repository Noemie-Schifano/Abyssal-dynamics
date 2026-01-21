'''
NS: Need "compute_k_buoyancy_balance_16T.py" to run before to create "file_buoyb"
    Need "compute_k_buoyancy_balance_16T_6h.py" to run before to create "file_buoyb6h
    Create diagnostics of buoyancy and the role of mixing as a function of the time for each TREs
    The same 16 TREs are released but with 6 hours-intervals    
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
from croco_simulations_jon_hist_last6h import Croco_6h
import cartopy.crs as ccrs
import gsw as gsw
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
matplotlib.rcParams.update({'font.size': 18})

# ------------ file BBL -------------
file_buoyb      = 'rrexnumsb200-rsup5_16tracers_k_buoy_rhop_tracer_balance.nc'
file_buoyb6h    = 'rrexnumsb200-rsup5_16tracers_6h_k_buoy_rhop_tracer_balance.nc'


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
kmin,kmax = 7e-6,7e-3
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
    cmap_data     = plt.cm.Paired
    cfmap         = cmap_data(np.linspace(0.,1.,5))
    cfmap         = ['r','b','c','m']
    cfc           = colors.to_rgba('darkgrey')
    
    tpas2c      = [1,3,5,6] # --> tracers to color
    
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

# ------------ read buoyancy balance for tracer and in time ------------
# buoy(ntpas,nt)
nc = Dataset(file_buoyb,'r')
w        = nc.variables['w_avg'][:].T
N2       = nc.variables['N2_avg'][:].T
c_rhs    = nc.variables['c_rhs_avg'][:].T
b_rhs    = nc.variables['b_rhs_avg'][:].T
b_adv    = nc.variables['b_adv_avg'][:].T
buoy     = nc.variables['buoy_avg'][:,:].T     
buoy_var = nc.variables['buoy_var'][:,:].T     
nc.close()

buoy_corrected = buoy


# ------------ read buoyancy balance for tracer and in time ------------
# buoy(ntpas,nt)
nc6 = Dataset(file_buoyb6h,'r')
w6h        = nc6.variables['w_avg'][:].T
N26h       = nc6.variables['N2_avg'][:].T
c_rhs6h    = nc6.variables['c_rhs_avg'][:].T
b_rhs6h    = nc6.variables['b_rhs_avg'][:].T
b_adv6h    = nc6.variables['b_adv_avg'][:].T
buoy6h     = nc6.variables['buoy_avg'][:,:].T     
buoy_var6h = nc6.variables['buoy_var'][:,:].T    
nc6.close()

buoy_corrected6h = buoy6h



if plot_tracer_b == True:
    # --> TREs originels
    int_brhs = np.zeros((np.shape(b_rhs)))
    int_brhs[0,:] = buoy_corrected[0,:]
    int_brhs_crhs = np.zeros((np.shape(b_rhs)))
    int_brhs_crhs[0,:] = buoy_corrected[0,:]
    for i in range(1,nt):
        int_brhs[i,:]        = buoy_corrected[0,:]+ 2*np.sum(b_rhs[:(i+1),:],axis=0)*dt
        int_brhs_crhs[i,:]   = buoy_corrected[0,:]+ np.sum(b_rhs[:(i+1),:],axis=0)*dt+np.sum(c_rhs[:(i+1),:],axis=0)*dt
        print(i,len(b_rhs[:i,:]),int_brhs[i-1,0],buoy[1,0])
    # --> TREs 6h
    int_brhs6h = np.zeros((np.shape(b_rhs6h)))
    int_brhs6h[0,:] = buoy_corrected6h[0,:]
    int_brhs_crhs6h = np.zeros((np.shape(b_rhs6h)))
    int_brhs_crhs6h[0,:] = buoy_corrected6h[0,:]
    for i in range(1,nt):
        int_brhs6h[i,:]        = buoy_corrected6h[0,:]+ 2*np.sum(b_rhs6h[:(i+1),:],axis=0)*dt
        int_brhs_crhs6h[i,:]   = buoy_corrected6h[0,:]+ np.sum(b_rhs6h[:(i+1),:],axis=0)*dt+np.sum(c_rhs6h[:(i+1),:],axis=0)*dt
        print(i,len(b_rhs[:i,:]),int_brhs[i-1,0],buoy[1,0])
    ntimes      = np.arange(nt)
    ntimessbbl  = (ntimes[1:]+ntimes[:-1])/2
    print('--------- make plot ---------')
    count_x, count_y = 0,0
    plt.figure(figsize=(20,20))
    gs     = gridspec.GridSpec(int(ntpas/4),int(ntpas/4),hspace=0.15,wspace=0.25)
    for itpas in range(ntpas):
        ax = plt.subplot(gs[count_x,count_y])
        # --> TRE originel
        ax.plot(ntimes,buoy_corrected[:,itpas]-buoy_corrected[0,itpas],color='b',linewidth=lw,label=r'$\langle$b$\rangle$-$\langle$b$\rangle$$_{t=0}$, TREs$_{ref}$')
        ax.plot(ntimes,int_brhs[:,itpas]-int_brhs[0,itpas],color='b',linestyle='dotted',linewidth=lw,label=r'2$\int$ $\langle$b$_{rhs}$$\rangle$dt, TREs$_{ref}$')
        ax.plot(ntimes,int_brhs_crhs[:,itpas]-int_brhs_crhs[0,itpas],color='b',linestyle='dashed',linewidth=lw,label=r'$\int$ $\langle$b$_{rhs}$$\rangle$+ $\langle b*$c$_{rhs}$$\rangle$/$\langle$c$\rangle$  dt, TREs$_{ref}$')
        # --> TRE, 6h
        ax.plot(ntimes,buoy_corrected6h[:,itpas]-buoy_corrected6h[0,itpas],color='r',linewidth=lw,label=r'$\langle$b$\rangle$-$\langle$b$\rangle$$_{t=0}$, TREs$_{6h}$')
        ax.plot(ntimes,int_brhs6h[:,itpas]-int_brhs6h[0,itpas],color='r',linestyle='dotted',linewidth=lw,label=r'2$\int$ $\langle$b$_{rhs}$$\rangle$dt, TREs$_{6h}$')
        ax.plot(ntimes,int_brhs_crhs6h[:,itpas]-int_brhs_crhs6h[0,itpas],color='r',linestyle='dashed',linewidth=lw,label=r'$\int$ $\langle$b$_{rhs}$$\rangle$+ $\langle b*$c$_{rhs}$$\rangle$/$\langle$c$\rangle$  dt, TREs$_{6h}$')
        if itpas==0:
            plt.legend(bbox_to_anchor=[4.0, 1.6],ncol=2)
        ax.axhline(y=0,c='k',alpha=0.5)
        ax.set_ylim(b_lim[itpas][0],b_lim[itpas][1])
        if count_x==int(ntpas/4)-1:
            ax.set_xlabel('Time since release [h]')
        else:
            ax.set_xticklabels([])
        if count_y==0:
            ax.set_ylabel(r' [$m^2$ $s^{-3}$]')
        plt.title(tname_plot[itpas])
        if count_y==int(ntpas/4)-1:
            count_x+=1
            count_y=0
        else:
            count_y+=1
    plt.savefig('figure9.png',dpi=180,bbox_inches='tight')
    plt.close()


