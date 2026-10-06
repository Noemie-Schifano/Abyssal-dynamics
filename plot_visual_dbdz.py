'''
NS: Need "compute_k_buoyancy_balance_16T.py" to run before to create "file_buoyb"
    Need "compute_k_buoyancy_balance_16T_6h.py" to run before to create "file_buoyb6h"
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
import cartopy.crs as ccrs
import gsw as gsw
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
matplotlib.rcParams.update({'font.size': 18})

# ------------ file BBL -------------
file_dbdz      = '/home/schifano/Documents/THESE/rrexnumsb200-rsup5_16tracers_dbdz.nc'

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
tpas_list  = ['tpas0'+str(i) for i in range(1,10)]+['tpas'+str(i) for i in range(10,17)]
tname      = ['Tracer '+str(i) for i in range(1,17)]
ntpas      = len(tpas_list)
ntimes     = np.arange(0,nt)

# --------- plot options -------
plot_tracer_dbdz       = True
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

if plot_tracer_dbdz == True:
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
nc = Dataset(file_dbdz,'r')
dbdz    = nc.variables['dbdz_avg'][:,:].T     
dbdz_6h = nc.variables['dbdz_6h_avg'][:,:].T     
nc.close()

if plot_tracer_dbdz == True:
    ntimes      = np.arange(nt)
    ntimessbbl  = (ntimes[1:]+ntimes[:-1])/2
    print('--------- make plot ---------')
    count_x, count_y = 0,0
    plt.figure(figsize=(20,20))
    gs     = gridspec.GridSpec(int(ntpas/4),int(ntpas/4),hspace=0.15,wspace=0.25)
    for itpas in range(ntpas):
        ax = plt.subplot(gs[count_x,count_y])
        # --> TRE originel
        ax.plot(ntimes,dbdz[:,itpas],color='b',linewidth=lw,label=r'$\langle$db/dz$\rangle$, TREs$_{ref}$')
        # --> TRE 6h
        ax.plot(ntimes,dbdz_6h[:,itpas],color='r',linewidth=lw,label=r'$\langle$db/dz$\rangle$, TREs$_{6h}$')
        if itpas==0:
            plt.legend(bbox_to_anchor=[4.0, 1.6],ncol=2)
        ax.axhline(y=0,c='k',alpha=0.5)
        #ax.set_ylim(b_lim[itpas][0],b_lim[itpas][1])
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
    plt.savefig('/home/schifano/Documents/THESE/dbdz_tres.png',dpi=180,bbox_inches='tight')
    plt.close()


