'''
NS: Need "compute_k_buoyancy_balance_16T_6h.py" to run before to create "file_buoyb"
    Need "extract_tracer_bbl_6h.py" to run before to create "rrexnumsb200-rsup5_16T_6h_porcentage_in_BBL.nc"
    Time-evolution of several fields experienced by the center of mass of each TRE
    Fields are: - Vertical velocity
                - Stratification
                - Buoyancy-evolution of tracer center of mass compared to the initial buoyancy
                - Depth-evolution of tracer center of mass
                - Change of buoancy due mmixng (b_rhs)
                - Change of buoyancy due advection (b_adv)
    This code is for the TRE_6h simulations
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
file_buoyb      = 'rrexnumsb200-rsup5_16tracers_6h_k_buoy_rhop_tracer_balance.nc'


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
plot_tracer_b          = False
brhs_crhs              = False
ncol_tpas = 3
lw        = 2
lw0       = 1
fs        = 10
kmin,kmax = 7e-6,7e-3
alphab    = ['a) ','b) ','c) ','d) ','e) ','f) ','g) ','h) ','i) ','j) ','k) ','l) ','m) ','n) ','o) ','p) ']
tname_plot= [alphab[i]+tname[i] for i in range(0,16)]
cf1 = colors.to_rgba('mediumpurple')
cf2,cf3,cf4=colors.to_rgba('teal'),colors.to_rgba('crimson'),colors.to_rgba('chocolate')

b_lim = [ [-0.1e-5,3.e-5] ,[-1e-5,2.5e-5] ,[-0.1e-6,5e-6]  ,[-2.5e-5,0.5e-5],
          [-7.5e-6,0.2e-6],[-3.e-6,4e-6],[-0.1e-6,1.5e-5],[-1e-6,1e-5],
          [-6e-6,1.1e-5]  ,[-0.1e-6,5e-5],[-3e-6,7e-6]    ,[-0.1e-6,3e-5],
          [-0.5e-6,2e-6]  ,[-0.1e-6,7e-5],[-1.1e-5,0.1e-5],[-0.1e-6,8.3e-6] ]

if plot_buoy_balance == True:
    ls  = ['solid','solid','solid','solid',
           'solid','solid','solid','solid',
           'solid','solid','solid','solid',
           'solid','solid','solid','solid']

    cmap_data     = plt.cm.Paired
    cfmap         = cmap_data(np.linspace(0.,1.,5))
    cfmap         = ['r','b','c','m']
    cfc           = colors.to_rgba('darkgrey')
    cfm           = colors.to_rgba('dimgray')

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


# --- Make plot ---
if plot_buoy_balance==True:
	nc2         = Dataset('rrexnumsb200-rsup5_16T_6h_porcentage_in_BBL.nc','r')
	z_r_tpas    = nc2.variables['z_r_tpas'][:].T
	nc2.close()
	

	figure = plt.figure(figsize=(20,20))
	gs     = gridspec.GridSpec(4,2,hspace=0.3,wspace=0.2)

	ax  =  plt.subplot(gs[0,0]) # ---> w
	plt.title('a)')
	
	for itpas in range(len(tpas_list)):
		plt.plot(ntimes,w[:,itpas],c=cfbb[itpas],linestyle=ls[itpas],zorder=zordert[itpas],linewidth=lwt[itpas])
	plt.plot(ntimes,np.nanmedian(w,axis=1),c=cfm,linestyle='solid' ,label='Median value',zorder=2,linewidth=3*lw0)
	plt.plot(ntimes,np.nanmean(w,axis=1),  c=cfm,linestyle='dashed',label='Mean value'  ,zorder=1,linewidth=3*lw0)
	plt.legend(bbox_to_anchor=(1.5,1.4),ncol=5)
	plt.axhline(y=0,c='k',alpha=0.8)
	plt.xlabel('Time since release [h]')
	plt.ylabel(r'$\langle$w$\rangle$ [m.day$^{-1}]$')

	ax  =  plt.subplot(gs[0,1]) # ---> N2
	plt.title('b)')
	for itpas in range(len(tpas_list)):
		plt.plot(ntimes,N2[:,itpas],c=cfbb[itpas],linestyle=ls[itpas],label=tname[itpas],zorder=zordert[itpas],linewidth=lwt[itpas])
	plt.plot(ntimes,np.nanmedian(N2,axis=1),c=cfm,linestyle='solid' ,label='Median value',zorder=2,linewidth=3*lw0)
	plt.plot(ntimes,np.nanmean(N2,axis=1),  c=cfm,linestyle='dashed',label='Mean value'  ,zorder=1,linewidth=3*lw0)
	plt.axhline(y=0,c='k',alpha=0.8)
	plt.xlabel('Time since release [h]')
	plt.ylabel(r'$\langle$$N^2$$\rangle$ [$s^{-2}]$')

	ax  =  plt.subplot(gs[1,0]) # ---> b
	plt.title('c)')
	for itpas in range(len(tpas_list)):
	    plt.plot(ntimes,buoy_corrected[:,itpas]-buoy_corrected[0,itpas],c=cfbb[itpas],linestyle=ls[itpas],label=tname[itpas],zorder=zordert[itpas],linewidth=lwt[itpas])
	plt.plot(ntimes,np.nanmedian(buoy_corrected[:,:]-buoy_corrected[0,:],axis=1),c=cfm,linestyle='solid' ,label='Median value',zorder=2,linewidth=3*lw0)
	plt.plot(ntimes,np.nanmean(buoy_corrected[:,:]-buoy_corrected[0,:],axis=1),  c=cfm,linestyle='dashed',label='Mean value'  ,zorder=1,linewidth=3*lw0)
	plt.axhline(y=0,c='k',alpha=0.8)
	plt.xlabel('Time since release [h]')
	plt.ylabel(r'$\langle$b$\rangle$-$\langle$b$\rangle$$_{t=0}$ [$m^2$ $s^{-3}$]')


	ax  =  plt.subplot(gs[1,1]) # ---> z
	plt.title('d)')
	for itpas in range(len(tpas_list)):
	    print(z_r_tpas[0,itpas])
	    plt.plot(ntimes,z_r_tpas[:,itpas]-z_r_tpas[0,itpas],c=cfbb[itpas],linestyle=ls[itpas],label=tname[itpas],zorder=zordert[itpas],linewidth=lwt[itpas])
	plt.plot(ntimes,np.nanmedian(z_r_tpas-z_r_tpas[0,:],axis=1),c=cfm,linestyle='solid' ,label='Median value',zorder=2,linewidth=3*lw0)
	plt.plot(ntimes,np.nanmean(z_r_tpas-z_r_tpas[0,:],axis=1),  c=cfm,linestyle='dashed',label='Mean value'  ,zorder=1,linewidth=3*lw0)
	plt.axhline(y=0,c='k',alpha=0.8)
	plt.xlabel('Time since release [h]')
	plt.ylabel(r'$\langle$z$\rangle$-$\langle$z$\rangle$$_{t=0}$ [m]')
      
	ax  =  plt.subplot(gs[2,0]) # ---> b_rhs
	plt.title('e)')
	for itpas in range(len(tpas_list)):
	    plt.plot(ntimes,b_rhs[:,itpas],c=cfbb[itpas],linestyle=ls[itpas],label=tname[itpas],zorder=zordert[itpas],linewidth=lwt[itpas])
	plt.plot(ntimes,np.nanmedian(b_rhs,axis=1),c=cfm,linestyle='solid' ,label='Median value',zorder=2,linewidth=3*lw0)
	plt.plot(ntimes,np.nanmean(b_rhs,axis=1),  c=cfm,linestyle='dashed',label='Mean value'  ,zorder=1,linewidth=3*lw0)
	plt.axhline(y=0,c='k',alpha=0.8)
	plt.xlabel('Time since release [h]')
	plt.ylabel(r'$\langle$$b_{rhs}$$\rangle$ [$m^2$ $s^{-3}$]')

	ax  =  plt.subplot(gs[2,1]) # ---> b_adv
	plt.title('f)')
	for itpas in range(len(tpas_list)):
	    plt.plot(ntimes,b_adv[:,itpas],c=cfbb[itpas],linestyle=ls[itpas],label=tname[itpas],zorder=zordert[itpas],linewidth=lwt[itpas])
	plt.plot(ntimes,np.nanmedian(b_adv,axis=1),c=cfm,linestyle='solid' ,label='Median value',zorder=2,linewidth=3*lw0)
	plt.plot(ntimes,np.nanmean(b_adv,axis=1),  c=cfm,linestyle='dashed',label='Mean value'  ,zorder=1,linewidth=3*lw0)
	plt.axhline(y=0,c='k',alpha=0.8)
	plt.xlabel('Time since release [h]')
	plt.ylabel(r'$\langle$$b_{adv}$$\rangle$ [$m^2$ $s^{-3}$]')


	plt.savefig('figure12.png',dpi=200,bbox_inches='tight')
