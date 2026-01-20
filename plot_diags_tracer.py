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
from croco_simulations_jon_hist_last import Croco
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
           #'dashed','dashed','dashed','dashed',
           #'dashed','dashed','dashed','dashed']
        #    'dotted','dotted','dotted','dotted',
        #    'dashed','dashed','dashed','dashed',
        #    'dashdot','dashdot','dashdot','dashdot']
    cmap_data     = plt.cm.Paired
    cfmap         = cmap_data(np.linspace(0.,1.,5))
    cfmap         = ['r','b','c','m']
    cfc           = colors.to_rgba('darkgrey')
    cfm           = colors.to_rgba('dimgray')
    #tpas2c      = [] #[1,3,5,6] # --> tracers to color
    #tpasn2c     = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] # --> tracers not 2 colored

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

    #for i in range(len(tpas2c)):       
    #    cfbb[tpas2c[i]]    = cfmap[i]
    #    zordert[tpas2c[i]] = 1
    #    lwt[tpas2c[i]]     = lw0+0.5
# ------------ read buoyancy balance for tracer and in time ------------
nc = Dataset(file_buoyb,'r')
w        = nc.variables['w_avg'][:].T
N2       = nc.variables['N2_avg'][:].T
c_rhs    = nc.variables['c_rhs_avg'][:].T
b_rhs    = nc.variables['b_rhs_avg'][:].T
b_adv    = nc.variables['b_adv_avg'][:].T
buoy     = nc.variables['buoy_avg'][:,:].T    # buoy(ntpas,nt) 
buoy_var = nc.variables['buoy_var'][:,:].T    # buoy(ntpas,nt) 
nc.close()

buoy_corrected = buoy
#buoy_corrected = buoy_dv

# --- Make plot ---
if plot_buoy_balance==True:
	nc2         = Dataset('/home/datawork-lops-rrex/nschifan/DIAGS/rrexnumsb200-rsup5_16T_porcentage_in_BBL_noemie.nc','r')
	hbbl        = nc2.variables['hbbl'][:].T
	z_r_tpas    = nc2.variables['z_r_tpas'][:].T
	nc2.close()
	hbbl[0,:]=np.nan*np.ones(np.shape(hbbl[0,:]))

	figure = plt.figure(figsize=(20,20))
	gs     = gridspec.GridSpec(4,2,hspace=0.3,wspace=0.2)

	ax  =  plt.subplot(gs[0,0]) # ---> w
	plt.title('a)')
	#for itpas in tpasn2c:
	#    plt.plot(ntimes,w[:,itpas],c=cfbb[itpas],linestyle=ls[itpas],zorder=zordert[itpas],linewidth=lwt[itpas])
	#for itpas in tpas2c:
	#    plt.plot(ntimes,w[:,itpas],c=cfbb[itpas],linestyle=ls[itpas],label=tname[itpas],zorder=zordert[itpas],linewidth=lwt[itpas])
	for itpas in range(len(tpas_list)):
		plt.plot(ntimes,w[:,itpas],c=cfbb[itpas],linestyle=ls[itpas],zorder=zordert[itpas],linewidth=lwt[itpas])
	plt.plot(ntimes,np.nanmedian(w,axis=1),c=cfm,linestyle='solid' ,label='Median value',zorder=2,linewidth=3*lw0)
	plt.plot(ntimes,np.nanmean(w,axis=1),  c=cfm,linestyle='dashed',label='Mean value'  ,zorder=1,linewidth=3*lw0)
	plt.legend(bbox_to_anchor=(1.5,1.4),ncol=2)
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
	#plt.ylim(-9.5e-7,9e-7)

	#ax  =  plt.subplot(gs[3,0]) # ---> hbbl
	#plt.title('g)')
	#for itpas in range(len(tpas_list)):
	#    plt.plot(ntimes,hbbl[:,itpas],c=cfbb[itpas],linestyle=ls[itpas],label=tname[itpas],zorder=zordert[itpas],linewidth=lwt[itpas])
	#plt.xlabel('Time since release [h]')
	#plt.ylabel(r'<$hbbl_{KPP}$> [m]')

	plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/WMT/Deep_tracer/diags_tracer_w_N2_buoy_balance.png',dpi=200,bbox_inches='tight')

if brhs_crhs == True:
    print('--------- make plot ---------')
    count_x, count_y = 0,0
    plt.figure(figsize=(20,20))
    gs     = gridspec.GridSpec(int(ntpas/4),int(ntpas/4),hspace=0.3,wspace=0.35)
    for itpas in range(ntpas):
        ax = plt.subplot(gs[count_x,count_y])
        ax.plot(ntimes,b_rhs[:,itpas],color='r',linewidth=lw,label=r'b$_{rhs}$')
        ax.plot(ntimes,c_rhs[:,itpas],color='b',linestyle='dotted',linewidth=lw,label=r'c$_{rhs}$')
        #ax.plot(ntimes,-(abs(buoy_corrected[:,itpas])-abs(buoy_corrected[0,itpas])),color='b',linewidth=lw,label='|$<b>_{t=0}$|-|<b>|')
        #ax.plot(ntimes,-(abs(int_brhs[:,itpas])-abs(int_brhs[0,itpas])),color='b',linestyle='dotted',linewidth=lw,label=r'|2$\int b_{rhs}dt$$|_{(t=0)}$-|2$\int b_{rhs}dt$|')
        #plt.yscale('log')
        if itpas==0:
            plt.legend(bbox_to_anchor=[3.0, 1.3],ncol=ncol_tpas+1)
        ax.axhline(y=0,c='k',alpha=0.5)
        if count_x==int(ntpas/4)-1:
            plt.xlabel('Time since release [h]')
        else:
            ax.set_xticklabels([])
        if count_y==0:
            ax.set_ylabel(r' [$m^2$ $s^{-3}$]',color='b')
        if count_y==int(ntpas/4)-1:
            count_x+=1
            count_y=0
        else:
            count_y+=1
        plt.title(tname_plot[itpas])
        ax.tick_params(axis='y', colors='b')
    plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/WMT/Deep_tracer_6h/16tracer_brhs_crhs.png',dpi=180,bbox_inches='tight')
    plt.close()


if plot_tracer_b == True:
    nc2         = Dataset('/home/datawork-lops-rrex/nschifan/DIAGS/rrexnumsb200-rsup5_16T_porcentage_in_BBL.nc','r')
    porc_inbbl  = nc2.variables['porc_tpas_inbbl'][:].T
    nc2.close()
    # -- int(2*b_rhs)dt
    int_brhs = np.zeros((np.shape(b_rhs)))
    int_brhs[0,:] = buoy_corrected[0,:]
    int_brhs_crhs = np.zeros((np.shape(b_rhs)))
    int_brhs_crhs[0,:] = buoy_corrected[0,:]
    for i in range(1,nt):
        #int_brhs[i-1,:]   = 0.5*(buoy[i-1,:]+buoy[i,:])+ 2*np.sum(b_rhs[:(i+1),:],axis=0)*((i+1)*dt)
        int_brhs[i,:]        = buoy_corrected[0,:]+ 2*np.sum(b_rhs[:(i+1),:],axis=0)*dt
        int_brhs_crhs[i,:]   = buoy_corrected[0,:]+ np.sum(b_rhs[:(i+1),:],axis=0)*dt+np.sum(c_rhs[:(i+1),:],axis=0)*dt
        print(i,len(b_rhs[:i,:]),int_brhs[i-1,0],buoy[1,0])
    # hbbl = np.nan at t=0
    porc_inbbl[0,:] = np.nan*np.ones(np.shape(porc_inbbl[0,:]))
    # -- compute sign of dbdt ---
    dbdt  = np.diff(buoy_corrected,axis=0)/dt
    sdbdt = np.sign(dbdt)
    print(dbdt[:,4])
    print(sdbdt[:,4])
    ntimes      = np.arange(nt)
    ntimessbbl  = (ntimes[1:]+ntimes[:-1])/2
    print('--------- make plot ---------')
    count_x, count_y = 0,0
    plt.figure(figsize=(20,20))
    gs     = gridspec.GridSpec(int(ntpas/4),int(ntpas/4),hspace=0.3,wspace=0.35)
    for itpas in range(ntpas):
        ax = plt.subplot(gs[count_x,count_y])
        ax.plot(ntimes,buoy_corrected[:,itpas]-buoy_corrected[0,itpas],color='b',linewidth=lw,label=r'$\langle$b$\rangle$-$\langle$b$\rangle$$_{t=0}$')
        ax.plot(ntimes,int_brhs[:,itpas]-int_brhs[0,itpas],color='b',linestyle='dotted',linewidth=lw,label=r'2$\int$ $\langle$b$_{rhs}$$\rangle$dt')
        ax.plot(ntimes,int_brhs_crhs[:,itpas]-int_brhs_crhs[0,itpas],color='b',linestyle='dashed',linewidth=lw,label=r'$\int$ $\langle$b$_{rhs}$$\rangle$+ $\langle b*$c$_{rhs}$$\rangle$/$\langle$c$\rangle$  dt')
        if itpas==0:
            plt.legend(bbox_to_anchor=[3.0, 1.3],ncol=ncol_tpas+1)
        ax2 = ax.twinx()
        ntimessbblp = np.copy(ntimessbbl)
        ntimessbbln = np.copy(ntimessbbl)
        ntimessbblp[sdbdt[:,itpas]==-1.0] = np.nan
        ntimessbbln[sdbdt[:,itpas]==1.0] = np.nan
        ax2.plot(ntimes,porc_inbbl[:,itpas],color='r',linewidth=lw,alpha=0.8)
        ax.axhline(y=0,c='k',alpha=0.5)
        # --> if sdbdt>0, the majority of the tracer is in the BBL
        #ax2.fill_between(ntimessbblp,50,100,color='r',alpha=0.1)
        #ax2.fill_between(ntimessbbln,0,50,color='r',alpha=0.1)
        ax.set_ylim(b_lim[itpas][0],b_lim[itpas][1])
        ax2.set_ylim(20,100)
        if count_x==int(ntpas/4)-1:
            plt.xlabel('Time since release [h]')
        else:
            ax.set_xticklabels([])
        if count_y==0:
            ax.set_ylabel(r' [$m^2$ $s^{-3}$]',color='b')
        plt.title(tname_plot[itpas])
        if count_y==int(ntpas/4)-1:
            ax2.set_ylabel(r'% of tracer in the BBL',color='r')
            count_x+=1
            count_y=0
        else:
            count_y+=1
            ax2.set_yticklabels([])
        ax.tick_params(axis='y', colors='b')
        ax2.tick_params(axis='y', colors='r')
    plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/WMT/Deep_tracer/16tracer_b.png',dpi=180,bbox_inches='tight')
    plt.close()
