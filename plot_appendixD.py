'''
NS 23/07/2024: - Using 7 months time-mean netcdf file
               - Select areas of upwelling and downwelling
               - make buoyancy balance in those two areas 
'''

import matplotlib
matplotlib.use('Agg') 
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
import time as time
import calendar as cal
import datetime as datetime
from croco_simulations_jonathan_ncra_longrun import Croco_longrun
#from croco_simulations_hist import Croco_hist
import cartopy.crs as ccrs
import gsw as gsw
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
matplotlib.rcParams.update({'font.size': 28})

# ------------ file BBL -------------
file_bbl        = 'pathway2data/BBL_height_N2.nc'
file_n2_bbl     = 'pathway2data/rrexnumsb200_bottom_bvf.nc'

# ------------ parameters ------------ 
name_exp    = 'rrexnum200' 
name_exp_path ='rrexnums200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_NOFILT_T'
name_exp_grd= ''
nbr_levels  = '200'
name_exp_saveup = name_exp_path
var_list = ['zeta','AKt','bvf','hbbl']
ndfiles     = 1  # number of days per netcdf
rho0        = 1027.4
g           = 9.81
eps_b       = 1e-8
# --- plot options --- 
fs      = 28      
lwc     = 0.5
# ------------ bin for hab ------
#h_e = np.array([ 0,   3,   6,   9,  12,  15,  18,  21,  24,  27,  30,  33,  36, 66,  96, 126, 156, 186, 216, 246, 276, 306])
h_e = np.array([  0,   3,   6,   9,  12,  15,  18,  21,  26,  36,  46,  56,  66,
        76,  86,  96, 106, 116, 126, 136, 146, 156, 166, 176, 186, 196,
       206, 216, 226, 236, 246, 256, 266, 276, 286, 296, 306, 316, 326,
       336])
h_c = 0.5*(h_e[1:]+h_e[:-1]) # hab centres
nbin      = h_c.shape[0]
gg = 9.81
rhoref   = 1027.4

neg1 = 0
neg2 = 0
maxg = 0
ming = 0


# --------- defined areas ------------------
choice_list = ['Subdomain A','Subdomain B','Subdomain C','Subdomain D','All grid']
#choice_list = [r'Upwelling $N^2_{BBL}$>0',r'Downwelling $N^2_{BBL}$>0',r'Upwelling $N^2_{BBL}$<0',r'Downwelling $N^2_{BBL}$<0','All grid']

label_plot = ['a) '+choice_list[0],'f)','k)',
              'b) '+choice_list[1],'g)','l)',
              'c) '+choice_list[2],'h)','m)',
              'd) '+choice_list[3],'i)','n)',
              'e) '+choice_list[4],'j)','o)']

label_plot2 = ['a) '+choice_list[0],'f)',
               'b) '+choice_list[1],'g)',
               'c) '+choice_list[2],'h)',
               'd) '+choice_list[3],'i)',
               'e) '+choice_list[4],'j)']


##############################################################################################
# ---------------- function --------------------------
def find_upwelling_downwelling(w,N2,lonr,latr):
    # create an area of dimensions [x-axis,y-axis]
    # --> area == 1     : upwelling,  N2>=0
    # --> area == 9999  : downwelling, N2>=0
    # --> area == -1    : upwelling,  N2<0
    # --> area == -9999 : downwelling, N2<0
    # w          : vertical velocity at the bottom 
    # N2         : stratification at the bottom
    # lonr, latr : longitude and latitude at rho-coordinates from CROCO
    area      = np.zeros(np.shape(lonr))
    area_N2   = np.zeros(np.shape(lonr))
    direction = w*N2
    area[np.sign(direction)==1]=1
    area[np.sign(direction)==-1]=9999
    area_N2[N2<0] = -area[N2<0]
    area_N2[N2>0] = area[N2>0]
    return area_N2



##############################################################################################
# ------------ read BBL ------------ 
nc        = Dataset(file_bbl,'r')
hab_bbl   = np.ravel(nc.variables['hab_bbl'][:,:])
N2_bbl    = nc.variables['N2_bbl'][:,:]
nc.close()


##############################################################################################
# ------------ read data ------------ 
data = Croco_longrun(name_exp,nbr_levels,['0'],name_exp_grd,name_pathdata)
data.get_grid()
# ------------ make plot ------------ 
print(' ... read avg file data + make plot ... ')
# ------------ read data ------------ 
data.get_outputs(0,var_list,get_date=False)
[z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
h_tile = np.transpose(np.tile(data.h,(200,1,1)),(1,2,0))
hab    = h_tile + z_r
# --> hab w 
hw_tile = np.transpose(np.tile(data.h,(201,1,1)),(1,2,0))
habw    = hw_tile + z_w
kkpp = tools.w2rho(np.asfortranarray(data.var['AKt']))
print( ' ############ -------------------------------------------------------------------- ################')
keff = data.get_diffusivity(0,'diffusivity')
N2   = data.var['bvf']
hbbl = data.var['hbbl']



# --> find bottom upwelling and downwelling 
area_tot  = find_upwelling_downwelling(w[:,:,0],N2_bbl,data.lonr,data.latr)

# --> ravel variables 
area_r             = np.ravel(area_tot)
area_r3d           = np.ravel(np.transpose(np.tile(area_tot,(200,1,1)),(1,2,0)))
area_r3dw          = np.ravel(np.transpose(np.tile(area_tot,(201,1,1)),(1,2,0)))
hab_ri             = np.ravel(hab)
habw_ri            = np.ravel(habw)
hbbl_ri            = np.ravel(hbbl)
kkpp_ri            = np.ravel(kkpp)
keff_ri            = np.ravel(keff)
N2_ri              = np.ravel(N2)


# --> statistics of variables in subdomains
nz = nbin 
kkpp_r  = np.zeros((len(choice_list),nz))
keff_r  = np.zeros((len(choice_list),nz))
N2_r    = np.zeros((len(choice_list),nz))
hbbl_r  = np.zeros(len(choice_list))
hab_bblt= np.zeros(len(choice_list))



for choice in range(len(choice_list)):
    if choice==0:  # upwelling N2>0
        hab_bblt[choice]   = np.nanmean(hab_bbl[area_r==1])
        hab_up             = np.copy(hab_ri[area_r3d==1])
        habw_up            = np.copy(habw_ri[area_r3dw==1])
        hbbl_up            = np.copy(hbbl_ri[area_r==1])
        kkpp_up            = np.copy(kkpp_ri[area_r3d==1])
        keff_up            = np.copy(keff_ri[area_r3d==1])
        N2_up              = np.copy(N2_ri[area_r3dw==1])
        # binned in hab space 
        h_r              = hab_up
        hw_r             = habw_up 
        hbbl_r[choice]   = np.mean(hbbl_up)
        kkpp_r[choice,:] = stats.binned_statistic(h_r,kkpp_up,statistic='mean',bins=h_e)[0]
        keff_r[choice,:] = stats.binned_statistic(h_r,keff_up,statistic=np.nanmean,bins=h_e)[0]
        # on w-grid
        N2_r[choice,:]       = stats.binned_statistic(hw_r,N2_up,statistic='mean',bins=h_e)[0]


    elif choice==1 :         # downwelling N2>0
        hab_bblt[choice]     = np.nanmean(hab_bbl[area_r==9999])
        hab_down             = np.copy(hab_ri[area_r3d==9999])
        print('downwelling :',len(hab_down))
        habw_down            = np.copy(habw_ri[area_r3dw==9999])
        hbbl_down            = np.copy(hbbl_ri[area_r==9999])
        kkpp_down            = np.copy(kkpp_ri[area_r3d==9999])
        keff_down            = np.copy(keff_ri[area_r3d==9999])
        N2_down              = np.copy(N2_ri[area_r3dw==9999])
        # binned in hab space 
        h_r              = hab_down
        hw_r             = habw_down
        hbbl_r[choice]   = np.mean(hbbl_down)
        kkpp_r[choice,:] = stats.binned_statistic(h_r,kkpp_down,statistic='mean',bins=h_e)[0]
        keff_r[choice,:] = stats.binned_statistic(h_r,keff_down,statistic=np.nanmean,bins=h_e)[0]
        # on w-grid
        N2_r[choice,:]       = stats.binned_statistic(hw_r,N2_down,statistic='mean',bins=h_e)[0]

    elif choice==2:  # upwelling N2<0
        hab_bblt[choice]     = np.nanmean(hab_bbl[area_r==-1])
        hab_up             = np.copy(hab_ri[area_r3d==-1])
        print('upwelling N2<0:',len(hab_up))
        habw_up            = np.copy(habw_ri[area_r3dw==-1])
        hbbl_up            = np.copy(hbbl_ri[area_r==-1])
        kkpp_up            = np.copy(kkpp_ri[area_r3d==-1])
        keff_up            = np.copy(keff_ri[area_r3d==-1])
        N2_up              = np.copy(N2_ri[area_r3dw==-1])
        # binned in hab space 
        h_r              = hab_up
        hw_r             = habw_up
        hbbl_r[choice]   = np.mean(hbbl_up)
        kkpp_r[choice,:] = stats.binned_statistic(h_r,kkpp_up,statistic='mean',bins=h_e)[0]
        keff_r[choice,:] = stats.binned_statistic(h_r,keff_up,statistic=np.nanmean,bins=h_e)[0]
        # on w-grid
        N2_r[choice,:]       = stats.binned_statistic(hw_r,N2_up,statistic='mean',bins=h_e)[0]
        
    elif choice==3 :         # downwelling N2<0
        hab_bblt[choice]     = np.nanmean(hab_bbl[area_r==-9999])
        hab_down             = np.copy(hab_ri[area_r3d==-9999])
        print('downwelling N2<0:',len(hab_down))
        habw_down            = np.copy(habw_ri[area_r3dw==-9999])
        hbbl_down            = np.copy(hbbl_ri[area_r==-9999])
        kkpp_down            = np.copy(kkpp_ri[area_r3d==-9999])
        keff_down            = np.copy(keff_ri[area_r3d==-9999])
        N2_down              = np.copy(N2_ri[area_r3dw==-9999])
        # binned in hab space 
        h_r              = hab_down
        hw_r             = habw_down
        hbbl_r[choice]   = np.mean(hbbl_down)
        kkpp_r[choice,:] = stats.binned_statistic(h_r,kkpp_down,statistic='mean',bins=h_e)[0]
        keff_r[choice,:] = stats.binned_statistic(h_r,keff_down,statistic=np.nanmean,bins=h_e)[0]
        # on w-grid
        N2_r[choice,:]       = stats.binned_statistic(hw_r,N2_down,statistic='mean',bins=h_e)[0]
        

    else:  # all grid
        hab_bblt[choice]   = np.nanmean(hab_bbl)
        hab_ag             = np.copy(hab_ri)
        print('all grid :')
        habw_ag            = np.copy(habw_ri)
        hbbl_ag            = np.copy(hbbl_ri)
        kkpp_ag            = np.copy(kkpp_ri)
        keff_ag            = np.copy(keff_ri)
        N2_ag              = np.copy(N2_ri)
        # binned in hab space
        h_r              = hab_ag
        hw_r             = habw_ag
        hbbl_r[choice]   = np.mean(hbbl_ag)
        kkpp_r[choice,:] = stats.binned_statistic(h_r,kkpp_ag,statistic='mean',bins=h_e)[0]
        keff_r[choice,:] = stats.binned_statistic(h_r,keff_ag,statistic=np.nanmean,bins=h_e)[0]
        # on w-grid
        N2_r[choice,:]       = stats.binned_statistic(hw_r,N2_ag,statistic='mean',bins=h_e)[0]

            
# -- masks ---
kkpp_r[kkpp_r<1e-5]=1e-5


#############################################################################
print(' --- make plot appendix ---')
count=0
# ---- Make plot ----
figure = plt.figure(figsize=(40,30))
gs = gridspec.GridSpec(2,5,wspace=0.4,hspace=0.2) #0.4
for choice in range(len(choice_list)):
    ic = choice
    ax = plt.subplot(gs[0,ic]) # --> KKPP, KEFF
    plt.plot(kkpp_r[choice,:],h_c,color='k',linewidth = 2,label=r'$\kappa_{KPP}$')
    plt.plot(keff_r[choice,:],h_c,color='r',linewidth = 2,label=r'$\kappa_{eff}$')
    plt.axvline(x=1e-5,color='k',linewidth=lwc,alpha=0.5)
    plt.axvline(x=1e-4,color='k',linewidth=lwc,alpha=0.5)
    plt.axvline(x=1e-3,color='k',linewidth=lwc,alpha=0.5)
    plt.axvline(x=1e-2,color='k',linewidth=lwc,alpha=0.5)
    plt.axvline(x=1e-1,color='k',linewidth=lwc,alpha=0.5)
    plt.axhline(y=hab_bblt[choice],color='k',linewidth=lwc+1,linestyle='dashed',alpha=0.5)
    plt.axhline(y=hbbl_r[choice],color='r',linewidth=lwc+1,linestyle='dashed',alpha=0.5)
    plt.legend(fontsize=fs)
    plt.ylim(0,200)
    plt.xlim(7e-6,1)
    ax.set_xticks([1e-5,1e-3,1e-1])
    plt.xscale('log')
    plt.xlabel(r'$\kappa$ [$m^2$ $s^{-1}$]',fontsize=fs)
    plt.ylabel('hab [m]',fontsize=fs)    
    plt.title(label_plot2[count],fontsize=fs)
    ax.tick_params(labelsize=fs)
    count+=1

    ax = plt.subplot(gs[1,ic]) # --> N2
    plt.plot(N2_r[choice,:],h_c,color='k',linewidth = 2)
    plt.axvline(x=0,color='k',linewidth=lwc,alpha=0.8)
    plt.axvline(x=0.5e-6,color='k',linewidth=lwc,alpha=0.5)
    plt.axvline(x=1e-6,color='k',linewidth=lwc,alpha=0.5)
    plt.axhline(y=hab_bblt[choice],color='k',linewidth=lwc+1,linestyle='dashed',alpha=0.5)
    plt.axhline(y=hbbl_r[choice],color='r',linewidth=lwc+1,linestyle='dashed',alpha=0.5)
    plt.ylim(0,200)
    plt.xlim(-0.2e-6,1.5e-6)
    ax.set_xticks([0,0.5e-6,1e-6,1.5e-6])
    plt.ylabel('hab [m]',fontsize=fs)
    plt.xlabel(r'$N^2$ [$s^{-2}$]',fontsize=fs)
    ax.tick_params(labelsize=fs)
    plt.title(label_plot2[count],fontsize=fs)
    count+=1

plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/WMT/rrexnumsb200-rsup5_buoyancy_balance_mean_upwelling_downelling_appendix.png',dpi=200,bbox_inches='tight')
plt.close()

