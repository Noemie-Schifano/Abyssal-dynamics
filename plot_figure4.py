''''
NS 2024/04/12: Need "extract_bottom_w_moorings_croco_notracer.py" to run before   
               Histogramms of the deepest vertical velocity from the no-normal flow condition w at mooring locations
                       and interpolated with croco model output
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
from croco_simulations_jonathan_ncra import Croco
from croco_simulations_jonathan_ncra_longrun import Croco_longrun
import gsw as gsw
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
matplotlib.rcParams.update({'font.size': 14})
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import FormatStrFormatter

# ------------ parameters in-situ ------------ 
choice_mooring  = ['IRW','IRM','IRE','RRT','ICW','ICM','ICE']
file_data_d     = 'rrex_dD_10km.nc'
file_data       = 'RREX_'
file_croco      = 'rrex_bottom_w_mooring_notracer.nc'
label_croco     = ['-2104 m','-1603 m', '-1519 m', '-1449 m', '-2104 m',
                         '-2212 m', '-2379 m' ]  # Dd
hab_moorings    = ['100','93', '116','81','57','46','282']  
end_nc          = '_CURR_hourly.nc'
file_out        = 'rrex_u_v_full.nc'

# ------------ parameters croco ------------ 
name_exp    = 'rrexnum200' 
name_exp_path ='rrexnum200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_NOFILT_T'
nbr_levels  = '200'
name_exp_grd= ''
ndfiles     = 1  # number of days per netcdf
time= ['20']
var_list = ['w','zeta','u','v']

# --- plot options --- 
jsec    = 400
fs      = 20    # fontsize 
lw      = 2     # linewidth
ms      = 20    # markersize 
lw_c    = 0.2   # linewidth coast 
my_bbox = dict(fc='w',ec='k',pad=2,lw=0.,alpha=0.5)
res   = 'h' # resolution of the coastline: c (coarse), l (low), i (intermediate), h (high)
proj  = 'lcc'  # map projection. lcc = lambert conformal 
paral = np.arange(0,80,5)
merid = np.arange(0,360,5)
lon_0,lat_0   = -31.5,58.6 # centre of the map 
Lx,Ly         = 1400e3,1500e3 # [km] zonal and meridional extent of the map 
scale_km      = 400 # [km] map scale 
cf0           = colors.to_rgba('teal')
alpha_plot    = 0.5
plot_log      = False

# --- bin hab
dz = 10. # [m] 
zbine = np.arange(-5000,dz,dz)     # bin edges 
zbinc = 0.5*(zbine[1:]+zbine[:-1]) # bin centres 
nz    = zbinc.shape[0]
 

# --- bin w
dw  = 2. # [m/day]
w_e = np.arange(-750,750+dw,dw)

# ---------> choose w or u.grad(h)
list_choice = ['w','w_from_u']
choice      = list_choice[1]
if choice == 'w':
    # --- w ---
    label_w     = r'w [m day$^{-1}$]'
    cf0         = colors.to_rgba('teal')
else:
    # --- w from u ---
    label_w     = r'-$\overline{u}^h$ $\cdot$ $\nabla^h$ H [m day$^{-1}$]'
    cf0         = colors.to_rgba('royalblue')

# ----------- read in-situ currents ---------
######### IRW 
nc       = Dataset(file_data+choice_mooring[0]+end_nc,'r')
uirw_qc  = nc.variables['UCUR_QC'][:,-1]
virw_qc  = nc.variables['VCUR_QC'][:,-1]
u_irw    = nc.variables['UCUR'][:,-1]
v_irw    = nc.variables['VCUR'][:,-1]
d_irw    = nc.variables['DEPTH'][:,-1]
nc.close()
# --> QUALITFY FLAG
u_irw[uirw_qc>2]=np.nan
v_irw[virw_qc>2]=np.nan

######### IRM
nc       = Dataset(file_data+choice_mooring[1]+end_nc,'r')
uirm_qc  = nc.variables['UCUR_QC'][:,-1]
virm_qc  = nc.variables['VCUR_QC'][:,-1]
u_irm    = nc.variables['UCUR'][:,-1]
v_irm    = nc.variables['VCUR'][:,-1]
d_irm    = nc.variables['DEPTH'][:,-1]
nc.close()
# --> QUALITFY FLAG
u_irm[uirm_qc>2]=np.nan
v_irm[virm_qc>2]=np.nan

######### IRE
nc       = Dataset(file_data+choice_mooring[2]+end_nc,'r')
uire_qc  = nc.variables['UCUR_QC'][:,-1]
vire_qc  = nc.variables['VCUR_QC'][:,-1]
u_ire    = nc.variables['UCUR'][:,-1]
v_ire    = nc.variables['VCUR'][:,-1]
d_ire    = nc.variables['DEPTH'][:,-1]
nc.close()
# --> QUALITFY FLAG
u_irw[uirw_qc>2]=np.nan
v_irw[virw_qc>2]=np.nan

######### RRT
nc       = Dataset(file_data+choice_mooring[3]+end_nc,'r')
urrt_qc  = nc.variables['UCUR_QC'][:,-1]
vrrt_qc  = nc.variables['VCUR_QC'][:,-1]
u_rrt    = nc.variables['UCUR'][:,-1]
v_rrt    = nc.variables['VCUR'][:,-1]
d_rrt    = nc.variables['DEPTH'][:,-1]
nc.close()
# --> QUALITFY FLAG
u_rrt[urrt_qc>2]=np.nan
v_rrt[vrrt_qc>2]=np.nan

######### ICW
nc       = Dataset(file_data+choice_mooring[4]+end_nc,'r')
uicw_qc  = nc.variables['UCUR_QC'][:,-1]
vicw_qc  = nc.variables['VCUR_QC'][:,-1]
u_icw    = nc.variables['UCUR'][:,-1]
v_icw    = nc.variables['VCUR'][:,-1]
d_icw    = nc.variables['DEPTH'][:,-1]
nc.close()
# --> QUALITFY FLAG
u_icw[uicw_qc>2]=np.nan
v_icw[vicw_qc>2]=np.nan

######### ICM
nc       = Dataset(file_data+choice_mooring[5]+end_nc,'r')
uicm_qc  = nc.variables['UCUR_QC'][:,-1]
vicm_qc  = nc.variables['VCUR_QC'][:,-1]
u_icm    = nc.variables['UCUR'][:,-1]
v_icm    = nc.variables['VCUR'][:,-1]
d_icm    = nc.variables['DEPTH'][:,-1]
nc.close()
# --> QUALITFY FLAG
u_icm[uicm_qc>2]=np.nan
v_icm[vicm_qc>2]=np.nan

######### ICE
nc       = Dataset(file_data+choice_mooring[6]+end_nc,'r')
uice_qc  = nc.variables['UCUR_QC'][:,-1]
vice_qc  = nc.variables['VCUR_QC'][:,-1]
u_ice    = nc.variables['UCUR'][:,-1]
v_ice    = nc.variables['VCUR'][:,-1]
d_ice    = nc.variables['DEPTH'][:,-1]
nc.close()
# --> QUALITFY FLAG
u_ice[uice_qc>2]=np.nan
v_ice[vice_qc>2]=np.nan



# ----------- read in-situ dDdx and dDdy ---------
ncd     = Dataset(file_data_d,'r')
dDdx    = ncd.variables['dDdx'][:] 
dDdy    = ncd.variables['dDdy'][:]
D       = ncd.variables['D'][:]
ncd.close()

print('depth ')

# ----------- compute_w  ---------
w_irw = -1*(u_irw*dDdx[0]+v_irw*dDdy[0])*3600*24
w_irm = -1*(u_irm*dDdx[1]+v_irm*dDdy[1])*3600*24
w_ire = -1*(u_ire*dDdx[2]+v_ire*dDdy[2])*3600*24
w_rrt = -1*(u_rrt*dDdx[3]+v_rrt*dDdy[3])*3600*24
w_icw = -1*(u_icw*dDdx[4]+v_icw*dDdy[4])*3600*24
w_icm = -1*(u_icm*dDdx[5]+v_icm*dDdy[5])*3600*24
w_ice = -1*(u_ice*dDdx[6]+v_ice*dDdy[6])*3600*24

# --> compute median and std
print('-----> STATISTICS IN-SITU')
W     = [w_irw,w_irm,w_ire,
         w_rrt,w_icw,w_icm,
         w_ice]
med = np.zeros(np.shape(W)[0])
std = np.zeros(np.shape(W)[0])
for m in range(np.shape(W)[0]):
    med[m]   = np.median(W[m])
    std[m]   = np.std(W[m],axis=0)


# --------- read bottom w from CROCO ---------
ncc    =  Dataset(file_croco,'r')
if choice == 'w':
    wc_irw = ncc.variables['wb0'][:]
    wc_irm = ncc.variables['wb1'][:]
    wc_ire = ncc.variables['wb2'][:]
    wc_rrt = ncc.variables['wb3'][:]
    wc_icw = ncc.variables['wb4'][:]
    wc_icm = ncc.variables['wb5'][:]
    wc_ice = ncc.variables['wb6'][:]
else:
    wc_irw = ncc.variables['wub0'][:]
    wc_irm = ncc.variables['wub1'][:]
    wc_ire = ncc.variables['wub2'][:]
    wc_rrt = ncc.variables['wub3'][:]
    wc_icw = ncc.variables['wub4'][:]
    wc_icm = ncc.variables['wub5'][:]
    wc_ice = ncc.variables['wub6'][:]
ncc.close()

# --> compute median and std
print('-----> STATISTICS IN-SITU')
Wc     = [wc_irw,wc_irm,wc_ire,
         wc_rrt,wc_icw,wc_icm,
         wc_ice]
medc = np.zeros(np.shape(Wc)[0])
stdc = np.zeros(np.shape(Wc)[0])
for m in range(np.shape(Wc)[0]):
    medc[m]   = np.median(Wc[m])
    stdc[m]   = np.std(Wc[m],axis=0)



print(' ... make plot ... ')
plt.figure(figsize=(20,20))
gs = gridspec.GridSpec(7,1,hspace=0.5)

ax = plt.subplot(gs[0])  ########################## IRW
plt.title('a) IRW, hab = '+hab_moorings[0] +' m',fontsize=fs)
data = w_irw
plt.hist(data,weights=np.ones(len(data)) / len(data),color='m',alpha=alpha_plot,label='Mooring')
data = wc_irw
plt.hist(data,weights=np.ones(len(data))/len(data),color=cf0,alpha=alpha_plot,label='CROCO')
plt.ylabel(r' pdf [$\%$]')
plt.legend(loc='upper right')
plt.axvline(x=0,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=100,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=200,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=300,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-100,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-200,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-300,color='k',linewidth=1,alpha=0.2)
m= 0
plt.axvline(x=med[m],color='r',linewidth=2)
plt.axvline(x=med[m]+std[m],color='r',linewidth=2,linestyle='dashed')
plt.axvline(x=med[m]-std[m],color='r',linewidth=2,linestyle='dashed')
plt.axvline(x=medc[m],color='k',linewidth=2)
plt.axvline(x=medc[m]+stdc[m],color='k',linewidth=2,linestyle='dashed')
plt.axvline(x=medc[m]-stdc[m],color='k',linewidth=2,linestyle='dashed')
plt.xlim(-310,310)

if plot_log==True:
    plt.yscale('log')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))

ax = plt.subplot(gs[1])  ########################## IRM
plt.title('b) IRM, hab = '+hab_moorings[1] +' m',fontsize=fs)
data = w_irm
plt.hist(data,bins=w_e,weights=np.ones(len(data)) / len(data),color='m',alpha=alpha_plot,label='-2060 m')
data = wc_irm
plt.hist(data,bins=w_e, weights=np.ones(len(data)) / len(data),color=cf0,alpha=alpha_plot,label=label_croco[0])
plt.ylabel(r' pdf [$\%$]')
plt.axvline(x=0,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=100,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=200,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=300,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-100,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-200,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-300,color='k',linewidth=1,alpha=0.2)
m= 1
plt.axvline(x=med[m],color='r',linewidth=2)
plt.axvline(x=med[m]+std[m],color='r',linewidth=2,linestyle='dashed')
plt.axvline(x=med[m]-std[m],color='r',linewidth=2,linestyle='dashed')
plt.axvline(x=medc[m],color='k',linewidth=2)
plt.axvline(x=medc[m]+stdc[m],color='k',linewidth=2,linestyle='dashed')
plt.axvline(x=medc[m]-stdc[m],color='k',linewidth=2,linestyle='dashed')
plt.xlim(-310,310)

if plot_log==True:
    plt.yscale('log')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))


ax = plt.subplot(gs[2])  ########################## IRE
plt.title('c) IRE, hab = '+hab_moorings[2] +' m',fontsize=fs)
data = w_ire
plt.hist(data,bins=w_e,weights=np.ones(len(data)) / len(data),color='m',alpha=alpha_plot,label='-2060 m')
data = wc_ire
plt.hist(data,bins=w_e, weights=np.ones(len(data)) / len(data),color=cf0,alpha=alpha_plot,label=label_croco[0])
plt.ylabel(r' pdf [$\%$]')
plt.axvline(x=0,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=100,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=200,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=300,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-100,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-200,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-300,color='k',linewidth=1,alpha=0.2)
m= 2
plt.axvline(x=med[m],color='r',linewidth=2)
plt.axvline(x=med[m]+std[m],color='r',linewidth=2,linestyle='dashed')
plt.axvline(x=med[m]-std[m],color='r',linewidth=2,linestyle='dashed')
plt.axvline(x=medc[m],color='k',linewidth=2)
plt.axvline(x=medc[m]+stdc[m],color='k',linewidth=2,linestyle='dashed')
plt.axvline(x=medc[m]-stdc[m],color='k',linewidth=2,linestyle='dashed')
plt.xlim(-310,310)
if plot_log==True:
    plt.yscale('log')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))


ax = plt.subplot(gs[3])  ########################## RRT
plt.title('d) RRT, hab = '+hab_moorings[3] +' m',fontsize=fs)
data = w_rrt
plt.hist(data,bins=w_e,weights=np.ones(len(data)) / len(data),color='m',alpha=alpha_plot,label='-2060 m')
data = wc_rrt
plt.hist(data,bins=w_e, weights=np.ones(len(data)) / len(data),color=cf0,alpha=alpha_plot,label=label_croco[0])
plt.ylabel(r' pdf [$\%$]')
plt.axvline(x=0,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=100,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=200,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=300,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-100,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-200,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-300,color='k',linewidth=1,alpha=0.2)
m= 3
plt.axvline(x=med[m],color='r',linewidth=2)
plt.axvline(x=med[m]+std[m],color='r',linewidth=2,linestyle='dashed')
plt.axvline(x=med[m]-std[m],color='r',linewidth=2,linestyle='dashed')
plt.axvline(x=medc[m],color='k',linewidth=2)
plt.axvline(x=medc[m]+stdc[m],color='k',linewidth=2,linestyle='dashed')
plt.axvline(x=medc[m]-stdc[m],color='k',linewidth=2,linestyle='dashed')
plt.xlim(-310,310)
if plot_log==True:
    plt.yscale('log')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))


ax = plt.subplot(gs[4])  ########################## ICW
plt.title('e) ICW, hab = '+hab_moorings[4] +' m',fontsize=fs)
data = w_icw
plt.hist(data,bins=w_e,weights=np.ones(len(data)) / len(data),color='m',alpha=alpha_plot,label='-2060 m')
data = wc_icw
plt.hist(data,bins=w_e, weights=np.ones(len(data)) / len(data),color=cf0,alpha=alpha_plot,label=label_croco[0])
plt.ylabel(r' pdf [$\%$]')
plt.axvline(x=0,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=100,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=200,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=300,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-100,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-200,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-300,color='k',linewidth=1,alpha=0.2)
m= 4
plt.axvline(x=med[m],color='r',linewidth=2)
plt.axvline(x=med[m]+std[m],color='r',linewidth=2,linestyle='dashed')
plt.axvline(x=med[m]-std[m],color='r',linewidth=2,linestyle='dashed')
plt.axvline(x=medc[m],color='k',linewidth=2)
plt.axvline(x=medc[m]+stdc[m],color='k',linewidth=2,linestyle='dashed')
plt.axvline(x=medc[m]-stdc[m],color='k',linewidth=2,linestyle='dashed')
plt.xlim(-310,310)
if plot_log==True:
    plt.yscale('log')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))


ax = plt.subplot(gs[5])  ########################## ICM
plt.title('f) ICM, hab = '+hab_moorings[5] +' m',fontsize=fs)
data = w_icm
plt.hist(data,bins=w_e,weights=np.ones(len(data)) / len(data),color='m',alpha=alpha_plot,label='-2060 m')
data = wc_icm
plt.hist(data,bins=w_e, weights=np.ones(len(data)) / len(data),color=cf0,alpha=alpha_plot,label=label_croco[0])
plt.ylabel(r' pdf [$\%$]')
plt.axvline(x=0,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=100,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=200,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=300,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-100,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-200,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-300,color='k',linewidth=1,alpha=0.2)
m= 5
plt.axvline(x=med[m],color='r',linewidth=2)
plt.axvline(x=med[m]+std[m],color='r',linewidth=2,linestyle='dashed')
plt.axvline(x=med[m]-std[m],color='r',linewidth=2,linestyle='dashed')
plt.axvline(x=medc[m],color='k',linewidth=2)
plt.axvline(x=medc[m]+stdc[m],color='k',linewidth=2,linestyle='dashed')
plt.axvline(x=medc[m]-stdc[m],color='k',linewidth=2,linestyle='dashed')
plt.xlim(-310,310)
if plot_log==True:
    plt.yscale('log')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))


ax = plt.subplot(gs[6])  ########################## ICE
plt.title('g) ICE, hab = '+hab_moorings[6] +' m',fontsize=fs)
data = w_ice
plt.hist(data,bins=w_e,weights=np.ones(len(data)) / len(data),color='m',alpha=alpha_plot,label='-2060 m')
data = wc_ice
plt.hist(data,bins=w_e, weights=np.ones(len(data)) / len(data),color=cf0,alpha=alpha_plot,label=label_croco[0])
plt.ylabel(r' pdf [$\%$]')
plt.xlabel(label_w)
plt.axvline(x=0,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=100,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=200,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=300,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-100,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-200,color='k',linewidth=1,alpha=0.2)
plt.axvline(x=-300,color='k',linewidth=1,alpha=0.2)
m= 6
plt.axvline(x=med[m],color='r',linewidth=2)
plt.axvline(x=med[m]+std[m],color='r',linewidth=2,linestyle='dashed')
plt.axvline(x=med[m]-std[m],color='r',linewidth=2,linestyle='dashed')
plt.axvline(x=medc[m],color='k',linewidth=2)
plt.axvline(x=medc[m]+stdc[m],color='k',linewidth=2,linestyle='dashed')
plt.axvline(x=medc[m]-stdc[m],color='k',linewidth=2,linestyle='dashed')
plt.xlim(-310,310)
if plot_log==True:
    plt.yscale('log')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
if plot_log == True:
    plt.savefig('figure4.png',dpi=180,bbox_inches='tight')
else:
    plt.savefig('figure4_nolog.png',dpi=180,bbox_inches='tight')
plt.close()

















