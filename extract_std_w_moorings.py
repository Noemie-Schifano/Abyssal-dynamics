''''
NS 2024/04/12: make composite plots of tracer dispersion   
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
sys.path.append('/home/datawork-lops-rrex/nschifan/Python_Modules_p3/')
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

# ------------ parameters in-situ ------------ 
choice_mooring  = ['IRW','IRM','IRE','RRT','ICW','ICM','ICE']
nr              = [30,  3, 66, 93, 63, 29, 29] # number of measures for each mooring
file_data_d     = '/home/datawork-lops-rrex/nschifan/Data_in_situ_Rene/rrex_dD_10km.nc'
file_data       = '/home/datawork-lops-rrex/nschifan/Data_in_situ_Rene/RREX_'
end_nc          = '_CURR_hourly.nc'
file_out        = '/home/datawork-lops-rrex/nschifan/Data_in_situ_Rene/moorings_w_std.nc'
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
fs      = 20       # fontsize 
lw      = 2     # linewidth
ms      = 20      # markersize 
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

# --- bin hab
dz = 10. # [m] 
zbine = np.arange(-5000,dz,dz)     # bin edges 
zbinc = 0.5*(zbine[1:]+zbine[:-1]) # bin centres 
nz    = zbinc.shape[0]
# - regrid in height-above-bottom (hab) coordinates - 


# --- w ---
label_w     = r'w [m day$^{-1}$]'

# ----------- read in-situ currents ---------
######### IRW 
nc       = Dataset(file_data+choice_mooring[0]+end_nc,'r')
uirw_qc  = nc.variables['UCUR_QC'][:,:]
virw_qc  = nc.variables['VCUR_QC'][:,:]
u_irw    = nc.variables['UCUR'][:,:]
v_irw    = nc.variables['VCUR'][:,:]
d_irw    = nc.variables['DEPTH'][:,:]
nc.close()
# --> QUALITFY FLAG
u_irw[uirw_qc>2]=np.nan
v_irw[virw_qc>2]=np.nan

######### IRM
nc       = Dataset(file_data+choice_mooring[1]+end_nc,'r')
uirm_qc  = nc.variables['UCUR_QC'][:,:]
virm_qc  = nc.variables['VCUR_QC'][:,:]
u_irm    = nc.variables['UCUR'][:,:]
v_irm    = nc.variables['VCUR'][:,:]
d_irm    = nc.variables['DEPTH'][:,:]
nc.close()
# --> QUALITFY FLAG
u_irm[uirm_qc>2]=np.nan
v_irm[virm_qc>2]=np.nan

######### IRE
nc       = Dataset(file_data+choice_mooring[2]+end_nc,'r')
uire_qc  = nc.variables['UCUR_QC'][:,:]
vire_qc  = nc.variables['VCUR_QC'][:,:]
u_ire    = nc.variables['UCUR'][:,:]
v_ire    = nc.variables['VCUR'][:,:]
d_ire    = nc.variables['DEPTH'][:,:]
nc.close()
# --> QUALITFY FLAG
u_irw[uirw_qc>2]=np.nan
v_irw[virw_qc>2]=np.nan

######### RRT
nc       = Dataset(file_data+choice_mooring[3]+end_nc,'r')
urrt_qc  = nc.variables['UCUR_QC'][:,:]
vrrt_qc  = nc.variables['VCUR_QC'][:,:]
u_rrt    = nc.variables['UCUR'][:,:]
v_rrt    = nc.variables['VCUR'][:,:]
d_rrt    = nc.variables['DEPTH'][:,:]
nc.close()
# --> QUALITFY FLAG
u_rrt[urrt_qc>2]=np.nan
v_rrt[vrrt_qc>2]=np.nan

######### ICW
nc       = Dataset(file_data+choice_mooring[4]+end_nc,'r')
uicw_qc  = nc.variables['UCUR_QC'][:,:]
vicw_qc  = nc.variables['VCUR_QC'][:,:]
u_icw    = nc.variables['UCUR'][:,:]
v_icw    = nc.variables['VCUR'][:,:]
d_icw    = nc.variables['DEPTH'][:,:]
nc.close()
# --> QUALITFY FLAG
u_icw[uicw_qc>2]=np.nan
v_icw[vicw_qc>2]=np.nan

######### ICM
nc       = Dataset(file_data+choice_mooring[5]+end_nc,'r')
uicm_qc  = nc.variables['UCUR_QC'][:,:]
vicm_qc  = nc.variables['VCUR_QC'][:,:]
u_icm    = nc.variables['UCUR'][:,:]
v_icm    = nc.variables['VCUR'][:,:]
d_icm    = nc.variables['DEPTH'][:,:]
nc.close()
# --> QUALITFY FLAG
u_icm[uicm_qc>2]=np.nan
v_icm[vicm_qc>2]=np.nan


######### ICE
nc       = Dataset(file_data+choice_mooring[6]+end_nc,'r')
uice_qc  = nc.variables['UCUR_QC'][:,:]
vice_qc  = nc.variables['VCUR_QC'][:,:]
u_ice    = nc.variables['UCUR'][:,:]
v_ice    = nc.variables['VCUR'][:,:]
d_ice    = nc.variables['DEPTH'][:,:]
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


# ----------- compute_w  ---------
w_irw = -1*(u_irw*dDdx[0]+v_irw*dDdy[0])*3600*24
w_irm = -1*(u_irm*dDdx[1]+v_irm*dDdy[1])*3600*24
w_ire = -1*(u_ire*dDdx[2]+v_ire*dDdy[2])*3600*24
w_rrt = -1*(u_rrt*dDdx[3]+v_rrt*dDdy[3])*3600*24
w_icw = -1*(u_icw*dDdx[4]+v_icw*dDdy[4])*3600*24
w_icm = -1*(u_icm*dDdx[5]+v_icm*dDdy[5])*3600*24
w_ice = -1*(u_ice*dDdx[6]+v_ice*dDdy[6])*3600*24

print(np.shape(w_irw),np.shape(w_irm))

# -------- compute statistics w ------
stdw0 = np.zeros(nr[0])
stdw1 = np.zeros(nr[1])
stdw2 = np.zeros(nr[2])
stdw3 = np.zeros(nr[3])
stdw4 = np.zeros(nr[4])
stdw5 = np.zeros(nr[5])
stdw6 = np.zeros(nr[6])


std   = [ stdw0, stdw1, stdw2, stdw3, stdw4, stdw5, stdw6 ]
w     = [ w_irw, w_irm, w_ire, w_rrt, w_icw, w_icm, w_ice ]

for u in range(len(nr)):
    for k in range(nr[u]):
        print(u,k)
        std[u][k]    = np.std(w[u][:,k])


print('===================================================')
print(' ... save in netcdf file ... ')
print('===================================================')
nc = Dataset(file_out,'w')
nc.createDimension('nr0',nr[0])
nc.createDimension('nr1',nr[1])
nc.createDimension('nr2',nr[2])
nc.createDimension('nr3',nr[3])
nc.createDimension('nr4',nr[4])
nc.createDimension('nr5',nr[5])
nc.createDimension('nr6',nr[6])

nc.createVariable('std0','f',('nr0'))
nc.createVariable('std1','f',('nr1'))
nc.createVariable('std2','f',('nr2'))
nc.createVariable('std3','f',('nr3'))
nc.createVariable('std4','f',('nr4'))
nc.createVariable('std5','f',('nr5'))
nc.createVariable('std6','f',('nr6'))


nc.variables['std0'][:]     = stdw0
nc.variables['std1'][:]     = stdw1
nc.variables['std2'][:]     = stdw2
nc.variables['std3'][:]     = stdw3
nc.variables['std4'][:]     = stdw4
nc.variables['std5'][:]     = stdw5
nc.variables['std6'][:]     = stdw6


nc.close()








