"""
NS: Extract for each tracer, the depth, the porcentage of tracer in the BBL and the height of the BBL 
    For TRE_ref set of simulation 
"""


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
import gsw as gsw
import obsfit1d as fit
#import time as time
from croco_simulations_jon_hist_last import Croco
matplotlib.rcParams.update({'font.size': 14})

# ------------ parameters ------------ 
name_exp    = 'rrexnum200'
name_exp_path ='rrexnum200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_T'
nbr_levels  = '200'
name_exp_grd= ''
test_nc    = False
if test_nc == True :
    ndfiles     = 2
    time        = np.arange(0,1)
    file_diag_k = 'test_rrexnumsb200-rsup5_16T_porcentage_in_BBL.nc'
else:
    ndfiles     = 24
    time        = np.arange(0,25,24)
    file_diag_k = 'rrexnumsb200-rsup5_16T_porcentage_in_BBL.nc'
nt          = len(time)*ndfiles
dt          = 1*3600 #1h 
# - select tracers for analysis - 
tpas_list =  ['tpas0'+str(i) for i in range(1,10)]+['tpas'+str(i) for i in range(10,17)]
ntpas = len(tpas_list)
var_list  = ['zeta','hbbl']
var_list += tpas_list 


# --- variables to be saved ---
porc_tpas_inbbl = np.zeros((ntpas,nt))
hbbl_tpas       = np.zeros((ntpas,nt))
z_r_tpas        = np.zeros((ntpas,nt))

# --- function ---
def tracer_avg(var,tpas,dvol):
    # dvol is the cell volume  
    #return np.nansum(var*tpas)/np.nansum(tpas)
    return np.nansum(var*tpas*dvol)/np.nansum(tpas*dvol)


def def_mask_BBL(z_r,zbbl):
    #return a 3D mask
    # mask_bbl = 0 is depth > depth BBL
    # mask_bbl = 1 otherwise
    # z_r : 3D depth
    # zbbl: 2D (x,y) depth of BBL
    mask_bbl = np.zeros(np.shape(z_r))
    for i in range(np.shape(z_r)[0]):
        for j in range(np.shape(z_r)[1]):
            for k in range(np.shape(z_r)[2]):
                if z_r[i,j,k]<=zbbl[i,j]:
                    mask_bbl[i,j,k]=1
    return mask_bbl

# --- read data ---
tt = 0
for t_nc in range(len(time)):
    data = Croco_6h(name_exp,nbr_levels,str(int(time[t_nc])),name_exp_grd,name_pathdata)
    data.get_grid()
    dsurf   = 1./np.transpose(np.tile(data.pm*data.pn,(int(nbr_levels),1,1)),(1,2,0)) # horizontal surface area
    for t in range(0,ndfiles):
        print('=====================  time index %.4i ====================='%t)
        print('    ---> read outputs ')
        data.get_outputs(t,var_list)
        hbbl      = data.var['hbbl']
        zbbl      = -data.h+hbbl
        [z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
        # --- compute mask BBL---
        mask_BBL = def_mask_BBL(z_r,zbbl)
        # --- compute dvol ---
        dvol    = np.diff(z_w,axis=-1)*dsurf
        for i in range(ntpas):
            tpas                  = np.asfortranarray(data.var[tpas_list[i]][:,:,:])
            tpas[tpas<1e-6]       = np.nan
            tpas_z                = np.nansum(tpas*dvol,axis=-1)/np.nansum(dvol,axis=-1)
            # quantity of tpas in BBL
            Qtpas_bbl             = np.nansum(tpas*mask_BBL*dvol)
            porc_tpas_inbbl[i,tt] = 100*Qtpas_bbl/np.nansum(tpas*dvol)
            hbbl_tpas[i,tt]       = tracer_avg(hbbl,tpas_z,np.nansum(dvol,axis=-1))
            z_r_tpas[i,tt]        = tracer_avg(z_r,tpas,dvol)
        tt+=1


# ----------- save parameters in netcdf file ------------ 
print(' ... save in netcdf file ... ')
#print(file_diag)
nc = Dataset(file_diag_k,'w')
nc.createDimension('time',nt)
nc.createDimension('ntpas',ntpas)
nc.createVariable('time','f',('time'))
var = nc.createVariable('porc_tpas_inbbl','f',('ntpas','time'))
var.long_name = 'porcentage of grid cells with tracer in the BBL (defined with KPP)'
var = nc.createVariable('hbbl','f',('ntpas','time'))
var.long_name = 'hbbl (defined with KPP), weighted by tracer concentration'
var = nc.createVariable('z_r_tpas','f',('ntpas','time'))
var.long_name = 'absolute depth, z_r, weighted by tracer concentration'


nc.variables['porc_tpas_inbbl'][:]    = porc_tpas_inbbl
nc.variables['hbbl'][:]               = hbbl_tpas
nc.variables['z_r_tpas'][:]           = z_r_tpas
nc.close()

