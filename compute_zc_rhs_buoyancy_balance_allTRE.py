'''
NS: Compute and save change in depth for each TRE due to mixing
    The same 16 TREs are released but with 6 hours-intervals, both are analysed    
'''

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
from croco_simulations_jon_hist_last import Croco
from croco_simulations_jon_hist_last6h import Croco_6h
matplotlib.rcParams.update({'font.size': 14})

# ------------ parameters ------------ 
name_exp    = 'rrexnum200'
name_pathdata = 'RREXNUMSB200_RSUP5_T'
nbr_levels  = '200'
name_exp_grd= ''
test_nc    = False
if test_nc == True :
    ndfiles     = 2
    time        = np.arange(0,1)
    file_diag_k = 'test_rrexnumsb200-rsup5_allTREs_zc_rhs.nc'
else:
    ndfiles     = 24
    time        = np.arange(0,25,24)
    file_diag_k = 'rrexnumsb200-rsup5_allTREs_zc_rhs.nc'
nt          = len(time)*ndfiles
dt          = 1*3600 #1h 
# - select tracers for analysis - 
tpas_list =  ['tpas0'+str(i) for i in range(1,10)]+['tpas'+str(i) for i in range(10,17)]
ntpas = len(tpas_list)
var_list  = ['zeta']
var_list += tpas_list 

gg = 9.81
rhoref = 1027.4

# --- variables to be saved ---
zc_rhs_avg    = np.zeros((ntpas,nt))
zc_rhs_avg6h  = np.zeros((ntpas,nt))

# --- function ---
def z_avg_rhs(var,z,tpas,dvol):
    return np.nansum(var*z)/np.nansum(tpas*dvol)

# --- read data ---
print(' ............ time loop ...............................................  ')
tt = 0
for t_nc in range(len(time)):
    data   = Croco(name_exp,nbr_levels,str(int(time[t_nc])),name_exp_grd,name_pathdata)
    data6h = Croco_6h(name_exp,nbr_levels,str(int(time[t_nc])),name_exp_grd,name_pathdata)
    data.get_grid()
    data6h.get_grid()
    dsurf   = 1./np.transpose(np.tile(data.pm*data.pn,(int(nbr_levels),1,1)),(1,2,0)) # horizontal surface area
    dsurf6h   = 1./np.transpose(np.tile(data6h.pm*data6h.pn,(int(nbr_levels),1,1)),(1,2,0))
    for t in range(0,ndfiles):
        print('=====================  time index %.4i ====================='%t)
        print('    ---> read outputs ')
        # --- TRE_ref
        data.get_outputs(t,var_list)
        [z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
        print("z_r",z_r[100,100,0],z_r[100,100,-1])
        dvol    = np.diff(z_w,axis=-1)*dsurf
        c_rhs         = data.get_crhs(tt)
        # --- TRE_6h
        data6h.get_outputs(t,var_list)
        [z_r6h,z_w6h] = toolsF.zlevs(data6h.h,data6h.var['zeta'],data6h.hc,data6h.Cs_r,data6h.Cs_w)
        dvol6h    = np.diff(z_w6h,axis=-1)*dsurf6h
        c_rhs6h         = data6h.get_crhs(tt)
        for i in range(ntpas):
            # --- TRE_ref
            tpas            = np.asfortranarray(data.var[tpas_list[i]][:,:,:])
            tpas[tpas<1e-6] = np.nan
            zc_rhs_avg[i,tt] = z_avg_rhs(c_rhs[i][:,:,:],z_r,tpas,dvol)
            # --- TRE_6h
            tpas6h            = np.asfortranarray(data6h.var[tpas_list[i]][:,:,:])
            tpas6h[tpas6h<1e-6] = np.nan
            zc_rhs_avg6h[i,tt] = z_avg_rhs(c_rhs6h[i][:,:,:],z_r6h,tpas6h,dvol6h)
        tt+=1

# ----------- save parameters in netcdf file ------------ 
print(' ... save in netcdf file ... ')
nc = Dataset(file_diag_k,'w')
nc.createDimension('time',nt)
nc.createDimension('ntpas',ntpas)
var = nc.createVariable('zc_rhs_avg','f',('ntpas','time'))
var.long_name = 'volume integral of z*c_rhs divided by the volume integral of tracer concentration, using TRE_ref'
var = nc.createVariable('zc_rhs_avg6h','f',('ntpas','time'))
var.long_name = 'same as zc_rhs_avg but for TRE_6h'
nc.variables['zc_rhs_avg'][:]     = zc_rhs_avg
nc.variables['zc_rhs_avg6h'][:]   = zc_rhs_avg6h
nc.close()

