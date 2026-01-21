'''
NS: Compute and save diagnostics for each TRE
    nota bene: The same 16 TREs are released but with 6 hours-intervals, in this code it is the 
                 16 TREs released 6h after the referenced TREs that are analysed    
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
from croco_simulations_jon_hist_last6h import Croco_6h
matplotlib.rcParams.update({'font.size': 14})

# ------------ parameters ------------ 
name_exp    = 'rrexnum200'
name_exp_path ='rrexnum200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_T'
nbr_levels  = '200'
name_exp_grd= ''
test_nc    = False
sigma2     = False
rhop_tracer= True
if test_nc == True :
    ndfiles     = 2
    time        = np.arange(0,1)
    file_diag_k = 'test_rrexnumsb200-rsup5_16tracers_6h_k_buoy_balance.nc'
else:
    ndfiles     = 24
    time        = np.arange(0,25,24)
    if sigma2 == True:
        file_diag_k = 'rrexnumsb200-rsup5_16tracers_6h_k_buoy_sigma2_balance.nc'
    elif rhop_tracer == True:
        file_diag_k = 'rrexnumsb200-rsup5_16tracers_6h_k_buoy_rhop_tracer_balance.nc'
    else:
        file_diag_k = 'rrexnumsb200-rsup5_16tracers_6h_k_buoy_balance.nc'
    
nt          = len(time)*ndfiles
dt          = 1*3600 #1h 
# - select tracers for analysis - 
tpas_list =  ['tpas0'+str(i) for i in range(1,10)]+['tpas'+str(i) for i in range(10,17)]
ntpas = len(tpas_list)
var_list  = ['zeta','rho','u','v','bvf','salt','temp']
var_list += tpas_list 

gg = 9.81
rhoref = 1027.4

# --- variables to be saved ---
buoy_avg     = np.zeros((ntpas,nt))      # first order moment of tracer in buoyancy space  
buoy_var     = np.zeros((ntpas,nt))      # second order moment of tracer in buoyancy space (variance)    
b_rhs_avg    = np.zeros((ntpas,nt))
b_rhs_dv_avg    = np.zeros((ntpas,nt))
c_rhs_avg    = np.zeros((ntpas,nt))
b_adv_avg    = np.zeros((ntpas,nt))
w_avg        = np.zeros((ntpas,nt))
p_ref        = np.zeros(ntpas)
N2_avg       = np.zeros((ntpas,nt)) 

# --- functions ---
def tracer_avg(var,tpas,dvol):
    # dvol is the cell volume  
    #return np.nansum(var*tpas)/np.nansum(tpas)
    return np.nansum(var*tpas*dvol)/np.nansum(tpas*dvol)


def tracer_avg_rhs(var,tpas,dvol):
    return np.nansum(var*tpas)/np.nansum(tpas*dvol)


def buoy_avg_rhs(var,buoy,tpas,dvol):
    return np.nansum(var*buoy)/np.nansum(tpas*dvol)


# --- read data ---
print(' ............ time loop ...............................................  ')
tt = 0
for t_nc in range(len(time)):
    data = Croco_6h(name_exp,nbr_levels,str(int(time[t_nc])),name_exp_grd,name_pathdata)
    data.get_grid()
    dsurf   = 1./np.transpose(np.tile(data.pm*data.pn,(int(nbr_levels),1,1)),(1,2,0)) # horizontal surface area
    dsurf_w = 1./np.transpose(np.tile(data.pm*data.pn,(int(nbr_levels)-1,1,1)),(1,2,0)) # horizontal surface area at w-points
    for t in range(0,ndfiles):
        print('=====================  time index %.4i ====================='%t)
        print('    ---> read outputs ')
        data.get_outputs(t,var_list)
        [z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
        dvol    = np.diff(z_w,axis=-1)*dsurf
        dvol_w  = np.diff(z_r,axis=-1)*dsurf_w
        w         = 3600*24*toolsF.get_wvlcty(data.var['u'],data.var['v'],z_r,z_w,data.pm,data.pn)
        print(' ... compute potential density referenced at 1000 meters... ')
        p               = gsw.p_from_z(z_r,np.nanmean(data.latr))
        # --- compute initial reference pressure 
        if tt==0:
            print(' --- compute p_ref ---')
            for itpas in range(ntpas):
                tpas            = np.asfortranarray(data.var[tpas_list[itpas]][:,:,:])
                tpas[tpas<1e-6] = np.nan
                p_ref[itpas] = tracer_avg(p,tpas,dvol)
            print(p_ref)
        SA      = gsw.SA_from_SP(data.var['salt'],p,np.nanmean(data.lonr),np.nanmean(data.latr))
        CT      = gsw.CT_from_pt(SA,data.var['temp'])
        temp    = data.var['temp'][:]
        N2      = tools.w2rho(data.var['bvf'])
        # --- buoyancy balance
        [b_rhs,b_adv] = data.get_buoyancy_balance(tt)
        c_rhs         = data.get_crhs(tt)
        print('     --> compute the moments of tracer and N2 ')
        for i in range(ntpas):
            # --- compute buoyancy 
            rho_po  = gsw.rho(SA,CT,p_ref[i])
            buoy    = -gg*(rho_po-rhoref)/rhoref
            # --- compute tracer
            tpas            = np.asfortranarray(data.var[tpas_list[i]][:,:,:])
            tpas[tpas<1e-6] = np.nan
            tpas_w          = 0.5*(tpas[:,:,:-1]+tpas[:,:,1:])
            buoy_avg[i,tt]  = tracer_avg(buoy,tpas,dvol)                        # called nu in (24) in H19    
            buoy_var[i,tt]  = tracer_avg(buoy**2,tpas,dvol) - buoy_avg[i,tt]**2 # called sigma^2 in (25) in H19
            w_avg[i,tt]     = tracer_avg(w,tpas,dvol)
            N2_avg[i,tt]    = tracer_avg(N2,tpas,dvol)
            # rhs is already inegrated over volume
            b_rhs_avg[i,tt] = tracer_avg_rhs(b_rhs,tpas,dvol)
            b_rhs_dv_avg[i,tt] = tracer_avg(b_rhs,tpas,dvol)
            b_adv_avg[i,tt] = tracer_avg_rhs(b_adv,tpas,dvol)
            c_rhs_avg[i,tt] = buoy_avg_rhs(c_rhs[i][:,:,:],buoy,tpas,dvol)
        tt+=1


# ----------- save parameters in netcdf file ------------ 
print(' ... save in netcdf file ... ')
nc = Dataset(file_diag_k,'w')
nc.rhoref      = rhoref
nc.createDimension('time',nt)
nc.createDimension('ntpas',ntpas)

nc.createVariable('time','f',('time'))
nc.createVariable('buoy_avg','f',('ntpas','time'))
nc.createVariable('buoy_var','f',('ntpas','time'))
var = nc.createVariable('b_rhs_avg','f',('ntpas','time'))
var.long_name = 'b_rhs weighted by the tracer concentration'
var = nc.createVariable('b_rhs_dv_avg','f',('ntpas','time'))
var = nc.createVariable('c_rhs_avg','f',('ntpas','time'))
var.long_name = 'c_rhs weighted by the tracer concentration'
var = nc.createVariable('b_adv_avg','f',('ntpas','time'))
var.long_name = 'b_adv weighted by the tracer concentration'
var = nc.createVariable('w_avg','f',('ntpas','time'))
var.long_name = 'w weighted by the tracer concentration'
var = nc.createVariable('N2_avg','f',('ntpas','time'))
var.long_name = 'N2 weighted by the tracer concentration'

nc.variables['buoy_avg'][:]     = buoy_avg
nc.variables['buoy_var'][:]     = buoy_var
nc.variables['b_rhs_avg'][:]    = b_rhs_avg
nc.variables['b_rhs_dv_avg'][:] = b_rhs_dv_avg
nc.variables['c_rhs_avg'][:]    = c_rhs_avg
nc.variables['b_adv_avg'][:]    = b_adv_avg
nc.variables['w_avg'][:]        = w_avg
nc.variables['N2_avg'][:]       = N2_avg

nc.close()

