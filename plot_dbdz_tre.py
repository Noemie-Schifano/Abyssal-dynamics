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
from croco_simulations_jon_hist_last import Croco
from croco_simulations_jon_hist_last6h import Croco_6h
matplotlib.rcParams.update({'font.size': 14})

# ------------ parameters ------------ 
name_exp    = 'rrexnum200'
name_exp_path ='rrexnum200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_T'
nbr_levels  = '200'
name_exp_grd= ''
test_nc    = True
sigma2     = False
rhop_tracer= True
if test_nc == True :
    ndfiles     = 2
    time        = np.arange(0,1)
    file_diag_k = 'test_rrexnumsb200-rsup5_16tracers_dbdz.nc'
else:
    ndfiles     = 24
    time        = np.arange(0,25,24)
    if rhop_tracer == True:
        file_diag_k = 'rrexnumsb200-rsup5_16tracers_dbdz.nc'
    else:
        file_diag_k = 'rrexnumsb200-rsup5_16tracers_dbdz.nc'
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
dbdz_avg        = np.zeros((ntpas,nt))      # first order moment of tracer in buoyancy space  
dbdz_6h_avg     = np.zeros((ntpas,nt))      # first order moment of tracer in buoyancy space  
p_ref, p_ref_6h = np.zeros(ntpas),np.zeros(ntpas)
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
    # ---- TRE ref --- 
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
        del dsurf, dsurf_w
        print(' ... compute potential density referenced at 1000 meters... ')
        # --- compute initial reference pressure 
        if tt==0:
            print(' --- compute p_ref ---')
            p  = gsw.p_from_z(z_r,np.nanmean(data.latr))
            for itpas in range(ntpas):
                tpas            = np.asfortranarray(data.var[tpas_list[itpas]][:,:,:])
                tpas[tpas<1e-6] = np.nan
                p_ref[itpas] = tracer_avg(p,tpas,dvol)
            print(p_ref)
        SA      = gsw.SA_from_SP(data.var['salt'],p,np.nanmean(data.lonr),np.nanmean(data.latr))
        CT      = gsw.CT_from_pt(SA,data.var['temp'])
        print('     --> compute the moments of tracer and N2 ')
        for i in range(ntpas):
            # --- compute buoyancy 
            rho_po  = gsw.rho(SA,CT,p_ref[i])
            buoy    = -gg*(rho_po-rhoref)/rhoref
            dbdz    = np.diff(buoy,axis=-1)/np.diff(z_r,axis=-1)
            # --- compute tracer
            tpas            = np.asfortranarray(data.var[tpas_list[i]][:,:,:])
            tpas[tpas<1e-6] = np.nan
            tpas_w          = 0.5*(tpas[:,:,:-1]+tpas[:,:,1:])
            print(np.shape(dbdz),np.shape(tpas_w),np.shape(dvol_w))
            dbdz_avg[i,tt]  = tracer_avg(dbdz,tpas_w,dvol_w)     
    del data, SA, CT, dvol, dvol_w, z_r, z_w, tpas, tpas_w 

    # ---- TRE 6h -----
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
        print(' ... compute potential density referenced at 1000 meters... ')
        # --- compute initial reference pressure 
        if tt==0:
            print(' --- compute p_ref ---')
            p  = gsw.p_from_z(z_r,np.nanmean(data.latr))
            for itpas in range(ntpas):
                tpas            = np.asfortranarray(data.var[tpas_list[itpas]][:,:,:])
                tpas[tpas<1e-6] = np.nan
                p_ref_6h[itpas] = tracer_avg(p,tpas,dvol)
            print(p_ref)
        SA      = gsw.SA_from_SP(data.var['salt'],p,np.nanmean(data.lonr),np.nanmean(data.latr))
        CT      = gsw.CT_from_pt(SA,data.var['temp'])
        print('     --> compute the moments of tracer and N2 ')
        for i in range(ntpas):
            # --- compute buoyancy 
            rho_po  = gsw.rho(SA,CT,p_ref_6h[i])
            buoy    = -gg*(rho_po-rhoref)/rhoref
            dbdz    = np.diff(buoy,axis=-1)/np.diff(z_r,axis=-1)
            print(np.shape(dbdz),np.shape(tpas_w),np.shape(dvol_w))
            # --- compute tracer
            tpas            = np.asfortranarray(data.var[tpas_list[i]][:,:,:])
            tpas[tpas<1e-6] = np.nan
            tpas_w          = 0.5*(tpas[:,:,:-1]+tpas[:,:,1:])
            print(np.shape(dbdz),np.shape(tpas_w),np.shape(dvol_w))
            dbdz_6h_avg[i,tt]  = tracer_avg(dbdz,tpas,dvol)                           
        del data, SA, CT, dvol, dvol_w, z_r, z_w, tpas, tpas_w 
        tt+=1


# ----------- save parameters in netcdf file ------------ 
print(' ... save in netcdf file ... ')
nc = Dataset(file_diag_k,'w')
nc.rhoref      = rhoref
nc.createDimension('time',nt)
nc.createDimension('ntpas',ntpas)

nc.createVariable('time','f',('time'))
nc.createVariable('dbdz_avg','f',('ntpas','time'))
var = nc.createVariable('dbdz_avg','f',('ntpas','time'))
var.long_name = 'dbdz weighted by the tracer concentration TRE_ref'

nc.createVariable('dbdz_6h_avg','f',('ntpas','time'))
var = nc.createVariable('dbdz_6h_avg','f',('ntpas','time'))
var.long_name = 'dbdz weighted by the tracer concentration TRE_6h'

nc.variables['dbdz_avg'][:]     = dbdz_avg
nc.variables['dbdz_6h_avg'][:]     = dbdz_6h_avg

nc.close()

