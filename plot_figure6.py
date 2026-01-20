'''
NS : Need "extract_height_BBL_N2_ijcoord.py" to run before
     Using 7 months time-mean netcdf file
     - Select areas of upwelling and downwelling
     - make buoyancy balance in those two areas 
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
import time as time
import calendar as cal
import datetime as datetime
from croco_simulations_jonathan_ncra_longrun import Croco_longrun
import cartopy.crs as ccrs
import gsw as gsw
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
matplotlib.rcParams.update({'font.size': 32})  
from matplotlib.ticker import FormatStrFormatter

# ------------ file BBL -------------
file_bbl        = 'BBL_height_N2.nc'

# ------------ parameters ------------ 
name_exp    = 'rrexnum200' 
name_exp_path ='rrexnums200_rsup5'
name_pathdata = 'RREXNUMSB200_RSUP5_NOFILT_T'
name_exp_grd= ''
nbr_levels  = '200'
name_exp_saveup = name_exp_path
var_list = ['zeta','w','AKt','bvf','salt','temp','u','v','hbbl']

ndfiles     = 1  # number of days per netcdf
rho0        = 1027.4
g           = 9.81
# --- plot options --- 
fs      = 30      # fontsize 
lwc     = 0.5
#choice_list = [r'Upwelling $N^2_{BBL}$>0',r'Downwelling $N^2_{BBL}$>0',r'Upwelling $N^2_{BBL}$<0',r'Downwelling $N^2_{BBL}$<0','All grid']
choice_list = ['Subdomain A','Subdomain B','Subdomain C','Subdomain D','All grid']
title_figw  = ['b) '+choice_list[0],'c) '+choice_list[1],'d) '+choice_list[2],'e) '+choice_list[3], 'f) '+choice_list[4]] 
title_figb  = ['g)','h)','i)','j)','k)']
levels_h    = np.arange(0,4200,200)
cf_up_N2pos   = colors.to_rgba('orange')
cf_down_N2pos = colors.to_rgba('purple')
cf_up_N2neg   = colors.to_rgba('red')
cf_down_N2neg = colors.to_rgba('blue')
cf_plot       = [cf_up_N2pos,cf_down_N2pos,cf_up_N2neg,cf_down_N2neg,'k']
zorder_choice = [1,4,2,3,5]

# ------------ bin for hab ------
h_e = np.array([ 0,   3,   6,   9,  12,  15,  18,  21,  26,36, 66,  96, 126, 156, 186, 216, 246, 276, 306, 336])
h_c = 0.5*(h_e[1:]+h_e[:-1]) # hab centres
nbin      = h_c.shape[0]
gg = 9.81
rhoref   = 1027.4

neg1 = 0
neg2 = 0
maxg = 0
ming = 0

# ------------ bin for slope  ------
minc      = 0
maxc      = 0.12
nce       = 0.0012       # 0.0085
slope_e    = np.arange(minc,maxc,nce)

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

def index_up_down(area,criteria):
    # in area defined with the function "find_upwelling_downwelling",
    # return the index in ij-coordinates where area==criteria
    A = np.where(area==criteria)
    return [A[0][:],A[1][:]]


def define_var_at_area(var,points):
    # select a variable at the area define between points
    # points containes (x,y) coordinates of the limits of the square 
    return var[points[0]:points[1],points[2]:points[3]]

def compute_tsadv(T,u,v,w,z_r,z_w,f,g,rho0,pm,pn,newadv):
    '''
    check alternate formulation for r.h.s. of tracer equation
    '''
    if newadv:
        advx = tools.u2rho(u * tools.diffx(T,pm))
        advy = tools.v2rho(v * tools.diffy(T,pn))
        omega = w
        print(np.shape(omega),np.shape(T),np.shape(z_r))
        advz = omega[:,:,1:-1] * (T[:,:,1:]-T[:,:,:-1])/(z_r[:,:,1:]-z_r[:,:,:-1])
        advz = tools.vinterp(advz,z_r,z_w[:,:,1:-1],z_r)
    else:
        advx = tools.u2rho(u * tools.diffxi(T,pm,z_r,z_w))
        advy = tools.v2rho(v * tools.diffeta(T,pn,z_r,z_w))
        omega =w
        advz = omega[:,:,:] * tools.diffz_sig(T,z_r,z_w)
    return advx + advy, advz


def get_brhs(simul,T,S, t_rhs, s_rhs,z_r,z_w, nlbrhs = False):
    if 'NONLIN_EOS' in simul.cpp:
        [alpha,beta] = toolsF.alfabeta_3d(T,S,z_r,z_w,simul.rho0);
    else:
        print('using LIN_EOS')
        [alpha,beta] = [simul.Tcoef,simul.Scoef]/simul.rho0
    if nlbrhs:
        b_rhs = nonlin_brhs(T,S, t_rhs, s_rhs, simul.rho0, simul.g)
    else:
        b_rhs = simul.g * (alpha * t_rhs - beta * s_rhs)  # rho-rho grid
    return b_rhs


##############################################################################################
# ---------------- load simulation as Jonathan to compute buoyancy from advection -----------
mean = 'mean.00010-000229'
year=0; time_his=20; time_diag=time_his; time_avg=int(time_his/2); section_lat = False
mysimul = 'rrexnumsb200_RSUP5_NOFILT_T_avg_' + mean; 

simul0 = load(mysimul,time=time_his,output=True)

if mysimul == 'rrexnum100_UP3':
    title = 'UP3'
elif mysimul == 'rrexnum100':
    title = 'RSUP3'
else:
    title = mysimul

fifig = './Figures/' 
zoom = [simul0.x.min(), simul0.x.max(), simul0.y.min(), simul0.y.max()]; domain = 'all';

lon1,lat1 = zoom[0], zoom[2]
lon2,lat2 = zoom[1], zoom[3]
# Only load area around the section
ix1,iy1 = tools.find_nearest_points(simul0.x,simul0.y,lon1,lat1)
ix2,iy2 = tools.find_nearest_points(simul0.x,simul0.y,lon1,lat2)
ix3,iy3 = tools.find_nearest_points(simul0.x,simul0.y,lon2,lat1)
ix4,iy4 = tools.find_nearest_points(simul0.x,simul0.y,lon2,lat2)
ixs = [ix1,ix2,ix3,ix4]
iys = [iy1,iy2,iy3,iy4]

i_sec = 500 #- 200#+300
dx_sec = 400;
j_sec = 350 #- 200#+300
dy_sec = 200; 

coordinates= ' [' + format(np.max([j_sec-dy_sec,0])) + ','\
              + format(j_sec+dy_sec) + ','\
              + format(np.max([i_sec-dx_sec,0])) + ','\
              + format(i_sec+dx_sec) + ',[1,300,1]] '
simul = simul0
# --- load variables  ---
[z_r,z_w] = tools.get_depths(simul)
t=var('temp',simul,depths=[0])
[TXadv,TYadv,TVadv] = t.get_tracer_advection(simul); del t
T = var('temp',simul).data; S = var('salt',simul).data
b_adv_mean_save     = get_brhs(simul,T[:,:,:],S[:,:,:],\
                                TXadv[:,:,:,0]+TYadv[:,:,:,0]+TVadv[:,:,:,0],\
                                TXadv[:,:,:,1]+TYadv[:,:,:,1]+TVadv[:,:,:,1],z_r,z_w)
#b_adv_mean_save[np.isnan(b_adv_mean_save)==1]=0
print('b_adv_mean',b_adv_mean_save)

##############################################################################################
# ------------ read BBL ------------ 
# --- read bbl --- 
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
keff = data.get_diffusivity('0','diffusivity')
N2   = data.var['bvf']
w    = 3600*24*data.var['w']
u    = tools.u2rho(data.var['u'])
v    = tools.v2rho(data.var['v'])
hbbl = data.var['hbbl']
# ---> buoyancy balance 
[b_rhs,b_adv] = data.get_buoyancy_balance(0)
# ---> compute topostrophy
dhdx = tools.u2rho((data.h[1:,:]-data.h[:-1,:]))*data.pm
dhdy = tools.v2rho((data.h[:,1:]-data.h[:,:-1]))*data.pn
gradhi =np.sqrt(dhdx**2 + dhdy**2)
gradh  = np.transpose(np.tile(gradhi,(200,1,1)),(1,2,0))
print(' --- compute topostrophy ---')
to      = 100*(tools.u2rho(data.var['u'])*np.transpose(np.tile(dhdy,(200,1,1)),(1,2,0)) - tools.v2rho(data.var['v'])*np.transpose(np.tile(dhdx,(200,1,1)),(1,2,0)))
# --> find bottom upwelling and downwelling 
area_tot  = find_upwelling_downwelling(w[:,:,0],N2_bbl,data.lonr,data.latr)

porc_up_N2pos    = 100*len(area_tot[area_tot==1])/len(np.ravel(area_tot))
porc_down_N2pos  = 100*len(area_tot[area_tot==9999])/len(np.ravel(area_tot))
porc_up_N2neg    = 100*len(area_tot[area_tot==-1])/len(np.ravel(area_tot))
porc_down_N2neg  = 100*len(area_tot[area_tot==-9999])/len(np.ravel(area_tot))

[lon_up_N2pos, lat_up_N2pos]     = index_up_down(area_tot,1)
[lon_down_N2pos, lat_down_N2pos] = index_up_down(area_tot,9999)
[lon_up_N2neg, lat_up_N2neg]     = index_up_down(area_tot,-1)
[lon_down_N2neg, lat_down_N2neg] = index_up_down(area_tot,-9999)


# --> ravel variables 
area_r             = np.ravel(area_tot)
area_r3d           = np.ravel(np.transpose(np.tile(area_tot,(200,1,1)),(1,2,0)))
area_r3dw          = np.ravel(np.transpose(np.tile(area_tot,(201,1,1)),(1,2,0)))
hab_ri             = np.ravel(hab)
habw_ri            = np.ravel(habw)
hbbl_ri            = np.ravel(hbbl)
w_ri               = np.ravel(w) 
u_ri               = np.ravel(u)
v_ri               = np.ravel(v)
to_ri              = np.ravel(to)
kkpp_ri            = np.ravel(kkpp)
keff_ri            = np.ravel(keff)
N2_ri              = np.ravel(N2)
b_rhs_ri           = np.ravel(b_rhs)
b_adv_ri           = np.ravel(b_adv)
b_adv_mean_save_ri = np.ravel(b_adv_mean_save)

# --> statistics of variables in subdomains
nz      = nbin 
w10     = np.zeros((len(choice_list),nz))
w90     = np.zeros((len(choice_list),nz))
w_r     = np.zeros((len(choice_list),nz)) 
u_r     = np.zeros((len(choice_list),nz))
v_r     = np.zeros((len(choice_list),nz))
to_r    = np.zeros((len(choice_list),nz))
kkpp_r  = np.zeros((len(choice_list),nz))
keff_r  = np.zeros((len(choice_list),nz))
N2_r    = np.zeros((len(choice_list),nz))
b_rhs_r = np.zeros((len(choice_list),nz))
b_adv_r = np.zeros((len(choice_list),nz))
b_mean_r= np.zeros((len(choice_list),nz))
hbbl_r  = np.zeros(len(choice_list))
hab_bblt= np.zeros(len(choice_list))



for choice in range(len(choice_list)):
    if choice==0:  # upwelling N2>0
        hab_bblt[choice]   = np.nanmean(hab_bbl[area_r==1])
        hab_up             = np.copy(hab_ri[area_r3d==1])
        print('upwelling :',len(hab_up))
        habw_up            = np.copy(habw_ri[area_r3dw==1])
        hbbl_up            = np.copy(hbbl_ri[area_r==1])
        w_up               = np.copy(w_ri[area_r3d==1])
        u_up               = np.copy(u_ri[area_r3d==1])
        v_up               = np.copy(v_ri[area_r3d==1])
        to_up              = np.copy(to_ri[area_r3d==1])
        kkpp_up            = np.copy(kkpp_ri[area_r3d==1])
        keff_up            = np.copy(keff_ri[area_r3d==1])
        print(keff_up)
        N2_up              = np.copy(N2_ri[area_r3dw==1])
        b_rhs_up           = np.copy(b_rhs_ri[area_r3d==1])
        b_adv_up           = np.copy(b_adv_ri[area_r3d==1])
        b_adv_mean_save_up = np.copy(b_adv_mean_save_ri[area_r3d==1])
        print(b_adv_mean_save_up)
        # binned in hab space 
        h_r              = hab_up
        hw_r             = habw_up 
        hbbl_r[choice]   = np.median(hbbl_up)
        w10[choice,:]    = stats.binned_statistic(h_r,w_up,statistic=lambda y: np.nanpercentile(y, 10),bins=h_e)[0]
        w90[choice,:]    = stats.binned_statistic(h_r,w_up,statistic=lambda y: np.nanpercentile(y, 90),bins=h_e)[0]
        w_r[choice,:]    = stats.binned_statistic(h_r,w_up,statistic='median',bins=h_e)[0]
        u_r[choice,:]    = stats.binned_statistic(h_r,u_up,statistic='median',bins=h_e)[0]
        v_r[choice,:]    = stats.binned_statistic(h_r,v_up,statistic='median',bins=h_e)[0]
        to_r[choice,:]   = stats.binned_statistic(h_r,to_up,statistic='median',bins=h_e)[0]
        kkpp_r[choice,:] = stats.binned_statistic(h_r,kkpp_up,statistic='median',bins=h_e)[0]
        keff_r[choice,:] = stats.binned_statistic(h_r,keff_up,statistic=np.nanmedian,bins=h_e)[0]
        # on w-grid
        N2_r[choice,:]       = stats.binned_statistic(hw_r,N2_up,statistic='median',bins=h_e)[0]
        # on rho-grid
        b_rhs_r[choice,:]    = stats.binned_statistic(h_r,b_rhs_up,statistic='median',bins=h_e)[0]
        b_adv_r[choice,:]    = stats.binned_statistic(h_r,b_adv_up,statistic='median',bins=h_e)[0]
        b_mean_r[choice,:]   = stats.binned_statistic(h_r,b_adv_mean_save_up,statistic=np.nanmedian,bins=h_e)[0]

    elif choice==1 :         # downwelling N2>0
        hab_bblt[choice]     = np.nanmean(hab_bbl[area_r==9999])
        hab_down             = np.copy(hab_ri[area_r3d==9999])
        print('downwelling :',len(hab_down))
        habw_down            = np.copy(habw_ri[area_r3dw==9999])
        hbbl_down            = np.copy(hbbl_ri[area_r==9999])
        w_down               = np.copy(w_ri[area_r3d==9999])
        u_down               = np.copy(u_ri[area_r3d==9999])
        v_down               = np.copy(v_ri[area_r3d==9999])
        to_down              = np.copy(to_ri[area_r3d==9999])
        kkpp_down            = np.copy(kkpp_ri[area_r3d==9999])
        keff_down            = np.copy(keff_ri[area_r3d==9999])
        N2_down              = np.copy(N2_ri[area_r3dw==9999])
        b_rhs_down           = np.copy(b_rhs_ri[area_r3d==9999])
        b_adv_down           = np.copy(b_adv_ri[area_r3d==9999])
        b_adv_mean_save_down = b_adv_mean_save_ri[area_r3d==9999]
        # binned in hab space 
        h_r              = hab_down
        hw_r             = habw_down
        hbbl_r[choice]   = np.median(hbbl_down)
        w10[choice,:]    = stats.binned_statistic(h_r,w_down,statistic=lambda y: np.nanpercentile(y, 10),bins=h_e)[0]
        w90[choice,:]    = stats.binned_statistic(h_r,w_down,statistic=lambda y: np.nanpercentile(y, 90),bins=h_e)[0]
        w_r[choice,:]    = stats.binned_statistic(h_r,w_down,statistic='median',bins=h_e)[0]
        u_r[choice,:]    = stats.binned_statistic(h_r,u_down,statistic='median',bins=h_e)[0]
        v_r[choice,:]    = stats.binned_statistic(h_r,v_down,statistic='median',bins=h_e)[0]
        to_r[choice,:]   = stats.binned_statistic(h_r,to_down,statistic='median',bins=h_e)[0]
        kkpp_r[choice,:] = stats.binned_statistic(h_r,kkpp_down,statistic='median',bins=h_e)[0]
        keff_r[choice,:] = stats.binned_statistic(h_r,keff_down,statistic=np.nanmedian,bins=h_e)[0]
        # on w-grid
        N2_r[choice,:]       = stats.binned_statistic(hw_r,N2_down,statistic='median',bins=h_e)[0]
        # on rho-grid
        b_rhs_r[choice,:]    = stats.binned_statistic(h_r,b_rhs_down,statistic='median',bins=h_e)[0]
        b_adv_r[choice,:]    = stats.binned_statistic(h_r,b_adv_down,statistic='median',bins=h_e)[0]
        b_mean_r[choice,:]   = stats.binned_statistic(h_r,b_adv_mean_save_down,statistic=np.nanmedian,bins=h_e)[0]

    elif choice==2:  # upwelling N2<0
        hab_bblt[choice]     = np.nanmean(hab_bbl[area_r==-1])
        hab_up             = np.copy(hab_ri[area_r3d==-1])
        print('upwelling N2<0:',len(hab_up))
        habw_up            = np.copy(habw_ri[area_r3dw==-1])
        hbbl_up            = np.copy(hbbl_ri[area_r==-1])
        w_up               = np.copy(w_ri[area_r3d==-1])
        u_up               = np.copy(u_ri[area_r3d==-1])
        v_up               = np.copy(v_ri[area_r3d==-1])
        to_up              = np.copy(to_ri[area_r3d==-1])
        kkpp_up            = np.copy(kkpp_ri[area_r3d==-1])
        keff_up            = np.copy(keff_ri[area_r3d==-1])
        print(keff_up)
        N2_up              = np.copy(N2_ri[area_r3dw==-1])
        b_rhs_up           = np.copy(b_rhs_ri[area_r3d==-1])
        b_adv_up           = np.copy(b_adv_ri[area_r3d==-1])
        b_adv_mean_save_up = np.copy(b_adv_mean_save_ri[area_r3d==-1])
        # binned in hab space 
        h_r              = hab_up
        hw_r             = habw_up
        hbbl_r[choice]   = np.median(hbbl_up)
        w10[choice,:]    = stats.binned_statistic(h_r,w_up,statistic=lambda y: np.nanpercentile(y, 10),bins=h_e)[0]
        w90[choice,:]    = stats.binned_statistic(h_r,w_up,statistic=lambda y: np.nanpercentile(y, 90),bins=h_e)[0]
        w_r[choice,:]    = stats.binned_statistic(h_r,w_up,statistic='median',bins=h_e)[0]
        u_r[choice,:]    = stats.binned_statistic(h_r,u_up,statistic='median',bins=h_e)[0]
        v_r[choice,:]    = stats.binned_statistic(h_r,v_up,statistic='median',bins=h_e)[0]
        to_r[choice,:]   = stats.binned_statistic(h_r,to_up,statistic='median',bins=h_e)[0]
        kkpp_r[choice,:] = stats.binned_statistic(h_r,kkpp_up,statistic='median',bins=h_e)[0]
        keff_r[choice,:] = stats.binned_statistic(h_r,keff_up,statistic=np.nanmedian,bins=h_e)[0]
        # on w-grid
        N2_r[choice,:]       = stats.binned_statistic(hw_r,N2_up,statistic='median',bins=h_e)[0]
        # on rho-grid
        b_rhs_r[choice,:]    = stats.binned_statistic(h_r,b_rhs_up,statistic='median',bins=h_e)[0]
        b_adv_r[choice,:]    = stats.binned_statistic(h_r,b_adv_up,statistic='median',bins=h_e)[0]
        b_mean_r[choice,:]   = stats.binned_statistic(h_r,b_adv_mean_save_up,statistic=np.nanmedian,bins=h_e)[0]

    elif choice==3 :         # downwelling N2<0
        hab_bblt[choice]     = np.nanmean(hab_bbl[area_r==-9999])
        hab_down             = np.copy(hab_ri[area_r3d==-9999])
        print('downwelling N2<0:',len(hab_down))
        habw_down            = np.copy(habw_ri[area_r3dw==-9999])
        hbbl_down            = np.copy(hbbl_ri[area_r==-9999])
        w_down               = np.copy(w_ri[area_r3d==-9999])
        u_down               = np.copy(u_ri[area_r3d==-9999])
        v_down               = np.copy(v_ri[area_r3d==-9999])
        to_down              = np.copy(to_ri[area_r3d==-9999])
        kkpp_down            = np.copy(kkpp_ri[area_r3d==-9999])
        keff_down            = np.copy(keff_ri[area_r3d==-9999])
        N2_down              = np.copy(N2_ri[area_r3dw==-9999])
        b_rhs_down           = np.copy(b_rhs_ri[area_r3d==-9999])
        b_adv_down           = np.copy(b_adv_ri[area_r3d==-9999])
        b_adv_mean_save_down = np.copy(b_adv_mean_save_ri[area_r3d==-9999])
        # binned in hab space 
        h_r              = hab_down
        hw_r             = habw_down
        hbbl_r[choice]   = np.median(hbbl_down)
        w10[choice,:]    = stats.binned_statistic(h_r,w_down,statistic=lambda y: np.nanpercentile(y, 10),bins=h_e)[0]
        w90[choice,:]    = stats.binned_statistic(h_r,w_down,statistic=lambda y: np.nanpercentile(y, 90),bins=h_e)[0]
        w_r[choice,:]    = stats.binned_statistic(h_r,w_down,statistic='median',bins=h_e)[0]
        u_r[choice,:]    = stats.binned_statistic(h_r,u_down,statistic='median',bins=h_e)[0]
        v_r[choice,:]    = stats.binned_statistic(h_r,v_down,statistic='median',bins=h_e)[0]
        to_r[choice,:]   = stats.binned_statistic(h_r,to_down,statistic='median',bins=h_e)[0]
        kkpp_r[choice,:] = stats.binned_statistic(h_r,kkpp_down,statistic='median',bins=h_e)[0]
        keff_r[choice,:] = stats.binned_statistic(h_r,keff_down,statistic=np.nanmedian,bins=h_e)[0]
        # on w-grid
        N2_r[choice,:]       = stats.binned_statistic(hw_r,N2_down,statistic='median',bins=h_e)[0]
        # on rho-grid
        b_rhs_r[choice,:]    = stats.binned_statistic(h_r,b_rhs_down,statistic='median',bins=h_e)[0]
        b_adv_r[choice,:]    = stats.binned_statistic(h_r,b_adv_down,statistic='median',bins=h_e)[0]
        b_mean_r[choice,:]   = stats.binned_statistic(h_r,b_adv_mean_save_down,statistic=np.nanmedian,bins=h_e)[0]

    else:  # all grid
        hab_bblt[choice]   = np.nanmean(hab_bbl)
        hab_ag             = np.copy(hab_ri)
        print('all grid :')
        habw_ag            = np.copy(habw_ri)
        hbbl_ag            = np.copy(hbbl_ri)
        w_ag               = np.copy(w_ri)
        u_ag               = np.copy(u_ri)
        v_ag               = np.copy(v_ri)
        to_ag              = np.copy(to_ri)
        kkpp_ag            = np.copy(kkpp_ri)
        keff_ag            = np.copy(keff_ri)
        N2_ag              = np.copy(N2_ri)
        b_rhs_ag           = np.copy(b_rhs_ri)
        b_adv_ag           = np.copy(b_adv_ri)
        b_adv_mean_save_ag = np.copy(b_adv_mean_save_ri)
        # binned in hab space
        h_r              = hab_ag
        hw_r             = habw_ag
        hbbl_r[choice]   = np.median(hbbl_ag)
        w10[choice,:]    = stats.binned_statistic(h_r,w_ag,statistic=lambda y: np.nanpercentile(y, 10),bins=h_e)[0]
        w90[choice,:]    = stats.binned_statistic(h_r,w_ag,statistic=lambda y: np.nanpercentile(y, 90),bins=h_e)[0]
        w_r[choice,:]    = stats.binned_statistic(h_r,w_ag,statistic='median',bins=h_e)[0]
        u_r[choice,:]    = stats.binned_statistic(h_r,u_ag,statistic='median',bins=h_e)[0]
        v_r[choice,:]    = stats.binned_statistic(h_r,v_ag,statistic='median',bins=h_e)[0]
        to_r[choice,:]   = stats.binned_statistic(h_r,to_ag,statistic='median',bins=h_e)[0]
        kkpp_r[choice,:] = stats.binned_statistic(h_r,kkpp_ag,statistic='median',bins=h_e)[0]
        keff_r[choice,:] = stats.binned_statistic(h_r,keff_ag,statistic=np.nanmedian,bins=h_e)[0]
        # on w-grid
        N2_r[choice,:]       = stats.binned_statistic(hw_r,N2_ag,statistic='median',bins=h_e)[0]
        # on rho-grid
        b_rhs_r[choice,:]    = stats.binned_statistic(h_r,b_rhs_ag,statistic='median',bins=h_e)[0]
        b_adv_r[choice,:]    = stats.binned_statistic(h_r,b_adv_ag,statistic='median',bins=h_e)[0]
        b_mean_r[choice,:]   = stats.binned_statistic(h_r,b_adv_mean_save_ag,statistic=np.nanmedian,bins=h_e)[0]

 
           
# -- masks ---
kkpp_r[kkpp_r<1e-5]=1e-5
print('bbl N2')
print(hab_bblt)

print('bbl KPP')
print(hbbl_r)


print(' --------- make plot -------')
figure = plt.figure(figsize=(40,40))
gs = gridspec.GridSpec(3,5,height_ratios=[2,1,1],wspace=0.5,hspace=0.4)


ax = plt.subplot(gs[0,:3]) # -------------------------------------------- horizontal map N2>0 upwelling
plt.gca().set_aspect('equal', adjustable='box')
plt.title('a)')
plt.scatter(lon_up_N2pos,lat_up_N2pos,color = cf_up_N2pos,marker='D',s=1,linewidth=0.1)       # N2>0, upwelling
plt.scatter(lon_down_N2pos,lat_down_N2pos,color = cf_down_N2pos,marker='D',s=1,linewidth=0.1) # N2>0, downwelling
plt.scatter(lon_up_N2neg,lat_up_N2neg,color = cf_up_N2neg,marker='D',s=1,linewidth=0.1)       # N2<0, upwelling
plt.scatter(lon_down_N2neg,lat_down_N2neg,color = cf_down_N2neg,marker='D',s=1,linewidth=0.1) # N2<0, downwelling
# --- for legend --- 
plt.scatter(lon_up_N2pos[0],lat_up_N2pos[0],color = cf_up_N2pos,marker='D',s=100,linewidth=0.1,alpha=1,
                              label=choice_list[0]+': '+str(round(porc_up_N2pos,2))+r' $\%$ of the grid')   # N2>0, upwelling
plt.scatter(lon_down_N2pos[0],lat_down_N2pos[0],color = cf_down_N2pos,marker='D',s=100,linewidth=0.1,alpha=1,
                              label=choice_list[1]+': '+str(round(porc_down_N2pos,2))+r' $\%$ of the grid') # N2>0, downwelling
plt.scatter(lon_up_N2neg[0],lat_up_N2neg[0],color = cf_up_N2neg,marker='D',s=100,linewidth=0.1,alpha=1,
                              label=choice_list[2]+': '+str(round(porc_up_N2neg,2))+r' $\%$ of the grid')   # N2<0, upwelling
plt.scatter(lon_down_N2neg[0],lat_down_N2neg[0],color = cf_down_N2neg,marker='D',s=100,linewidth=0.1,alpha=1,
                              label=choice_list[3]+': '+str(round(porc_down_N2neg,2))+r' $\%$ of the grid') # N2<0, downwelling
ct  = ax.contour(data.h.T,levels=levels_h,colors='k',linewidths=0.8)
plt.legend(bbox_to_anchor=(1.5,0.75), loc='upper center', ncol=1,fontsize=36) #28
ax.set_xticks([0,250,500,750,1000],['0','200','400','600','800'])
ax.set_yticks([0,250,500,750],['0','200','400','600'])
plt.xlabel(r'km in $\xi$-direction',fontsize=fs)
plt.ylabel(r'km in $\eta$-direction',fontsize=fs)

### second line, w
for choice in range(len(choice_list)):
    ic = choice
    ax = plt.subplot(gs[1,ic])  # --> w_r
    plt.title(title_figw[ic])
    plt.plot(w_r[choice,:],h_c,color='k',linewidth = 2)
    plt.plot(w10[choice,:],h_c,color='k',linestyle='dashed',linewidth = 2)
    plt.plot(w90[choice,:],h_c,color='k',linestyle='dashed',linewidth = 2)
    plt.axvline(x=0,color='k',linewidth=lwc,alpha=0.8)
    plt.axhline(y=hab_bblt[choice],color='k',linewidth=lwc+1,linestyle='dashed',alpha=0.5)
    plt.axhline(y=hbbl_r[choice],color='r',linewidth=lwc+1,linestyle='dashed',alpha=0.5)
    plt.ylim(0,200)
    if choice==0:
        l = np.arange(0,70,10)
        for i in l:
            plt.axvline(x=i,color='k',linewidth=lwc,alpha=0.5)
        ax.set_xticks(l.tolist())
        plt.ylabel('hab [m]')
    elif choice==1:
        l = np.arange(-80,30,20)
        for i in l:
            plt.axvline(x=i,color='k',linewidth=lwc,alpha=0.5)
        ax.set_xticks(l.tolist())

    elif choice==2:
        l = np.arange(-150,70,50)
        for i in l:
            plt.axvline(x=i,color='k',linewidth=lwc,alpha=0.5)
        ax.set_xticks(l.tolist())

    elif choice==3:
        l = np.arange(0,250,40)
        for i in l:
            plt.axvline(x=i,color='k',linewidth=lwc,alpha=0.5)
        l2plot = [0,80,160,240]
        ax.set_xticks(l2plot)

    else:
        l = np.arange(-60,50,20)
        for i in l:
            plt.axvline(x=i,color='k',linewidth=lwc,alpha=0.5)
        ax.set_xticks(l.tolist())

    plt.xlabel(r'w [m $day^{-1}$]',fontsize=fs)
    ax.tick_params(labelsize=fs)


    ax = plt.subplot(gs[2,ic]) # --> b
    plt.title(title_figb[ic])
    plt.plot((1e11)*b_rhs_r[choice,:],h_c,color='b',linewidth = 2,label=r'$b_{rhs}$')
    plt.plot((1e11)*b_adv_r[choice,:],h_c,color='r',linewidth = 2,label=r'$b_{adv}$')
    plt.plot((1e11)*b_mean_r[choice,:],h_c,color='y',linewidth = 2,label=r'$b_{adv}^{mean}$')
    plt.plot((1e11)*(b_adv_r[choice,:]-b_mean_r[choice,:]),h_c,color='c',linewidth = 2,label=r'$b_{adv}^{eddy}$')
    plt.axvline(x=0,color='k',linewidth=lwc,alpha=0.8)
    plt.axhline(y=hab_bblt[choice],color='k',linewidth=lwc+1,linestyle='dashed',alpha=0.5)
    plt.axhline(y=hbbl_r[choice],color='r',linewidth=lwc+1,linestyle='dashed',alpha=0.5)
    plt.ylim(0,200)
    if ic==0:
        plt.legend(bbox_to_anchor=(3.5,1.3),loc='upper center',ncol=4)
    if choice==0:
        plt.ylabel('hab [m]')
        plt.xlim(-6,6)
        ax.set_xticks([-6,0,6])
    elif choice==2:
        plt.axvline(x=20,color='k',linewidth=lwc,alpha=0.5) 
        plt.axvline(x=40,color='k',linewidth=lwc,alpha=0.5)
        plt.axvline(x=-20,color='k',linewidth=lwc,alpha=0.5)
        plt.axvline(x=-40,color='k',linewidth=lwc,alpha=0.5)
        plt.xlim(-25,60)
        ax.set_xticks([-40,-20,0,20,40])
    elif choice==3:
        plt.axvline(x=10,color='k',linewidth=lwc,alpha=0.5)
        plt.axvline(x=20,color='k',linewidth=lwc,alpha=0.5)
        plt.axvline(x=-10,color='k',linewidth=lwc,alpha=0.5)
        plt.axvline(x=-20,color='k',linewidth=lwc,alpha=0.5)
        plt.xlim(-25,30)
        ax.set_xticks([-20,-10,0,10,20])
    else:
        plt.axvline(x=2,color='k',linewidth=lwc,alpha=0.5)
        plt.axvline(x=4,color='k',linewidth=lwc,alpha=0.5)
        plt.axvline(x=-2,color='k',linewidth=lwc,alpha=0.5)
        plt.axvline(x=-4,color='k',linewidth=lwc,alpha=0.5)
        plt.axvline(x=-6,color='k',linewidth=lwc,alpha=0.5)
        plt.axvline(x=-8,color='k',linewidth=lwc,alpha=0.5)
        plt.xlim(-10,6)
        ax.set_xticks([-10,-6,0,6])
    plt.xlabel(r'b [$10^{-11}$ $m^2$ $s^{-3}$]',fontsize=fs)
    ax.tick_params(labelsize=fs)


plt.savefig('figure6.png',dpi=200,bbox_inches='tight')
plt.close()


