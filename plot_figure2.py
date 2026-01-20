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
matplotlib.rcParams.update({'font.size': 18}) 

# ------------ parameters ------------ 
region = 'rrex'  

path_data  = '/home/datawork-lops-rrex/nschifan/'

#file_grd   = path_data+'GRD/rrextra_grd.nc'
file_grd   = path_data+'GRD/rrexnum_grd.nc'
begin      = '10'
date_end   = '30'
file_his   = '/home/datawork-lops-megatl/RREXNUM/RREXNUMSB200_RSUP5_NOFILT_T/HIS/rrexnum200_avg.mean.00010-00049.nc'  # prefix 
file0      = path_data+'HIST/rrex100-up3_avg.00000.nc'  # prefix 
date       = ['00010-00049']#,'2008-08-27-2008-08-28'] 
date_real  = [' 09/2008']

# ------------ parameters ------------ 
name_exp    = 'rrexnum200' #['rrex100-up3','rrex100-up5','rrex100-weno5','rrex200-up3','rrex200-up5','rrex200-weno5','rrex300-up3','rrex300-500cpu-up5','rrex300-500cpu-weno5']
name_exp_path ='rrexnum200_rsup5'#-rsup5'
#title_exp   = ['a) rrex100-up3','b) rrex100-up5','c) rrex100-weno5','d) rrex200-up3','e) rrex200-up5','f) rrex200-weno5','g) rrex300-up3','h) rrex300-up5','i) rrex300-weno5']
name_pathdata = 'RREXNUMSB200_RSUP5_NOFILT_T'#RSUP5_NOFILT_T'
nbr_levels  = '200'
name_exp_grd= 'rrex200-up5'
nbr_levels  = '200'

date_ini    = (2008,9,5,0,0,0) # year,month,day,hour,minutes,seconds for initial state  
date_end    = (2008,11,30,0,0,0) # year,month,day,hour,minutes,seconds for end     state 
output_freq = '1h'
ndfiles     = 1  # number of days per netcdf
time= ['20']
var_list = ['w','zeta','u','v']
# - select tracers to plot - 
sig_min,sig_max = 27.9,27.55 # axis limits 

# --- plot options --- 
jsec    = 400
fs      = 18    # fontsize 
lw      = 2     # linewidth
ms      = 20    # markersize 
lw_c    = 0.4   # 0.2 linewidth coast 
my_bbox = dict(fc='w',ec='k',pad=2,lw=0.,alpha=0.5)
res   = 'h' # resolution of the coastline: c (coarse), l (low), i (intermediate), h (high)
proj  = 'lcc'  # map projection. lcc = lambert conformal 
paral = np.arange(0,80,5)
merid = np.arange(0,360,5)
lon_0,lat_0   = -31.5,58.6 # centre of the map 
Lx,Ly         = 1400e3,1500e3 # [km] zonal and meridional extent of the map 
scale_km      = 400 # [km] map scale 
# --> for Quiver horizontal currents
xlon        = np.arange(0,1003)
ylat        = np.arange(0,803)
q_x, q_y    = np.meshgrid(xlon,ylat, indexing='ij')
cfq         = 'k' 
# --W depth to interpolate w 
depth = [-1000,-2000]

# --- bin hab
dz = 10. # [m] 
zbine = np.arange(-5000,dz,dz)     # bin edges 
zbinc = 0.5*(zbine[1:]+zbine[:-1]) # bin centres 
nz    = zbinc.shape[0]
# - regrid in height-above-bottom (hab) coordinates - 
#habe      = np.arange(0,3000+dz,dz)
habe =np.array([  0,   3,   6,   9,  12,  15,  18,  21,  24,  27,  30,  33,  36,
        39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,
        78,  81,  84,  87,  90,  93,  96,  99, 102,
         112,  122,  132,  142,  152,  162,  172,  182,  192,  202,
        212,  222,  232,  242,  252,  262,  272,  282,  292,  302,  312,
        322,  332,  342,  352,  362,  372,  382,  392,  402,  412,  422,
        432,  442,  452,  462,  472,  482,  492,  502,  512,  522,  532,
        542,  552,  562,  572,  582,  592,  602,  612,  622,  632,  642,
        652,  662,  672,  682,  692,  702,  712,  722,  732,  742,  752,
        762,  772,  782,  792,  802,  812,  822,  832,  842,  852,  862,
        872,  882,  892,  902,  912,  922,  932,  942,  952,  962,  972,
        982,  992, 1002, 1012, 1022, 1032, 1042, 1052, 1062, 1072, 1082,
       1092, 1102, 1112, 1122, 1132, 1142, 1152, 1162, 1172, 1182, 1192,
       1202, 1212, 1222, 1232, 1242, 1252, 1262, 1272, 1282, 1292, 1302,
       1312, 1322, 1332, 1342, 1352, 1362, 1372, 1382, 1392, 1402, 1412,
       1422, 1432, 1442, 1452, 1462, 1472, 1482, 1492, 1502, 1512, 1522,
       1532, 1542, 1552, 1562, 1572, 1582, 1592, 1602, 1612, 1622, 1632,
       1642, 1652, 1662, 1672, 1682, 1692, 1702, 1712, 1722, 1732, 1742,
       1752, 1762, 1772, 1782, 1792, 1802, 1812, 1822, 1832, 1842, 1852,
       1862, 1872, 1882, 1892, 1902, 1912, 1922, 1932, 1942, 1952, 1962,
       1972, 1982, 1992, 2002, 2012, 2022, 2032, 2042, 2052, 2062, 2072,
       2082, 2092, 2102, 2112, 2122, 2132, 2142, 2152, 2162, 2172, 2182,
       2192, 2202, 2212, 2222, 2232, 2242, 2252, 2262, 2272, 2282, 2292,
       2302, 2312, 2322, 2332, 2342, 2352, 2362, 2372, 2382, 2392, 2402,
       2412, 2422, 2432, 2442, 2452, 2462, 2472, 2482, 2492, 2502, 2512,
       2522, 2532, 2542, 2552, 2562, 2572, 2582, 2592, 2602, 2612, 2622,
       2632, 2642, 2652, 2662, 2672, 2682, 2692, 2702, 2712, 2722, 2732,
       2742, 2752, 2762, 2772, 2782, 2792, 2802, 2812, 2822, 2832, 2842,
       2852, 2862, 2872, 2882, 2892, 2902, 2912, 2922, 2932, 2942, 2952,
       2962, 2972, 2982, 2992, 3002])


habc = 0.5*(habe[1:]+habe[:-1]) # hab centres
nhab      = habc.shape[0]

# --- bin topostrophy [cm/s]
toe = np.arange(0,0.2,0.002)
toc = 0.5*(toe[1:]+toe[:-1])
nbin = 100

# bin w [m/day] 
we = np.arange(-25,5,0.3)
wc = 0.5*(we[1:]+we[:-1])


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))#pmin,pmax,pint  = -0.0005, 0.0005,0.00001


# --- w ---
cmap_w     =  plt.cm.RdBu_r #seismic
pmin,pmax,pint   = -50, 50,1#-5e-3,5e-3,5e-5
levels_w      = np.arange(pmin,pmax+pint,pint)
norm_w        = colors.Normalize(vmin=pmin,vmax=pmax)
cbticks_w     =  [pmin,0,pmax]
cblabel_w     = r'w [m day$^{-1}$]'

sigmin,sigmax = -0.006,0.003

# --- to ---
cmap_to         =  cmap_w#plt.cm.Spectral_r 
pmin,pmax,pint  = -1,1,0.01
norm_to         = colors.SymLogNorm(linthresh=0.003, linscale=0.003,
                                              vmin=pmin, vmax=pmax)
cbticks_to      =  [-1,-0.1,0,0.1,1]
cblabel_to      = r'$\tau$ [cm s$^{-1}$]'

levels_hplot   = np.arange(0,4000,500)


# --- define areas ---
#         [imin,imax,jmin,jmax]
points_rs = [225,425,100,200]
points_rn = [250,450,600,700]
# --> area abyssal plain
points_ap = [700,900,400,500]

# --- color for bar plot 
cfg  = 'b'
# ridge north
cfrn  = 'r'
# ridge south
cfrs  = colors.to_rgba('dimgray')
# ridge north
cfap  = 'm'

cf = [cfg, cfrn, cfrs, cfap]
cfw = [cfg, cfg, cfrn, cfrn, cfrs, cfrs, cfap, cfap]
 
def vert_int(var,z_w,depth1,depth2):

    cff2 = np.minimum(depth1,depth2)
    cff1 = np.maximum(depth1,depth2)

    Hz = z_w[:,:,1:] - z_w[:,:,:-1]
    
    cff = (z_w[:,:,:-1].T - cff1.T).T
    Hz[cff>0] = 0.
    Hz[np.logical_and(np.abs(cff)<Hz,cff<0)] = -cff[np.logical_and(np.abs(cff)<Hz,cff<0)]
    
    cff = (z_w[:,:,1:].T - cff2.T).T
    Hz[cff<0] = 0.
    Hz[np.logical_and(cff<Hz,cff>0)] = cff[np.logical_and(cff<Hz,cff>0)]
    
    varint = np.nansum(Hz * var,2)
    #varint = np.nansum( var,2)    
    
    return varint

    
def var_vol(var,vol):
    #new_var = np.zeros(np.shape(var))
    #for k in range(np.shape(var)[2]):
    #    new_var[:,:] = np.nansum(var[:,:,k]*vol[:,:,k],axis=-1)/np.nansum(vol[:,:,k],axis=-1))
    #for i in range(np.shape(var)[0]):
    #    for j in range(np.shape(var)[1]):
    #        new_var[i,j,:] = var[i,j,:]*vol[i,j,:]/np.nansum(vol[i,j,:]))
    #        if (i==10)and(j==10):
    #            print(var[i,j,:],vol[i,j,:],np.nansum(vol[i,j,:]))
    #new_var[var==np.nan]=np.nan
    #return new_var
    return np.nansum(var*vol,axis=-1)/np.nansum(vol,axis=-1)


def define_var_at_area(var,points):
    # select a variable at the area define between points
    # points containes (x,y) coordinates of the limits of the square 
    return var[points[0]:points[1],points[2]:points[3]]

def plot_k_oneline(ax,K,name_x_jon,fs,cf,cfw,sigmin,sigmax):    
    for expk in range(np.shape(K)[0]):
        print(expk)
    data = [K[expk][:] for expk in range(np.shape(K)[0])]
    # --- deal with Nan issues when using boxplot
    mask = ~np.isnan(data)
    filtered_data = [d[m] for d, m in zip(data, mask)]
    flierprop = dict( markersize=2)
    bp = ax.boxplot(filtered_data,medianprops=dict(color='k'),flierprops=flierprop,patch_artist = True, showfliers=False)
    for patch, colors in zip(bp['boxes'], cf):
        patch.set_facecolor(colors)
        patch.set(color = colors)
    for moustache, colors in zip(bp['whiskers'], cfw):
        moustache.set(color = colors)
    for cap, colors in zip(bp['caps'], cfw):
        cap.set(color = colors)
    plt.axhline(y=0)
    plt.xticks(ticks=[1,2,3,4],labels=name_x_jon,fontsize=fs)
    plt.xticks(rotation=30)
    return




# --- create variables ---
w_stat = np.zeros((3))
to_stat = np.zeros((3))
wrs_stat = np.zeros((3))
tors_stat = np.zeros((3))
wrn_stat = np.zeros((3))
torn_stat = np.zeros((3))
wap_stat = np.zeros((3))
toap_stat = np.zeros((3))
choice_list = ['all grid','ridge north','ridge south','abyssal plain']



# ------------ read data ------------ 
omega_T = 7.2921e-5 # --> rotation of the Earth
tt      = 0
for t_nc in range(len(time)):
    count_NL=0
    for t in range(ndfiles):
        data         = Croco(name_exp,nbr_levels,time[0],name_exp_grd,name_pathdata)
        data_longrun = Croco_longrun(name_exp,nbr_levels,time[0],name_exp_grd,name_pathdata) 
        #data = Croco(name_exp,nbra_levels,taime[t_nc],name_exp_grd,name_pathdata)
        data.get_grid()
        # ------------ make plot ------------ 
        print(' ... read avg file data + make plot ... ')
        # -- valeurs min et max longitude et latitude
        # --- [ lon min, lon max, lat min, lat max]               
        extent     = [-37.5,-21.2,53,62.5] # °E °N
        data.get_outputs(t,var_list,get_date=False)
        data_longrun.get_outputs(t,var_list,get_date=False)
        # ------------ read data ------------ 
        [z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
        # --- volume cell ---
        dsurf   = 1./np.transpose(np.tile(data.pm*data.pn,(int(nbr_levels),1,1)),(1,2,0))
        dvol   = np.diff(z_w,axis=-1)*dsurf
        # -- w [m/day] ---
        w   = 3600*24*data.var['w']
        u_rho = tools.u2rho(data.var['u']) #np.zeros([1002,802,200])
        v_rho = tools.v2rho(data.var['v']) 
        #w0  = w[:,:,0]
        # --- coriolis frequency
        k    = 2*omega_T*np.sin(data.latr)
        print(np.shape(k))
        # --- compute hab ---
        h_tile = np.transpose(np.tile(data.h,(200,1,1)),(1,2,0))
        hab    = h_tile + z_r
   
        # ---> compute topostrophy [cm/s]
        dhdx = tools.u2rho((data.h[1:,:]-data.h[:-1,:]))*data.pm
        dhdy = tools.v2rho((data.h[:,1:]-data.h[:,:-1]))*data.pn
        dhdx3d = np.transpose(np.tile(dhdx,(200,1,1)),(1,2,0))
        dhdy3d = np.transpose(np.tile(dhdy,(200,1,1)),(1,2,0))
        to   = 100*(tools.u2rho(data.var['u'])*dhdy3d - tools.v2rho(data.var['v'])*dhdx3d)

        # --> bin in hab space  
        # w [m/day], to [cm/s]
        w_bin  =  stats.binned_statistic(np.ravel(hab), np.ravel(w), 'mean', bins=habe)[0]
        to_bin =  stats.binned_statistic(np.ravel(hab), np.ravel(to), 'mean', bins=habe)[0]
        w_binmd  =  stats.binned_statistic(np.ravel(hab), np.ravel(w), 'median', bins=habe)[0]
        to_binmd =  stats.binned_statistic(np.ravel(hab), np.ravel(to), 'median', bins=habe)[0]

        print(np.shape(w_bin),np.shape(to_bin))

        # --> read longrun mean 
        w_longrun        = data_longrun.var['w']
        to_longrun       = (tools.u2rho(data_longrun.var['u'])*dhdy3d - tools.v2rho(data_longrun.var['v'])*dhdx3d)
        w0               = 3600*24*w_longrun[:,:,0]
        to0              = 100*to_longrun[:,:,0]
        w_bin_longrun    =  stats.binned_statistic(np.ravel(hab), np.ravel(3600*24*w_longrun), 'mean', bins=habe)[0]
        to_bin_longrun   =  stats.binned_statistic(np.ravel(hab), np.ravel(100*to_longrun), 'mean', bins=habe)[0]
        w_binmd_longrun  =  stats.binned_statistic(np.ravel(hab), np.ravel(3600*24*w_longrun), 'median', bins=habe)[0]
        to_binmd_longrun =  stats.binned_statistic(np.ravel(hab), np.ravel(100*to_longrun), 'median', bins=habe)[0]

        # --> statistics 
        w_stat = np.nanpercentile(np.ravel(w0),[10,50,90])
        to_stat = np.nanpercentile(np.ravel(to0),[10,50,90])
        print(' ----------------------------------------------------------------------------')
        print('--> statistics over the domain, w  = ',w_stat[1])
        print('--> statistics over the domain, to = ',to_stat[1])
 
        wrn_stat = np.nanpercentile(np.ravel(define_var_at_area(w0,points_rn)),[10,50,90])
        torn_stat = np.nanpercentile(np.ravel(define_var_at_area(to0,points_rn)),[10,50,90])

        wrs_stat = np.nanpercentile(np.ravel(define_var_at_area(w0,points_rs)),[10,50,90])
        tors_stat = np.nanpercentile(np.ravel(define_var_at_area(to0,points_rs)),[10,50,90])

        wap_stat = np.nanpercentile(np.ravel(define_var_at_area(w0,points_ap)),[10,50,90])
        toap_stat = np.nanpercentile(np.ravel(define_var_at_area(to0,points_ap)),[10,50,90])

        # --> interpolate w
        u_z1 = tools.vinterp(u_rho[:,:,:],[depth[0]],z_r,z_w)
        v_z1 = tools.vinterp(v_rho[:,:,:],[depth[0]],z_r,z_w)
        w_z1 = tools.vinterp(w,[depth[0]],z_r,z_w)
        u_z2 = tools.vinterp(u_rho[:,:,:],[depth[1]],z_r,z_w)
        v_z2 = tools.vinterp(v_rho[:,:,:],[depth[1]],z_r,z_w)
        w_z2 = tools.vinterp(w,[depth[1]],z_r,z_w)

        print(' ... get vertical levels ... ')
        latr = data.latr
        lonr = data.lonr
        h    = data.h
        print('time:',tt)
        print( '--- MAKE PLOT ---')
        # ------------ make plot  ------------ 
        count_x=0 #  line
        count_y=0 #  column
        last_line=0
        plt.figure(figsize=(15,15))                                  #0.4
        gs = gridspec.GridSpec(3,2,height_ratios=[1,0.05,1],hspace=0.45,wspace=0.3) #0.7
        
        # ---> topostrophy(s0)
        ax = plt.subplot(gs[0,0])
        plt.gca().set_aspect('equal', adjustable='box')
        ctf02 = ax.pcolormesh(to0.T,cmap=cmap_to,zorder=1,norm=norm_to)
        bathy = ax.contour(h.T,levels =[1000,2000,3000,4000] , colors='k',linewidths= lw_c,zorder=2)
        eps=25
        #Q =  plt.quiver(q_x[::eps,::eps].T, q_y[::eps,::eps].T, 2*u_rho[::eps,::eps,0].T, 2*v_rho[::eps,::eps,0].T, units='width',scale=7,color=cfq,zorder=3)
        Q =  plt.quiver(q_x[::eps,::eps].T, q_y[::eps,::eps].T, 3*u_rho[::eps,::eps,0].T, 3*v_rho[::eps,::eps,0].T, units='width',scale=7,color=cfq,zorder=3)
        ax.quiverkey(Q, X=0.9, Y=1.05, U=0.2,label=r'$20 \frac{cm}{s}$', labelpos='E')
        # ---> area above ridge north
        index  = [250,250,450,450,250]
        index2 = [600,700,700,600,600]
        x,y    = index, index2
        #x = [data.lonr[index[0],index2[0]],data.lonr[index[1],index2[1]],data.lonr[index[2],index2[2]],data.lonr[index[3],index2[3]],data.lonr[index[4],index2[4]]]
        #y = [data.latr[index[0],index2[0]],data.latr[index[1],index2[1]],data.latr[index[2],index2[2]],data.latr[index[3],index2[3]],data.latr[index[4],index2[4]]]
        ax.fill(x, y, color='r',linewidth=2,fill=False,zorder=3)#, hatch='\\\\\\\\\\\\')#,alpha=0.5)

        # ---> area above ridge
        index  = [225,225,425,425,225]
        index2 = [100,200,200,100,100]
        x,y    = index, index2
        #x = [data.lonr[index[0],index2[0]],data.lonr[index[1],index2[1]],data.lonr[index[2],index2[2]],data.lonr[index[3],index2[3]],data.lonr[index[4],index2[4]]]
        #y = [data.latr[index[0],index2[0]],data.latr[index[1],index2[1]],data.latr[index[2],index2[2]],data.latr[index[3],index2[3]],data.latr[index[4],index2[4]]]
        ax.fill(x, y, color='k',linewidth=2,fill=False,zorder=3)#, hatch='\\\\\\\\\\\\')#,alpha=0.5)

        # --> area above abyssal plain
        index  = [700,700,900,900,700]
        index2 = [400,500,500,400,400]
        x,y    = index, index2
        #x = [data.lonr[index[0],index2[0]],data.lonr[index[1],index2[1]],data.lonr[index[2],index2[2]],data.lonr[index[3],index2[3]],data.lonr[index[4],index2[4]]]
        #y = [data.latr[index[0],index2[0]],data.latr[index[1],index2[1]],data.latr[index[2],index2[2]],data.latr[index[3],index2[3]],data.latr[index[4],index2[4]]]
        ax.fill(x, y, color='m',linewidth=2,fill=False,zorder=3)#, hatch='\\\\\\\\\\\\')#,alpha=0.5)

        #plt.ylabel('grid-cells in j-direction',fontsize=fs)
        #plt.xlabel('grid-cells in i-direction',fontsize=fs)
        ax.set_xticks([0,250,500,750,1000],['0','200','400','600','800'])
        ax.set_yticks([0,250,500,750],['0','200','400','600'])
        plt.xlabel(r'km in $\xi$-direction',fontsize=fs)
        plt.ylabel(r'km in $\eta$-direction',fontsize=fs)

        plt.title(r'a) $\tau$($s_0$)',fontsize=fs)



        # --> w(s0)
        ax = plt.subplot(gs[0,1]) # --> w(s0)
        plt.gca().set_aspect('equal', adjustable='box')
        ctf01 = ax.pcolormesh(w0.T,cmap=cmap_w,zorder=1,norm=norm_w)
        bathy = ax.contour(h.T,levels =[1000,2000,3000,4000] , colors='k',linewidths= lw_c,zorder=2)
        #eps=25
        #Q =  plt.quiver(q_x[::eps,::eps].T, q_y[::eps,::eps].T, 2*u_rho[::eps,::eps,0].T, 2*v_rho[::eps,::eps,0].T, units='width',scale=7,color=cfq,zorder=3)
        #ax.quiverkey(Q, X=0.9, Y=1.05, U=0.2,label=r'$20 \frac{cm}{s}$', labelpos='E')
        # ---> area above ridge north
        index  = [250,250,450,450,250]
        index2 = [600,700,700,600,600]
        x,y    = index, index2
        #x = [data.lonr[index[0],index2[0]],data.lonr[index[1],index2[1]],data.lonr[index[2],index2[2]],data.lonr[index[3],index2[3]],data.lonr[index[4],index2[4]]]
        #y = [data.latr[index[0],index2[0]],data.latr[index[1],index2[1]],data.latr[index[2],index2[2]],data.latr[index[3],index2[3]],data.latr[index[4],index2[4]]]
        ax.fill(x, y, color='r',linewidth=2,fill=False,zorder=3)#, hatch='\\\\\\\\\\\\')#,alpha=0.5)

        # ---> area above ridge
        index  = [225,225,425,425,225]
        index2 = [100,200,200,100,100]
        x,y    = index, index2
        #x = [data.lonr[index[0],index2[0]],data.lonr[index[1],index2[1]],data.lonr[index[2],index2[2]],data.lonr[index[3],index2[3]],data.lonr[index[4],index2[4]]]
        #y = [data.latr[index[0],index2[0]],data.latr[index[1],index2[1]],data.latr[index[2],index2[2]],data.latr[index[3],index2[3]],data.latr[index[4],index2[4]]]
        ax.fill(x, y, color='k',linewidth=2,fill=False,zorder=3)#, hatch='\\\\\\\\\\\\')#,alpha=0.5)

        # --> area above abyssal plain
        index  = [700,700,900,900,700]
        index2 = [400,500,500,400,400]
        x,y    = index, index2
        #x = [data.lonr[index[0],index2[0]],data.lonr[index[1],index2[1]],data.lonr[index[2],index2[2]],data.lonr[index[3],index2[3]],data.lonr[index[4],index2[4]]]
        #y = [data.latr[index[0],index2[0]],data.latr[index[1],index2[1]],data.latr[index[2],index2[2]],data.latr[index[3],index2[3]],data.latr[index[4],index2[4]]]
        ax.fill(x, y, color='m',linewidth=2,fill=False,zorder=3)#, hatch='\\\\\\\\\\\\')#,alpha=0.5)

        #plt.ylabel('grid-cells in j-direction',fontsize=fs)
        #plt.xlabel('grid-cells in i-direction',fontsize=fs)
        ax.set_xticks([0,250,500,750,1000],['0','200','400','600','800'])
        ax.set_yticks([0,250,500,750],['0','200','400','600'])
        plt.xlabel(r'km in $\xi$-direction',fontsize=fs)
        plt.ylabel(r'km in $\eta$-direction',fontsize=fs)

        plt.title(r'b) w($s_0$)',fontsize=fs)



        cax = plt.subplot(gs[1,1]) # --> w(s0)
        cb     = plt.colorbar(ctf01,cax,orientation='horizontal',extend='both',ticks=cbticks_w)
        cb.set_label(cblabel_w,fontsize=fs,labelpad=-65)
        cb.ax.tick_params(labelsize=fs)


        # - bounding box 'aext' -  # --> to(so)
        cax = plt.subplot(gs[1,0])
        cb     = plt.colorbar(ctf02,cax,orientation='horizontal',extend='both',ticks=cbticks_to)
        cb.set_label(cblabel_to,fontsize=fs,labelpad=-65)
        cb.ax.tick_params(labelsize=fs)



        # --> statistics to(s0)
        ax = plt.subplot(gs[2,0])
        W  = [to_stat,torn_stat,tors_stat,toap_stat]
        plot_k_oneline(ax,W,choice_list,fs,cf,cfw,sigmin,sigmax)
        plt.title(r'c)',fontsize=fs)
        plt.ylabel(r'$\tau$($s_0$) [cm s$^{-1}$]')


        # --> statistics w(s0)
        ax = plt.subplot(gs[2,1])
        W  = [w_stat,wrn_stat,wrs_stat,wap_stat]
        plot_k_oneline(ax,W,choice_list,fs,cf,cfw,sigmin,sigmax)
        plt.title('d)',fontsize=fs)
        plt.ylabel(r'w($s_0$) [m day$^{-1}$]')

        #cax = plt.subplot(gs[2,0]) # --> to(s0)
        #cb     = plt.colorbar(ctf01,cax,orientation='horizontal',extend='both',ticks=cbticks_w)
        #cb.set_label(cblabel_w,fontsize=fs,labelpad=-60)
        #cb.ax.tick_params(labelsize=fs)

        ## --> w(-2000m)
        #ax = plt.subplot(gs[2,0])  # ---> 2000 m
        #plt.gca().set_aspect('equal', adjustable='box')
        #ax.set_title('e) w(z=-2000m)',fontsize=fs)
        #eps = 25
        #plt.ylabel('grid-cells in j-direction',fontsize=fs)
        #plt.xlabel('grid-cells in i-direction',fontsize=fs)
        #ct  = ax.contour(data.h.T,levels=levels_hplot,colors='grey',linewidths=lw_c,zorder=2)
        #ctf = ax.pcolormesh(w_z2[:,:,0].T,cmap=cmap_w,zorder=1,norm=norm_w)
        #Q =  plt.quiver(q_x[::eps,::eps].T, q_y[::eps,::eps].T,2*u_z2[::eps,::eps,0].T,2*v_z2[::eps,::eps,0].T, units='width',scale=7,color=cfq,zorder=3)      
        ## --- legend quiver ---
        #ax.quiverkey(Q, X=0.9, Y=1.05, U=0.2,label=r'$20 \frac{cm}{s}$', labelpos='E')#,coordinates=figure)

        ## --> w(-1000m)
        #ax = plt.subplot(gs[2,1])  # ---> 1000 m
        #plt.gca().set_aspect('equal', adjustable='box')
        #ax.set_title('f) w(z=-1000m)',fontsize=fs)
        #eps = 25
        #plt.ylabel('grid-cells in j-direction',fontsize=fs)
        #plt.xlabel('grid-cells in i-direction',fontsize=fs)
        #ct  = ax.contour(data.h.T,levels=levels_hplot,colors='grey',linewidths=lw_c,zorder=2)
        #ctf = ax.pcolormesh(w_z1[:,:,0].T,cmap=cmap_w,zorder=1,norm=norm_w)
        #Q =  plt.quiver(q_x[::eps,::eps].T, q_y[::eps,::eps].T, 2*u_z1[::eps,::eps,0].T, 2*v_z1[::eps,::eps,0].T, units='width',scale=7,color=cfq,zorder=3)
        #ax.quiverkey(Q, X=0.9, Y=1.05, U=0.2,label=r'$20 \frac{cm}{s}$', labelpos='E')#,coordinates=figure)



        plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/WMT/rrexnumsb200-rsup5_horizontal_slice_w_avg_last_sigma_mean00010-00229.png',dpi=180,bbox_inches='tight')
        plt.close()



        tt+=1
