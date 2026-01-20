'''
NS: Horizontal map of:
        - the slope
        - the standard deviation of the bottom vertical velocity
    Standard deviation of the vertical velocity as a function of the slope
    PSD of the vertical velocity at the deepest measurement of the mooring
        RRT and the interpolation from CROCO, both following the no-normal flow condition
        
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal as sig 
import scipy.stats  as stats
import scipy.interpolate  as itp
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors 
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable # colorbar 
from netCDF4 import Dataset
import gsw as gsw 
import sys
sys.path.append('Python_Modules_p3/')
import R_tools as tools
from croco_simulations_jonathan_ncra_longrun import Croco_longrun
matplotlib.rcParams.update({'font.size': 20})


# ------------ parameters ------------ 
# ---------> choose w or u.grad(h)
list_choice = ['w','w_from_u']
choice      = list_choice[1]
if choice == 'w':
    # --- w ---
    label_w     = r'w [m.day$^{-1}$]'
    cf0         = colors.to_rgba('teal')
else:
    # --- w from u ---
    label_w     = r'-$\overline{u}^h$ $\cdot$ $\nabla^h$ H'
    cf0         = colors.to_rgba('royalblue')

# ------------ parameters in-situ ------------ 
choice_mooring  = ['IRW','IRM','IRE','RRT','ICW','ICM','ICE']
stations        = len(choice_mooring)
nr              = [30,  3, 66, 93, 63, 29, 29] # number of measures for each mooring
file_data_d     = 'rrex_dD_10km.nc'
file_data       = 'RREX_'
end_nc          = '_CURR_hourly.nc'
file_croco      = 'rrex_bottom_w_mooring_notracer.nc'
idx_lon, idx_lat= [191, 266, 309, 373, 438, 503, 599],[363, 387, 402, 416, 400, 385, 361]


ti,tf = 24166,24166+365  
list_moorings = ['RRT'] 
M2 = 1./44700. # [sec] 
S2 = 1./43200. 
O1 = 1./92950.  
K1 = 1./86164
Nbut = 4 # order of Butterworth filter
wkb_stretching = False # WKB stretching 

# --- plot options --- 
plot_couple_psd      = False # can't choose this and the following (different files)  
plot_rotary_psd      = False # rotary  (clockwise and counter-clockwise)              
plot_psd_z_L4        = False # from L4, i.e., interpolated data on the vertical 
plot_psd_z_L3        = False # from L3, i.e., data at original depth level 
plot_psd_z_L3_rotary = False # from L3, i.e., data at original depth level 
plot_psd_monthly     = False 
plot_psd_season      = False  
plot_psd_all         = True
plot_psd_std         = True
title_mooring        = ['a) '+choice_mooring[0] , 'b) '+choice_mooring[1] , 'c) '+choice_mooring[2] ,
                                                  'd) '+choice_mooring[3] , 'e) '+choice_mooring[4] , 
                                                  'f) '+choice_mooring[5] , 'g) '+choice_mooring[6]]
fs             = 20 #  psd season 
raster         = True 


# ----------- read in-situ currents ---------
######### IRW 
nc       = Dataset(file_data+choice_mooring[0]+end_nc,'r')
uirw_qc  = nc.variables['UCUR_QC'][:,:]
virw_qc  = nc.variables['VCUR_QC'][:,:]
u_irw    = nc.variables['UCUR'][:,:]
v_irw    = nc.variables['VCUR'][:,:]
d_irw    = nc.variables['DEPTH'][:,:]
lat_irw  = nc.variables['LATITUDE'][:]
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
lat_irm  = nc.variables['LATITUDE'][:]
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
lat_ire  = nc.variables['LATITUDE'][:]
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
lat_rrt  = nc.variables['LATITUDE'][:]
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
lat_icw  = nc.variables['LATITUDE'][:]
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
lat_icm  = nc.variables['LATITUDE'][:]
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
lat_ice  = nc.variables['LATITUDE'][:]
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
w_irw = -1*(u_irw*dDdx[0]+v_irw*dDdy[0])
w_irm = -1*(u_irm*dDdx[1]+v_irm*dDdy[1])
w_ire = -1*(u_ire*dDdx[2]+v_ire*dDdy[2])
w_rrt = -1*(u_rrt*dDdx[3]+v_rrt*dDdy[3])
w_icw = -1*(u_icw*dDdx[4]+v_icw*dDdy[4])
w_icm = -1*(u_icm*dDdx[5]+v_icm*dDdy[5])
w_ice = -1*(u_ice*dDdx[6]+v_ice*dDdy[6])

w   = [ w_irw, w_irm, w_ire, w_rrt,
               w_icw, w_icm, w_ice ]

lat = [ lat_irw, lat_irm, lat_ire, lat_rrt,
               lat_icw, lat_icm, lat_ice ]

# --------- read bottom w from CROCO ---------
ncc    =  Dataset(file_croco,'r')
if choice == 'w':
    wc_irw = ncc.variables['wb0'][:]/(3600*24)
    wc_irm = ncc.variables['wb1'][:]/(3600*24)
    wc_ire = ncc.variables['wb2'][:]/(3600*24)
    wc_rrt = ncc.variables['wb3'][:]/(3600*24)
    wc_icw = ncc.variables['wb4'][:]/(3600*24)
    wc_icm = ncc.variables['wb5'][:]/(3600*24)
    wc_ice = ncc.variables['wb6'][:]/(3600*24)
else:
    wc_irw = ncc.variables['wub0'][:]/(3600*24)
    wc_irm = ncc.variables['wub1'][:]/(3600*24)
    wc_ire = ncc.variables['wub2'][:]/(3600*24)
    wc_rrt = ncc.variables['wub3'][:]/(3600*24)
    wc_icw = ncc.variables['wub4'][:]/(3600*24)
    wc_icm = ncc.variables['wub5'][:]/(3600*24)
    wc_ice = ncc.variables['wub6'][:]/(3600*24)
ncc.close()


wc   = [ wc_irw, wc_irm, wc_ire, wc_rrt,
               wc_icw, wc_icm, wc_ice ]

print('bottom')
print(wc_irw)

# mooring : 1h
dt      = 3600 
fsamp   = 1./dt #[Hz] sampling frequency
## CROCO  : 12h
#fsampc  = fsamp/12
#CROCO  : 1h
fsampc  = fsamp
# number of days for echantillonage
ndays   = 30*24*3600

nperseg_m  = np.int(ndays*fsamp) #256 # each hour
nperseg_c  = np.int(ndays*fsampc) # each 12h 
print('nperseg',nperseg_m,nperseg_c)
freqr    = np.nan*np.zeros((stations,nperseg_m//2+1))
freqrc   = np.nan*np.zeros((stations,nperseg_c//2+1))
psdr     = np.nan*np.zeros((stations,nperseg_m//2+1)) # rotary 
psdrc    = np.nan*np.zeros((stations,nperseg_c//2+1)) # rotary 
k        = -1 # deepest measure of mooring

# --> loop on moorings
for mooring in range(stations):
    w_m    = np.copy(w[mooring][:,k])
    wc_m   = np.copy(wc[mooring][:])
    print(mooring)
    print(wc_m)
    if plot_psd_all:
        tmp =  np.copy(w_m)
        if tmp[~np.isnan(tmp)].shape[0]==tmp.shape[0]: # length of no-nan values of tmp
            [freqr[mooring,:],psdr[mooring,:]] = sig.welch(w_m,fsamp, window='hanning',nperseg=nperseg_m)

        tmp =  np.copy(wc_m)
        if tmp[~np.isnan(tmp)].shape[0]==tmp.shape[0]: # length of no-nan values of tmp
             [freqrc[mooring,:],psdrc[mooring,:]] = sig.welch(wc_m,fsampc, window='hanning',nperseg=nperseg_c)
    

# --- band-pass filtering --- 
fi    = gsw.f(lat[mooring])/(2*np.pi)
c     = 1.07                                # Alford's parameter is 1.25 
Wn_M2 = np.array([(1./c)*M2,c*M2])*(2*dt)   # cutoff freq, normalized [0,1], 1 is the Nyquist freq 
Wn_fi = np.array([(1./c)*fi,c*fi])*(2*dt)   # cutoff freq, normalized [0,1], 1 is the Nyquist freq 
SD = 0.5*(M2+S2)                            # semi-diurnal frequencies 
Wn_SD = np.array([(1./c)*SD,c*SD])*(2*dt)   # cutoff freq, normalized [0,1], 1 is the Nyquist freq 

# ------------ make plots ------------ 
print(' ... make plot ... ') 
if plot_psd_all   :
    cpd   = 2*freqr/M2
    cpdc  = 2*freqrc/M2
    cpdM2 = 2. # [sec] 
    cpdS2 = 2*S2/M2
    cpdO1 = 2*O1/M2
    cpdK1 = 2*K1/M2
    cpdfi = 2*fi/M2
    # --- actually make plot --- 
    print(' --- make plot ---')
    line, column = 0,0
    plt.figure(figsize=(20,20))
    gs = gridspec.GridSpec(3,3,hspace=0.3,wspace=0.3)
    for mooring in range(stations):
        ax = plt.subplot(gs[line,column])
        print(line,column)
        ax.text(0.8,30,title_mooring[mooring])
        plt.loglog(abs(cpd[mooring,:]),abs(psdr[mooring,:]),'m',lw=1,label='Mooring')
        plt.loglog(abs(cpdc[mooring,:]),abs(psdrc[mooring,:]),c=cf0,lw=1,alpha=0.6,label='CROCO, '+label_w )
        if mooring==0:
            plt.legend()

        plt.xlabel('Cycle per day',fontsize=fs)
        plt.ylabel('PSD [(m s$^{-1}$)$^2$ Hz$^{-1}$]',fontsize=fs)
        
        # - primary frequencies - 
        ymin,ymax = 1e-7,10
        plt.plot([cpdM2,cpdM2],[ymin,ymax],'k--',lw=0.8)
        plt.plot([cpdS2,cpdS2],[ymin,ymax],'k--',lw=0.8)
        plt.plot([cpdfi,cpdfi],[ymin,ymax],'k--',lw=0.8)
        plt.plot([cpdO1,cpdO1],[ymin,ymax],'k--',lw=0.8)
        plt.plot([cpdK1,cpdK1],[ymin,ymax],'k--',lw=0.8)

        plt.text(cpdfi,ymax*1.1,'$f$',ha='center',va='bottom',fontsize=0.8*fs)
        plt.text(0.5*(cpdO1+cpdK1),ymax*1.1,'O$_1$,K$_1$',ha='center',va='bottom',fontsize=0.8*fs)
        plt.text(0.5*(cpdM2+cpdS2),ymax*1.1,'M$_2$,S$_2$',ha='left',va='bottom',rotation=0,fontsize=0.8*fs)

        # - grid plot -
        plt.axvline(x=0.1,c='k',lw=0.3,alpha=0.5)
        plt.axvline(x=1,c='k',lw=0.3,alpha=0.5)
        plt.axvline(x=10,c='k',lw=0.3,alpha=0.5)
        plt.axhline(y=1e-5,c='k',lw=0.3,alpha=0.5)
        plt.axhline(y=1e-3,c='k',lw=0.3,alpha=0.5)
        plt.axhline(y=0.1,c='k',lw=0.3,alpha=0.5)


        plt.ylim(ymin,ymax)
        ax.tick_params(labelsize=fs)

        if column==2:
            column=0
            line+=1
        else: 
            column+=1
             

    plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/WMT/PSD_mooring_croco.png',dpi=200,bbox_inches='tight')


if plot_psd_std == True:
    # ------------ parameters ------------ 
    name_exp    = 'rrexnum200'
    name_pathdata = 'RREXNUMSB200_RSUP5_NOFILT_T'
    name_pathdata_nosmooth = 'RREXNUM200_RSUP5_NOFILT_T'
    nbr_levels  = '200'
    name_exp_grd= 'rrex200-up5'
    nbr_levels  = '200'
    ndfiles     = 1  # number of days per netcdf
    time= ['20']
    var_list = ['zeta']

    # ---> data 
    file_w          = '/home/datawork-lops-rrex/nschifan/Data_in_situ_Rene/rrexnumsb200_bottom_w.nc'
    file_data       = '/home/datawork-lops-rrex/nschifan/Data_in_situ_Rene/moorings_w_std.nc'

    # --- plot options ---
    lw      = 1.5     # linewidth
    ms      = 20      # markersize 
    lw_c    = 0.2   # linewidth coast 
    cf0           = colors.to_rgba('teal')
    cmap_data     = plt.cm.magma
    cf_data       = cmap_data(np.linspace(0.2,0.85,len(choice_mooring))) #plt.cm.get_cmap(cmap_data,len(choice_mooring))
    gradh_text    = [0.02,0.03,0.04,0.05,0.06,0.07,0.08]

    
    # --- STD w ---
    cmap_w     =  plt.cm.nipy_spectral 
    pmin,pmax,pint   = 0,200,1
    levels_w      = np.arange(pmin,pmax+pint,pint)
    norm_w        = colors.Normalize(vmin=pmin,vmax=pmax)
    cbticks_w     =  [0,50,100,150,pmax]
    cblabel_w     = r'STD(w($s_0$)) [m.day$^{-1}$]'

    # - bin for slope ------
    minc      = 0
    maxc      = 0.122
    nce       = 0.002      
    slope_e    = np.arange(minc,maxc,nce) 
    slope_c    = 0.5*(slope_e[1:]+slope_e[:-1])  # bin center 

    # ------------ norm gradh ----
    cmap_gradh    = plt.cm.copper_r
    norm_gradh    = colors.Normalize(vmin=0,vmax=0.12)
    levels_gradh  = np.arange(0,0.121,0.01)
    cbticks_gradh = [0,0.06,0.12]
    cblabel_gradh = r'slope'
    levels_hplot  = np.arange(0,4500,500)

    # ------------ read deepest std mooring ------------ 
    stations = 7
    std_data = np.zeros(stations)
    var_name = ['std0','std1','std2','std3','std4','std5','std6']
    nc       = Dataset(file_data,'r')
    for u in range(stations):
         std_data[u]  = nc.variables[var_name[u]][-1]
    nc.close()

    # ------------ read bottom w ------------ 
    nc  = Dataset(file_w,'r')
    w   = nc.variables['w'][:,:,:]
    nc.close()
    # --> compute statistics w 
    std_w  = np.std(w,axis=-1)

    # ------------ read grid smooth ------------ 
    data = Croco_longrun(name_exp,nbr_levels,time[0],name_exp_grd,name_pathdata) 
    data.get_grid()
    data.get_outputs(0,var_list)
    data.get_zlevs()
    # --> compute slopes 
    dhdx   = tools.u2rho(data.h[1:,:]-data.h[:-1,:])*data.pm
    dhdy   = tools.v2rho(data.h[:,1:]-data.h[:,:-1])*data.pn
    gradh  = np.sqrt(dhdx**2 + dhdy**2)
    latr = data.latr
    lonr = data.lonr
    h    = data.h

    # ------------ read data no smooth ------------ 
    data_ns = Croco_longrun(name_exp,nbr_levels,time[0],name_exp_grd,name_pathdata_nosmooth)
    data_ns.get_grid()
    # --> compute slopes  
    dhdx_ns   = tools.u2rho(data.h[1:,:]-data.h[:-1,:])*data.pm
    dhdy_ns   = tools.v2rho(data.h[:,1:]-data.h[:,:-1])*data.pn
    gradh_ns  = np.sqrt(dhdx**2 + dhdy**2)


    # --> bin std(w) in slope space
    std_bin_md_slope      =  stats.binned_statistic(np.ravel(gradh[np.isnan(std_w)==0]),np.ravel(std_w[np.isnan(std_w)==0]),'median',bins=slope_e)[0]
    std_bin_10p_slope     =  stats.binned_statistic(np.ravel(gradh),np.ravel(std_w),statistic=lambda y: np.nanpercentile(y, 10),bins=slope_e)[0]
    std_bin_90p_slope     =  stats.binned_statistic(np.ravel(gradh),np.ravel(std_w),statistic=lambda y: np.nanpercentile(y, 90),bins=slope_e)[0]
    std_bin_min_slope     =  stats.binned_statistic(np.ravel(gradh[np.isnan(std_w)==0]),np.ravel(std_w[np.isnan(std_w)==0]),statistic=np.nanmin,bins=slope_e)[0]
    std_bin_max_slope     =  stats.binned_statistic(np.ravel(gradh[np.isnan(std_w)==0]),np.ravel(std_w[np.isnan(std_w)==0]),statistic=np.nanmax,bins=slope_e)[0]

    # --------------> make plot <----------------- #
    print( '--- MAKE PLOT ---')
    plt.figure(figsize=(20,20)) 
    gs = gridspec.GridSpec(3,2,height_ratios=[1,0.05,1],wspace=0.15)
                                                                  
    ax = plt.subplot(gs[0,0])  # ---> sigma slope
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('a)')
    ctf00 = ax.contourf(gradh_ns.T,levels=levels_gradh,cmap=cmap_gradh,norm=norm_gradh)
    ax.contour(h.T,levels=levels_hplot,colors='k',linewidths=lw_c)
    ax.set_xticks([0,250,500,750,1000],['0','200','400','600','800'])
    ax.set_yticks([0,250,500,750],['0','200','400','600'])
    plt.xlabel(r'km in $\xi$-direction',fontsize=fs)
    plt.ylabel(r'km in $\eta$-direction',fontsize=fs)

    ax.tick_params(labelsize=fs)

    # ------------------ colorbar slope 
    cax = plt.subplot(gs[1,0])
    cb  = plt.colorbar(ctf00,cax,orientation='horizontal',ticks=cbticks_gradh)
    cb.set_label(cblabel_gradh,fontsize=fs,labelpad=-80) 
    cb.ax.tick_params(labelsize=fs)


    ax = plt.subplot(gs[0,1]) # ------------------------------------- std(w)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('b)')
    ctf01 = ax.pcolormesh(std_w.T,cmap=cmap_w,zorder=1,norm=norm_w,)
    bathy = ax.contour(h.T,levels =[1000,2000,3000,4000] , colors='gray',linewidths= lw_c)
    ax.set_xticks([0,250,500,750,1000],['0','200','400','600','800'])
    ax.set_yticks([0,250,500,750])
    plt.xlabel(r'km in $\xi$-direction',fontsize=fs)
    ax.set_yticklabels([])
    ax.tick_params(labelsize=fs)

    cax = plt.subplot(gs[1,1])  # ----------------------------------------------------------------------------- colorbar std(w)
    cb     = plt.colorbar(ctf01,cax,orientation='horizontal',extend='both',ticks=cbticks_w)
    cb.set_label(cblabel_w,fontsize=fs,labelpad=-80)
    cb.ax.tick_params(labelsize=fs)

    ax = plt.subplot(gs[2,0])   # -----------------------------------------------------------------------------> std(w) = f(slope) 
    plt.plot(std_bin_md_slope,slope_c,color=cf0,linewidth=lw,label=r'Median')
    plt.plot(std_bin_10p_slope,slope_c,color=cf0,linestyle='dashed',linewidth=lw,label=r'$10^{th}$,$90^{th}$ percentiles')
    plt.plot(std_bin_90p_slope,slope_c,color=cf0,linestyle='dashed',linewidth=lw)
    plt.legend() 
    print('----> DATA')
    print(std_data,gradh_ns[idx_lon,idx_lat])
    for u in range(stations):
        if u==0:
            plt.scatter(std_data[u],gradh_ns[idx_lon[u],idx_lat[u]],s=15,marker='D',c='k',label='Deepest measure from moorings')
            texta = choice_mooring[u]
            ax.text(4,gradh_text[u],texta,fontsize=fs,color='k') 
        else:
            plt.scatter(std_data[u],gradh_ns[idx_lon[u],idx_lat[u]],s=15,marker='D',c=cf_data[u-1])
            texta = choice_mooring[u]
            ax.text(4,gradh_text[u],texta,fontsize=fs,color=cf_data[u-1])
    for k in range(1,10):
        plt.axvline(x=k,color='k',linewidth=0.5,alpha=0.2)
    for k in range(10,100,10):
        plt.axvline(x=k,color='k',linewidth=0.5,alpha=0.2)
    for k in range(100,1000,100):
        plt.axvline(x=k,color='k',linewidth=0.5,alpha=0.2)
    plt.axvline(x=10,color='k',linewidth=0.5,alpha=0.5)
    plt.axvline(x=100,color='k',linewidth=0.5,alpha=0.5)
    plt.xscale('log')
    plt.title('c) ',fontsize=fs)
    plt.xlabel(r'STD(w($s_0$)) [m $day^{-1}$]',fontsize=fs)
    plt.ylabel('Slope [%]',fontsize=fs)
    plt.xlim(2,1000)
    ax.set_xticks([10,100,1000])
    ax.set_yticks([0,0.02,0.04,0.06,0.08,0.1,0.12])
    ax.set_yticklabels(['0','2','4','6','8','10','12'],fontsize=fs)
    ax.tick_params(labelsize=fs)

    ax = plt.subplot(gs[2,1])
    mooring = 3
    cpd   = 2*freqr/M2
    cpdc  = 2*freqrc/M2
    cpdM2 = 2. # [sec] 
    cpdS2 = 2*S2/M2
    cpdO1 = 2*O1/M2
    cpdK1 = 2*K1/M2
    cpdfi = 2*fi/M2
    ax.text(0.8,30,title_mooring[mooring])
    plt.loglog(abs(cpd[mooring,:]),abs(psdr[mooring,:]),'m',lw=1,label='Mooring')
    plt.loglog(abs(cpdc[mooring,:]),abs(psdrc[mooring,:]),c=cf0,lw=1,alpha=0.6,label='CROCO, '+label_w )
    
    plt.legend()

    plt.xlabel('Cycle per day',fontsize=fs)
    plt.ylabel('PSD [(m s$^{-1}$)$^2$ Hz$^{-1}$]',fontsize=fs)

    # - primary frequencies - 
    
    ymin,ymax = 1e-7,10
    plt.plot([cpdM2,cpdM2],[ymin,ymax],'k--',lw=0.8)
    plt.plot([cpdS2,cpdS2],[ymin,ymax],'k--',lw=0.8)
    plt.plot([cpdfi,cpdfi],[ymin,ymax],'k--',lw=0.8)
    plt.plot([cpdO1,cpdO1],[ymin,ymax],'k--',lw=0.8)
    plt.plot([cpdK1,cpdK1],[ymin,ymax],'k--',lw=0.8)

    plt.text(cpdfi,ymax*1.1,'$f$',ha='center',va='bottom',fontsize=0.8*fs)
    plt.text(0.5*(cpdO1+cpdK1),ymax*1.1,'O$_1$,K$_1$',ha='center',va='bottom',fontsize=0.8*fs)
    plt.text(0.5*(cpdM2+cpdS2),ymax*1.1,'M$_2$,S$_2$',ha='left',va='bottom',rotation=0,fontsize=0.8*fs)

    # - grid plot -
    plt.axvline(x=0.1,c='k',lw=0.3,alpha=0.5)
    plt.axvline(x=1,c='k',lw=0.3,alpha=0.5)
    plt.axvline(x=10,c='k',lw=0.3,alpha=0.5)
    plt.axhline(y=1e-5,c='k',lw=0.3,alpha=0.5)
    plt.axhline(y=1e-3,c='k',lw=0.3,alpha=0.5)
    plt.axhline(y=0.1,c='k',lw=0.3,alpha=0.5)


    plt.ylim(ymin,ymax)
    #plt.xlim(1e-7,1e-4)
    ax.tick_params(labelsize=fs)
    

    plt.savefig('figure5.png',dpi=180,bbox_inches='tight')
    plt.close()






