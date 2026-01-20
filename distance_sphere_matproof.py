import numpy as np
def dist_sphere(lat0,lon0,lat1,lon1):
    # --- http://www.movable-type.co.uk/scripts/latlong.html --- 
    lat0 = lat0*np.pi/180.
    lat1 = lat1*np.pi/180.
    lon0 = lon0*np.pi/180.
    lon1 = lon1*np.pi/180.
    aa  = np.sin(0.5*(lat1-lat0))**2 + np.cos(lat1)*np.cos(lat0)*np.sin(0.5*(lon1-lon0))**2
    rad = 6371000. # earth's radius
    dd  = 2*np.arctan2(aa**0.5,(1-aa)**0.5)*rad
    return dd

