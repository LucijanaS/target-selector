import numpy as np

class Layout():

    def raddeg(self, d, m, s):
        return (d + m/60 + s/3600) * np.pi/180

    def radhr(self, hr, m, s):
        return (hr + m/60 + s/3600) * np.pi/12 

    def set_radec(self,ra,dec):
        self.ra = self.radhr(ra[0],ra[1],ra[2])
        self.dec = self.raddeg(dec[0],dec[1],dec[2])

    def set_latlon(self,lat,lon):
        self.lat = self.raddeg(lat[0],lat[1],lat[2])
        self.lon = self.raddeg(lon[0],lon[1],lon[2])

    def rotx(self,x,y,z,a):
        cs,sn = np.cos(a), np.sin(a)
        return x, cs*y - sn*z, sn*y + cs*z

    def roty(self,x,y,z,a):
        cs,sn = np.cos(a), np.sin(a)
        return cs*x + sn*z, y, -sn*x + cs*z

    def get_uvw(self,jday,dx,dy,dz):
        gsid = 18.697374558 + 24.06570982441908*(jday - 2451545)   
        sid = (gsid % 24)*np.pi/12 + self.lon  # 1 hr has 15 deg of rotation
        ha = sid - self.ra  # hour angle
        dx,dy,dz = self.rotx(dx,dy,dz,-self.lat)
        dx,dy,dz = self.roty(dx,dy,dz,ha)
        u,v,w = self.rotx(dx,dy,dz,self.dec)
        return u,v,w

    def get_xyz(self,jday,u,v,w):
        gsid = 18.697374558 + 24.06570982441908*(jday - 2451545)   
        sid = (gsid % 24)*np.pi/12 + self.lon  # 1 hr has 15 deg of rotation
        ha = sid - self.ra  # hour angle
        dx,dy,dz = self.rotx(u,v,w,-self.dec)
        dx,dy,dz = self.roty(dx,dy,dz,-ha)
        dx,dy,dz = self.rotx(dx,dy,dz,self.lat)
        return dx,dy,dz


        
