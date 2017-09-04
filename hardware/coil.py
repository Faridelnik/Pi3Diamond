import time
import visa
import numpy

rm=visa.ResourceManager()

from enthought.traits.api import HasTraits, SingletonHasTraits, Property, Float, Bool, Instance, DelegatesTo, Tuple

channel_map = {'x': 'gpib0::15', 'y': 'gpib0::16', 'z': 'gpib0::17'}
#v_max_map = {'x': 20.0, 'y': 40.0, 'z': 12.0}
#i_max_map = {'x': 4.0, 'y': 4.0, 'z': 1.0}

gauss_per_amp = {'x': 47./4.0, 'y': 58./4.0, 'z': 36.0/1.0}

# rough calibration.
#  - field measured about 1cm above focus
#  - objective and sample holder are present
# x (pair with smaller radius): 47G / 4 A, field direction for positive current: from table edge to inside of table
# y (pair with smaller radius): 58G / 4 A, field direction for positive current: from inside of table to table edge
# z (single coil): 30G / A, (36G/A on the axis at the coil surface) field direction for positive current: into table surface

class Coil( HasTraits ):
    
    limit_current = Property( trait=Float )
    limit_voltage = Property( trait=Float )
    current = Property( trait=Float )
    field = Property( trait=Float )
    current = Property( trait=Float )
    voltage = Property( trait=Float )
        
    capacitor = Property( trait=Bool )

    def __init__(self, axis):
        HasTraits.__init__(self)
        self.axis=axis
        #self.visa = visa.instrument(channel_map[axis])
        self.visa = rm.get_instrument(channel_map[axis])
        
    def _get_limit_current(self):
        return float(self.visa.ask('IMAX?'))

    def _set_limit_current(self, val):
        self.visa.write('IMAX %.4f' % float(val))

    def _get_limit_voltage(self):
        return float(self.visa.ask('VMAX?'))

    def _set_limit_voltage(self, val):
        self.visa.write('VMAX %.4f' % float(val))

    def _get_voltage(self):
        return float(self.visa.ask('VOUT?'))*(-1)**int(self.visa.ask('INV?'))

    def _set_voltage(self, val):
        self.visa.write('VSET %.4f' % float(val))
        

    def _get_current(self):
        return float(self.visa.ask('IOUT?'))*(-1)**int(self.visa.ask('INV?'))
        
    def _set_current(self, val):
        inverted = bool( int(self.visa.ask('INV?')) )
        if val == 0: # shut down coils
            self.visa.write('ISET 0')
            self.visa.write('OUT 0')
            self.visa.write('INV 0')
        else:
            #sign = val>0
            sign = val<0
            if sign != inverted: # need to shut down and set invert first, was ==
                self.visa.write('ISET 0')
                self.visa.write('OUT 0')
                self.visa.write('INV %i'%sign)
            self.visa.write('OUT 1') # turn on (if it was not already turned on)
            self.visa.write('ISET %.4f' % abs(float(val)))
#        time.sleep(0.5) # wait a little until current is stabilized

    def _get_capacitor(self):
        return bool(self.visa.ask('CAP?'))
    
    def _set_capacitor(self, val):
        if val == 0:
            self.visa.write('CAP 0')
        elif val == 1:
            self.visa.write('CAP 1')

    def _set_field(self, val):
        self.current = val / gauss_per_amp[self.axis]

    def _get_field(self):
        return self.current * gauss_per_amp[self.axis]
    
    def get_error(self):
        return bool(self.visa.ask('ACK?'))
    
    def reset_error(self):
        self.visa.write('ACK')
        
    def get_inv(self):
        return bool(self.visa.ask('INV?'))
    
    def set_inv(self,val):
        if val == 1:
            self.visa.write('INV 1')
        elif val == 0:
            self.visa.write('INV 0')
        

class Coils( SingletonHasTraits ):
    
    x_coil = Instance( Coil )
    y_coil = Instance( Coil )
    z_coil = Instance( Coil )
    
    def _x_coil_default(self):
        return Coil('x')
    
    def _y_coil_default(self):
        return Coil('y')
    
    def _z_coil_default(self):
        return Coil('z')
    
    x_current = DelegatesTo(delegate='x_coil', prefix='current')
    y_current = DelegatesTo(delegate='y_coil', prefix='current')
    z_current = DelegatesTo(delegate='z_coil', prefix='current')
    
    polar = Property( trait=Tuple )
    cartesian = Property( trait=Tuple )
    
    def _set_polar(self, tup):
        r = tup[0]
        t = tup[1]*numpy.pi/180.
        p = tup[2]*numpy.pi/180.
        self.x_coil.field = numpy.round(r * numpy.sin(t) * numpy.cos(p),6)
        self.y_coil.field = numpy.round(r * numpy.sin(t) * numpy.sin(p),6)
        self.z_coil.field = numpy.round(r * numpy.cos(t),6)

    def _get_polar(self):
        x = self.x_coil.field
        y = self.y_coil.field
        z = self.z_coil.field
        r = (x**2+y**2+z**2)**0.5
        t = numpy.arccos(z/r)*180./numpy.pi
        p = numpy.arctan2(y,x)*180./numpy.pi
        return r, t, p
    
    def _set_cartesian(self, tup):
        self.x_coil.field = tup[0]
        self.y_coil.field = tup[1]
        self.z_coil.field = tup[2]

    def _get_cartesian(self):
        return self.x_coil.field, self.y_coil.field, self.z_coil.field
    
