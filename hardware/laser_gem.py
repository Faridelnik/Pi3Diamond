
from traits.api import SingletonHasTraits, Trait, Instance, Property, String, Range, Float, Int, Bool, Enum, Button, on_trait_change, cached_property, Code, List, NO_COMPARE
from traitsui.api import View, Item, HGroup, VGroup, Tabbed, EnumEditor, TextEditor, Group
from enable.api import Component, ComponentEditor
from traitsui.menu import Action, Menu, MenuBar
import time
import visa
from tools.emod import ManagedJob
from tools.utility import GetSetItemsHandler, GetSetItemsMixin

class LaserGem(ManagedJob, GetSetItemsMixin):
    """Provides control of Gem laser with COM via visa."""
    _slew_rate_min = 0.01
    
    laser = Enum('off', 'on', desc='switching laser on and off', label='Laser', auto_set=False, enter_set=True)
    switch = Enum('power', 'current', desc='control by power or current', label='power/current')
    power = Range(low=0., high=500., value=0., desc='power of laser [mW]', label='power [mW]', mode='text', auto_set=False, enter_set=True)
    current = Range(low=0.0, high=100.0, value=0., desc='Current of red 0-100%', label='Current [%]', auto_set=False, enter_set=True)

    laser_monitor = Enum('off', 'on', desc='Gem laser status', label='laser Status')
    power_monitor = Range(low=0.0, high=500.0, value=0.0, desc='power of Gem [mW]', label='Gem Power [mW]')
    current_monitor = Range(low=0.0, high=100.0, value=0.0, desc='current [%]', label='current [%]')
    laser_temperature = Range(low=0.0, high=100.0, value=0.0, desc='laser temperature', label='laser temperature')
    PSU_temperature = Range(low=0.0, high=100.0, value=0.0, desc='PSU temperature', label='PSU temperature')
    
    def __init__(self, visa_address='COM5'):
    	super(LaserGem, self).__init__()
        self.visa_address = visa_address
        self.instr = visa.instrument(self.visa_address)
        #self.instr.term_chars = '\r'
        self.instr.delay = 0.5
        self.on_trait_change(self._open_laser, 'laser', dispatch='ui')
        self.on_trait_change(self._set_laser, 'current,power', dispatch='ui')
        
    def _write(self, string):
        try: # if the connection is already open, this will work
            self.instr.write(string)
        except: # else we attempt to open the connection and try again
            try: # silently ignore possible exceptions raised by del
                del self.instr
            except Exception:
                pass
            self.instr = visa.instrument(self.visa_address)
            self.instr.delay = 0.5
            self.instr.write(string)
        
    def _ask(self, str):
        try:
            val = self.instr.ask(str)
        except:
            self.instr = visa.instrument(self.visa_address)
            self.instr.delay = 0.5
            val = self.instr.ask(str)
        return val
        
    def on(self):
        self._write('ON\r')
        self.instr.read()
        
    def off(self):
        self._write('OFF\r')  
        self.instr.read()  
        
    def check_status(self):
        temp = self._ask('STAT?\r')
        
        if temp == 'ENABLED' or temp == '\nENABLED':
            return 'on'
        else:
            return 'off'
    
    def get_current(self):
        """in unit of %"""
        
        return float(self._ask('CURRENT?\r')[:-1])
        
    def get_power(self):
        """in unit of mW"""
        return float(self._ask('POWER?\r')[:-2])
    
    def set_current(self, current):
        self._write('CURRENT=%f\r' % float(current))
        self.instr.read()
        
    def set_power(self, power):
        self._write('POWER=%f\r' % float(power))
        self.instr.read()
        
    def get_laser_temperature(self):
        """in unit of C"""
        return float(self._ask('LASTEMP?\r')[:-1])
        
    def get_PSU_temperature(self):
        """in unit of C"""
        return float(self._ask('PSUTEMP?\r')[:-1])
 
    @on_trait_change('laser')
    def _open_laser(self):
        if self.laser == 'on':
            self.on()
            time.sleep(5.0)
            self.laser_monitor = self.check_status()
        elif self.laser == 'off':
            self.off()
            time.sleep(1.0)
            self.laser_monitor = self.check_status()

    @on_trait_change('current,power')
    def _set_laser(self):
        if self.switch == 'power':
            self.set_power(self.power)
        else:
            self.set_current(self.current)
        time.sleep(5.0)
        self.power_monitor = self.get_power() 
        self.current_monitor = self.get_current() 
        self.laser_temperature = self.get_laser_temperature()
        self.PSU_temperature = self.get_PSU_temperature() 
    
       
    traits_view = View(VGroup(HGroup(Item('laser', style='custom'),
                                      Item('switch', style='custom'),
                                      Item('laser_monitor', width= -20, style='readonly'),
                                      ),
                               HGroup(Item('current', width= -40, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                                       Item('current_monitor', width= -40, style='readonly'),
                                       Item('power', width= -40, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                                       Item('power_monitor', width= -40, style='readonly'),
                                       ),
                               HGroup(Item('laser_temperature', width= -40, style='readonly'),
                                      Item('PSU_temperature', width= -40, style='readonly'),
                                      ),
                              ),
                      title='LaserGem', height=120, width=700, buttons=[], resizable=True)
    

 
if __name__ == '__main__':
     gem = LaserGem()
   
