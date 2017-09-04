import visa
"""from traits.api import SingletonHasTraits, Range
from traitsui.api import View, Item, HGroup, VGroup
"""

class DtgOptical(SingletonHasTraits):
    """Provides control of DTG for optical pulse measurements via visa."""
    
    def __init__(self, visa_address='GPIB0::1'):
    	self.visa_address = visa_address
        self.instr = visa.instrument(self.visa_address)
        self.instr.delay = 0.5
        
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
        self._write('PGENA:CH1:OUTP ON') # open the output of physical channel 1 in mainframe
        self._write('OUTP:DC:STAT ON') # turn on dc output
        self._write('TBAS:RUN ON')  # open output from software side
        
    def off(self):
        self._write('TBAS:RUN OFF')    
        self._write('PGENA:CH1:OUTP OFF')
        
    def setup_burst_mode(self):
        self._write('TBAS:OMOD PULS')    
        self._write('TBAS:MODE BURS')
        self._write('TBAS:TIN:SOUR EXT')
        self._write('TBAS:TIN:IMP 50')
    
    def set_dc_level(self,dc_level):
        self._write('OUTP:DC:LEV 0,%f' % float(dc_level))
    
    def set_burst_count(self, burst_count):
        self._write('TBAS:COUN %i' % int(burst_count))
        
    def set_trigger_level(self, trigger_level):
        self._write('TBAS:TIN:LEV %f' % float(trigger_level))
        
    def set_pulse_length(self, pulse_length):
        self._write('TBAS:FREQ %i MHZ' % int(500 / pulse_length))
    
    def set_frequency(self, frequency):
        self._write('TBAS:FREQ %i MHZ' % int(frequency))
        
    def set_duty_cycle(self, duty_cycle):
        self._write('PGENA:CH1:DCYC %f' % float(duty_cycle))
    
    def set_time_delay(self, time_delay):
        self._write('PGENA:CH1:LDEL %i ns' % int(time_delay))
    
    def set_high_level(self, high_level):
        self._write('PGENA:CH1:HIGH %f' % float(high_level))
    
    def set_low_level(self, low_level):
        self._write('PGENA:CH1:LOW %f' % float(low_level))
        
    def set_parameters(self, frequency, duty_cycle,time_delay, high_level, low_level, burst_count=1, trigger_level=1.1,dc_level=1.0):
        self.setup_burst_mode()
        self.set_burst_count(burst_count)
        self.set_trigger_level(trigger_level)
        self.set_frequency(frequency)
        self.set_duty_cycle(duty_cycle)
        self.set_time_delay(time_delay)
        self.set_high_level(high_level)
        self.set_low_level(low_level)
        self.set_dc_level(dc_level)
    
"""    view = View(VGroup(HGroup(Item('burst_count'),
                              Item('trigger_level'), 
                              ),
                       HGroup(Item('frequency'), 
                              Item('duty_cycle'), 
                              Item('time_delay'),
                              ),
                       HGroup(Item('high_level'), 
                              Item('low_level'),
                              ),
                       ),
                title='DTG Optical', resizable=True)
"""        
if __name__ == '__main__':
     dtg = DTGOPTICAL()
     
     
