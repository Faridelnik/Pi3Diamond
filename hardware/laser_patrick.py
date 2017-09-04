import numpy as np
import visa

from nidaq import AnalogOutSyncCount 

class LaserRed():
    """Provides control of velocity laser with GPIB via visa."""
    _slew_rate_min = 0.01
    
    def __init__(self, visa_address='GPIB0::2', ao_chan='/Dev2/ao0', co_dev='/Dev2/Ctr1', ci_dev='/Dev2/Ctr2', ci_port='/Dev2/PFI0'):
    	self.visa_address = visa_address
        self.ni_task = AnalogOutSyncCount(ao_chan, co_dev, ci_dev, ci_port, ao_range=(-10.0, 10.0), duty_cycle=0.9)
        self.set_detuning_voltage(0.0)
        self.set_piezo_voltage(50.0)
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
        self._write(':OUTP ON')
        
    def off(self):
        self._write(':OUTP OFF')    
        # self._write(':CURR %f' % self._output_threshold)
    def check_status(self):
        temp = self._ask(':OUTP?')
        if temp == '1':
            return 'on'
        else:
            return 'off'
    
    def get_current(self):
        return float(self._ask(':SENS:CURR:DIOD'))
        
    def get_power(self):
        return float(self._ask(':SENS:POW:FRON'))
    
    def set_current(self, current):
        self._write(':CURR %f' % float(current))
        
    def get_wavelength(self):
        return float(self._ask(':SENS:WAVE'))

    def set_wavelength(self, wavelength):
        self._write(':WAVE %e' % wavelength)
        self._write(':OUTP:TRAC OFF')

    def set_output(self, current, wavelength):
        self.set_current(current)
        self.set_wavelength(wavelength)
    
    def scan(self, start_wavelength, stop_wavelength, slew_rate):
        self._write(':WAVE:SLEW:FORW %e' % max(slew_rate, self._slew_rate_min))
        self._write(':WAVE:START %e' % start_wavelength)
        self._write(':WAVE:STOP %e' % stop_wavelength)
        self._write(':OUTP:SCAN:START')

    def stop_scan(self):
        self._write(':OUTP:SCAN:RESET')
      #  self._write(':OUTP:TRAC OFF')

    def pause_scan(self):
        self._write(':OUTP:SCAN:STOP')
      #  self._write(':OUTP:TRAC OFF')
        
    def resume_scan(self):
        self._write(':OUTP:SCAN:START')

    def piezo_scan(self, detuning, seconds_per_point):
        voltage = self._detuning_to_voltage(np.array(detuning, dtype=float))
        #if voltage[0]>-2.99:
        #    self.ni_task.line(np.arange(-3.0, voltage[0], self._detuning_to_voltage(0.1)),0.001)
        return self.ni_task.line(voltage, seconds_per_point)
    
    def set_piezo_voltage(self, voltage):
        self._write(':VOLT %f' % float(voltage))
        
    def set_detuning(self, detuning):
        voltage = self._detuning_to_voltage(detuning)
        current_voltage = self.get_piezo_voltage()
        #if current_voltage != -3.0:
        #    self.ni_task.line(np.arange(current_voltage, -3.0, self._detuning_to_voltage(-0.5)),0.001)
        """self.ni_task.point(-3.0)"""
        #if detuning>-45.0:
        #    self.ni_task.line(np.arange(-3.0, voltage, self._detuning_to_voltage(0.1)),0.001)
        self.ni_task.point(voltage)

    def set_detuning_voltage(self, detuning_voltage):
        self.ni_task.point(detuning_voltage)
        
    def _detuning_to_voltage(self, detuning):
        """Convert detuning into voltage."""
        return detuning 
        
    def get_piezo_voltage(self):
        return (50.0 - float(self._ask(':SENS:VOLT:PIEZ'))) * 0.06
    
    def get_detuning(self):
        return (50.0 - float(self._ask(':SENS:VOLT:PIEZ'))) * 0.90
    
    # def getCavityTemperature(self):
        # return float(self._ask(':SENS:TEMP:LEV:CAV'))
        
    # def getDiodeTemperature(self):
        # return float(self._ask(':SENS:TEMP:LEV:DIOD'))
        
 
if __name__ == '__main__':
     l = LaserRed()
     #print np.array([np.any(l.piezo_scan(np.arange(0.,50.,0.1),1e-3)) for i in range(10)]).mean()
     #freq = 7. # frequency [Hz]
     #ramp = np.arange(-45.,45.,.1)
     #spp = 1./(2*len(ramp)*freq)
     #trace = np.hstack(100*(ramp,ramp[::-1]))
     #print l.piezo_scan(trace,spp)
     
