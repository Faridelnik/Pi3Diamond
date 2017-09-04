"""
This file is part of pi3diamond.

pi3diamond is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pi3diamond is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with diamond. If not, see <http://www.gnu.org/licenses/>.

Copyright (C) 2009-2011 Helmut Fedder <helmut.fedder@gmail.com>
"""

import visa
import numpy

class SMIQ_RF():
    """Provides control of SMIQ family RF sources from Rhode und Schwarz with GPIB via visa."""
    _output_threshold = -90.0
    
    def __init__(self, visa_address='GPIB0::28'):
        self.visa_address = visa_address
        
    def _write(self, string):
        try: # if the connection is already open, this will work
            self.instr.write(string)
        except: # else we attempt to open the connection and try again
            try: # silently ignore possible exceptions raised by del
                del self.instr
            except Exception:
                pass
            self.instr = visa.instrument(self.visa_address)
            self.instr.write(string)
        
    def _ask(self, str):
        try:
            val = self.instr.ask(str)
        except:
            self.instr = visa.instrument(self.visa_address)
            val = self.instr.ask(str)
        return val

    def getPower(self):
        return float(self._ask(':POW?'))

    def setPower(self, power):
        if power is None or power < self._output_threshold:
            self._write(':OUTP OFF')
            return
        self._write(':FREQ:MODE CW')
        self._write(':POW %f' % float(power))
        self._write(':OUTP ON')

    def getFrequency(self):
        return float(self._ask(':FREQ?'))

    def setFrequency(self, frequency):
        self._write(':FREQ:MODE CW')
        self._write(':FREQ %e' % frequency)

    def setOutput(self, power, frequency):
        self.setPower(power)
        self.setFrequency(frequency)

    def initSweep(self, frequency, power):
        if len(frequency) != len(power):
            raise ValueError('Length mismatch between list of frequencies and list of powers.')
        self._write(':FREQ:MODE CW')
        self._write(':LIST:DEL:ALL')
        self._write('*WAI')
        self._write(":LIST:SEL 'ODMR'")
        FreqString = ''
        for f in frequency[:-1]:
            FreqString += ' %f,' % f
        FreqString += ' %f' % frequency[-1]
        self._write(':LIST:FREQ' + FreqString)
        self._write('*WAI')
        PowerString = ''
        for p in power[:-1]:
            PowerString += ' %f,' % p
        PowerString += ' %f' % power[-1]
        self._write(':LIST:POW' + PowerString)
        self._write(':LIST:LEAR')
        self._write(':TRIG1:LIST:SOUR EXT')
        # we switch frequency on negative edge. Thus, the first square pulse of the train
        # is first used for gated count and then the frequency is increased. In this way
        # the first frequency in the list will correspond exactly to the first acquired count. 
        self._write(':TRIG1:SLOP NEG') 
        self._write(':LIST:MODE STEP')
        self._write(':FREQ:MODE LIST')
        self._write('*WAI')
        N = int(numpy.round(float(self._ask(':LIST:FREQ:POIN?'))))
        if N != len(frequency):
            raise RuntimeError, 'Error in SMIQ with List Mode'

    def resetListPos(self):
        self._write(':ABOR:LIST')
        self._write('*WAI')

class HP33120A():
    """Provides control of HP33120A with GPIB via visa."""
    _output_threshold = -22.0
    
    def __init__(self, visa_address='GPIB0::11'):
        self.visa_address = visa_address
        self._write('FUNC:SHAP SIN')
        self._write('TRIG:SOUR EXT')
        self._write('BM:SOUR EXT')
        self._write('BM:STAT ON')
        
    def _write(self, string):
        try: # if the connection is already open, this will work
            self.instr.write(string)
        except: # else we attempt to open the connection and try again
            try: # silently ignore possible exceptions raised by del
                del self.instr
            except Exception:
                pass
            self.instr = visa.instrument(self.visa_address)
            self.instr.write(string)
        
    def _ask(self, str):
        try:
            val = self.instr.ask(str)
        except:
            self.instr = visa.instrument(self.visa_address)
            val = self.instr.ask(str)
        return val

    def setMode(self):
        self._write('TRIG:SOUR EXT')
        self._write('BM:SOUR EXT')
        self._write('BM:STAT ON')
        
    def getPower(self):
        return float(self._ask('VOLT?'))

    def setPower(self, power):
        if power is None or power < self._output_threshold:
            self._write('VOLT -22.0 DBM')
            return
        #self._write('FUNC:SHAP SIN')
        self._write('VOLT %f DBM' % float(power))

    def getFrequency(self):
        return float(self._ask('FREQ?'))

    def setFrequency(self, frequency):
        #self._write('FUNC:SHAP SIN')
        self._write('FREQ %e HZ' % frequency)

    def setOutput(self, power, frequency):
        if power is None or power < self._output_threshold:
            power = self._output_threshold
        self._write('VOLT %f DBM' % float(power))
        self._write('FREQ %e HZ' % frequency)
        """#self._write('TRIG:SOUR EXT')
        #self._write('BM:SOUR EXT')
        #self._write('BM:STAT ON')
        self._write('APPL:SIN %e HZ, %f DBM' % (frequency, float(power)))
        """
    def Off(self):
        self._write('VOLT -22.0 DBM')
        return self._ask('VOLT?')

class Rigol1022():
    """Provides control of Rigol DG1022 function generator with USB via visa."""
    _output_threshold = -90

    def __init__(self, visa_address='Rigol1022'): #visa_address='USB0::0x1AB1::0x0588::DG1D121801379::INSTR'
        self.visa_address = visa_address
        
    def _write(self, string):
        try: # if the connection is already open, this will work
            self.instr.write(string)
        except Exception, e: # else we attempt to open the connection and try again
            print 'Rigol1022 write:', e
            try: # silently ignore possible exceptions raised by del
                del self.instr
            except Exception:
                pass
            self.instr = visa.instrument(self.visa_address, delay=0.04)
            self.instr.write(string)
        
    def _ask(self, s):
        try:
            val = self.instr.ask(s)
        except Exception, e:
            print 'Rigol1022 read:', e
            self.instr = visa.instrument(self.visa_address, delay=0.04)
            val = self.instr.ask(s)
        return val
        
    def On(self):
        self._write('OUTP ON')
        return self._ask('OUTP?')

    def Off(self):
        self._write('OUTP OFF')
        return self._ask('OUTP?')

    def _voltToPower(self, u):
        """
        Converts p-p voltage to dBm assuming 50 Ohm load.
        
        dBm = 20 log10( U / U_0 2**0.5 )
        
        where U_0 = sqrt(50 * 10^-3) * 2**0.5  = 0.31622776601683794
        """
        return 20 * numpy.log10(u / 0.31622776601683794)
        
    def _powerToVolt(self, p):
        """
        Converts power in dBm to p-p Volt assuming 50 Ohm load.
        
        U = U_0 10^(dBm / 20)
        """
        return .1 ** 0.5 * 10 ** (p / 20.)

    def getPower(self):
        return self._voltToPower(float(self._ask('VOLT?').split(':')[1]))

    def setPower(self, power):
        if power is None or power < self._output_threshold:
            self._write('OUTP OFF')
            return
        self._write('VOLT %f' % self._powerToVolt(power))
        self._write('OUTP ON')
 
    def getFrequency(self):
        return float(self._ask('FREQ?').split(':')[1])

    def setFrequency(self, frequency):
        self._write('FREQ %e' % frequency)

    def setOutput(self, power, frequency):
        self.setPower(power)
        self.setFrequency(frequency)
        s = self._ask('FUNC?')
        if s != 'CH1:SIN':
            self._write('FUNC SIN')
