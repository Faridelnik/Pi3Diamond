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
rm = visa.ResourceManager()
import numpy
import logging

class SMIQ_HMC():
    """Provides control of SMIQ family microwave sources from Rhode und Schwarz with GPIB via visa."""
    _output_threshold = -90.0
    
    def __init__(self, visa_address='GPIB1::30::INSTR'):
        self.visa_address = visa_address
        
    def _write(self, string):
        try: # if the connection is already open, this will work
            self.instr.write(string)
        except: # else we attempt to open the connection and try again
            try: # silently ignore possible exceptions raised by del
                del self.instr
            except Exception:
                pass
            self.instr = rm.get_instrument(self.visa_address)
            self.instr.write(string)
        
    def _ask(self, str):
        try:
            val = self.instr.ask(str)
        except:
            self.instr = rm.get_instrument(self.visa_address)
            val = self.instr.ask(str)
        return val 

    def getPower(self):
        return float(self._ask(':POW?'))

    def setPower(self, power):
        if power is None or power < self._output_threshold:
            logging.getLogger().debug('SMIQ at '+str(self.visa_address)+' turning off.')
            self._write(':FREQ:MODE CW')
            self._write(':OUTP OFF')
            return
        logging.getLogger().debug('SMIQ at '+str(self.visa_address)+' setting power to '+str(power))
        self._write(':FREQ:MODE CW')
        self._write(':POW %f' % float(power))
        self._write(':OUTP ON')

    def getFrequency(self):
        return float(self._ask(':FREQ?'))

    def setFrequency(self, frequency):
        self._write(':FREQ:MODE CW')
        self._write(':FREQ %e' % frequency)

        
        """
        ODMR based on sweep mode with one initial trigger
        
        1. set microwave source to sweep mode, with 1 sweep per trigger
        2. create timedifferences thread on timetagger with binwidth = dwell time, waiting for start trigger
        3. generate one start trigger
        4. wait until timetagger is done
        5. fetch data from timetagger
        """
        
        
        
    def initSweep(self, frequency, dt=1):
        """
        determine f0, f1, df
        
        
        """
        #if len(frequency) != len(power):
            #raise ValueError('Length mismatch between list of frequencies and list of powers.')
            
        #self._write("POW %f" % power)
        #self._write("OUTP ON")    
        f0 = frequency[0]
        f1 = frequency[-1]
        df = frequency[1]-frequency[0]
        # set sweep mode with df, f0, f1
        # set dwell time
        # set trigger to external
        
        self._write(":FREQ:STAR %e" %f0)
        self._write(":FREQ:STOP %e" % f1)
        self._write(":FREQ:STEP %e" %df)
        #print instr.ask("FREQ:STAR?;STOP?;STEP?")

        self._write(":FREQ:MODE SWE")
        self._write(":SWE:DWEL %f" %dt)
        self._write(":SWE:DIR UP")
        # each trigger do one sweep
        self._write(":SWE:COUNT 1")

        #instr.write("TRIG:SOUR EXT; SLOP POS")
        #self._write("TRIG:SOUR IMM")

        self._write(":INIT:CONT ON")
        self._write(":INIT")
        time.sleep(0.1)
        
        self._write('*WAI')
        
        #N = int(numpy.round(float(self._ask(':LIST:FREQ:POIN?'))))
        #if N != len(frequency):
            #raise RuntimeError, 'Error in SMIQ with List Mode'

       
          
    def Off(self):
        
        self._write(':OUTP OFF')
        self._write('*WAI')    

class SMR20():
    """Provides control of SMR20 microwave source from Rhode und Schwarz with GPIB via visa."""
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
        self._write(':LIST:POW'  +  PowerString)
        self._write(':TRIG1:LIST:SOUR EXT')
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

from nidaq import SquareWave

class HybridMicrowaveSourceSMIQNIDAQ():
    """Provides a microwave source that can do frequency sweeps
    with pixel clock output using SMIQ and nidaq card."""

    def __init__(self, visa_address, square_wave_device):
        self.source = SMIQ( visa_address )
        self.square_wave = SquareWave( square_wave_device )

    def setOutput(self, power, frequency, seconds_per_point=1e-2):
        """Sets the output of the microwave source.
        'power' specifies the power in dBm. 'frequency' specifies the
        frequency in Hz. If 'frequency' is a single number, the source
        is set to cw. If 'frequency' contains multiple values, the
        source sweeps over the frequencies. 'seconds_per_point' specifies
        the time in seconds that the source spends on each frequency step.
        A sweep is excecute by the 'doSweep' method."""
        
        # in any case set the CW power
        self.source.setPower(power)
        self.square_wave.setTiming(seconds_per_point)
        
        try: length=len(frequency)
        except TypeError: length=0

        self._length=length

        if length:
            self.source.setFrequency(frequency[0])
            self.source.initSweep(frequency, power*numpy.ones(length))
        else:
            self.source.setFrequency(frequency)

    def doSweep(self):
        """Perform a single sweep."""
        if not self._length:
            raise RuntimeError('Not in sweep mode. Change to sweep mode and try again.')
        #self.source.resetListPos()
        self.square_wave.setLength(self._length)
        self.square_wave.output()


