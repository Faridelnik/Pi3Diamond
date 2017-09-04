
"""
Hardware API is defined here.

Example of usage:

from hardware.api import PulseGenerator
PG = PulseGenerator  

Default hardware api hooks are dummy classes.

Provide a file 'custom_api.py' to define actual hardware API hooks.
This can be imported names, modules, factory functions and factory functions
that emulate singleton behavior.

See 'custom_api_example.py' for examples.
"""

import numpy as np
import logging
import time

class Scanner(  ):
    def getXRange(self):
        return (0.,100.)
    def getYRange(self):
        return (0.,100.)
    def getZRange(self):
        return (-20.,20.)
    def setx(self, x):
        pass
    def sety(self, y):
        pass
    def setz(self, z):
        pass
    def setPosition(self, x, y, z):
        """Move stage to x, y, z"""
        pass
    def scanLine(self, Line, SecondsPerPoint, return_speed=None):
        time.sleep(0.1)
        return (1000*np.sin(Line[0,:])*np.sin(Line[1,:])*np.exp(-Line[2,:]**2)).astype(int)
        #return np.random.random(Line.shape[1])

class Counter(  ):
    def configure(self, n, SecondsPerPoint, DutyCycle=0.8):
        x = np.arange(n)
        a = 100.
        c = 50.
        x0 = n/2.
        g = n/10.
        y = np.int32( c - a / np.pi * (  g**2 / ( (x-x0)**2 + g**2 )  ) )
        Counter._sweeps = 0
        Counter._y = y
    def run(self):
        time.sleep(1)
        Counter._sweeps+=1
        return np.random.poisson(Counter._sweeps*Counter._y)
    def clear(self):
        pass

class Microwave(  ):
    def setPower(self, power):
        logging.getLogger().debug('Setting microwave power to '+str(power)+'.')
    def setOutput(self, power, frequency):
        logging.getLogger().debug('Setting microwave to p='+str(power)+' f='+str(frequency)+'.')
    def initSweep(self, f, p):
        logging.getLogger().debug('Setting microwave to sweep between frequencies %e .. %e with power %f.'%(f[0],f[-1],p[0]))
    def resetListPos(self):
        pass

MicrowaveA = Microwave
MicrowaveB = Microwave
MicrowaveC = Microwave
MicrowaveD = Microwave
MicrowaveE = Microwave

class RFSource():
    def setOutput(self, power, frequency):
        pass

class PulseGenerator():
    def Sequence(self, sequence, loop=True):
        pass
    def Light(self):
        pass
    def Night(self):
        pass
    def Open(self):
        pass
    def High(self):
        pass
    def checkUnderflow(self):
        return False
        #return np.random.random()<0.1

class Laser():
    """Provides control of the laser power."""
    voltage = 0.

class PowerMeter():
    """Provides an optical power meter."""
    power = 0.
    def getPower(self):
        """Return the optical power in Watt."""
        PowerMeter.power += 1
        return PowerMeter.power*1e-3

class Coil():
    def set_output(self,channel,current):
        pass

class RotationStage():
    def set_angle(self, angle):
        pass
    
import dummy_time_tagger as TimeTagger

# if customized hardware factory is present run it
# Provide this file to overwrite / add your own factory functions, classes, imports
import os
if os.access('hardware/custom_api.py', os.F_OK):
    execfile('hardware/custom_api.py')
