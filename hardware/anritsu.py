"""
This file is part of Diamond. Diamond is a confocal scanner written
in python / Qt4. It combines an intuitive gui with flexible
hardware abstraction classes.

Diamond is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Diamond is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with diamond. If not, see <http://www.gnu.org/licenses/>.

Copyright (C) 2009 Helmut Rathgen <helmut.rathgen@gmail.com>
"""

from visa import instrument

ANRITSU = instrument('TCPIP::192.168.0.3::inst0::INSTR')

def On():
    ANRITSU.write('OUTP:STAT ON')
    ANRITSU.write('*WAI')
    return ANRITSU.ask('OUTP:STAT?')

def Off():
    ANRITSU.write('OUTP:STAT OFF')
    return ANRITSU.ask('OUTP:STAT?')

def Power(power=None):
    if power != None:
        ANRITSU.write(':POW %f' % power)
    return float(ANRITSU.ask(':POW?'))

def Freq(f=None):
    if f != None:
        ANRITSU.write(':FREQ %e' % f)
    return float(ANRITSU.ask(':FREQ?'))

def CW(f=None, power=None):
    ANRITSU.write(':FREQ:MODE CW')
    if f != None:
        ANRITSU.write(':FREQ %e' % f)
    if power != None:
        ANRITSU.write(':POW %f' % power)

def List(freq, power):
    ANRITSU.write(':POW %f' % power)
    ANRITSU.write(':LIST:TYPE FREQ')
    ANRITSU.write(':LIST:IND 0')
    s = ''
    for f in freq[:-1]:
        s += ' %f,' % f
    s += ' %f' % freq[-1]
    ANRITSU.write(':LIST:FREQ' + s)
    ANRITSU.write(':LIST:STAR 0')
    ANRITSU.write(':LIST:STOP %i' % (len(freq)-1) )
    ANRITSU.write(':LIST:MODE MAN')
    ANRITSU.write('*WAI')

def Trigger(source, pol):
    ANRITSU.write(':TRIG:SOUR '+source)
    ANRITSU.write(':TRIG:SLOP '+pol)
    ANRITSU.write('*WAI')

def ResetListPos():
    ANRITSU.write(':LIST:IND 0')
    ANRITSU.write('*WAI')

def ListOn():
    ANRITSU.write(':FREQ:MODE LIST')
    ANRITSU.write(':OUTP ON')
    ANRITSU.write('*WAI')

def Modulation(flag=None):
    if flag is not None:
        if flag:
            ANRITSU.write('PULM:SOUR EXT')
            ANRITSU.write('PULM:STAT ON')
        else:
            ANRITSU.write('PULM:STAT OFF')
    return ANRITSU.ask('PULM:STAT?')

Modulation(True)
