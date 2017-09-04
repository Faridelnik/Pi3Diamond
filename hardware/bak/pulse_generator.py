# -*- coding: utf-8 -*-
import numpy
import struct
from pulser import Pulser
from nidaq import DOTask

import logging

class PulseGenerator():
    def __init__(self, serial='', channel_map={'ch0':0,'ch1':1,'ch2':2,'ch3':3,'ch4':4,'ch5':5,'ch6':6,'ch7':7,'ch8':8,'ch9':9,'ch10':10,'ch11':11} ):
        self.channel_map = channel_map
        pulser=Pulser(serial=serial)
        self.N_CHANNELS = pulser.getInfo()
        pulser.setResetValue(0x0000)
        pulser.disableDecoder()
        pulser.reset()
        self.pulser=pulser
        
        dotask=DOTask(DOChannels='/Dev2/port0/line0')
        self.dotask=dotask


    def reset(self):
        self.pulser.reset_fpga_com()
        
        
    def Sequence(self,seq,loop=None):
        #print self.pulser.loadPages(self.convertSequenceToBinary(seq)), "Bytes written."
        self.pulser.loadPages(self.convertSequenceToBinary(seq))
        self.pulser.run()

    def Run(self):
        self.pulser.run()

    def Halt(self):
        self.pulser.halt()

    def Night(self):
        self.pulser.setResetValue(0x0000)
        self.dotask.Write(numpy.array((0)))
        self.pulser.halt()

    def Light(self):
        self.Continuous(['green'])
        self.dotask.Write(numpy.array((1)))
        self.pulser.halt()

    def Open(self):
        self.pulser.setResetValue(0xffff)
        self.pulser.halt()

    def Continuous(self, channels):
        bits = 0
        for channel in channels:
            bits = bits | (1 << self.channel_map[channel]) 
        self.pulser.setResetValue(bits)
        self.pulser.halt()

    def checkUnderflow(self):
        return self.pulser.checkUnderflow()
    
    def getInfo(self):
        return self.pulser.getInfo()

#########PATTERN CALCULATION##########
    
    # time is specified in ns

    dt=1.5 # timing step length (1.6ns)

    def createBitsFromChannels(self,channels):
        bits = numpy.zeros(self.N_CHANNELS,dtype=bool)
        for channel in channels:
            bits[self.channel_map[channel]] = True
        return bits

    def setBits(self,integers,start,count,bits):
        """Sets the bits in the range start:start+count in integers[i] to bits[i]."""
        # ToDo: check bit order (depending on whether least significant or most significant bit is shifted out first from serializer)
        for i in range(self.N_CHANNELS):
            if bits[i]:
                integers[i] = integers[i] | (2**count-1) << start

    def pack(self,mult,pattern):
        s = struct.pack('>I%iB'%len(pattern), mult, *pattern[::-1])
        swap = ''
        for i in range(len(s)):
            swap += s[i-1 if i%2 else i+1]
        return swap

    def convertSequenceToBinary(self,sequence):
        """Converts a pulse sequence into a series of pulser instructions,
        taking into account the 8bit minimal pattern length. The pulse sequence
        is described as a list of tuples (channels, time). The pulser instructions
        are of the form 'repetition (32 bit) | ch0 pattern (8bit), ..., ch3 pattern (8bit)'.
        If necessary, high level pulse commands are split into a series of
        suitable low level pulser instructions."""
        buf = ''
        blank = numpy.zeros(self.N_CHANNELS,dtype=int)
        pattern = blank.copy()
        index = 0
        for channels, time in sequence:
            ticks = int(round(time/self.dt))
            if ticks is 0:
                continue
            bits = self.createBitsFromChannels(channels)
            if index + ticks < 8: # if pattern does not fill current block, append to current block and continue
                self.setBits(pattern,index,ticks,bits)
                index += ticks
                continue
            if index > 0: # else fill current block with pattern, reduce ticks accordingly, write block and start a new block 
                self.setBits(pattern,index,8-index,bits)
                buf += self.pack(0,pattern)
                ticks -= ( 8 - index )
                pattern = blank.copy()
            repetitions = ticks / 8 # number of full blocks
            index = ticks % 8 # remainder will make the beginning of a new block
            if repetitions > 0:
                buf += self.pack(repetitions-1,255*bits)
            if index > 0:
                pattern = blank.copy()
                self.setBits(pattern,0,index,bits)
        if index > 0: # fill up incomplete block with zeros and write it
            self.setBits(pattern,index,8-index,numpy.zeros(self.N_CHANNELS,dtype=bool))
            buf += self.pack(0,pattern)
        #print "buf has",len(buf)," bytes"
        buf=buf+((1024-len(buf))%1024)*'\x00' # pad buffer with zeros so it matches SDRAM / FIFO page size
        #logging.getLogger().debug('buffer: '+buf)
        #print "buf has",len(buf)," bytes"
        return buf


########## TESTCODE############


if __name__ == '__main__':
    
    PG = PulseGenerator()    
    PG.Sequence(100*[(['ch0'],1000.),([],1000.)])
    
#    PG.Sequence([(['laser'],3),([],6*1.5),(['mw'],127)] )
    
#    def high(self,x):
#        return([(['laser'],x) , ( ['mw'],x ) ] )
#    
#    def low(self,x):
#        return([ (['mw'],x), ( ['laser'],x ) ] )
