# -*- coding: utf-8 -*-
import logging
import ok
import os
import time
bitfile = os.path.join(os.path.dirname(__file__),'pulser.bit')

class Pulser():
    """LOWLEVEL PULSER CONTROL"""

    command_map = {'IDLE':0,'RUN':1,'LOAD':2,'RESET_READ':3,'RESET_SDRAM':4,'RESET_WRITE':5}

    def __init__(self, serial=''):
        xem = ok.FrontPanel()
        if (xem.OpenBySerial(serial) != 0):
            raise RuntimeError, 'Failed to open USB connection.'
        PLL = ok.PLL22150()
        xem.GetPLL22150Configuration(PLL)
        PLL.SetVCOParameters(333,48)            #set VCO to 333MHz
        PLL.SetOutputSource(0,5)                #clk0@166MHz
        PLL.SetDiv1(1,6)                        #needs to be set for DIV1=3
        PLL.SetOutputEnable(0,1)
        logging.getLogger().info('Pulser base clock input (clk2xin, stage two): '+str(PLL.GetOutputFrequency(0)))
        xem.SetPLL22150Configuration(PLL)

        if (xem.ConfigureFPGA(bitfile) != 0):
            raise RuntimeError, 'Failed to upload bit file to fpga.'

        self.xem = xem
        self.serial = serial
    
    def getInfo(self):
        self.xem.UpdateWireOuts()
        return self.xem.GetWireOutValue(0x20)

    def ctrlPulser(self,command):
        self.xem.SetWireInValue(0x00,self.command_map[command], 0x07)
        self.xem.UpdateWireIns()
        #print self.command_map[command]

    def enableDecoder(self):
        self.xem.SetWireInValue(0x00,0x00,0x08)
        self.xem.UpdateWireIns()

    def disableDecoder(self):
        self.xem.SetWireInValue(0x00,0xFF,0x08)
        self.xem.UpdateWireIns()

    def run(self):
        self.ctrlPulser('RUN')
        self.enableDecoder()

    def halt(self):
        self.disableDecoder()
        self.ctrlPulser('IDLE')

    def loadPages(self,buf):
        if len(buf) % 1024 != 0:
            raise RuntimeError('Only full SDRAM pages supported. Pad your buffer with zeros such that its length is a multiple of 1024.')
        self.disableDecoder()
        self.ctrlPulser('RESET_WRITE')
        self.ctrlPulser('RESET_READ')
        self.ctrlPulser('LOAD')
        bytes = self.xem.WriteToBlockPipeIn(0x80,1024,buf)
        self.ctrlPulser('IDLE')
        self.xem.UpdateTriggerOuts()
        return bytes

    def reset_fpga_com(self):
        del self.xem
        self.__init__(self.serial)
        time.sleep(0.1)
        
    def reset(self):     
        self.disableDecoder()
        self.ctrlPulser('RESET_WRITE')
        self.ctrlPulser('RESET_READ')
        self.ctrlPulser('RESET_SDRAM')
        self.ctrlPulser('IDLE')

    def setResetValue(self,bits):
        self.xem.SetWireInValue(0x00,bits<<4,0xfff0)
        self.xem.UpdateWireIns()

    def checkUnderflow(self):
        self.xem.UpdateTriggerOuts()
        return self.xem.IsTriggered(0x60,1)

    ########OLD CONTROL UNIT######
    """  
    def enableDecoder(self, state=False):
        if(state==True):
            self.xem.SetWireInValue(0x00,0x00,0x40)
        else:
            self.xem.SetWireInValue(0x00,0xFF,0x40)
        self.xem.UpdateWireIns()

    def enableRun(self, state=False):
        if( state==True ):
            self.xem.SetWireInValue(0x00,0xFF,0x20)
        else:
             self.xem.SetWireInValue(0x00,0x00,0x20)
        self.xem.UpdateWireIns()

    def enableLoad(self, state=False):
        if( state==True ):
            self.xem.SetWireInValue(0x00,0xFF,0x10)
        else:
             self.xem.SetWireInValue(0x00,0x00,0x10)
        self.xem.UpdateWireIns()

    def resetRead(self, state=True):
        if( state==True ):
            self.xem.SetWireInValue(0x00,0xFF,0x08)
        else:
             self.xem.SetWireInValue(0x00,0x00,0x08)
        self.xem.UpdateWireIns()

    def resetWrite(self,state=True):
        if( state==True ):
            self.xem.SetWireInValue(0x00,0xFF,0x04)
        else:
             self.xem.SetWireInValue(0x00,0x00,0x04)
        self.xem.UpdateWireIns()
    
    def resetSDRAM(self,state=True):
        if( state==True ):
            self.xem.SetWireInValue(0x00,0xFF,0x02)
        else:
             self.xem.SetWireInValue(0x00,0x00,0x02)
        self.xem.UpdateWireIns()


    def setIdle(self,state=True):
        if( state==True ):
            self.xem.SetWireInValue(0x00,0xFF,0x01)
        else:
             self.xem.SetWireInValue(0x00,0x00,0x01)
        self.xem.UpdateWireIns()
    
    def resetState(self,state):
        if( state==True):
            self.xem.SetWireInValue(0x00,0xFF,0x80)
        else:
            self.xem.SetWireInValue(0x00,0x00,0x80)
        self.xem.UpdateWireIns()
    """

