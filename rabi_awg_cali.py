
from tools.utility import edit_singleton
from datetime import date
import os

from traits.api import Bool
import imp
import numpy as np
import time
import threading

from tools.emod import ManagedJob
from tools.utility import GetSetItemsHandler, GetSetItemsMixin

import hardware.api as ha

class awg_rabi_calibration(ManagedJob):

    def __init__(self,psawg):
        super(awg_rabi_calibration, self).__init__()     
        self.psawg = psawg
        
    def _cali_high(self,freqs,fnum):
        file_pos = 'D:/data/ProgGate/Images/x_3_y_1_pair/control_cali/NV_direction/f_' 
        file_pos = file_pos + fnum
        os.path.exists(file_pos)
        
        if(freqs is not None):
            self.e_freqs = freqs
        
        from measurements.pulsed_awg import Rabi
        self.psawg.measurement = Rabi()
            
        self.psawg.measurement.power = 16
        self.psawg.measurement.freq_center = 2.61e9
        self.psawg.measurement.freq = self.e_freqs
        self.psawg.measurement.tau_begin = 15
       # self.psawg.measurement.tau_end = 3000
       # self.psawg.measurement.tau_delta = 20
        #self.psawg.measurement.sweeps = 6e5
        self.psawg.measurement.wait = 5e3
        
        
        #awg_amp =[ 0.050, 0.060, 0.070, 0.080, 0.090, 0.100, 0.110,0.130,0.150,0.170,0.190,0.210,0.230,0.250,0.270,0.290,0.310,0.350, 0.390, 0.430, 0.470, 0.510, 0.550]
        #tau_end = [3000,3000,3000,3000,3000,2500,2500,2500, 2500,2000,2000,2000,1500,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
        #tau_delta = [50,50, 40, 40,30,20,15,15,15,15,15,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
        #sweeps = [1.5e6, 1.5e6, 1.0e6, 5.0e5, 2e5, 2e5,2e5, 2e5, 2e5, 2e5, 2e5, 2e5,2e5, 2e5, 2e5, 2e5, 2e5 ,2e5, 2e5, 2e5 ,2e5, 2e5, 2e5]
        awg_amp =[0.5, 0.6, 0.7]
        tau_end = [2000,2000,2000,2000]
        tau_delta = [10,10,10,10]
        sweeps = [5e5 ,5e5, 5e5, 5e5]
        for t in range(len(awg_amp)):
            self.psawg.measurement.sweeps = sweeps[t]
            self.psawg.measurement.amp = awg_amp[t]
            self.psawg.measurement.tau_end = tau_end[t]
            self.psawg.measurement.tau_delta = tau_delta[t]
            time.sleep(1.0)
            self.psawg.measurement.load()
            time.sleep(35.0)
            while self.psawg.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
        #time.sleep(25.0)
            self.psawg.measurement.submit()
               
            while self.psawg.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
             
            file_name = file_pos + '/amp_' + "%1.3f"%(awg_amp[t]) + '_'
            self.psawg.save_line_plot(file_name + '.png')
            self.psawg.save(file_name + '.pyd')
            time.sleep(10.0)
            self.psawg.measurement.remove()
            self.psawg.measurement.state = 'idle'            
            time.sleep(50.0)
                         
                    
           
    def _cali_low(self,freqs,fnum):
        file_pos = 'D:/data/ProgGate/Images/x_3_y_1_pair/control_cali/NV_direction/f_' 
        file_pos = file_pos + fnum
        os.path.exists(file_pos)
        
        if(freqs is not None):
            self.e_freqs = freqs
        
        from measurements.pulsed_awg import Rabi
        self.psawg.measurement = Rabi()
            
        self.psawg.measurement.power = 16
        self.psawg.measurement.freq_center = 2.61e9
        self.psawg.measurement.freq = self.e_freqs
        self.psawg.measurement.tau_begin = 15
      #  self.psawg.measurement.tau_end = 3000
       # self.psawg.measurement.tau_delta = 20
        #self.psawg.measurement.sweeps = 6e5
        self.psawg.measurement.wait = 5e3
        
        
        #awg_amp = [0.050, 0.060, 0.070, 0.080, 0.090, 0.100, 0.110,0.130,0.150,0.170,0.190,0.210,0.230,0.250,0.270,0.290,0.310,0.350, 0.390, 0.430, 0.470, 0.510, 0.550]
        #tau_end = [3000,3000,3000,3000,3000,2500,2500,2500, 2500,2000,2000,2000,1500,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
        #tau_delta = [50,50, 40, 40,30,20,15,15,15,15,15,10,10,10,10,10,10,10,10,10,10,10,10]
        #sweeps = [2.0e6, 1.5e6, 1.5e6, 1.0e6, 6e5, 5e5,5e5, 2e5, 2e5, 2e5, 2e5, 2e5, 2e5, 2e5 ,2e5, 2e5, 2e5 ,2e5, 2e5, 2e5 ,2e5, 2e5, 2e5]
        awg_amp =[0.4, 0.5, 0.6, 0.7]
        tau_end = [2000,2000,2000,2000]
        tau_delta = [10,10,10,10]
        sweeps = [5e5 ,5e5, 5e5, 5e5]
        for t in range(len(awg_amp)):
            self.psawg.measurement.sweeps = sweeps[t]
            self.psawg.measurement.amp = awg_amp[t]
            self.psawg.measurement.tau_end = tau_end[t]
            self.psawg.measurement.tau_delta = tau_delta[t]
            time.sleep(1.0)
            self.psawg.measurement.load()
            time.sleep(35.0)
            while self.psawg.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
        #time.sleep(25.0)
            self.psawg.measurement.submit()
               
            while self.psawg.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
             
            file_name = file_pos + '/amp_' + "%1.3f"%(awg_amp[t]) + '_'
            self.psawg.save_line_plot(file_name + '.png')
            self.psawg.save(file_name + '.pyd')
            time.sleep(10.0)
            self.psawg.measurement.remove()
            self.psawg.measurement.state = 'idle'            
            time.sleep(50.0)
       
    def _run(self):
       # awg_amp = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        #freq = [2.905179e9,2.837093e9] #[2.905127e9,2.837051e9]
        #freq = [2.902940e9,2.839210e9] #[2.902974e9,2.839217e9]
        #freq = [2.955537e9,2.791676e9] #[2.900797e9,2.841390e9]
        freq = [2.7954905e9, 2.9607435e9] #[2.900797e9,2.841390e9]
        #for t in range(len(awg_amp)):          
        self._cali_high(freq[0],'1')
        self._cali_low(freq[1],'2')
        
        