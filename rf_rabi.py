
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

class RF_rabi_check(ManagedJob):

    def __init__(self,psawg):
        super(RF_rabi_check, self).__init__()     
        self.psawg = psawg
        
    def _check(self):
        file_pos = 'D:/data/QuantumSim/Luck_1/rf_rabi' 
        os.path.exists(file_pos)
        
        from measurements.pulsed_awg import Single_RF_Rabi
        self.psawg.measurement = Single_RF_Rabi()
        
        self.psawg.measurement.power = -26
        self.psawg.measurement.amp = 0.2
        self.psawg.measurement.tau_begin = 300
        self.psawg.measurement.tau_end = 30000
        self.psawg.measurement.tau_delta = 1000
        self.psawg.measurement.wait_time = 60e3
        self.psawg.measurement.sweeps = 3e5
        self.psawg.measurement.laser = 10e3
        self.psawg.measurement.wait = 10e3
        self.psawg.measurement.rf_freq = 2.78e6
        
        freqs = [2.845929e9, 2.847990e9, 2.850066e9, 2.899368e9, 2.901468e9,2.903502e9]
        t_pis = [2350,
                 2400,
                 2540,
                 3290,
                 3585,
                 3636
                ]      
        for t in range(len(freqs)):
           self.psawg.measurement.freq = freqs[t]         
           self.psawg.measurement.pi_1 = t_pis[t] 
           self.psawg.measurement.load()
           time.sleep(12.0)
           self.psawg.measurement.submit()
           
           while self.psawg.measurement.state != 'done':
                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                   break
            
           file_name = file_pos + '/freq_' + str(t)
           self.psawg.save_line_plot(file_name + '.png')
           self.psawg.save(file_name + '.pyd')          
           time.sleep(20)


    def _run(self):
        #freq = [2.828757e9,2.908415e9]
        self._check()
       # self._cali(freq[1],'/high_frequency')
        
        