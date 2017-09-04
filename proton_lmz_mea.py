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

class proton_lmz_measure(ManagedJob):
    def __init__(self, pdawg=None):
        super(proton_lmz_measure, self).__init__()     
            
        if pdawg is not None:
            self.pdawg = pdawg       
            

    def _run(self):
        file_pos = 'D:/data/protonNMR/quardrople_dec/search_8(PDMS)/x_5_10_y_5_10/NV1/lmz'
        os.path.exists(file_pos)
       
        power = 10.2
        freq_center = 1.46e9
        freq = 1.34147e9
        half_pi = 44
        pi = 88
        tau_echo = 14e3
        wait_time = 40e3
        
        from measurements.shallow_NV import Proton_longmagzationdet_freq
        self.pdawg.measurement = Proton_longmagzationdet_freq()
        self.pdawg.measurement.power = power
        self.pdawg.measurement.freq_center = freq_center
        self.pdawg.measurement.freq = freq
        self.pdawg.measurement.pi2_1 = half_pi
        self.pdawg.measurement.pi_1 = pi
        self.pdawg.measurement.tau_echo = tau_echo
        self.pdawg.measurement.wait_time = wait_time
        #self.pdawg.measurement.pulse_num = 10
        self.pdawg.measurement.tau_begin = 2.0e6
        self.pdawg.measurement.tau_end = 2.8e6
        self.pdawg.measurement.tau_delta = 25e3
        self.pdawg.measurement.sweeps =1.0e6
        
        for nk in range(7):
                
            self.pdawg.measurement.N_period = nk + 2
            
            self.pdawg.measurement.load()
            time.sleep(100.0)
            
            while self.pdawg.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
            
            self.pdawg.measurement.submit()
            
            while self.pdawg.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                    
            file_hahn = file_pos + '/lmz_N_' + str(nk+2)
            self.pdawg.save_line_plot(file_hahn + '.png')
            self.pdawg.save(file_hahn + '.pyd')
            time.sleep(10.0)
            self.pdawg.measurement.remove()
            self.pdawg.measurement.state = 'idle'
            time.sleep(20.0)       
                    
                        
                
