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

class Hahn_AC_phase_test(ManagedJob):

    def __init__(self, pdawg=None):
        super(Hahn_AC_phase_test, self).__init__()     

        if pdawg is not None:
            self.pdawg = pdawg    
        
           
        
    def _run(self):
        file_pos = 'D:/data/protonNMR/quardrople_dec/search_8(PDMS)/x_5_10_y_5_10/NV1/T2_AC_phase_dependence'
        os.path.exists(file_pos)
       
        power = 11
        freq_center = 1.46e9
        freq = 1.3423e9
        half_pi = 42.4
        pi = 84.8
        tau_echo = 8e3
        wait_time = 30e3
        
        from measurements.shallow_NV import Hahn_AC_phase
        self.pdawg.measurement = Hahn_AC_phase()
        self.pdawg.measurement.power = power
        self.pdawg.measurement.freq_center = freq_center
        
        self.pdawg.measurement.pi2_1 = half_pi
        self.pdawg.measurement.pi_1 = pi
        self.pdawg.measurement.tau_echo = tau_echo
        self.pdawg.measurement.wait_time = wait_time
        self.pdawg.measurement.N_period = 1
        
        self.pdawg.measurement.amp_rf = 1
        self.pdawg.measurement.tau_begin = 0
        self.pdawg.measurement.tau_end = 6.3
        self.pdawg.measurement.tau_delta = 0.2
        self.pdawg.measurement.sweeps = 2.0e5
        
       
        for nk in range(40):
            self.pdawg.measurement.freq_rf = 2.0e6 + nk * 2.0e5
            

            self.pdawg.measurement.load()
            time.sleep(15.0)
            
            while self.pdawg.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
            
            self.pdawg.measurement.submit()
            
            while self.pdawg.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                    
            file_hahn = file_pos + '/freq_' + str(self.pdawg.measurement.freq_rf) + '_amp_1.0_N_1_echo_8us'
            self.pdawg.save_line_plot(file_hahn + '.png')
            self.pdawg.save(file_hahn + '.pyd')
            time.sleep(10.0)
            self.pdawg.measurement.remove()
            self.pdawg.measurement.state = 'idle'
            time.sleep(20.0)       

                        
                
