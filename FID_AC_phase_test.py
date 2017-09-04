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

class FID_AC_test(ManagedJob):
    def __init__(self, odmr = None, pdawg=None,  pdawg2=None):
        super(FID_AC_test, self).__init__()     
        
        if odmr is not None:
            self.odmr = odmr    
        
        if pdawg is not None:
            self.pdawg = pdawg    

        if pdawg2 is not None:
            self.pdawg2 = pdawg2           
            
    def _odmr(self,fst,fend):
        t_pi = 900
        power_p = -30
        stop_time = 120
        
        self.odmr.pulsed = True
        self.odmr.perform_fit = True
        #self.odmr.number_of_resonances=2
        self.odmr.power_p = power_p
        self.odmr.t_pi = t_pi
        self.odmr.stop_time = stop_time 
        
        self.odmr.frequency_begin_p = fst
        self.odmr.frequency_end_p = fend
        self.odmr.frequency_delta_p = 1.0e5
        self.odmr.number_of_resonances = 1
        self.odmr.threshold = -60
        
    def _run(self):
        file_pos = 'D:/data/protonNMR/quardrople_dec/search_8(PDMS)/x_5_10_y_5_10/NV1/FID_AC_test'
        os.path.exists(file_pos)
       
        power = 10.2
        freq_center = 1.46e9
        freq = 1.3421e9
        half_pi = 44
        #pi = 88
        #tau_echo = 14e3
        #wait_time = 40e3
        
        fstart= 1.338e9     
        fend= 1.347e9
        self._odmr(fstart,fend)
        time.sleep(1.0)    
        
        from measurements.shallow_NV import FID_AC
        self.pdawg.measurement = FID_AC()
        self.pdawg.measurement.power = power
        self.pdawg.measurement.freq_center = freq_center
        
        self.pdawg.measurement.pi2_1 = half_pi
        #self.pdawg.measurement.pi_1 = pi
        #self.pdawg.measurement.tau_echo = tau_echo
        #self.pdawg.measurement.wait_time = wait_time
        self.pdawg.measurement.N_period = 1
        self.pdawg.measurement.freq_rf = 5.0e6
        #self.pdawg.measurement.amp_rf = 1
        self.pdawg.measurement.tau_begin = 600
        self.pdawg.measurement.tau_end = 24e3
        self.pdawg.measurement.tau_delta = 600
        self.pdawg.measurement.sweeps = 5.0e5
        
        
        from measurements.shallow_NV import FID_Db
        self.pdawg2.measurement = FID_Db()
        self.pdawg2.measurement.power = power
        self.pdawg2.measurement.freq_center = freq_center
        #self.pdawg2.measurement.freq = freq
        self.pdawg2.measurement.pi2_1 = half_pi
        #self.pdawg.measurement.pi_1 = pi
        #self.pdawg.measurement.tau_echo = tau_echo
        #self.pdawg.measurement.wait_time = wait_time
        #self.pdawg.measurement.N_period = 1
        #self.pdawg.measurement.freq_rf = 5.0e6
        #self.pdawg.measurement.amp_rf = 1
        self.pdawg2.measurement.tau_begin = 600
        self.pdawg2.measurement.tau_end = 24e3
        self.pdawg2.measurement.tau_delta = 600
        self.pdawg2.measurement.sweeps = 5.0e5
        
        for nk in range(4):
            
            self.pdawg.measurement.amp_rf = nk * 0.2 + 0.2
            
            self.odmr.submit()   
            time.sleep(2.0)
            
            while self.odmr.state != 'done':
                #print threading.currentThread().getName()

                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break
                     
            file_odmr = file_pos + '/Odmr_amp_rf_' + str(self.pdawg.measurement.amp_rf)
            self.odmr.save_line_plot(file_odmr + '.png')
            self.odmr.save(file_odmr + '.pyd')
            time.sleep(5.0)  
            
            freq = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)
            self.pdawg.measurement.freq = freq
            self.pdawg2.measurement.freq = freq
            
            self.pdawg.measurement.load()
            time.sleep(15.0)
            
            while self.pdawg.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
            
            self.pdawg.measurement.submit()
            
            while self.pdawg.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                    
            file_hahn = file_pos + '/FID_AC_amp_rf_' + str(self.pdawg.measurement.amp_rf)
            self.pdawg.save_line_plot(file_hahn + '.png')
            self.pdawg.save(file_hahn + '.pyd')
            time.sleep(10.0)
            self.pdawg.measurement.remove()
            self.pdawg.measurement.state = 'idle'
            time.sleep(20.0)       
            
            self.pdawg2.measurement.load()
            time.sleep(15.0)
            
            while self.pdawg2.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
            
            self.pdawg2.measurement.submit()
            
            while self.pdawg2.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                    
            file_hahn = file_pos + '/FID_Db_amp_rf_' + str(self.pdawg.measurement.amp_rf)
            self.pdawg.save_line_plot(file_hahn + '.png')
            self.pdawg.save(file_hahn + '.pyd')
            time.sleep(10.0)
            self.pdawg.measurement.remove()
            self.pdawg.measurement.state = 'idle'
            time.sleep(20.0)       
                    
                        
                
