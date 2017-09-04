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

class dynamics_measure(ManagedJob):
    def __init__(self,auto_focus, confocal,odmr=None,psawg=None,sensing=None,pdawg=None,sf=None,gs=None):
        super(dynamics_measure, self).__init__()     
        self.auto_focus=auto_focus   
        self.confocal=confocal
        if odmr is not None:
            self.odmr = odmr
            
        if psawg is not None:
            self.psawg = psawg

        if sensing is not None:
            self.sensing = sensing
            
        if pdawg is not None:
            self.pdawg = pdawg     
            
        if sf is not None:
            self.sf = sf    
            
        if gs is not None:
            self.gs = gs      
            
    def _odmr(self,fst,fend):
        t_pi = 1400
        power_p = -33
        stop_time = 200
        
        self.odmr.pulsed = True
        self.odmr.perform_fit = True
        #self.odmr.number_of_resonances=2
        self.odmr.power_p = power_p
        self.odmr.t_pi = t_pi
        self.odmr.stop_time = stop_time 
        
        self.odmr.frequency_begin_p = fst
        self.odmr.frequency_end_p = fend
        self.odmr.frequency_delta_p = 2.0e5
        self.odmr.number_of_resonances = 1
        self.odmr.threshold = -60

    def _run(self):
        file_pos = 'D:/data/protonNMR/quardrople_dec/search/x_24_34_y_26_36/NV2'
        file_name = file_pos
        os.path.exists(file_pos)
        
        power = 9
        freq_center = 1.76e9
        fstart= 1.476e9     
        fend= 1.490e9
        self._odmr(fstart,fend)
        time.sleep(1.0)                
        
        from measurements.pulsed_awg import Rabi 
        self.psawg.measurement = Rabi()
        from analysis.pulsedawgan import RabiFit_phase   
        self.psawg.fit = RabiFit_phase()
        
        self.psawg.measurement.power = power
        self.psawg.measurement.freq_center = freq_center

        self.psawg.measurement.tau_begin = 10
        self.psawg.measurement.tau_end = 1000
        self.psawg.measurement.tau_delta = 10
        self.psawg.measurement.sweeps = 1.0e5       

        self.psawg.fit = RabiFit_phase()
        time.sleep(5.0)    

 
        self.auto_focus.periodic_focus = True
        
        time.sleep(1.0)
        self.odmr.submit()   
        time.sleep(2.0)
        
        while self.odmr.state != 'done':
            #print threading.currentThread().getName()

            threading.currentThread().stop_request.wait(1.0)
            if threading.currentThread().stop_request.isSet():
                 break
                 
        file_odmr = file_name + '/Odmr'
        self.odmr.save_line_plot(file_odmr + '.png')
        self.odmr.save(file_odmr + '.pyd')
        time.sleep(5.0)  
        
            
        freq = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)

        
        # Rabi #####################
        
        self.psawg.measurement.freq = freq
        #self.psawg.measurement.load()

        self.psawg.fit = RabiFit_phase()
        time.sleep(5.0)    
        rabi_flag = True
        if np.isnan(power):
            power = 9
        while rabi_flag:
            self.psawg.measurement.power = power
            self.psawg.measurement.load()
            time.sleep(5.0)
            self.psawg.fit = RabiFit_phase()
            time.sleep(5.0)
        
            while self.psawg.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
            
            self.psawg.measurement.submit()
           
            while self.psawg.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
            if(self.psawg.fit.contrast[0] < 18):
                continue
            half_pi = self.psawg.fit.t_pi2[0]
            pi = self.psawg.fit.t_pi[0]   
            if pi < 80.5 and pi > 79.8:
                rabi_flag = False
            else:
                amp = 80.0/pi
                #amp^2 = power/power_next
                power = power - 10*np.log10(amp**2)
            if np.isnan(power):
                power = 9   
     
         
        file_rabi = file_name + '/Rabi'
        self.psawg.save_line_plot(file_rabi + '.png')
        self.psawg.save(file_rabi + '.pyd')
        time.sleep(10.0)
        self.psawg.measurement.remove()
        self.psawg.measurement.state = 'idle'            
        time.sleep(5.0)
        half_pi = self.psawg.fit.t_pi2[0]
        pi = self.psawg.fit.t_pi[0]
        
            
        from measurements.shallow_NV import XY8_Ref
        self.pdawg.measurement = XY8_Ref()
        self.pdawg.measurement.power = power
        self.pdawg.measurement.freq_center = freq_center
        self.pdawg.measurement.freq = freq
        self.pdawg.measurement.pi2_1 = half_pi
        self.pdawg.measurement.pi_1 = pi
        self.pdawg.measurement.pulse_num = 12
        self.pdawg.measurement.tau_begin = 60
        self.pdawg.measurement.tau_end = 110
        self.pdawg.measurement.tau_delta = 2
        self.pdawg.measurement.sweeps = 4.0e5
        
        self.pdawg.measurement.load()
        time.sleep(100.0)
        
        while self.pdawg.measurement.reload == True:
            threading.currentThread().stop_request.wait(1.0)
        
        self.pdawg.measurement.submit()
        
        while self.pdawg.measurement.state != 'done':
             threading.currentThread().stop_request.wait(1.0)
             if threading.currentThread().stop_request.isSet():
                break
                
        file_hahn = file_name + '/dynamics_XY8_' + str(self.pdawg.measurement.pulse_num) + '_' + str(int(pi)) + 'ns'
        self.pdawg.save_line_plot(file_hahn + '.png')
        self.pdawg.save(file_hahn + '.pyd')
        time.sleep(10.0)
        self.pdawg.measurement.remove()
        self.pdawg.measurement.state = 'idle'
        time.sleep(20.0)       
        
        from measurements.shallow_NV import Hahn
        self.pdawg.measurement = Hahn()
        self.pdawg.measurement.power = power
        self.pdawg.measurement.freq_center = freq_center
        self.pdawg.measurement.freq = freq
        self.pdawg.measurement.pi2_1 = half_pi
        self.pdawg.measurement.pi_1 = pi
        
        self.pdawg.measurement.tau_begin = 300
        self.pdawg.measurement.tau_end = 15000
        self.pdawg.measurement.tau_delta = 600
        self.pdawg.measurement.sweeps = 2.0e5
        
        self.pdawg.measurement.load()
        time.sleep(20.0)
        
        while self.pdawg.measurement.reload == True:
            threading.currentThread().stop_request.wait(1.0)
        
        self.pdawg.measurement.submit()
        
        while self.pdawg.measurement.state != 'done':
             threading.currentThread().stop_request.wait(1.0)
             if threading.currentThread().stop_request.isSet():
                break
                
        file_hahn = file_name + '/T2'
        self.pdawg.save_line_plot(file_hahn + '.png')
        self.pdawg.save(file_hahn + '.pyd')
        time.sleep(10.0)
        self.pdawg.measurement.remove()
        self.pdawg.measurement.state = 'idle'
        time.sleep(20.0)       
        
        from measurements.shallow_NV import T1
        self.pdawg.measurement = T1()
        self.pdawg.measurement.power = power
        self.pdawg.measurement.freq_center = freq_center
        self.pdawg.measurement.freq = freq
        self.pdawg.measurement.pi_1 = pi
        self.pdawg.measurement.tau_begin = 300
        self.pdawg.measurement.tau_end = 1000000
        self.pdawg.measurement.tau_delta = 60000
        self.pdawg.measurement.sweeps = 1.0e5
        
        self.pdawg.measurement.load()
        time.sleep(15.0)
        
        while self.pdawg.measurement.reload == True:
            threading.currentThread().stop_request.wait(1.0)
        
        self.pdawg.measurement.submit()
        
        while self.pdawg.measurement.state != 'done':
             threading.currentThread().stop_request.wait(1.0)
             if threading.currentThread().stop_request.isSet():
                break
                
        file_hahn = file_name + '/T1'
        self.pdawg.save_line_plot(file_hahn + '.png')
        self.pdawg.save(file_hahn + '.pyd')
        time.sleep(10.0)
        self.pdawg.measurement.remove()
        self.pdawg.measurement.state = 'idle'
        time.sleep(20.0)       
        
        x_last = self.confocal.x
        y_last = self.confocal.y
        
        self.auto_focus.periodic_focus = False
        
   
                    
                        
                
