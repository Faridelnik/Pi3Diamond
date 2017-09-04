from tools.utility import edit_singleton
from datetime import date
import os

from traits.api import Bool
import imp
import numpy as np
import time
import threading
import sys
from tools.emod import ManagedJob
from tools.utility import GetSetItemsHandler, GetSetItemsMixin

import hardware.api as ha

class get_data(ManagedJob):
    def __init__(self,file_name,auto_focus, confocal,odmr=None,psawg=None,pair=None):
        super(get_data, self).__init__()     
        self.auto_focus=auto_focus   
        self.confocal=confocal
        self.file_name = file_name
        if odmr is not None:
            self.odmr = odmr
            
        if psawg is not None:
            self.psawg = psawg

        if pair is not None:
            self.pair = pair
            
    def _odmr(self,fst,fend):
        t_pi = 2700
        power_p = -30
        stop_time = 1000
        
        self.odmr.pulsed = True
        self.odmr.perform_fit = True
        #self.odmr.number_of_resonances=2
        self.odmr.power_p = power_p
        self.odmr.t_pi = t_pi
        self.odmr.stop_time = stop_time 
        
        self.odmr.frequency_begin_p = fst
        self.odmr.frequency_end_p = fend
        self.odmr.frequency_delta_p = 1.5e5

    def _run(self):
        file_name = self.file_name
        if not os.path.isdir(file_name):
            os.mkdir(file_name)
        
        self.auto_focus.submit()
        time.sleep(8.0)
        file_nv = file_name + '/image.png'
        self.confocal.save_image(file_nv)
        time.sleep(8.0)
        self.auto_focus.periodic_focus = True
        
        fstart= 2.951e9     
        fend= 2.965e9
        self._odmr(fstart,fend)
        time.sleep(1.0)
        self.odmr.submit()   
        time.sleep(2.0)
        
        while self.odmr.state != 'done':
            #print threading.currentThread().getName()

            threading.currentThread().stop_request.wait(1.0)
            if threading.currentThread().stop_request.isSet():
                 break
                 
        file_odmr = file_name + '/odmr_nv1_high'
        self.odmr.save_line_plot(file_odmr + '.png')
        self.odmr.save(file_odmr + '.pyd')
        time.sleep(5.0)  
        
        fhigh1 = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)
        
        # Rabi #####################
        from measurements.pair_search import Rabi
        self.psawg.measurement = Rabi()
        from analysis.pulsedawgan import RabiFit_phase   
        self.psawg.fit = RabiFit_phase()
        
        self.psawg.measurement.power = 16
        self.psawg.measurement.freq_center = 2.71e9

        self.psawg.measurement.tau_begin = 10
        self.psawg.measurement.tau_end = 500
        self.psawg.measurement.tau_delta = 5
        self.psawg.measurement.sweeps = 3e5
        
        self.psawg.measurement.freq = fhigh1
        self.psawg.measurement.load()
        time.sleep(15.0)
        while self.psawg.measurement.reload == True:
            threading.currentThread().stop_request.wait(1.0)
            
        self.psawg.measurement.submit()
        time.sleep(5.0)
        self.psawg.fit = RabiFit_phase()
        time.sleep(5.0)
        self.psawg.fit = RabiFit_phase()
           
        while self.psawg.measurement.state != 'done':
             threading.currentThread().stop_request.wait(1.0)
             if threading.currentThread().stop_request.isSet():
                break
         
        file_rabi = file_name + '/rabi_nv1_high'
        self.psawg.save_line_plot(file_rabi + '.png')
        self.psawg.save(file_rabi + '.pyd')
        time.sleep(10.0)
        self.psawg.measurement.remove()
        self.psawg.measurement.state = 'idle'            
        time.sleep(5.0)
        half_pi_nv1 = self.psawg.fit.t_pi2[0]
        
        condition = np.isnan(half_pi_nv1) or self.psawg.fit.contrast[0] < 7 or self.psawg.fit.contrast[0] > 25
        if condition:
            sys.exit(0)
            
        fstart= 2.784e9    
        fend= 2.798e9
        self.odmr.frequency_begin_p = fstart
        self.odmr.frequency_end_p = fend
        time.sleep(1.0)
        self.odmr.submit()   
        time.sleep(2.0)
        
        while self.odmr.state != 'done':
            #print threading.currentThread().getName()

            threading.currentThread().stop_request.wait(1.0)
            if threading.currentThread().stop_request.isSet():
                 break
                 
        file_odmr = file_name + '/odmr_nv1_low'
        self.odmr.save_line_plot(file_odmr + '.png')
        self.odmr.save(file_odmr + '.pyd')
        time.sleep(5.0)  
        
        flow1 = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)
        
        # Rabi #####################                   
        self.psawg.measurement.freq = flow1
        self.psawg.measurement.load()
        time.sleep(15.0)
        while self.psawg.measurement.reload == True:
            threading.currentThread().stop_request.wait(1.0)
            
        self.psawg.measurement.submit()
        time.sleep(5.0)
        self.psawg.fit = RabiFit_phase()
        time.sleep(5.0)
        self.psawg.fit = RabiFit_phase()
           
        while self.psawg.measurement.state != 'done':
             threading.currentThread().stop_request.wait(1.0)
             if threading.currentThread().stop_request.isSet():
                break
         
        file_rabi = file_name+ '/rabi_nv1_low'
        self.psawg.save_line_plot(file_rabi + '.png')
        self.psawg.save(file_rabi + '.pyd')
        time.sleep(10.0)
        self.psawg.measurement.remove()
        self.psawg.measurement.state = 'idle'            
        time.sleep(5.0)
        pi_p_nv1 = self.psawg.fit.t_pi[0]
        
        condition = np.isnan(pi_p_nv1) or self.psawg.fit.contrast[0] < 7 or self.psawg.fit.contrast[0] > 25
        if condition:
            sys.exit(0)
            
        # odmr    
        self.odmr.frequency_begin_p = 2.918e9
        self.odmr.frequency_end_p = 2.934e9
        self.odmr.submit()   
        time.sleep(10)

        while self.odmr.state != 'done':
            #print threading.currentThread().getName()

            threading.currentThread().stop_request.wait(1.0)
            if threading.currentThread().stop_request.isSet():
                 break

        file_odmr = file_name + '/odmr_nv2_high'
        self.odmr.save_line_plot(file_odmr + '.png')
        self.odmr.save(file_odmr + '.pyd')
        time.sleep(10)
        fhigh2 = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)    
        
        # Rabi #####################
        self.psawg.measurement.freq = fhigh2
        self.psawg.measurement.load()
        time.sleep(15.0)
        while self.psawg.measurement.reload == True:
            threading.currentThread().stop_request.wait(1.0)
            
        self.psawg.measurement.submit()
        time.sleep(5.0)
        self.psawg.fit = RabiFit_phase()
           
        while self.psawg.measurement.state != 'done':
             threading.currentThread().stop_request.wait(1.0)
             if threading.currentThread().stop_request.isSet():
                break
         
        file_rabi = file_name + '/rabi_nv2_high'
        self.psawg.save_line_plot(file_rabi + '.png')
        self.psawg.save(file_rabi + '.pyd')
        time.sleep(10.0)
        self.psawg.measurement.remove()
        self.psawg.measurement.state = 'idle'            
        time.sleep(5.0)
        half_pi_nv2 = self.psawg.fit.t_pi2[0]
        
        condition = np.isnan(half_pi_nv2) or self.psawg.fit.contrast[0] < 7 or self.psawg.fit.contrast[0] > 25
        if condition:
            sys.exit(0)
            
        self.odmr.frequency_begin_p = 2.820e9
        self.odmr.frequency_end_p = 2.837e9
        time.sleep(1.0)
        self.odmr.submit()   
        time.sleep(2.0)
        
        while self.odmr.state != 'done':
            #print threading.currentThread().getName()

            threading.currentThread().stop_request.wait(1.0)
            if threading.currentThread().stop_request.isSet():
                 break
                 
        file_odmr = file_name + '/odmr_nv2_low'
        self.odmr.save_line_plot(file_odmr + '.png')
        self.odmr.save(file_odmr + '.pyd')
        time.sleep(5.0)  
        
        flow2 = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)
        
        # Rabi #####################                   
        self.psawg.measurement.freq = flow2
        self.psawg.measurement.load()
        time.sleep(15.0)
        while self.psawg.measurement.reload == True:
            threading.currentThread().stop_request.wait(1.0)
            
        self.psawg.measurement.submit()
        time.sleep(5.0)
        self.psawg.fit = RabiFit_phase()
           
        while self.psawg.measurement.state != 'done':
             threading.currentThread().stop_request.wait(1.0)
             if threading.currentThread().stop_request.isSet():
                break
         
        file_rabi= file_name + '/rabi_nv2_low'
        self.psawg.save_line_plot(file_rabi + '.png')
        self.psawg.save(file_rabi + '.pyd')
        time.sleep(10.0)
        self.psawg.measurement.remove()
        self.psawg.measurement.state = 'idle'            
        time.sleep(5.0)
        pi_p_nv2 = self.psawg.fit.t_pi[0]
        
        condition = np.isnan(pi_p_nv2) or self.psawg.fit.contrast[0] < 7 or self.psawg.fit.contrast[0] > 25
        if condition:
            sys.exit(0)

        from measurements.pair_search import HahnPair
        self.pair.measurement = HahnPair()
        self.pair.measurement.power = 16
        self.pair.measurement.freq_center = 2.71e9
        self.pair.measurement.multihahn = True
        self.pair.measurement.freq = fhigh1
        self.pair.measurement.freq_4 = flow1
        self.pair.measurement.ffreq = fhigh2
        self.pair.measurement.ffreq_4 = flow2
        self.pair.measurement.pi2_1 = half_pi_nv1
        self.pair.measurement.pi2_2 = half_pi_nv2
        self.pair.measurement.pi_1_p = half_pi_nv1 * 2
        self.pair.measurement.pi_2_p = half_pi_nv2 * 2
        self.pair.measurement.pi_1_m = pi_p_nv1
        self.pair.measurement.pi_2_m = pi_p_nv2
        
        self.pair.measurement.tau_begin = 300
        self.pair.measurement.tau_end = 50000
        self.pair.measurement.tau_delta = 1000
        self.pair.measurement.sweeps = 3.0e5
        
        self.pair.measurement.load()
        time.sleep(35.0)
        
        while self.pair.measurement.reload == True:
            threading.currentThread().stop_request.wait(1.0)
        
        self.pair.measurement.submit()
        
        while self.pair.measurement.state != 'done':
             threading.currentThread().stop_request.wait(1.0)
             if threading.currentThread().stop_request.isSet():
                break
                
        file_hahn = file_name + '/hahnpair_double'
        self.pair.save_line_plot(file_hahn + '.png')
        self.pair.save(file_hahn + '.pyd')
        time.sleep(10.0)
        self.pair.measurement.remove()
        self.pair.measurement.state = 'idle'
        time.sleep(20.0)       
                