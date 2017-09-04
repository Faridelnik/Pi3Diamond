
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

class dephasing_vs_mag(ManagedJob):

    def __init__(self,odmr,pair,file_po):
        super(dephasing_vs_mag, self).__init__()    
        self.odmr = odmr
        self.pair = pair
        self.po=file_po
        
    def _run(self):
        file_pos = 'D:/data/ProgGate/Images/pair_search/10_14-18(search)/x_4_3/' + self.po
        os.path.exists(file_pos)
        
        self.odmr.frequency_begin_p_1 = 2.951e9
        self.odmr.frequency_end_p_1 = 2.964e9
        
        self.odmr.frequency_begin_p_2 = 2.875e9
        self.odmr.frequency_end_p_2 = 2.885e9
        
        t_pi = 3100
        power_p = -30
        stop_time = 500
        
        from analysis.pulsedawgan import RabiFit   
        
      
        for t in range(len([0])):
            
            # high frequency
            self.odmr.pulsed = True
            self.odmr.perform_fit = True
            #self.odmr.number_of_resonances=2
            self.odmr.power_p = power_p
            self.odmr.t_pi = t_pi
            self.odmr.stop_time = stop_time 
            
            self.odmr.frequency_begin_p = 2.951e9
            self.odmr.frequency_end_p = 2.964e9
            self.odmr.frequency_delta_p = 1.5e5
            self.odmr.submit()   
            time.sleep(10)
            
            while self.odmr.state != 'done':
                #print threading.currentThread().getName()

                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break

            file_name = file_pos + '_odmr_nv1_high'
            self.odmr.save_line_plot(file_name + '.png')
            self.odmr.save(file_name + '.pyd')
            time.sleep(10)
            fhigh1 = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)
            
            
            # rabi_high
            from measurements.pair_search import Rabi
            self.pair.measurement = Rabi()
            self.pair.fit = RabiFit()
            
            self.pair.measurement.power = 16
            self.pair.measurement.freq_center = 2.71e9

            self.pair.measurement.tau_begin = 15
            self.pair.measurement.tau_end = 500
            self.pair.measurement.tau_delta = 5
            self.pair.measurement.sweeps = 2e5
            
            self.pair.measurement.freq = fhigh1
            self.pair.measurement.load()
            time.sleep(15.0)
            while self.pair.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
                
            self.pair.measurement.submit()
            time.sleep(5.0)
            self.pair.fit = RabiFit()
               
            while self.pair.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
             
            file_name = file_pos + '_rabi_nv1_high'
            self.pair.save_line_plot(file_name + '.png')
            self.pair.save(file_name + '.pyd')
            time.sleep(10.0)
            self.pair.measurement.remove()
            self.pair.measurement.state = 'idle'            
            time.sleep(20.0)
            half_pi_nv1 = self.pair.fit.t_pi2[0]
            
            # low frequency          
            self.odmr.frequency_begin_p = 2.785e9
            self.odmr.frequency_end_p = 2.798e9
            self.odmr.frequency_delta_p = 1.5e5
            self.odmr.submit()   
            time.sleep(10)
    
            while self.odmr.state != 'done':
                #print threading.currentThread().getName()

                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break

            file_name = file_pos + '_odmr_nv1_low'
            self.odmr.save_line_plot(file_name + '.png')
            self.odmr.save(file_name + '.pyd')
            time.sleep(20)
            flow1 = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)
            
            # rabi_low
            
            self.pair.measurement.freq = flow1
            self.pair.measurement.load()
            time.sleep(15.0)
            while self.pair.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
                
            self.pair.measurement.submit()
            time.sleep(5.0)
            self.pair.fit = RabiFit()   
            while self.pair.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
             
            file_name = file_pos + '_rabi_nv1_low'
            self.pair.save_line_plot(file_name + '.png')
            self.pair.save(file_name + '.pyd')
            time.sleep(10.0)
            self.pair.measurement.remove()
            self.pair.measurement.state = 'idle'
            time.sleep(10.0)
            pi_p_nv1 = self.pair.fit.t_pi[0]
            
            # high frequency      #################################    
            self.odmr.frequency_begin_p = 2.919e9
            self.odmr.frequency_end_p = 2.931e9
            self.odmr.frequency_delta_p = 1.5e5
            self.odmr.submit()   
            time.sleep(10)
    
            while self.odmr.state != 'done':
                #print threading.currentThread().getName()

                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break

            file_name = file_pos + '_odmr_nv2_high'
            self.odmr.save_line_plot(file_name + '.png')
            self.odmr.save(file_name + '.pyd')
            time.sleep(20)
            fhigh2 = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)
            
            # rabi_low
            
            self.pair.measurement.freq = fhigh2
            self.pair.measurement.load()
            time.sleep(15.0)
            while self.pair.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
                
            self.pair.measurement.submit()
            time.sleep(5.0)
            self.pair.fit = RabiFit()   
            while self.pair.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
             
            file_name = file_pos + '_rabi_nv2_high'
            self.pair.save_line_plot(file_name + '.png')
            self.pair.save(file_name + '.pyd')
            time.sleep(10.0)
            self.pair.measurement.remove()
            self.pair.measurement.state = 'idle'
            time.sleep(10.0)
            half_pi_nv2 = self.pair.fit.t_pi2[0]
            
            # low frequency      #################################    
            self.odmr.frequency_begin_p = 2.822e9
            self.odmr.frequency_end_p = 2.836e9
            self.odmr.frequency_delta_p = 1.5e5
            self.odmr.submit()   
            time.sleep(10)
    
            while self.odmr.state != 'done':
                #print threading.currentThread().getName()

                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break

            file_name = file_pos + '_odmr_nv2_low'
            self.odmr.save_line_plot(file_name + '.png')
            self.odmr.save(file_name + '.pyd')
            time.sleep(20)
            flow2 = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)
            
            # rabi_low
            
            self.pair.measurement.freq = flow2
            self.pair.measurement.load()
            time.sleep(15.0)
            while self.pair.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
                
            self.pair.measurement.submit()
            time.sleep(5.0)
            self.pair.fit = RabiFit()   
            while self.pair.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
             
            file_name = file_pos + '_rabi_nv2_low'
            self.pair.save_line_plot(file_name + '.png')
            self.pair.save(file_name + '.pyd')
            time.sleep(10.0)
            self.pair.measurement.remove()
            self.pair.measurement.state = 'idle'
            time.sleep(10.0)
            pi_p_nv2 = self.pair.fit.t_pi[0]
            
            
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
                    
            file_name = file_pos + '_hahnpair_double_'
            self.pair.save_line_plot(file_name + '.png')
            self.pair.save(file_name + '.pyd')
            time.sleep(10.0)
            self.pair.measurement.remove()
            self.pair.measurement.state = 'idle'
            time.sleep(50.0)
                        