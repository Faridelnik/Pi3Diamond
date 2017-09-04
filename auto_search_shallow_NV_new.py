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

class auto_search_shallow(ManagedJob):
    def __init__(self,auto_focus, confocal,odmr=None,psrabi=None,psawg=None,sensing=None,pdawg=None,pdawg2=None,pdawg3=None,sf=None,gs=None):
        super(auto_search_shallow, self).__init__()     
        self.auto_focus=auto_focus   
        self.confocal=confocal
        if odmr is not None:
            self.odmr = odmr
            
        if psrabi is not None:
            self.psrabi = psrabi      
            
        if psawg is not None:
            self.psawg = psawg

        if sensing is not None:
            self.sensing = sensing
            
        if pdawg is not None:
            self.pdawg = pdawg  
            
        if pdawg2 is not None:
            self.pdawg2 = pdawg2 

        if pdawg3 is not None:
            self.pdawg3 = pdawg3                     
            
        if sf is not None:
            self.sf = sf    
            
        if gs is not None:
            self.gs = gs      
            
    def _odmr_pulsed(self,fst,fend,t_pi):
        #t_pi = 2400
        power_p = -20
        stop_time = 100
        
        self.odmr.pulsed = True
        self.odmr.perform_fit = True
        #self.odmr.number_of_resonances=2
        self.odmr.power_p = power_p
        self.odmr.t_pi = t_pi
        self.odmr.stop_time = stop_time 
        
        self.odmr.frequency_begin_p = fst
        self.odmr.frequency_end_p = fend
        self.odmr.frequency_delta_p = 1.5e5
        self.odmr.number_of_resonances = 1
        self.odmr.threshold = -65
        
    def _odmr_cw(self):
        power = -10
        stop_time = 180
        
        self.odmr.pulsed = False
        self.odmr.perform_fit = True
        #self.odmr.number_of_resonances=2
        self.odmr.power = power
        self.odmr.stop_time = stop_time 
        
        self.odmr.frequency_begin = 2.77e9
        self.odmr.frequency_end = 2.99e9
        self.odmr.frequency_delta = 1.5e6
        self.odmr.number_of_resonances = 2
        self.odmr.threshold = -60    

    def _run(self):
        #file_path = 'D:/data/protonNMR/membrane_2/scan/L11/11'
        #file_path = 'D:/data/protonNMR/membrane_2/scan/L11/22'
        file_path = 'D:/data/protonNMR/membrane_2/scan/M12(dirty)/13'
        os.path.exists(file_path)
        
        self.confocal.resolution = 150
        
        power = 16
        freq_center = 2.7e9
        #req = 1.49347e9
        #fstart= 1.532e9     
        #fend= 1.542e9
        
        
        from measurements.pulsed_awg import Rabi 
        self.psawg.measurement = Rabi()
        from analysis.pulsedawgan import RabiFit_phase   
        self.psawg.fit = RabiFit_phase()
        
        self.psawg.measurement.power = power
        self.psawg.measurement.vpp = 1.2
        #self.psawg.measurement.freq_center = freq_center

        self.psawg.measurement.tau_begin = 10
        self.psawg.measurement.tau_end = 400
        self.psawg.measurement.tau_delta = 10
        self.psawg.measurement.sweeps = 1.0e5       

        self.psawg.fit = RabiFit_phase()
        time.sleep(5.0)    
        
        from measurements.shallow_NV import Hahn
        self.pdawg.measurement = Hahn()
        self.pdawg.measurement.power = power
        self.pdawg.measurement.vpp = 1.2
        
        self.pdawg.measurement.tau_begin = 300
        self.pdawg.measurement.tau_end = 8000
        self.pdawg.measurement.tau_delta = 300
        self.pdawg.measurement.sweeps = 3.0e5
        
        from measurements.shallow_NV import T1
        self.pdawg2.measurement = T1()
        self.pdawg2.measurement.power = power
       
        self.pdawg2.measurement.vpp = 1.2

        self.pdawg2.measurement.tau_begin = 300
        self.pdawg2.measurement.tau_end = 1000000
        self.pdawg2.measurement.tau_delta = 100000
        self.pdawg2.measurement.sweeps = 1.0e5
        
        self.auto_focus.periodic_focus = False
        
        #if os.path.isfile(file_image1):
            #self.confocal.load(file_image1)
            #time.sleep(1.0)
        #else:
       
        self.confocal.remove_all_labels()
        self.auto_focus.remove_all_targets()
        
        x_shift = 0 
        y_shift = 0
        index = np.linspace(2,31,30)
        for ncenter in index:
            ncenter = int(ncenter)
            self.auto_focus.periodic_focus = False
            self.auto_focus.target_name = 'NV'+str(ncenter)
            time.sleep(1.0)
            
            self.confocal.y = self.sf.Centroids[ncenter][0] + y_shift
            self.confocal.x = self.sf.Centroids[ncenter][1] + x_shift
            if (self.confocal.y > 50 or self.confocal.y < 0 or self.confocal.x > 50 or self.confocal.x < 0):
                continue
            self.auto_focus.submit()
            time.sleep(12.0)
            
            file_name = file_path + '/NV' + str(ncenter)
            #file_name = file_path
            if not os.path.isdir(file_name):
                os.mkdir(file_name)
            #else:
                #continue
 
            self.confocal.resolution = 150   
            self.confocal.submit()
            time.sleep(155)
            
            file_nv = file_name + '/nv' + str(ncenter) + '_image.png'
            self.confocal.save_image(file_nv)
            
            
            self.auto_focus.submit()
            time.sleep(16.0)

            self.auto_focus.periodic_focus = True
            
            
            time.sleep(1.0)
            self._odmr_cw()
            time.sleep(1.0)
            self.odmr.submit()   
            time.sleep(2.0)
            
            while self.odmr.state != 'done':
                #print threading.currentThread().getName()

                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break
            self.odmr.remove()   
            self.odmr.state = 'idle'               
            file_odmr = file_name + '/Odmr_cw'
            self.odmr.save_line_plot(file_odmr + '.png')
            self.odmr.save(file_odmr + '.pyd')
            time.sleep(5.0)  
            
            if(len(self.odmr.fit_frequencies) > 2 or self.odmr.fit_contrast.max() < 5 or self.odmr.fit_contrast.max() > 43):
                continue
                
            freq = self.odmr.fit_frequencies[0]
            
            if(np.isnan(freq)):
                continue
                
            ''' 
            self.psrabi.measurement.freq = freq-1.5e6   
            self.psrabi.measurement.power = -20   
            self.psrabi.measurement.sweeps = 1.0e5               
            self.psrabi.measurement.fit = RabiFit_phase()   
            time.sleep(1.0)
            self.psrabi.submit()
            while self.psrabi.state != 'done':
                #print threading.currentThread().getName()

                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break
            self.psrabi.measurement.remove()   
            self.psrabi.measurement.state = 'idle'             
            file_rabi = file_name + '/Rabi_pulsed'
            self.psrabi.save_line_plot(file_rabi + '.png')
            self.psrabi.save(file_rabi + '.pyd')
            time.sleep(5.0)  
            
            
        
            t_pulse_pi = 1300
            
            fst= 1.532e9     
            fend= 1.544e9
            self._odmr_pulsed(fst,fend,t_pulse_pi)
            time.sleep(1.0)
            self.odmr.submit()   
            time.sleep(2.0)
            
            while self.odmr.state != 'done':
                #print threading.currentThread().getName()

                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break
            self.odmr.remove()   
            self.odmr.state = 'idle'                
            file_odmr = file_name + '/Odmr_pulsed'
            self.odmr.save_line_plot(file_odmr + '.png')
            self.odmr.save(file_odmr + '.pyd')
            time.sleep(5.0)  
            
            if(len(self.odmr.fit_frequencies) > 2 or self.odmr.fit_contrast.max() < 6 or self.odmr.fit_contrast.max() > 43):
                continue
                
            freq = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)
            
            if(np.isnan(freq)):
                continue    
                '''
            # Rabi #####################
            
            self.psawg.measurement.freq = freq
            freq_center = freq-100e6  
            self.psawg.measurement.freq_center = freq_center
            '''
            rabi_flag = True
            if np.isnan(power):
                power = 12
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
                        
                half_pi = self.psawg.fit.t_pi2[0]
                pi = self.psawg.fit.t_pi[0]   
                if pi < 96 and pi > 72:
                    rabi_flag = False
                else:
                    amp = 85.0/pi
                    #amp^2 = power/power_next
                    power = power - 10*np.log10(amp**2)
                    if power > 16 or power < 0:
                         rabi_flag=False
                if self.psawg.fit.contrast[0] < 10:
                    rabi_flag=False
                if np.isnan(power):
                    power = 12   
                    '''
                    
                  
            #self.psawg.measurement.power = power
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
                    
             
            self.psawg.measurement.progress = 0
            self.psawg.measurement.elapsed_sweeps = 0
         
            file_rabi = file_name + '/Rabi' + '_power_' + str(power)
            self.psawg.save_line_plot(file_rabi + '.png')
            self.psawg.save(file_rabi + '.pyd')
            time.sleep(10.0)
            self.psawg.measurement.remove()
            self.psawg.measurement.state = 'idle'            
            time.sleep(5.0)
            half_pi = self.psawg.fit.t_pi2[0]
            pi = self.psawg.fit.t_pi[0]
            
            condition = np.isnan(half_pi) or self.psawg.fit.contrast[0] < 12
            if condition:
                continue
                

            self.pdawg.measurement.freq_center = freq_center
            self.pdawg.measurement.freq = freq
            self.pdawg.measurement.pi2_1 = half_pi
            self.pdawg.measurement.pi_1 = pi
            
            self.pdawg.measurement.load()
            time.sleep(20.0)
            
            while self.pdawg.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
            
            self.pdawg.measurement.submit()
            
            while self.pdawg.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                    
            self.pdawg.measurement.progress = 0
            self.pdawg.measurement.elapsed_sweeps = 0                
                    
            file_hahn = file_name + '/T2'
            self.pdawg.save_line_plot(file_hahn + '.png')
            self.pdawg.save(file_hahn + '.pyd')
            time.sleep(5.0)
            self.pdawg.measurement.remove()
            self.pdawg.measurement.state = 'idle'
            time.sleep(5.0)       
            
            
            self.pdawg2.measurement.freq_center = freq_center
            self.pdawg2.measurement.freq = freq
            
            self.pdawg2.measurement.pi_1 = pi
            
            self.pdawg2.measurement.load()
            time.sleep(15.0)
            
            while self.pdawg2.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
            
            self.pdawg2.measurement.submit()
            
            while self.pdawg2.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
            self.pdawg2.measurement.progress = 0
            self.pdawg2.measurement.elapsed_sweeps = 0               
            file_T1 = file_name + '/T1'
            self.pdawg2.save_line_plot(file_T1 + '.png')
            self.pdawg2.save(file_T1 + '.pyd')
            time.sleep(10.0)
            self.pdawg2.measurement.remove()
            self.pdawg2.measurement.state = 'idle'
            time.sleep(10.0)   
            
            '''
            from measurements.shallow_NV import XY8_Ref
            self.pdawg3.measurement = XY8_Ref()
            self.pdawg3.measurement.power = power
            self.pdawg3.measurement.freq_center = freq_center
           
            self.pdawg3.measurement.pulse_num = 16
            self.pdawg3.measurement.tau_delta = 1.0
            self.pdawg3.measurement.sweeps = 0.5e6
            
            bfield = (2870 - self.psawg.measurement.freq/1.0e6)/2.8
            tau_s = int(1000000 / bfield / 4.255 / 4.0 - half_pi)
            
            self.pdawg3.measurement.freq = freq
            self.pdawg3.measurement.pi2_1 = half_pi
            self.pdawg3.measurement.pi_1 = pi
            self.pdawg3.measurement.tau_begin = tau_s - 8
            self.pdawg3.measurement.tau_end = tau_s + 8
            
            self.pdawg3.measurement.load()
            time.sleep(100.0)
            
            while self.pdawg3.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
            
            self.pdawg3.measurement.submit()
            
            while self.pdawg3.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
            self.pdawg3.measurement.progress = 0
            self.pdawg3.measurement.elapsed_sweeps = 0        
            
            file_hahn = file_name + '/dynamics_XY8_' + str(self.pdawg3.measurement.pulse_num) + '_' + 'ns_30_60ns' + str(int(pi))  + 'ns'
            self.pdawg3.save_line_plot(file_hahn + '.png')
            self.pdawg3.save(file_hahn + '.pyd')
            time.sleep(10.0)
            self.pdawg3.measurement.remove()
            self.pdawg3.measurement.state = 'idle'
            time.sleep(20.0)     
            
            
            ntau = [1,16,32]
            tau_end = [1500, 500, 400]
            tau_delta = [40, 15, 10]
            for k in range(3):   
        
                self.pdawg3.measurement.pulse_num = ntau[k]
                self.pdawg3.measurement.tau_begin = 20
                self.pdawg3.measurement.tau_end = tau_end[k]
                self.pdawg3.measurement.tau_delta = tau_delta[k]
                self.pdawg3.measurement.sweeps = 0.8e6
                
                self.pdawg3.measurement.load()
                time.sleep(100.0)
                
                while self.pdawg3.measurement.reload == True:
                    threading.currentThread().stop_request.wait(1.0)
                
                self.pdawg3.measurement.submit()
                
                while self.pdawg3.measurement.state != 'done':
                     threading.currentThread().stop_request.wait(1.0)
                     if threading.currentThread().stop_request.isSet():
                        break
                        
                self.pdawg3.measurement.progress = 0
                self.pdawg3.measurement.elapsed_sweeps = 0               
                file_hahn = file_name + '/dynamics_XY8_' + str(self.pdawg3.measurement.pulse_num) + '_' + 'full'
                self.pdawg3.save_line_plot(file_hahn + '.png')
                self.pdawg3.save(file_hahn + '.pyd')
                time.sleep(10.0)
                self.pdawg3.measurement.remove()
                self.pdawg3.measurement.state = 'idle'
                time.sleep(20.0)       
            
                '''
            
            '''
            self.pdawg.measurement.pi2_1 = half_pi
            self.pdawg.measurement.pi_1 = pi
            self.pdawg.measurement.tau_begin = tau_s - 15
            self.pdawg.measurement.tau_end = tau_s + 15
            
            self.pdawg.measurement.load()
            time.sleep(100.0)
            
            while self.pdawg.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
            
            self.pdawg.measurement.submit()
            
            while self.pdawg.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                    
            file_hahn = file_name + '/dynamics_XY8_' + str(self.pdawg.measurement.pulse_num) + '_' + 'ns_30_60ns' + str(int(pi))  + 'ns'
            self.pdawg.save_line_plot(file_hahn + '.png')
            self.pdawg.save(file_hahn + '.pyd')
            time.sleep(10.0)
            self.pdawg.measurement.remove()
            self.pdawg.measurement.state = 'idle'
            time.sleep(20.0)       
            '''
            self.auto_focus.periodic_focus = False
            x_shift = self.confocal.x - self.sf.Centroids[ncenter][1]
            y_shift = self.confocal.y - self.sf.Centroids[ncenter][0]

            '''
            

            if pi > 180 or pi < 100:
                continue
                
        
            half_pi = 49.4
            pi = 98.8
            file_name = file_path
            for k in range(3):   
            self.auto_focus.periodic_focus = True
    
            from measurements.shallow_NV import XY8_Ref
            self.pdawg3.measurement = XY8_Ref()
            self.pdawg3.measurement.power = power
            self.pdawg3.measurement.freq_center = freq_center
            self.pdawg3.measurement.freq = freq
            self.pdawg3.measurement.pi2_1 = half_pi
            self.pdawg3.measurement.pi_1 = pi
            self.pdawg3.measurement.pulse_num = 36 + k*8
            self.pdawg3.measurement.tau_begin = 50
            self.pdawg3.measurement.tau_end = 200 - 0 * 200
            self.pdawg3.measurement.tau_delta = 10
            self.pdawg3.measurement.sweeps = 0.8e6
            
            self.pdawg3.measurement.load()
            time.sleep(100.0)
            
            while self.pdawg3.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
            
            self.pdawg3.measurement.submit()
            
            while self.pdawg3.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                    
            file_hahn = file_name + '/dynamics_XY8_' + str(self.pdawg3.measurement.pulse_num) + '_' + str(int(pi)) + 'ns_30_60ns'
            self.pdawg3.save_line_plot(file_hahn + '.png')
            self.pdawg3.save(file_hahn + '.pyd')
            time.sleep(10.0)
            self.pdawg3.measurement.remove()
            self.pdawg3.measurement.state = 'idle'
            time.sleep(20.0)       
            
            
            
            from measurements.shallow_NV import XY8_Ref
            self.pdawg2.measurement = XY8_Ref()
            self.pdawg2.measurement.power = power
            self.pdawg2.measurement.freq_center = freq_center
            self.pdawg2.measurement.freq = freq
            self.pdawg2.measurement.pi2_1 = half_pi
            self.pdawg2.measurement.pi_1 = pi
            self.pdawg2.measurement.pulse_num = 40 + k*8
            self.pdawg2.measurement.tau_begin = 50
            self.pdawg2.measurement.tau_end = 200 - 0 * 200
            self.pdawg2.measurement.tau_delta = 10
            self.pdawg2.measurement.sweeps = 0.8e6
            
            self.pdawg2.measurement.load()
            time.sleep(100.0)
            
            while self.pdawg2.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
            
            self.pdawg2.measurement.submit()
            
            while self.pdawg2.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                    
            file_hahn = file_name + '/dynamics_XY8_' + str(self.pdawg2.measurement.pulse_num) + '_' + str(int(pi)) + 'ns_30_60ns'
            self.pdawg2.save_line_plot(file_hahn + '.png')
            self.pdawg2.save(file_hahn + '.pyd')
            time.sleep(10.0)
            self.pdawg2.measurement.remove()
            self.pdawg2.measurement.state = 'idle'
            time.sleep(20.0)       
            self.auto_focus.periodic_focus = False

            self.confocal.x = 45
            #self.auto_focus.periodic_focus = False  
            '''
                    
                        
                
