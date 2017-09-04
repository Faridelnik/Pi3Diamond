from tools.utility import edit_singleton
from datetime import date
import os

from traits.api import Bool
import imp
import numpy as np
import string
import time
import threading
import hardware.SMC_controller as smc
import hardware.api as ha
pg = ha.PulseGenerator()

Magnet=smc.SMC()

from tools.emod import ManagedJob
from tools.utility import GetSetItemsHandler, GetSetItemsMixin

import hardware.api as ha

class T1T2XY8fixedMF(ManagedJob):

    def __init__(self, auto_focus, confocal, mag_al, odmr=None,psrabi=None,psawg2=None,sensing=None,pdawg=None,pdawg2=None,pdawg3=None,sf=None,gs=None):
        super(T1T2XY8fixedMF, self).__init__()     
        
        self.auto_focus=auto_focus   
        self.confocal=confocal
        self.magnet_control = mag_al
        if odmr is not None:
            self.odmr = odmr
            
        if psrabi is not None:
            self.psrabi = psrabi      
            
        if psawg2 is not None:
            self.psawg2 = psawg2

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
            
    def _magnet_alignment(self, direction):
        
        self.magnet_control.choosed_align = 'fluoresence'
                         
        if direction == 'y':                 
            self.magnet_control.choosed_scan =  'fy_scan'   
        elif direction == 'x':   
            self.magnet_control.choosed_scan =  'fx_scan'   
            
        self.magnet_control.fitting_func = 'gaussian'
             
        self.magnet_control.start_positionX = 5.65
        self.magnet_control.end_positionX = 5.8
        self.magnet_control.stepX = 0.01
          
        self.magnet_control.start_positionY = 5.6
        self.magnet_control.end_positionY = 6.5
        self.magnet_control.stepY = 0.06
        
        self.magnet_control.number_of_resonances=1
        self.magnet_control.perform_fit=False
                 
    def _odmr_cw(self):
    
        self.odmr.pulsed = False
        self.odmr.perform_fit = True
      
        self.odmr.power = -28
        self.odmr.stop_time = 20
        
        self.odmr.frequency_begin= 1.1e+09
        self.odmr.frequency_end = 1.2e+09
        self.odmr.frequency_delta = 1.0e6
        self.odmr.number_of_resonances = 1
        self.odmr.threshold = -80
                 
    def _odmr_pulsed(self, freq):
        
        self.odmr.pulsed = True
        self.odmr.perform_fit = True
      
        self.odmr.power_p = -34
        self.odmr.t_pi = 650
        self.odmr.stop_time = 30
                
        self.odmr.frequency_begin_p = freq - 0.006e+09
        self.odmr.frequency_end_p = freq + 0.006e+09
        self.odmr.frequency_delta_p = 0.8e5
        self.odmr.number_of_resonances = 1
        self.odmr.threshold = -80

    def _run(self):
        
        file_path ='D:/data/sample #100/D1 Ar_SF6 p+/autopilot' 
        os.path.exists(file_path)
        
        self.confocal.resolution = 200
        power=-3
        vpp=0.6
        
        from measurements.pulsed_awg import Rabi 
        self.psawg2.measurement = Rabi()
        from analysis.pulsedawgan import RabiFit_phase   
        self.psawg2.fit = RabiFit_phase()
        

        self.psawg2.measurement.tau_begin = 300
        self.psawg2.measurement.tau_end = 1000
        self.psawg2.measurement.tau_delta = 10
        self.psawg2.measurement.sweeps = 0.6e5 
        self.psawg2.measurement.power = power   
        self.psawg2.measurement.vpp = vpp
        #self.psawg.measurement.freq_center = freq_center

        self.psawg2.fit = RabiFit_phase()
        time.sleep(5.0)          
        
        from measurements.shallow_NV import Hahn
        self.pdawg.measurement = Hahn()
        self.pdawg.measurement.power = power
        #self.pdawg.measurement.freq_center = freq_center
        self.pdawg.measurement.vpp = 0.6
        
        self.pdawg.measurement.tau_begin = 300
        self.pdawg.measurement.tau_end = 6000
        self.pdawg.measurement.tau_delta = 250
        self.pdawg.measurement.sweeps = 8.0e5
        
        from measurements.shallow_NV import T1
        self.pdawg2.measurement = T1()
        self.pdawg2.measurement.power = power
       
        self.pdawg2.measurement.vpp = 0.6
        #self.pdawg2.measurement.freq_center = freq_center
        self.pdawg2.measurement.tau_begin = 300
        self.pdawg2.measurement.tau_end = 1000000
        self.pdawg2.measurement.tau_delta = 100000
        self.pdawg2.measurement.sweeps = 0.4e+6
        
        from measurements.shallow_NV import XY8_Ref
        self.pdawg3.measurement = XY8_Ref()
                
        self.pdawg3.measurement.tau_begin = 40
        self.pdawg3.measurement.tau_end = 71
        self.pdawg3.measurement.tau_delta = 1
        self.pdawg3.measurement.sweeps = 2.1e5
        self.pdawg3.measurement.vpp = vpp
        self.pdawg3.measurement.power = power
        #self.pdawg3.measurement.freq_center = freq_center
    
        
        self.auto_focus.periodic_focus = False
        
               
        self.confocal.remove_all_labels()
        self.auto_focus.remove_all_targets()
        
        x_shift = 0 
        y_shift = 0
        index = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 31]
        
        # Ymax_ref =9.1472
        # Xmax_ref=15.8024
        
        # Magnet.move_absolute(1, Xmax_ref)
        # time.sleep(12.0)
        # Magnet.move_absolute(2, Ymax_ref)
        # time.sleep(12.0)
            
        for nind, ncenter in enumerate(index):
            ncenter = int(ncenter)
            self.auto_focus.target_name = 'NV'+str(nind+1)
            time.sleep(1.0)
            self.auto_focus.periodic_focus = False
            
            self.confocal.y = self.sf.Centroids[ncenter][0]
            self.confocal.x = self.sf.Centroids[ncenter][1]
            
            X0=self.confocal.x
            Y0=self.confocal.y
            
            if (self.confocal.y > 50 or self.confocal.y < 0 or self.confocal.x > 50 or self.confocal.x < 0):
                continue
            self.auto_focus.submit()
            time.sleep(12.0)
            self.auto_focus.submit()
            time.sleep(12.0)
            
            file_name = file_path + '/NV' + str(nind+1)

            if not os.path.isdir(file_name):
                os.mkdir(file_name)
            else:
                continue

        # confocal scan ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
         
            self.confocal.submit()                     
            time.sleep(220)
            
            file_nv = file_name + '/image.png'
            self.confocal.save_image(file_nv)
            
            self.auto_focus.submit()
            time.sleep(12.0)
            self.auto_focus.submit()
            time.sleep(12.0)
        
        # # magnet alignment  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     
        
        # self.auto_focus.periodic_focus = False    
        # self.auto_focus.submit()
        # time.sleep(12.0)
        
        # # Magnet.move_absolute(3, distance[nfield])
        # # time.sleep(10.0)
        # # self.auto_focus.submit()
        # # time.sleep(12.0)
        
        # Magnet.move_absolute(1, Xmax_ref)
        # time.sleep(12.0)
        # Magnet.move_absolute(2, Ymax_ref)
        # time.sleep(12.0)
        
        # self._magnet_alignment('y')
        # time.sleep(1.0)
        # Magnet.move_absolute(2, self.magnet_control.start_positionY)
        # time.sleep(12.0)
        # self.magnet_control.submit()
        
        # time.sleep(3.0)
        # if self.magnet_control.state == 'error':
            # time.sleep(3.0)
            # self.magnet_control.submit()
            
        # while self.magnet_control.state != 'done':
           
            # threading.currentThread().stop_request.wait(1.0)
            # if threading.currentThread().stop_request.isSet():
                 # break
        # #time.sleep(self.magnet_control.expected_duration+30)
        # self.magnet_control.perform_fit=True
        # time.sleep(1.0)
        # self.magnet_control.perform_fit=False
        # time.sleep(1.0)
        # self.magnet_control.perform_fit=True
        # Magnet.move_absolute(2, self.magnet_control.Ymax)
   
        # if not np.isnan(self.magnet_control.Ymax):
            # if abs(Ymax_ref - self.magnet_control.Ymax) < 0.2:
                # Magnet.move_absolute(2, self.magnet_control.Ymax)
                # Ymax_ref = self.magnet_control.Ymax
                
        # time.sleep(10)
        # self.magnet_control.save_fluor_y_plot(file_name+'/y')
        
        
        # self.auto_focus.submit()
        # time.sleep(12.0)
        # self._magnet_alignment('x')
        # time.sleep(1.0)
        # Magnet.move_absolute(1, self.magnet_control.start_positionX)
        # time.sleep(10.0)
        # self.magnet_control.submit()
        # time.sleep(3.0)
        # if self.magnet_control.state == 'error':
            # time.sleep(3.0)
            # self.magnet_control.submit()
        # while self.magnet_control.state != 'done':
           
            # threading.currentThread().stop_request.wait(1.0)
            # if threading.currentThread().stop_request.isSet():
                 # break
        # #time.sleep(self.magnet_control.expected_duration+30)
        # self.magnet_control.perform_fit=True
        # time.sleep(1.0)
        # self.magnet_control.perform_fit=False
        # time.sleep(1.0)
        # self.magnet_control.perform_fit=True
        
        # if not np.isnan(self.magnet_control.Xmax):
            # if abs(Xmax_ref - self.magnet_control.Xmax) < 0.2:
                # Magnet.move_absolute(1, self.magnet_control.Xmax)
                # Xmax_ref = self.magnet_control.Xmax
              
        # Magnet.move_absolute(1, self.magnet_control.Xmax)        
        # time.sleep(10)
        # self.magnet_control.save_fluor_x_plot(file_name+'/x')  
        
             
        # # pulsed odmr --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
            time.sleep(1.0)
            self._odmr_cw()
            time.sleep(1.0)
            self.odmr.submit()   
            time.sleep(2.0)
            
            while self.odmr.state != 'done':
               
                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break
            self.odmr.remove()   
            self.odmr.state = 'idle'
            
            if(self.odmr.fit_contrast.max() < 5 or self.odmr.fit_contrast.max() > 43):
                continue
                
            freq = self.odmr.fit_frequencies[0]
            
            if(np.isnan(freq)):
                continue 
            
            time.sleep(1.0)
            self._odmr_pulsed(freq)
            time.sleep(1.0)
            self.odmr.submit()   
            time.sleep(2.0)
            
                        
            while self.odmr.state != 'done':
               
                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break
                     
            self.odmr.remove()   
            self.odmr.state = 'idle'               
            file_odmr = file_name + '/odmr_' + string.replace(str(self.odmr.magnetic_field)[0:5], '.', 'd') + 'G'
            self.odmr.save_all(file_odmr)
            time.sleep(5.0)  
            
            self.auto_focus.periodic_focus = True
            
            if(len(self.odmr.fit_frequencies) > 2 or self.odmr.fit_contrast.max() < 5 or self.odmr.fit_contrast.max() > 43):
                continue
                
            freq = self.odmr.fit_frequencies[0]
            freq_center = freq-0.1e+9
         
        
            
            if(np.isnan(freq)):
                continue    
            
        # # Rabi ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            self.auto_focus.periodic_focus = True
            self.auto_focus.focus_interval = 3
            
            self.psawg2.measurement.freq = freq             
            self.psawg2.measurement.freq_center = freq_center     
            self.psawg2.measurement.load()
            time.sleep(5.0)
            self.psawg2.fit = RabiFit_phase()
            time.sleep(5.0)
        
            while self.psawg2.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
            
            self.psawg2.measurement.submit()
           
            while self.psawg2.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                    
             
            self.psawg2.measurement.progress = 0
            self.psawg2.measurement.elapsed_sweeps = 0
         
            file_rabi = file_name + '/Rabi' + '_power_' + str(power) + '_contrast_' + string.replace(str(self.psawg2.fit.contrast[0])[0:4], '.', 'd') 
            self.psawg2.save_all(file_rabi)
            time.sleep(10.0)
            
            self.psawg2.measurement.remove()
            self.psawg2.measurement.state = 'idle'            
            time.sleep(5.0)
            
            half_pi = self.psawg2.fit.t_pi2[0]
            pi = self.psawg2.fit.t_pi[0]
            
            condition = np.isnan(half_pi) or self.psawg2.fit.contrast[0] < 15
            if condition:
                continue
                
            self.pdawg.measurement.freq = freq
            self.pdawg.measurement.freq_center = freq_center
            self.pdawg.measurement.pi2_1 = half_pi
            self.pdawg.measurement.pi_1 = pi
            self.pdawg.measurement.rabi_contrast=self.psawg2.fit.contrast[0]
            
            self.pdawg.measurement.load()
            time.sleep(20.0)
            
            while self.pdawg.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
            
            self.pdawg.measurement.submit()
            
            while self.pdawg.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                    
                 if  self.pdawg.measurement.state == 'error':
                    time.sleep(4)
                    self.pdawg.measurement.resubmit()
         
                 
            self.pdawg.measurement.progress = 0
            self.pdawg.measurement.elapsed_sweeps = 0                
                    
            file_hahn = file_name + '/T2'
            self.pdawg.save_line_plot(file_hahn + '.png')
            self.pdawg.save(file_hahn + '.pyd')
            self.pdawg.save_processed_plot(file_hahn + '_ProcPlot.png')
            time.sleep(5.0)
            self.pdawg.measurement.remove()
            self.pdawg.measurement.state = 'idle'
            time.sleep(5.0)       
            
            self.pdawg2.measurement.freq = freq
            self.pdawg2.measurement.freq_center = freq_center
            self.pdawg2.measurement.pi_1 = pi
            self.pdawg2.measurement.rabi_contrast=self.psawg2.fit.contrast[0]
            
            self.pdawg2.measurement.load()
            time.sleep(15.0)
            
            self.auto_focus.focus_interval = 5
            
            while self.pdawg2.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
            
            self.pdawg2.measurement.submit()
            
            while self.pdawg2.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                 if  self.pdawg2.measurement.state == 'error':
                    time.sleep(4)
                    self.pdawg2.measurement.resubmit()   
                    
            self.pdawg2.measurement.progress = 0
            self.pdawg2.measurement.elapsed_sweeps = 0               
            file_T1 = file_name + '/T1'
            self.pdawg2.save_line_plot(file_T1 + '.png')
            self.pdawg2.save(file_T1 + '.pyd')
            self.pdawg2.save_processed_plot(file_T1 + '_ProcPlot.png')
            time.sleep(10.0)
            self.pdawg2.measurement.remove()
            self.pdawg2.measurement.state = 'idle'
            time.sleep(10.0)   
        
        #============================================================================================================================================================================================================
        
            self.pdawg3.measurement.freq_center = freq_center
            self.pdawg3.measurement.freq = freq
            self.pdawg3.measurement.pi2_1 = half_pi
            self.pdawg3.measurement.pi_1 = pi
            self.pdawg3.measurement.pulse_num = 12
            self.pdawg3.measurement.rabi_contrast=self.psawg2.fit.contrast[0]
            
            self.pdawg3.measurement.load()
            time.sleep(50.0)
            
            self.auto_focus.focus_interval = 2
            
            while self.pdawg3.measurement.reload == True:  
                threading.currentThread().stop_request.wait(1.0)
            
            self.pdawg3.measurement.submit()
            
            while self.pdawg3.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                    
                 if  self.pdawg3.measurement.state == 'error':
                     time.sleep(4)
                     self.pdawg3.measurement.resubmit()
                 

            self.pdawg3.measurement.progress = 0
            self.pdawg3.measurement.elapsed_sweeps = 0               
            file_XY = file_name + '/xy8-' + str(self.pdawg3.measurement.pulse_num)
            self.pdawg3.save_line_plot(file_XY + '.png')
            self.pdawg3.save_processed_plot(file_XY + '_ProcPlot.png')
            self.pdawg3.save(file_XY + '.pyd')
            time.sleep(10.0)
            self.pdawg3.measurement.remove()
            self.pdawg3.measurement.state = 'idle'
            time.sleep(10.0)   
        
        #===============================================================================================================================================================================================================
        
            self.pdawg3.measurement.freq_center = freq_center
            self.pdawg3.measurement.freq = freq
            self.pdawg3.measurement.pi2_1 = half_pi
            self.pdawg3.measurement.pi_1 = pi
            self.pdawg3.measurement.pulse_num = 24
            self.pdawg3.measurement.rabi_contrast=self.psawg2.fit.contrast[0]
            
            self.pdawg3.measurement.load()
            time.sleep(50.0)
            
            self.auto_focus.focus_interval = 2
            
            while self.pdawg3.measurement.reload == True:  
                threading.currentThread().stop_request.wait(1.0)
            
            self.pdawg3.measurement.submit()
            
            while self.pdawg3.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                    
                 if  self.pdawg3.measurement.state == 'error':
                     time.sleep(4)
                     self.pdawg3.measurement.resubmit()
                 

            self.pdawg3.measurement.progress = 0
            self.pdawg3.measurement.elapsed_sweeps = 0               
            file_XY = file_name + '/xy8-' + str(self.pdawg3.measurement.pulse_num)
            self.pdawg3.save_line_plot(file_XY + '.png')
            self.pdawg3.save_processed_plot(file_XY + '_ProcPlot.png')
            self.pdawg3.save(file_XY + '.pyd')
            time.sleep(10.0)
            self.pdawg3.measurement.remove()
            self.pdawg3.measurement.state = 'idle'
            time.sleep(10.0)   
        
        #===============================================================================================================================================================================================================
        
            self.pdawg3.measurement.freq_center = freq_center
            self.pdawg3.measurement.freq = freq
            self.pdawg3.measurement.pi2_1 = half_pi
            self.pdawg3.measurement.pi_1 = pi
            self.pdawg3.measurement.pulse_num = 30
            self.pdawg3.measurement.rabi_contrast=self.psawg2.fit.contrast[0]
            
            self.pdawg3.measurement.load()
            time.sleep(50.0)
            
            self.auto_focus.focus_interval = 2
            
            while self.pdawg3.measurement.reload == True:  
                threading.currentThread().stop_request.wait(1.0)
            
            self.pdawg3.measurement.submit()
            
            while self.pdawg3.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                    
                 if  self.pdawg3.measurement.state == 'error':
                     time.sleep(4)
                     self.pdawg3.measurement.resubmit()
                 

            self.pdawg3.measurement.progress = 0
            self.pdawg3.measurement.elapsed_sweeps = 0               
            file_XY = file_name + '/xy8-' + str(self.pdawg3.measurement.pulse_num)
            self.pdawg3.save_line_plot(file_XY + '.png')
            self.pdawg3.save_processed_plot(file_XY + '_ProcPlot.png')
            self.pdawg3.save(file_XY + '.pyd')
            time.sleep(10.0)
            self.pdawg3.measurement.remove()
            self.pdawg3.measurement.state = 'idle'
            time.sleep(10.0)   
        
        #===============================================================================================================================================================================================================
        
            self.auto_focus.periodic_focus = False
            
            self.auto_focus.submit()
            time.sleep(12.0)
            
            x_shift= self.confocal.x - X0
            y_shift= self.confocal.y - Y0
            
            for Ncent in index:
                self.sf.Centroids[Ncent][0] = self.sf.Centroids[Ncent][0] + y_shift
                self.sf.Centroids[Ncent][1] = self.sf.Centroids[Ncent][1] + x_shift
                
            self.auto_focus.periodic_focus = False
            pg.Night()
            print 'I have finished'
                        
                
