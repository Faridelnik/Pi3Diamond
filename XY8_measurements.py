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

Magnet=smc.SMC()

from tools.emod import ManagedJob
from tools.utility import GetSetItemsHandler, GetSetItemsMixin

import hardware.api as ha

class XY8_N(ManagedJob):

    def __init__(self, auto_focus, confocal, mag_al, odmr=None,psrabi=None,psawg2=None,sensing=None, pdawg3=None,sf=None,gs=None):
        super(XY8_N, self).__init__()     
        
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
             
        self.magnet_control.start_positionX = 10.4
        self.magnet_control.end_positionX = 11.6
        self.magnet_control.stepX = 0.03
          
        self.magnet_control.start_positionY = 7.9
        self.magnet_control.end_positionY = 8.2
        self.magnet_control.stepY = 0.01
        
        self.magnet_control.number_of_resonances=1
        self.magnet_control.perform_fit=False
                 
    def _odmr_pulsed(self):
        
        self.odmr.pulsed = True
        self.odmr.perform_fit = True
      
        self.odmr.power_p = -30
        self.odmr.t_pi = 590
        self.odmr.stop_time = 40
        
        self.odmr.frequency_begin_p = 1.163e+09
        self.odmr.frequency_end_p = 1.173e+09
        self.odmr.frequency_delta_p = 1.0e5
        self.odmr.number_of_resonances = 1
        self.odmr.threshold = -50

    def _run(self):
        
        file_path ='D:/data/Membrane 191 p+/PMMA'
        os.path.exists(file_path)
        
        self.confocal.resolution = 200
        power=0
        #freq_center = 1086401000.0
        vpp=0.6
        
        from measurements.pulsed_awg import Rabi 
        self.psawg2.measurement = Rabi()
        from analysis.pulsedawgan import RabiFit_phase   
        self.psawg2.fit = RabiFit_phase()
        

        self.psawg2.measurement.tau_begin = 300
        self.psawg2.measurement.tau_end = 1000
        self.psawg2.measurement.tau_delta = 10
        self.psawg2.measurement.sweeps = 0.5e5 
        self.psawg2.measurement.power = power   
        self.psawg2.measurement.vpp = vpp
        #self.psawg.measurement.freq_center = freq_center

        self.psawg2.fit = RabiFit_phase()
        time.sleep(5.0)          
        
        from measurements.shallow_NV import XY8_Ref
        self.pdawg3.measurement = XY8_Ref()
                
        self.pdawg3.measurement.tau_begin = 33
        self.pdawg3.measurement.tau_end = 54
        self.pdawg3.measurement.tau_delta = 1
        self.pdawg3.measurement.sweeps = 1.0e6
        self.pdawg3.measurement.vpp = vpp
        self.pdawg3.measurement.power = power
        #self.pdawg3.measurement.freq_center = freq_center
    
        
        self.auto_focus.periodic_focus = False
        
               
        self.confocal.remove_all_labels()
        self.auto_focus.remove_all_targets()
        
        x_shift = 0 
        y_shift = 0
        index = [1, 2, 3, 4, 5, 6, 10, 16, 17, 19, 23, 25, 27, 28, 29, 30, 32, 33, 34, 36, 43, 44, 45, 48, 50]
        
        # Ymax_ref =8.1
        # Xmax_ref=11
        
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
            #else:
                #continue

            # confocal scan ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
             
            self.confocal.submit()                     
            time.sleep(250)
            
            file_nv = file_name + '/image.png'
            self.confocal.save_image(file_nv)
            
            # # magnet control --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
         
            
            # # self.auto_focus.periodic_focus = False    
            # self.auto_focus.submit()
            # time.sleep(12.0)
            # self.auto_focus.submit()
            # time.sleep(12.0)
            
            # # Magnet.move_absolute(1, Xmax_ref)
            # # time.sleep(12.0)
            # # Magnet.move_absolute(2, Ymax_ref)
            # # time.sleep(12.0)
            
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
                     
            # time.sleep(self.magnet_control.expected_duration+30)
            # self.magnet_control.perform_fit=True
            # time.sleep(1.0)
            
            # Magnet.move_absolute(2, self.magnet_control.Ymax)
       
            # # if not np.isnan(self.magnet_control.Ymax):
                # # if abs(Ymax_ref - self.magnet_control.Ymax) < 0.2:
                    # # Magnet.move_absolute(2, self.magnet_control.Ymax)
                    # # Ymax_ref = self.magnet_control.Ymax
                    
            # time.sleep(10)
            # self.magnet_control.save_fluor_y_plot(file_name+'/y')
            
            # self.magnet_control.perform_fit=False
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
            # time.sleep(self.magnet_control.expected_duration+30)
            # self.magnet_control.perform_fit=True
            # time.sleep(1.0)
                        
            # # if not np.isnan(self.magnet_control.Xmax):
                # # if abs(Xmax_ref - self.magnet_control.Xmax) < 0.2:
                    # # Magnet.move_absolute(1, self.magnet_control.Xmax)
                    # # Xmax_ref = self.magnet_control.Xmax
                  
            # Magnet.move_absolute(1, self.magnet_control.Xmax)        
            # time.sleep(10)
            # self.magnet_control.save_fluor_x_plot(file_name+'/x')  
            
               
            # pulsed odmr --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
           
            self.auto_focus.submit()
            time.sleep(12.0)
            
            time.sleep(1.0)
            self._odmr_pulsed()
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
            
            if(len(self.odmr.fit_frequencies) > 2 or self.odmr.fit_contrast.max() < 8 or self.odmr.fit_contrast.max() > 43):
                continue
                
            freq = self.odmr.fit_frequencies[0]
            
            if(np.isnan(freq)):
                continue    
            # Rabi ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            self.auto_focus.periodic_focus = True
            
            self.psawg2.measurement.freq = freq             
                  
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
            
            condition = np.isnan(half_pi) or self.psawg2.fit.contrast[0] < 27
            if condition:
                continue
                
            pulse_num = [12]
            self.pdawg3.measurement.freq_center=self.psawg2.measurement.freq_center
            
            for i in pulse_num:
            
                self.pdawg3.measurement.rabi_contrast = self.psawg2.fit.contrast[0]
                self.pdawg3.measurement.freq = freq
                self.pdawg3.measurement.pi2_1 = half_pi
                self.pdawg3.measurement.pi_1 = pi
                self.pdawg3.measurement.pulse_num = i
                
                self.pdawg3.measurement.load()
                time.sleep(50.0)
                
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
                self.pdawg3.save(file_XY + '.pyd')
                time.sleep(10.0)
                self.pdawg3.measurement.remove()
                self.pdawg3.measurement.state = 'idle'
                time.sleep(10.0)   
            
            self.auto_focus.periodic_focus = False
            
            self.auto_focus.submit()
            time.sleep(12.0)
            
            x_shift= self.confocal.x - X0
            y_shift= self.confocal.y - Y0
            
            for Ncent in index:
                self.sf.Centroids[Ncent][0] = self.sf.Centroids[Ncent][0] + y_shift
                self.sf.Centroids[Ncent][1] = self.sf.Centroids[Ncent][1] + x_shift
            
  
                        
                
