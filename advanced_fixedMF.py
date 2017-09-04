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

pg = ha.PulseGenerator()

class AdvancedFixedMF(ManagedJob):

    def __init__(self, auto_focus, confocal, mag_al, odmr=None,psrabi=None,psawg=None,sensing=None,pdawg=None,pdawg2=None,pdawg3=None,sf=None,gs=None):
        super(AdvancedFixedMF, self).__init__()     
        
        self.auto_focus=auto_focus   
        self.confocal=confocal
        self.magnet_control = mag_al
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
            
    def _magnet_alignment(self, direction):
        
        self.magnet_control.choosed_align = 'fluoresence'
                         
        if direction == 'y':                 
            self.magnet_control.choosed_scan =  'fy_scan'   
        elif direction == 'x':   
            self.magnet_control.choosed_scan =  'fx_scan'   
            
        self.magnet_control.fitting_func = 'gaussian'
             
        self.magnet_control.start_positionX = 5.7
        self.magnet_control.end_positionX = 5.95
        self.magnet_control.stepX = 0.01
          
        self.magnet_control.start_positionY = 5.8
        self.magnet_control.end_positionY = 6.8
        self.magnet_control.stepY = 0.04
        
        self.magnet_control.number_of_resonances=1
        self.magnet_control.perform_fit=False
        
        self.magnet_control.periodic_focus = False
        self.magnet_control.Npoint_auto_focus = 10
        
        
    def _alignment_xy(self, Xmax_ref, Ymax_ref, file_name):
    
        self.auto_focus.periodic_focus = False    
        self.auto_focus.submit()
        time.sleep(20.0)
             
        self.magnet_control.perform_fit=False
        time.sleep(1.0)
                    
        # Magnet.move_absolute(1, Xmax_ref)
        # time.sleep(12.0)
        # Magnet.move_absolute(2, Ymax_ref)
        # time.sleep(12.0)
        
        self._magnet_alignment('y')
        time.sleep(5.0)
        #Magnet.move_absolute(2, self.magnet_control.start_positionY)
        #time.sleep(12.0)
        self.magnet_control.submit()
        
        time.sleep(self.magnet_control.expected_duration*2)
        
        if self.magnet_control.state == 'error':
            time.sleep(3.0)
            self.magnet_control.submit()
            time.sleep(self.magnet_control.expected_duration*2)
            
        # while self.magnet_control.state != 'done':
           
            # threading.currentThread().stop_request.wait(1.0)
            # if threading.currentThread().stop_request.isSet():
                 # break
                         
        self.magnet_control.perform_fit=True
        time.sleep(1.0)
        self.magnet_control.perform_fit=False
        time.sleep(1.0)
        self.magnet_control.perform_fit=True
        Magnet.move_absolute(2, self.magnet_control.Ymax)
        time.sleep(12)
           
        if not np.isnan(self.magnet_control.Ymax):
            if abs(Ymax_ref - self.magnet_control.Ymax) < 0.2:
                #Magnet.move_absolute(2, self.magnet_control.Ymax)
                Ymax_ref = self.magnet_control.Ymax
                
        time.sleep(10)
        self.magnet_control.save_fluor_y_plot(file_name+'/y')
        self.magnet_control.perform_fit=False
        time.sleep(1.0)
        
        self.auto_focus.submit()
        time.sleep(20.0)
        self._magnet_alignment('x')
        time.sleep(1.0)
        # Magnet.move_absolute(1, self.magnet_control.start_positionX)
        # time.sleep(10.0)
        self.magnet_control.submit()
        
        time.sleep(3.0)
        if self.magnet_control.state == 'error':
            time.sleep(3.0)
            self.magnet_control.submit()
        # while self.magnet_control.state != 'done':
           
            # threading.currentThread().stop_request.wait(1.0)
            # if threading.currentThread().stop_request.isSet():
                 # break
        time.sleep(self.magnet_control.expected_duration*2)
        self.magnet_control.perform_fit=True
        time.sleep(1.0)
        self.magnet_control.perform_fit=False
        time.sleep(1.0)
        self.magnet_control.perform_fit=True
        
        if not np.isnan(self.magnet_control.Xmax):
            if abs(Xmax_ref - self.magnet_control.Xmax) < 0.2:
                #Magnet.move_absolute(1, self.magnet_control.Xmax)
                Xmax_ref = self.magnet_control.Xmax
              
        Magnet.move_absolute(1, self.magnet_control.Xmax)        
        time.sleep(10)
        self.magnet_control.save_fluor_x_plot(file_name+'/x') 
        self.magnet_control.perform_fit=False
        time.sleep(1.0)        
        
        return Xmax_ref, Ymax_ref
        
    def _odmr_cw(self):
    
        self.odmr.pulsed = False
        self.odmr.perform_fit = True
      
        self.odmr.power = -22
        self.odmr.stop_time = 20
        
        self.odmr.frequency_begin= 1.43e+09
        self.odmr.frequency_end = 1.53e+09
        self.odmr.frequency_delta = 1.0e6
        self.odmr.number_of_resonances = 1
        self.odmr.threshold = -60
                 
    def _odmr_pulsed(self, freq):
        
        self.odmr.pulsed = True
        self.odmr.perform_fit = True
      
        self.odmr.power_p = -30
        self.odmr.t_pi = 528
        self.odmr.stop_time = 40
        
        self.odmr.frequency_begin_p = freq - 0.005e+09
        self.odmr.frequency_end_p = freq + 0.005e+09
        self.odmr.frequency_delta_p = 1.0e5
        self.odmr.number_of_resonances = 1
        self.odmr.threshold = -60
        
    def _apply_parameters_XY16(self, t1, freq_center):
        self.pdawg3.measurement.tau_begin = t1
        self.pdawg3.measurement.tau_end = t1+21
        self.pdawg3.measurement.freq_center = freq_center
        
    def _apply_parameters_XY8(self, t1, freq_center):
        self.pdawg2.measurement.tau_begin = t1
        self.pdawg2.measurement.tau_end = t1 + 21
        self.pdawg2.measurement.freq_center = freq_center
        
    def _odmr_rabi(self, file_name):
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
        
        # if(self.odmr.fit_contrast.max() < 5 or self.odmr.fit_contrast.max() > 43):
            # continue
            
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
        
        # if(len(self.odmr.fit_frequencies) > 2 or self.odmr.fit_contrast.max() < 5 or self.odmr.fit_contrast.max() > 43):
            # continue
            
        freq = self.odmr.fit_frequencies[0]
        
        # if(np.isnan(freq)):
            # continue    
            
        self.auto_focus.periodic_focus = True
        
        self.psawg.measurement.freq = freq    
        freq_center = freq-0.1e+9
         
        self.psawg.measurement.freq_center = freq_center         
        self.psawg.measurement.load()
        time.sleep(5.0)
        
        from analysis.pulsedawgan import RabiFit_phase   
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
     
        file_rabi = file_name + '/Rabi' + '_power_' + str(self.psawg.measurement.power) + '_contrast_' + string.replace(str(self.psawg.fit.contrast[0])[0:4], '.', 'd') 
        self.psawg.save_all(file_rabi)
        time.sleep(10.0)
        
        self.psawg.measurement.remove()
        self.psawg.measurement.state = 'idle'            
        time.sleep(5.0)
        
        half_pi = self.psawg.fit.t_pi2[0]
        pi = self.psawg.fit.t_pi[0]
        
        contrast=self.psawg.fit.contrast[0]
        
        # condition = np.isnan(half_pi) or self.psawg.fit.contrast[0] < 15
        # if condition:
            # continue
            
        return freq, freq_center, half_pi, pi, contrast

    def _run(self):
        
        file_path ='D:/data/BN_NMR/Isoya sample/before etching/reference 5 keV lowest dose/autopilot'
        
        self.confocal.resolution = 200
        power=-4
        vpp=0.6
        
        from measurements.pulsed_awg import Rabi 
        self.psawg.measurement = Rabi()
        from analysis.pulsedawgan import RabiFit_phase   
        self.psawg.fit = RabiFit_phase()
        

        self.psawg.measurement.tau_begin = 300
        self.psawg.measurement.tau_end = 1000
        self.psawg.measurement.tau_delta = 10
        self.psawg.measurement.sweeps = 0.8e5 
        self.psawg.measurement.power = power   
        self.psawg.measurement.vpp = vpp
        
        self.psawg.fit = RabiFit_phase()
        time.sleep(5.0)          
        
        from measurements.shallow_NV import Hahn
        self.pdawg.measurement = Hahn()
        self.pdawg.measurement.power = power
        self.pdawg.measurement.vpp = 0.6
        
        self.pdawg.measurement.tau_begin = 300
        self.pdawg.measurement.tau_end = 20000
        self.pdawg.measurement.tau_delta = 1000
        self.pdawg.measurement.sweeps = 1.00e6
        
        from measurements.shallow_NV import T1
        self.pdawg2.measurement = T1()
        self.pdawg2.measurement.power = power
       
        self.pdawg2.measurement.vpp = 0.6
        self.pdawg2.measurement.tau_begin = 300
        self.pdawg2.measurement.tau_end = 1000000
        self.pdawg2.measurement.tau_delta = 50000
        self.pdawg2.measurement.sweeps = 0.5e6
        
        from measurements.shallow_NV import XY16_Ref
        self.pdawg3.measurement = XY16_Ref()
        
        self.pdawg3.measurement.tau_delta = 1
        self.pdawg3.measurement.sweeps = 1.0e6
        self.pdawg3.measurement.vpp = vpp
        self.pdawg3.measurement.power = power
        #self.pdawg3.measurement.pulse_num = 5
        
        # from measurements.shallow_NV import XY8_Ref
        # self.pdawg2.measurement = XY8_Ref()
        
        # self.pdawg2.measurement.tau_delta = 1
        # self.pdawg2.measurement.sweeps = 1.0e6
        # self.pdawg2.measurement.vpp = vpp
        # self.pdawg2.measurement.power = power
        # self.pdawg2.measurement.pulse_num = 2*self.pdawg.measurement.pulse_num
        
        self.auto_focus.periodic_focus = False
        
               
        self.confocal.remove_all_labels()
        self.auto_focus.remove_all_targets()
        
        x_shift = 0 
        y_shift = 0
        index = [7,8, 11, 14, 17, 19, 27, 30, 36, 38, 39, 40, 44, 47, 48, 53, 55, 57, 58, 60, 61, 63, 67, 69]
        
        Ymax_ref =6.4624
        Xmax_ref=6.0913
                
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
            time.sleep(230)
            
            file_nv = file_name + '/image.png'
            self.confocal.save_image(file_nv)
            
            self.auto_focus.submit()
            time.sleep(12.0)
            self.auto_focus.submit()
            time.sleep(12.0)
            
            # magnet alignment --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
         
            #Xmax_ref, Ymax_ref = alignment_xy(Xmax_ref, Ymax_ref)
            
                 
            # pulsed odmr & Rabi -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
            freq, freq_center, half_pi, pi, contrast = self._odmr_rabi(file_name)
            
            
            self.pdawg.measurement.freq_center = freq_center    
            self.pdawg.measurement.freq = freq
            self.pdawg.measurement.pi2_1 = half_pi
            self.pdawg.measurement.pi_1 = pi
            self.pdawg.measurement.rabi_contrast=self.psawg.fit.contrast[0]
            
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
            
            self.pdawg2.measurement.freq_center = freq_center
            self.pdawg2.measurement.freq = freq
            
            self.pdawg2.measurement.pi_1 = pi
            self.pdawg2.measurement.rabi_contrast=self.psawg.fit.contrast[0]
            
            self.pdawg2.measurement.load()
            time.sleep(15.0)
            
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
            
            t1=57
            
            for m in range(1):
            
                          
                self.auto_focus.periodic_focus = True
            
                self._apply_parameters_XY16(t1, freq_center)
                
                self.pdawg3.measurement.freq = freq
                self.pdawg3.measurement.pi2_1 = half_pi
                self.pdawg3.measurement.pi_1 = pi
                self.pdawg3.measurement.rabi_contrast=self.psawg.fit.contrast[0]
                self.pdawg3.measurement.pulse_num = 20+m
                
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
                file_XY = file_name + '/xy16-' + str(self.pdawg3.measurement.pulse_num)+'_'+str(t1)+'-'+str(t1+20)+'ns'
                self.pdawg3.save_line_plot(file_XY + '.png')
                self.pdawg3.save_processed_plot(file_XY + '_ProcPlot.png')
                self.pdawg3.save(file_XY + '.pyd')
                time.sleep(10.0)
                self.pdawg3.measurement.remove()
                self.pdawg3.measurement.state = 'idle'
                time.sleep(10.0)   
                
                # self._apply_parameters_XY8(t1, freq_center)
                
                # self.pdawg2.measurement.freq = freq
                # self.pdawg2.measurement.pi2_1 = half_pi
                # self.pdawg2.measurement.pi_1 = pi
                # self.pdawg2.measurement.rabi_contrast=self.psawg.fit.contrast[0]
                
                # self.pdawg2.measurement.load()
                # time.sleep(50.0)
                
                # while self.pdawg2.measurement.reload == True:  
                    # threading.currentThread().stop_request.wait(1.0)
                
                # self.pdawg2.measurement.submit()
                
                # while self.pdawg2.measurement.state != 'done':
                     # threading.currentThread().stop_request.wait(1.0)
                     # if threading.currentThread().stop_request.isSet():
                        # break
                        
                     # if  self.pdawg2.measurement.state == 'error':
                         # time.sleep(4)
                         # self.pdawg2.measurement.resubmit()
                 

                # self.pdawg2.measurement.progress = 0
                # self.pdawg2.measurement.elapsed_sweeps = 0               
                # file_XY = file_name + '/xy8-' + str(self.pdawg2.measurement.pulse_num)+'_'+str(t1)+'-'+str(t1+20)+'ns'
                # self.pdawg2.save_line_plot(file_XY + '.png')
                # self.pdawg2.save_processed_plot(file_XY + '_ProcPlot.png')
                # self.pdawg2.save(file_XY + '.pyd')
                # time.sleep(10.0)
                # self.pdawg2.measurement.remove()
                # self.pdawg2.measurement.state = 'idle'
                # time.sleep(10.0)   
                
                
                # Check contrast
                
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
                time.sleep(10.0)
                
                self.psawg.measurement.remove()
                self.psawg.measurement.state = 'idle'            
                time.sleep(5.0)
                
                contrast2=self.psawg.fit.contrast[0]
                
                self.auto_focus.periodic_focus = False
                
                if contrast-contrast2 < 5:
                    print 'Yahoo!'
                    #t1=t1+20
                else:
                    Xmax_ref, Ymax_ref = self._alignment_xy(Xmax_ref, Ymax_ref, file_name)   
                    freq, freq_center, half_pi, pi, contrast = self._odmr_rabi(file_name)
                    t1=t1+20
                    
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
            
  
                        
                
