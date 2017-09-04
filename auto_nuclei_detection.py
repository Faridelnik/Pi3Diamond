from tools.utility import edit_singleton
from datetime import date
import os

from traits.api import Bool
import imp
import numpy as np
import time
import threading
import hardware.SMC_controller as smc

Magnet=smc.SMC()

from tools.emod import ManagedJob
from tools.utility import GetSetItemsHandler, GetSetItemsMixin

import hardware.api as ha

class Auto_decoupling(ManagedJob):

    def __init__(self, auto_focus, confocal, mag_al, odmr=None,psrabi=None,psawg=None,sensing=None,pdawg=None,pdawg2=None,pdawg3=None,sf=None,gs=None):
        super(Auto_decoupling, self).__init__()     
        
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
             
        self.magnet_control.start_positionX = 10.5
        self.magnet_control.end_positionX = 12
        self.magnet_control.stepX = 0.05
          
        self.magnet_control.start_positionY = 7.3
        self.magnet_control.end_positionY = 8
        self.magnet_control.stepY = 0.01
        
        self.magnet_control.number_of_resonances=1
        self.magnet_control.perform_fit=False
                 
    def _odmr_pulsed(self,fstart,fend, tpi):
        
        self.odmr.pulsed = True
        self.odmr.perform_fit = True
      
        self.odmr.power_p = -20
        self.odmr.t_pi = tpi
        self.odmr.stop_time = 50
        
        self.odmr.frequency_begin_p = fstart
        self.odmr.frequency_end_p = fend
        self.odmr.frequency_delta_p = 1.5e5
        self.odmr.number_of_resonances = 1
        self.odmr.threshold = -70

    def _run(self):
        
        file_path ='D:/data/protonNMR/membrane_2/micelle3/L11-13/direction1/'
        os.path.exists(file_path)
        
        self.confocal.resolution = 150
        
        from measurements.pulsed_awg import Rabi 
        self.psawg.measurement = Rabi()
        from analysis.pulsedawgan import RabiFit_phase   
        self.psawg.fit = RabiFit_phase()
        

        self.psawg.measurement.tau_begin = 10
        self.psawg.measurement.tau_end = 1000
        self.psawg.measurement.tau_delta = 10
        self.psawg.measurement.sweeps = 0.5e5       

        self.psawg.fit = RabiFit_phase()
        time.sleep(5.0)    
        
        
        
        distance = [11.5, 11.8, 12.0, 12.2, 12.35]
        fstart = [1.369e9, 1.264e9,1.188e9, 1.107e9, 1.045e9]
        fend = [1.389e9, 1.284e9, 1.208e9, 1.128e9, 1.065e9]
        tpi = [1150, 1690, 1000, 1180, 1473]
        power = [2, 9, 1 , 7 , 11]
        vpp = [1, 0.8, 0.6, 0.75, 1]
        freq_center = [1.28e9, 1.17e9, 1.10e9, 1.02e9, 0.95e9] 
        Nrep = [8, 8, 10, 42]
        file_list = ['532 Gauss H1, F19', '570 Gauss H1, F19', '597 Gauss H1, F19', '626 Gauss H1, F19', '648 Gauss H1, F19']
        
        
        
        # from measurements.shallow_NV import Hahn
        # self.pdawg.measurement = Hahn()
        # self.pdawg.measurement.power = power
        # self.pdawg.measurement.freq_center = freq_center
        # self.pdawg.measurement.vpp = 0.6
        
        # self.pdawg.measurement.tau_begin = 300
        # self.pdawg.measurement.tau_end = 8000
        # self.pdawg.measurement.tau_delta = 300
        # self.pdawg.measurement.sweeps = 3.0e5
        
        # from measurements.shallow_NV import T1
        # self.pdawg2.measurement = T1()
        # self.pdawg2.measurement.power = power
       
        # self.pdawg2.measurement.vpp = 0.6
        # self.pdawg2.measurement.freq_center = freq_center
        # self.pdawg2.measurement.tau_begin = 300
        # self.pdawg2.measurement.tau_end = 600000
        # self.pdawg2.measurement.tau_delta = 50000
        # self.pdawg2.measurement.sweeps = 2.0e5
        
        from measurements.shallow_NV import XY8_Ref
        self.pdawg3.measurement = XY8_Ref()
        
        
        
        self.pdawg3.measurement.tau_begin = 40
        self.pdawg3.measurement.tau_end = 64
        self.pdawg3.measurement.tau_delta = 1
        self.pdawg3.measurement.sweeps = 5.0e5
        
        
        self.auto_focus.periodic_focus = False
        
               
        self.confocal.remove_all_labels()
        self.auto_focus.remove_all_targets()
        
        x_shift = 0 
        y_shift = 0
        index = [22, 23, 26, 36]
        
        Ymax_ref =7.5744
        Xmax_ref=11.2122
        
        # Magnet.move_absolute(1, Xmax_ref)
        # time.sleep(12.0)
        # Magnet.move_absolute(2, Ymax_ref)
        # time.sleep(12.0)
        
        for nfield in range(4):
            self.psawg.measurement.vpp = vpp[nfield]
            self.psawg.measurement.power = power[nfield]
            self.psawg.measurement.freq_center = freq_center[nfield]
            
            self.pdawg3.measurement.vpp = vpp[nfield]
            self.pdawg3.measurement.power = power[nfield]
            self.pdawg3.measurement.freq_center = freq_center[nfield]
            
            
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
                
                file_name = file_path + file_list[nfield] + '/NV' + str(nind+1)
       
                if not os.path.isdir(file_name):
                    os.mkdir(file_name)
                #else:
                    #continue
     
                # confocal scan ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
                self.confocal.resolution = 100   
                self.confocal.submit()                     
                time.sleep(82)
                
                file_nv = file_name + '/image.png'
                self.confocal.save_image(file_nv)
                
                '''
                self.auto_focus.submit()
                time.sleep(12.0)

                 
                self.auto_focus.periodic_focus = True
                
                # pulsed odmr --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
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
                file_odmr = file_name + '/odmr'
                self.odmr.save_all(file_odmr)
                time.sleep(5.0)  
                
                if(len(self.odmr.fit_frequencies) > 2 or self.odmr.fit_contrast.max() < 5 or self.odmr.fit_contrast.max() > 43):
                    continue
                    
                freq = self.odmr.fit_frequencies[0]
                
                if(np.isnan(freq)):
                    continue
                    '''
                self.auto_focus.periodic_focus = False    
                self.auto_focus.submit()
                time.sleep(12.0)
                
                Magnet.move_absolute(3, distance[nfield])
                time.sleep(10.0)
                self.auto_focus.submit()
                time.sleep(12.0)
                
                Magnet.move_absolute(1, Xmax_ref)
                time.sleep(12.0)
                Magnet.move_absolute(2, Ymax_ref)
                time.sleep(12.0)
                
                self._magnet_alignment('y')
                time.sleep(1.0)
                Magnet.move_absolute(2, self.magnet_control.start_positionY)
                time.sleep(12.0)
                self.magnet_control.submit()
                
                time.sleep(3.0)
                if self.magnet_control.state == 'error':
                    time.sleep(3.0)
                    self.magnet_control.submit()
                    
                while self.magnet_control.state != 'done':
                   
                    threading.currentThread().stop_request.wait(1.0)
                    if threading.currentThread().stop_request.isSet():
                         break
                #time.sleep(self.magnet_control.expected_duration+30)
                self.magnet_control.perform_fit=True
                time.sleep(1.0)
                self.magnet_control.perform_fit=False
                time.sleep(1.0)
                self.magnet_control.perform_fit=True
                Magnet.move_absolute(2, self.magnet_control.Ymax)
                '''
                if not np.isnan(self.magnet_control.Ymax):
                    if abs(Ymax_ref - self.magnet_control.Ymax) < 0.2:
                        Magnet.move_absolute(2, self.magnet_control.Ymax)
                        Ymax_ref = self.magnet_control.Ymax
                        '''
                time.sleep(10)
                self.magnet_control.save_fluor_y_plot(file_name+'/y')
                
                
                self.auto_focus.submit()
                time.sleep(12.0)
                self._magnet_alignment('x')
                time.sleep(1.0)
                Magnet.move_absolute(1, self.magnet_control.start_positionX)
                time.sleep(10.0)
                self.magnet_control.submit()
                time.sleep(3.0)
                if self.magnet_control.state == 'error':
                    time.sleep(3.0)
                    self.magnet_control.submit()
                while self.magnet_control.state != 'done':
                   
                    threading.currentThread().stop_request.wait(1.0)
                    if threading.currentThread().stop_request.isSet():
                         break
                #time.sleep(self.magnet_control.expected_duration+30)
                self.magnet_control.perform_fit=True
                time.sleep(1.0)
                self.magnet_control.perform_fit=False
                time.sleep(1.0)
                self.magnet_control.perform_fit=True
                '''
                if not np.isnan(self.magnet_control.Xmax):
                    if abs(Xmax_ref - self.magnet_control.Xmax) < 0.2:
                        Magnet.move_absolute(1, self.magnet_control.Xmax)
                        Xmax_ref = self.magnet_control.Xmax
                        '''
                Magnet.move_absolute(1, self.magnet_control.Xmax)        
                time.sleep(10)
                self.magnet_control.save_fluor_x_plot(file_name+'/x')  
                
                '''
                self._magnet_alignment('y')
                time.sleep(1.0)
                Magnet.move_absolute(2, self.magnet_control.start_positionY)
                time.sleep(10.0)
                self.magnet_control.submit()
                
                time.sleep(3.0)
                if self.magnet_control.state == 'error':
                    time.sleep(3.0)
                    self.magnet_control.submit()
                    
                while self.magnet_control.state != 'done':
                   
                    threading.currentThread().stop_request.wait(1.0)
                    if threading.currentThread().stop_request.isSet():
                         break
                #time.sleep(self.magnet_control.expected_duration+30)
                self.magnet_control.perform_fit=True
                time.sleep(1.0)
                self.magnet_control.perform_fit=False
                time.sleep(1.0)
                self.magnet_control.perform_fit=True
                if not np.isnan(self.magnet_control.Ymax):
                    if abs(Ymax_ref - self.magnet_control.Ymax) < 0.2:
                        Magnet.move_absolute(2, self.magnet_control.Ymax)
                        Ymax_ref = self.magnet_control.Ymax
                time.sleep(10)
                self.magnet_control.save_fluor_y_plot(file_name+'/y')
                
                
                self.auto_focus.submit()
                time.sleep(12.0)
                self._magnet_alignment('x')
                time.sleep(1.0)
                Magnet.move_absolute(1, self.magnet_control.start_positionX)
                time.sleep(10.0)
                self.magnet_control.submit()
                time.sleep(3.0)
                if self.magnet_control.state == 'error':
                    time.sleep(3.0)
                    self.magnet_control.submit()
                while self.magnet_control.state != 'done':
                   
                    threading.currentThread().stop_request.wait(1.0)
                    if threading.currentThread().stop_request.isSet():
                         break
                #time.sleep(self.magnet_control.expected_duration+30)
                self.magnet_control.perform_fit=True
                time.sleep(1.0)
                self.magnet_control.perform_fit=False
                time.sleep(1.0)
                self.magnet_control.perform_fit=True
                if not np.isnan(self.magnet_control.Xmax):
                    if abs(Xmax_ref - self.magnet_control.Xmax) < 0.2:
                        Magnet.move_absolute(1, self.magnet_control.Xmax)
                        Xmax_ref = self.magnet_control.Xmax
                time.sleep(10)
                self.magnet_control.save_fluor_x_plot(file_name+'/x')    
                '''    
                
                
                # pulsed odmr --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
                time.sleep(1.0)
                self._odmr_pulsed(fstart[nfield], fend[nfield], tpi[nfield])
                time.sleep(1.0)
                self.odmr.submit()   
                time.sleep(2.0)
                
                while self.odmr.state != 'done':
                   
                    threading.currentThread().stop_request.wait(1.0)
                    if threading.currentThread().stop_request.isSet():
                         break
                self.odmr.remove()   
                self.odmr.state = 'idle'               
                file_odmr = file_name + '/odmr'
                self.odmr.save_all(file_odmr)
                time.sleep(5.0)  
                
                if(len(self.odmr.fit_frequencies) > 2 or self.odmr.fit_contrast.max() < 5 or self.odmr.fit_contrast.max() > 43):
                    continue
                    
                freq = self.odmr.fit_frequencies[0]
                
                if(np.isnan(freq)):
                    continue    
                # Rabi ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                self.auto_focus.periodic_focus = True
                
                self.psawg.measurement.freq = freq             
                      
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
             
                file_rabi = file_name + '/Rabi' + '_power_' + str(power) + '_contrast_' + string.replace(str(self.psawg.fit.contrast[0])[0:4], '.', 'd') 
                self.psawg.save_all(file_rabi)
                time.sleep(10.0)
                
                self.psawg.measurement.remove()
                self.psawg.measurement.state = 'idle'            
                time.sleep(5.0)
                
                half_pi = self.psawg.fit.t_pi2[0]
                pi = self.psawg.fit.t_pi[0]
                
                condition = np.isnan(half_pi) or self.psawg.fit.contrast[0] < 15
                if condition:
                    continue
                    
                # self.pdawg.measurement.freq = freq
                # self.pdawg.measurement.pi2_1 = half_pi
                # self.pdawg.measurement.pi_1 = pi
                
                # self.pdawg.measurement.load()
                # time.sleep(20.0)
                
                # while self.pdawg.measurement.reload == True:
                    # threading.currentThread().stop_request.wait(1.0)
                
                # self.pdawg.measurement.submit()
                
                # while self.pdawg.measurement.state != 'done':
                     # threading.currentThread().stop_request.wait(1.0)
                     # if threading.currentThread().stop_request.isSet():
                        # break
                        
                     # if  self.pdawg.measurement.state == 'error':
                        # time.sleep(4)
                        # self.pdawg.measurement.resubmit()
             
                     
                # self.pdawg.measurement.progress = 0
                # self.pdawg.measurement.elapsed_sweeps = 0                
                        
                # file_hahn = file_name + '/T2'
                # self.pdawg.save_line_plot(file_hahn + '.png')
                # self.pdawg.save(file_hahn + '.pyd')
                # time.sleep(5.0)
                # self.pdawg.measurement.remove()
                # self.pdawg.measurement.state = 'idle'
                # time.sleep(5.0)       
                
                # self.pdawg2.measurement.freq = freq
                
                # self.pdawg2.measurement.pi_1 = pi
                
                # self.pdawg2.measurement.load()
                # time.sleep(15.0)
                
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
                # file_T1 = file_name + '/T1'
                # self.pdawg2.save_line_plot(file_T1 + '.png')
                # self.pdawg2.save(file_T1 + '.pyd')
                # time.sleep(10.0)
                # self.pdawg2.measurement.remove()
                # self.pdawg2.measurement.state = 'idle'
                # time.sleep(10.0)   
                
                self.pdawg3.measurement.freq = freq
                self.pdawg3.measurement.pi2_1 = half_pi
                self.pdawg3.measurement.pi_1 = pi
                self.pdawg3.measurement.pulse_num = Nrep[nind]
                
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
                file_XY = file_name + '/xy8-' + str(Nrep[nind])
                self.pdawg3.save_line_plot(file_XY + '.png')
                self.pdawg3.save(file_XY + '.pyd')
                time.sleep(10.0)
                self.pdawg3.measurement.remove()
                self.pdawg3.measurement.state = 'idle'
                time.sleep(10.0)   
                
                self.auto_focus.periodic_focus = False
                # x_shift = self.confocal.x - self.sf.Centroids[ncenter][1]
                # y_shift = self.confocal.y - self.sf.Centroids[ncenter][0]
                
                self.auto_focus.submit()
                time.sleep(12.0)
                
                x_shift= self.confocal.x - X0
                y_shift= self.confocal.y - Y0
                
                for Ncent in index:
                    self.sf.Centroids[Ncent][0] = self.sf.Centroids[Ncent][0] + y_shift
                    self.sf.Centroids[Ncent][1] = self.sf.Centroids[Ncent][1] + x_shift
            
  
                        
                
