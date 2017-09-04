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

class auto_search_pillar(ManagedJob):
    def __init__(self,auto_focus, confocal,odmr=None,psrabi=None, sf=None,gs=None):
        super(auto_search_pillar, self).__init__()     
        self.auto_focus=auto_focus   
        self.confocal=confocal
        if odmr is not None:
            self.odmr = odmr
            
        if psrabi is not None:
            self.psrabi = psrabi
                
        if sf is not None:
            self.sf = sf    
            
        if gs is not None:
            self.gs = gs      
                
            
    def _odmr(self,fst,fend):
        t_pi = 260
        power_p = -10
        stop_time = 100
        
        self.odmr.pulsed = True
        self.odmr.perform_fit = True
        #self.odmr.number_of_resonances=2
        self.odmr.power_p = power_p
        self.odmr.t_pi = t_pi
        self.odmr.stop_time = stop_time 
        
        self.odmr.frequency_begin_p = fst
        self.odmr.frequency_end_p = fend
        self.odmr.frequency_delta_p = 8.0e5
        self.odmr.number_of_resonances = 2
        self.odmr.threshold = -60
        
    def _odmr_cw(self):
        power = -10
        stop_time = 100
        
        self.odmr.pulsed = False
        self.odmr.perform_fit = True
        #self.odmr.number_of_resonances=2
        self.odmr.power = power
        self.odmr.stop_time = stop_time 
        
        self.odmr.frequency_begin = 2.75e9
        self.odmr.frequency_end = 3.0e9
        self.odmr.frequency_delta = 1.5e6
        self.odmr.number_of_resonances = 'auto'
        self.odmr.threshold = -60       

    def _run(self):
        file_path = 'D:/data/protonNMR/mebrane/scan/nanoparticle_test/D_2_R_area/right_upper_part/ODMR'
        os.path.exists(file_path)
        
        #x_coil = ha.Coil()('x')
        #y_coil = ha.Coil()('y')
        self.confocal.resolution = 150
        
        #power = 6
   
        #fstart= 2.83e9     
        #fend= 2.91e9
        #self._odmr(fstart,fend)
        #time.sleep(1.0)                
        
        #from measurements.pulsed import Rabi 
        #self.psrabi.measurement = Rabi()
        #from analysis.pulsedan import RabiFit_phase
        #self.psrabi.fit = RabiFit_phase()
        '''
        self.psrabi.measurement.power = power
        #self.psrabi.measurement.freq_center = freq_center

        self.psrabi.measurement.tau_begin = 300
        self.psrabi.measurement.tau_end = 800
        self.psrabi.measurement.tau_delta = 10
        self.psrabi.measurement.sweeps = 4.0e5       

        self.psrabi.fit = RabiFit_phase()
        time.sleep(2.0)   
        
        from measurements.pulsed import FID 
        self.psfid.measurement = FID()
        
        self.psfid.measurement.power = power
        #self.psrabi.measurement.freq_center = freq_center

        self.psfid.measurement.tau_begin = 300
        self.psfid.measurement.tau_end = 3000
        self.psfid.measurement.tau_delta = 50
        self.psfid.measurement.sweeps = 5.0e5    

        from measurements.pulsed import Hahn 
        self.pshahn.measurement = Hahn()
        
        self.pshahn.measurement.power = power
        #self.psrabi.measurement.freq_center = freq_center

        self.pshahn.measurement.tau_begin = 300
        self.pshahn.measurement.tau_end = 12000
        self.pshahn.measurement.tau_delta = 250
        self.pshahn.measurement.sweeps = 3.0e5       
        '''
  
        self.auto_focus.periodic_focus = False
        
        #if os.path.isfile(file_image1):
            #self.confocal.load(file_image1)
            #time.sleep(1.0)
        #else:
       
        self.confocal.remove_all_labels()
        self.auto_focus.remove_all_targets()
        
        x_shift = 0 
        y_shift = 0
        index = np.linspace(0,96,97)
        #index = []
        for ncenter in index:
            #x_coil._set_current(0.5)
            #time.sleep(2.0)
            #y_coil._set_current(0.5)
            #time.sleep(2.0)
            ncenter = int(ncenter)
            self.auto_focus.periodic_focus = False
            self.auto_focus.target_name = 'NV'+str(ncenter)
            time.sleep(1.0)
            self.confocal.y = self.sf.Centroids[ncenter][0] + y_shift
            self.confocal.x = self.sf.Centroids[ncenter][1] + x_shift
            
            self.auto_focus.submit()
            time.sleep(12.0)
 
            self.confocal.resolution = 120   
            self.confocal.submit()
            time.sleep(90)
            
            file_nv = file_path + '/NV_' + str(ncenter+0) + '_image.png'
            self.confocal.save_image(file_nv)
            
            
            self.auto_focus.submit()
            time.sleep(16.0)

            self.auto_focus.periodic_focus = True
            
            time.sleep(1.0)
            self._odmr_cw()
            time.sleep(2.0)
            self.odmr.submit()
            
            while self.odmr.state != 'done':
                #print threading.currentThread().getName()

                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break
                     
            file_odmr = file_path + '/NV_' + str(ncenter+0) + '_odmr'
            self.odmr.save_line_plot(file_odmr + '.png')
            time.sleep(1.0)  
            self.odmr.save(file_odmr + '.pyd')
            time.sleep(5.0)  
            
            '''
            if self.odmr.fit_contrast.max() < 5 or self.odmr.fit_contrast.max() > 40:
                continue
                
            fre = self.odmr.fit_frequencies
            fre.sort()
            if abs(sum(fre[0:2])/2 - sum(fre[-2:])/2) < 5.0e6:
                x_coil._set_current(0.0)
                time.sleep(2.0)
                y_coil._set_current(0.35)
                time.sleep(2.0)
                self.odmr.submit()   
                time.sleep(2.0)
                
                while self.odmr.state != 'done':
                    #print threading.currentThread().getName()

                    threading.currentThread().stop_request.wait(1.0)
                    if threading.currentThread().stop_request.isSet():
                         break
                         
                file_odmr = file_path + '/NV_' + str(ncenter)  + '_Odmr_y_current_' + str(0.3) 
                self.odmr.save_line_plot(file_odmr + '.png')
                time.sleep(1.0)  
                self.odmr.save(file_odmr + '.pyd')
                time.sleep(10.0)  
                
                
            freq = self.odmr.fit_frequencies[self.odmr.fit_contrast.argmax()] - 1.5e6
            

            if(np.isnan(freq)):
                continue
            
            # Rabi #####################
            
            self.psrabi.measurement.freq = freq 
            
            self.psrabi.measurement.submit()
           
            while self.psrabi.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                        
            file_rabi = file_path + '/NV_' + str(ncenter) + '_Rabi'
            self.psrabi.save_line_plot(file_rabi + '.png')
            time.sleep(1.0)
            self.psrabi.save(file_rabi + '.pyd')
            time.sleep(1.0)
            self.psrabi.measurement.remove()
            self.psrabi.measurement.state = 'idle'            
            time.sleep(5.0)   
            
            if self.psrabi.fit.contrast[0] < 1.4:
                continue
            pi2_1 = self.psrabi.fit.t_pi2[0]
            pi = self.psrabi.fit.t_pi[0]
            if(np.isnan(pi2_1)):
                continue
            
            self.psfid.measurement.freq = freq
            self.psfid.measurement.t_pi2 = pi2_1
            
            self.psfid.measurement.submit()
           
            while self.psfid.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                        
            file_fid = file_path + '/NV_' + str(ncenter) + '_fid'
            self.psfid.save_line_plot(file_fid + '.png')
            time.sleep(1.0)  
            self.psfid.save(file_fid + '.pyd')
            time.sleep(1.0)
            self.psfid.measurement.remove()
            self.psfid.measurement.state = 'idle'            
            time.sleep(5.0)   
            
            
            self.pshahn.measurement.freq = freq
            self.pshahn.measurement.t_pi2 = pi2_1
            self.pshahn.measurement.t_pi = pi
            
            self.pshahn.measurement.submit()
           
            while self.pshahn.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                        
            file_hahn = file_path + '/NV_' + str(ncenter) + '_hahn'
            self.pshahn.save_line_plot(file_hahn + '.png')
            time.sleep(1.0)  
            self.pshahn.save(file_hahn + '.pyd')
            time.sleep(1.0)
            self.pshahn.measurement.remove()
            self.pshahn.measurement.state = 'idle'            
            time.sleep(5.0)   
            '''
            self.auto_focus.periodic_focus = False
            x_shift = self.confocal.x - self.sf.Centroids[ncenter][1]
            y_shift = self.confocal.y - self.sf.Centroids[ncenter][0]
