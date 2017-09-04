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
    def __init__(self,auto_focus, confocal,odmr=None, psawg=None, pdawg=None, pdawg2=None, pdawg3=None, sf=None,gs=None):
        super(auto_search_pillar, self).__init__()     
        self.auto_focus=auto_focus   
        self.confocal=confocal
        if odmr is not None:
            self.odmr = odmr
            
        if psawg is not None:
            self.psawg = psawg
            
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
                
            
    def _odmr(self,fst,fend):
        t_pi = 260
        power_p = -6
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
        
    def _odmr_zoom(self,fst,fend):
        t_pi = 1100
        power_p = -18
        stop_time = 100
        
        self.odmr.pulsed = True
        self.odmr.perform_fit = True
        #self.odmr.number_of_resonances=2
        self.odmr.power_p = power_p
        self.odmr.t_pi = t_pi
        self.odmr.stop_time = stop_time 
        
        self.odmr.frequency_begin_p = fst
        self.odmr.frequency_end_p = fend
        self.odmr.frequency_delta_p = 1.0e5
        self.odmr.number_of_resonances = 2
        self.odmr.threshold = -60   

    def _odmr_cw(self):
        power = -12
        stop_time = 80
        
        self.odmr.pulsed = False
        self.odmr.perform_fit = True
        #self.odmr.number_of_resonances=2
        self.odmr.power = power
        self.odmr.stop_time = stop_time 
        
        self.odmr.frequency_begin = 1.1e9
        self.odmr.frequency_end = 1.4e9
        self.odmr.frequency_delta = 1.0e6
        self.odmr.number_of_resonances = 1
        self.odmr.threshold = -70           

    def _run(self):
        file_path = 'D:/data/protonNMR/mebrane/scan/nanoparticle_test/D_3_R_area(np_PFI)/odmr_scan'
        os.path.exists(file_path)
        
        #x_coil = ha.Coil()('x')
        #y_coil = ha.Coil()('y')
        self.confocal.resolution = 150
        
        power = 16
        freq_center = 2.76e9
   
        fstart= 2.83e9     
        fend= 2.91e9
                    
        
        from measurements.pulsed_awg import Rabi 
        self.psawg.measurement = Rabi()
        from analysis.pulsedawgan import RabiFit_phase
        self.psawg.fit = RabiFit_phase()
        
        self.psawg.measurement.power = power
        self.psawg.measurement.freq_center = freq_center

        self.psawg.measurement.tau_begin = 10
        self.psawg.measurement.tau_end = 1000
        self.psawg.measurement.tau_delta = 10
        self.psawg.measurement.sweeps = 1.5e5       

        self.psawg.fit = RabiFit_phase()
        time.sleep(2.0)   
        
        from measurements.shallow_NV import FID_Db
        self.pdawg.measurement = FID_Db()
        
        self.pdawg.measurement.power = power
        self.pdawg.measurement.freq_center = freq_center

        self.pdawg.measurement.tau_begin = 100
        self.pdawg.measurement.tau_end = 2000
        self.pdawg.measurement.tau_delta = 30
        self.pdawg.measurement.sweeps = 2.0e5    

        from measurements.shallow_NV import Hahn 
        self.pdawg2.measurement = Hahn()
        
        self.pdawg2.measurement.power = power
        self.pdawg2.measurement.freq_center = freq_center

        self.pdawg2.measurement.tau_begin = 300
        self.pdawg2.measurement.tau_end = 10000
        self.pdawg2.measurement.tau_delta = 210
        self.pdawg2.measurement.sweeps = 2.0e5       
        
        from measurements.shallow_NV import T1 
        self.pdawg3.measurement = T1()
        
        self.pdawg3.measurement.power = power
        self.pdawg3.measurement.freq_center = freq_center

        self.pdawg3.measurement.tau_begin = 300
        self.pdawg3.measurement.tau_end = 150000
        self.pdawg3.measurement.tau_delta = 8000
        self.pdawg3.measurement.sweeps = 3.0e5    
  
  
        self.auto_focus.periodic_focus = False
        
        #if os.path.isfile(file_image1):
            #self.confocal.load(file_image1)
            #time.sleep(1.0)
        #else:
       
        self.confocal.remove_all_labels()
        self.auto_focus.remove_all_targets()
        self._odmr_cw()
        time.sleep(1.0)
        index = np.linspace(0,47,48)
        #index = [1,2,3,4,7,13,15,17,19,26,27,28,31,39,40,43,48,50,51,56]
        for ncenter in index:
            '''
            x_coil._set_current(0.5)
            time.sleep(1.0)
            y_coil._set_current(0.5)
            time.sleep(1.0)
            '''
            ncenter = int(ncenter)
            self.auto_focus.periodic_focus = False
            self.auto_focus.target_name = 'NV'+str(ncenter)
            time.sleep(1.0)
            self.confocal.y = self.sf.Centroids[ncenter][0]
            self.confocal.x = self.sf.Centroids[ncenter][1]
            self.auto_focus.submit()
            time.sleep(16.0)
            
            '''
            self.confocal.resolution = 120   
            self.confocal.submit()
            time.sleep(90)
            
            file_nv = file_path + '/NV_' + str(ncenter) + '_image.png'
            self.confocal.save_image(file_nv)
            '''
            
            
            self.auto_focus.submit()
            time.sleep(16.0)

            self.auto_focus.periodic_focus = True
            time.sleep(1.0)
            #self._odmr(fstart,fend)
            #time.sleep(1.0)    
            self.odmr.submit()   
            time.sleep(2.0)
            
            while self.odmr.state != 'done':
                #print threading.currentThread().getName()

                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break
                     
            self.odmr.remove()   
            self.odmr.state = 'idle'                
            file_odmr = file_path + '/NV_' + str(ncenter) + 'Odmr'
            self.odmr.save_line_plot(file_odmr + '.png')
            time.sleep(1.0)  
            self.odmr.save(file_odmr + '.pyd')
            time.sleep(5.0)  
            self.auto_focus.periodic_focus = False
            
            """
            if self.odmr.fit_contrast.max() < 9 or self.odmr.fit_contrast.max() > 40:
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
                '''
                
            freq = self.odmr.fit_frequencies[self.odmr.fit_contrast.argmax()]
            if np.isnan(freq):
                continue
            fstart1= freq - 4.0e6   
            fend1= freq + 4.0e6
            self._odmr_zoom(fstart1,fend1)
            time.sleep(1.0)  
            
            self.odmr.submit()   
            time.sleep(2.0)
            
            while self.odmr.state != 'done':
                #print threading.currentThread().getName()

                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break
                     
            file_odmr = file_path + '/NV_' + str(ncenter) + 'Odmr_zoom'
            self.odmr.save_line_plot(file_odmr + '.png')
            time.sleep(1.0)  
            self.odmr.save(file_odmr + '.pyd')
            time.sleep(1.0)  
            
            freq = self.odmr.fit_frequencies[self.odmr.fit_contrast.argmax()]
            

            if(np.isnan(freq)):
                continue
            
            # Rabi #####################
            
            self.psawg.measurement.freq = freq 
            self.psawg.measurement.load()
            time.sleep(5.0)
            self.psawg.measurement.submit()
           
            while self.psawg.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                        
            file_rabi = file_path + '/NV_' + str(ncenter) + '_Rabi'
            self.psawg.save_line_plot(file_rabi + '.png')
            time.sleep(1.0)
            self.psawg.save(file_rabi + '.pyd')
            time.sleep(1.0)
            self.psawg.measurement.remove()
            self.psawg.measurement.state = 'idle'            
            time.sleep(5.0)   
            
            if self.psawg.fit.contrast[0] < 1.4:
                continue
            pi2_1 = self.psawg.fit.t_pi2[0]
            pi = self.psawg.fit.t_pi[0]
            if(np.isnan(pi2_1)):
                continue
            
            self.pdawg.measurement.freq = freq
            self.pdawg.measurement.pi2_1 = pi2_1
            self.pdawg.measurement.load()
            time.sleep(5.0)
            self.pdawg.measurement.submit()
           
            while self.pdawg.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                        
            file_fid = file_path + '/NV_' + str(ncenter) + '_fid'
            self.pdawg.save_line_plot(file_fid + '.png')
            time.sleep(1.0)  
            self.pdawg.save(file_fid + '.pyd')
            time.sleep(1.0)
            self.pdawg.measurement.remove()
            self.pdawg.measurement.state = 'idle'            
            time.sleep(5.0)   
            
            
            self.pdawg2.measurement.freq = freq
            self.pdawg2.measurement.pi2_1 = pi2_1
            self.pdawg2.measurement.pi_1 = pi
            self.pdawg2.measurement.load()
            time.sleep(5.0)
            self.pdawg2.measurement.submit()
           
            while self.pdawg2.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                        
            file_hahn = file_path + '/NV_' + str(ncenter) + '_hahn'
            self.pdawg2.save_line_plot(file_hahn + '.png')
            time.sleep(1.0)  
            self.pdawg2.save(file_hahn + '.pyd')
            time.sleep(1.0)
            self.pdawg2.measurement.remove()
            self.pdawg2.measurement.state = 'idle'            
            time.sleep(5.0)   
            
            
            self.pdawg3.measurement.freq = freq
            #self.pdawg3.measurement.t_pi2 = pi2_1
            self.pdawg3.measurement.pi_1 = pi
            self.pdawg3.measurement.load()
            time.sleep(5.0)
            self.pdawg3.measurement.submit()
           
            while self.pdawg3.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
                        
            file_T1 = file_path + '/NV_' + str(ncenter) + '_T1'
            self.pdawg3.save_line_plot(file_T1 + '.png')
            time.sleep(1.0)  
            self.pdawg3.save(file_T1 + '.pyd')
            time.sleep(1.0)
            self.pdawg3.measurement.remove()
            self.pdawg3.measurement.state = 'idle'            
            time.sleep(5.0)   
            """
            
            
