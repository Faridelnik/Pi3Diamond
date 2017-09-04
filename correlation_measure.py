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

class correlation_measure(ManagedJob):
    def __init__(self,auto_focus, confocal,odmr=None,psawg=None,sensing=None,pdawg=None,sf=None,gs=None):
        super(correlation_measure, self).__init__()     
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
        t_pi = 900
        power_p = -36
        stop_time = 150
        
        self.odmr.pulsed = True
        self.odmr.perform_fit = True
        #self.odmr.number_of_resonances=2
        self.odmr.power_p = power_p
        self.odmr.t_pi = t_pi
        self.odmr.stop_time = stop_time 
        
        self.odmr.frequency_begin_p = fst
        self.odmr.frequency_end_p = fend
        self.odmr.frequency_delta_p = 1.0e5
        self.odmr.number_of_resonances = 1
        self.odmr.threshold = -60

    def _run(self):
        file_name = 'D:/data/protonNMR/NV search_Oxygen_etching2/x_13_18_y_30_40/NV2/corr'
        os.path.exists(file_name)
        
        power = 1.8
        freq_center = 1.76e9
        fstart= 1.587e9     
        fend= 1.597e9
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
        
        
        self.confocal.x1 = 14.8
        self.confocal.x2 = 16.8
        self.confocal.y1 = 35.1
        self.confocal.y2 = 37.1
        
        self.sf.BWimageMeth = 'advanced'    
        self.sf.size = 7   
 
        #t_corr = [8000, 9000, 11000, 14000, 17000, 20000, 25000, 30000, 40000, 60000, 100000, 150000, 180000, 200000, 220000, 320000, 260000, 290000, 420000, 520000, 720000, 820000, 920000]
        t_corr = [300000]
        for nl in range(len(t_corr)):

           
            from measurements.shallow_NV import Correlation_Spec_XY8
            self.sensing.fit = RabiFit_phase()
            self.sensing.measurement = Correlation_Spec_XY8()      
            self.sensing.measurement.freq_center = freq_center      
            self.sensing.measurement.pulse_num = 4
            self.sensing.measurement.tau_inte = 89.0
            self.sensing.measurement.dbtau_flag = True
            self.sensing.measurement.tau_begin1 = 3000
            self.sensing.measurement.tau_end1 = 40000
            self.sensing.measurement.tau_delta1 = 1555
            self.sensing.measurement.tau_begin = 44000
            self.sensing.measurement.tau_end = 300000
            self.sensing.measurement.tau_delta = 7259
                                     
            
            for nk in range(5):
                self.auto_focus.periodic_focus = False
                file_image = file_name + '/image_' + str(t_corr[nl]) + '_' + str(nk) + '.png'
                if os.path.isfile(file_image):
                    continue
                self.confocal.submit()
                time.sleep(88)
                #file_image = file_name + '/image_' + str(t_corr[nk]) + '.pyd'
                self.confocal.save_image(file_image)
                
                time.sleep(1.0)
                
                self.auto_focus.periodic_focus = True
                self.odmr.submit()   
                time.sleep(2.0)
                
                while self.odmr.state != 'done':
                    #print threading.currentThread().getName()

                    threading.currentThread().stop_request.wait(1.0)
                    if threading.currentThread().stop_request.isSet():
                         break
                         
                file_odmr = file_name + '/Odmr_' + str(t_corr[nl]) + '_' + str(nk) 
                self.odmr.save_line_plot(file_odmr + '.png')
                self.odmr.save(file_odmr + '.pyd')
                time.sleep(5.0)  
                
                if(len(self.odmr.fit_frequencies) > 3 or self.odmr.fit_contrast.max() < 5 or self.odmr.fit_contrast.max() > 40):
                    continue
                    
                freq = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)
                
                if(np.isnan(freq)):
                    continue
                
                # Rabi #####################
                self.psawg.measurement.freq = freq
                self.psawg.fit = RabiFit_phase()
                time.sleep(2.0)    
                rabi_flag = True
                if np.isnan(power):
                    power = 1.8
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
                    if pi < 80.5 and pi > 79.8:
                        rabi_flag = False
                    else:
                        amp = 80.0/pi
                        #amp^2 = power/power_next
                        power = power - 10*np.log10(amp**2)
                    if np.isnan(power):
                        power = 1.8    
                 
                file_rabi = file_name + '/Rabi_' + str(t_corr[nl]) + '_' +  str(nk) 
                self.psawg.save_line_plot(file_rabi + '.png')
                self.psawg.save(file_rabi + '.pyd')
                time.sleep(10.0)
                self.psawg.measurement.remove()
                self.psawg.measurement.state = 'idle'            
                time.sleep(5.0)
            
                self.sensing.measurement.power = power
                
                self.sensing.measurement.freq = freq
                self.sensing.measurement.pi2_1 = half_pi
                self.sensing.measurement.pi_1 = pi
                self.sensing.measurement.sweeps = (nk + 1) *1.0e6
                
                self.sensing.measurement.load()
                time.sleep(100.0)
                
                while self.sensing.measurement.reload == True:
                    threading.currentThread().stop_request.wait(1.0)
                
                if nk==0:
                    self.sensing.measurement.submit()
                else:
                    self.sensing.measurement.resubmit()
                self.sensing.fit = RabiFit_phase()
                
                while self.sensing.measurement.state != 'done':
                     threading.currentThread().stop_request.wait(1.0)
                     if threading.currentThread().stop_request.isSet():
                        break
                    
                file_corr = file_name + '/correlation_XY8_' + str(self.sensing.measurement.pulse_num) + '_' +str(80)+'ns'+'_' + str(t_corr[nl]/1000) + '_' + str(t_corr[nl]/1000+1) +'us_' +str(nk) 
                self.sensing.save_line_plot(file_corr + '.png')
                self.sensing.save(file_corr + '.pyd')
                time.sleep(10.0)
                self.sensing.measurement.remove()
                self.sensing.measurement.state = 'idle'
                time.sleep(20.0)       

                self.auto_focus.submit()
                time.sleep(16.0)
                self.auto_focus.submit()
                time.sleep(16.0)
                self.auto_focus.submit()
                time.sleep(16.0)
                
                cond1 = self.auto_focus.data_z.max() < 30
                
                if cond1:
                    pg = ha.PulseGenerator()
                    pg.Light()
                    self.auto_focus.submit()
                    time.sleep(16.0)
                    self.auto_focus.submit()
                    time.sleep(16.0)
                    self.auto_focus.submit()
                    time.sleep(16.0)
                       

                    
                        
                
