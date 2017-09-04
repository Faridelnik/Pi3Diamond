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
    def __init__(self,auto_focus, confocal,odmr=None,psawg=None,sensing=None,pdawg=None,sf=None,gs=None):
        super(auto_search_shallow, self).__init__()     
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
            
    def _odmr_pulsed(self,fst,fend):
        t_pi = 1100
        power_p = -26
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
        self.odmr.threshold = -60
        
    def _odmr_cw(self):
        power = -16
        stop_time = 40
        
        self.odmr.pulsed = False
        self.odmr.perform_fit = True
        #self.odmr.number_of_resonances=2
        self.odmr.power = power
        self.odmr.stop_time = stop_time 
        
        self.odmr.frequency_begin = 1.2e9
        self.odmr.frequency_end = 1.4e9
        self.odmr.frequency_delta = 2.0e6
        self.odmr.number_of_resonances = 1
        self.odmr.threshold = -70      

    def _run(self):
        file_pos = 'D:/data/protonNMR/NV search_Oxygen_etching4(T2rou)/search'
        os.path.exists(file_pos)
        
        self.confocal.resolution = 110
        
        power = 16
        freq_center = 1.57e9
        #fstart= 1.322e9     
        #fend= 1.350e9            
        
        from measurements.pulsed_awg import Rabi 
        self.psawg.measurement = Rabi()
        from analysis.pulsedawgan import RabiFit_phase   
        self.psawg.fit = RabiFit_phase()
        
        self.psawg.measurement.power = power
        self.psawg.measurement.freq_center = freq_center

        self.psawg.measurement.tau_begin = 10
        self.psawg.measurement.tau_end = 1000
        self.psawg.measurement.tau_delta = 10
        self.psawg.measurement.sweeps = 2.0e5       

        self.psawg.fit = RabiFit_phase()
        time.sleep(5.0)    

        self.sf.BWimageMeth = 'advanced'    
        self.sf.size = 4     
        
        for nk in range(8):
            for nl in range(8):
                if nk%2==1:
                    nl=8-nl
                x1 = 5 + (nk+0) * 5
                x2 = 5 + (nk+1) * 5
                y1 = 5 + (nl+0) * 5
                y2 = 5 + (nl+1) * 5
                
                file_path = file_pos + '/x_' + str(x1) + '_' + str(x2) + '_y_'  + str(y1) + '_' + str(y2)
                if not os.path.isdir(file_path):
                    os.mkdir(file_path)
                else:
                    continue
                    
                self.confocal.x1 = x1    
                self.confocal.x2 = x2
                self.confocal.y1 = y1
                self.confocal.y2 = y2
                
                self.auto_focus.periodic_focus = False
                file_image1 = file_path + '/image.pyd'
                
                #if os.path.isfile(file_image1):
                    #self.confocal.load(file_image1)
                    #time.sleep(1.0)
                #else:
                self.confocal.resolution = 150
                self.confocal.submit()
                time.sleep(160)
                self.confocal.remove_all_labels()
                self.auto_focus.remove_all_targets()
                self.sf._ImportImage_fired()
                time.sleep(2.0)
                self.sf.SpotMinInt = 100
                self.sf.SpotMaxInt = 280
                self.sf._ProcessBWimage_fired()
                time.sleep(15.0)
                    
                self.sf._ExportButton_fired()
                file_image = file_path + '/image.png'
                self.sf.confocal.save_image(file_image)
                file_image1 = file_path + '/image.pyd'
                self.sf.confocal.save(file_image1)
                
                for ncenter in range(0,len(self.sf.Centroids)):
                    self.auto_focus.periodic_focus = False
                    self.auto_focus.target_name = 'NV'+str(ncenter)
                    time.sleep(1.0)
                    self.confocal.y = self.sf.Centroids[ncenter][0]
                    self.confocal.x = self.sf.Centroids[ncenter][1]
                    self.auto_focus.submit()
                    time.sleep(15.0)
                    self.auto_focus.submit()
                    time.sleep(15.0)
                    #self.auto_focus.submit()
                    #time.sleep(16.0)
                    z_axis = np.sort(self.auto_focus.Z)
                    zlen=(z_axis[-1]-z_axis[0])/self.auto_focus.step_z
                    zpos=(self.auto_focus.zfit-z_axis[0])/self.auto_focus.step_z
                    zcondition = float(zpos/zlen)>0.4 and float(zpos)/zlen <0.6
                    for nfocus in range(4):
                        if zcondition and self.auto_focus.data_z.max()-self.auto_focus.data_z.min()>70:
                            break
                        else:
                            self.auto_focus.submit()
                            time.sleep(15.0)
                            z_axis = np.sort(self.auto_focus.Z)
                            zlen=(z_axis[-1]-z_axis[0])/self.auto_focus.step_z
                            zpos=(self.auto_focus.zfit-z_axis[0])/self.auto_focus.step_z
                            zcondition = float(zpos/zlen)>0.4 and  float(zpos/zlen)<0.6
                    self.auto_focus.fit_xy()
                    time.sleep(1.0)      

                       
                    cond1 = self.auto_focus.data_z.max() < 70 or self.auto_focus.xfit - 0.25<0 or self.auto_focus.yfit - 0.25<0 or self.auto_focus.xfit + 0.25>50 or self.auto_focus.yfit + 0.25>50
                    if cond1:
                        continue
                    else:    
                        self.confocal.x1=self.auto_focus.xfit - 0.25
                        self.confocal.x2=self.auto_focus.xfit + 0.25
                        self.confocal.y1=self.auto_focus.yfit - 0.25
                        self.confocal.y2=self.auto_focus.yfit + 0.25
                    self.confocal.resolution = 110    
                    self.confocal.submit()
                    time.sleep(88)
                    
                    self.gs._ImportImage_fired()
                    time.sleep(2.0)
                    self.gs._Getfitimage_fired()
                    file_image = file_path + '/NV' + str(ncenter) + '.png'
                    self.gs.save_image(file_image)
                    fitx=abs(self.gs.fitparemeter[4])
                    fity=abs(self.gs.fitparemeter[3])
                    spotcondition = fitx>0.25 or fity>0.25 or fitx > 1.6*fity or fity>1.6*fitx
                    if spotcondition:
                        continue
                        
                    
                    file_name = file_path + '/NV' + str(ncenter)
                    if not os.path.isdir(file_name):
                        os.mkdir(file_name)
                    else:
                        continue
                    
                    self.auto_focus.submit()
                    time.sleep(15.0)
                    self.confocal.x1 = x1    
                    self.confocal.x2 = x2
                    self.confocal.y1 = y1
                    self.confocal.y2 = y2
                    self.confocal.resolution = 110
                    self.confocal.submit()
                    time.sleep(88)
                    self.confocal.x = self.auto_focus.xfit
                    self.confocal.y = self.auto_focus.yfit
                    file_nv = file_name + '/image.png'
                    self.sf.confocal.save_image(file_nv)
                    time.sleep(16.0)
                    self.auto_focus.periodic_focus = True
                    '''
                    self._odmr_cw()
                    time.sleep(1.0)    
                    self.odmr.submit()   
                    time.sleep(1.0)
                    
                    while self.odmr.state != 'done':
                        #print threading.currentThread().getName()

                        threading.currentThread().stop_request.wait(1.0)
                        if threading.currentThread().stop_request.isSet():
                             break
                             
                    file_odmr = file_name + '/Odmr_cw'
                    self.odmr.save_line_plot(file_odmr + '.png')
                    self.odmr.save(file_odmr + '.pyd')
                    time.sleep(2.0)  
                    
                    if(len(self.odmr.fit_frequencies) > 3 or self.odmr.fit_contrast.max() < 14 or self.odmr.fit_contrast.max() > 40):
                        continue
                        
                    freq = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)
                    
                    if(np.isnan(freq)):
                        continue
                        '''
                    fst = 1.486e9   
                    fend = 1.496e9
                    self._odmr_pulsed(fst,fend)
                    time.sleep(1.0)    
                    self.odmr.submit()   
                    time.sleep(1.0)
                    
                    while self.odmr.state != 'done':
                        #print threading.currentThread().getName()

                        threading.currentThread().stop_request.wait(1.0)
                        if threading.currentThread().stop_request.isSet():
                             break
                             
                    file_odmr = file_name + '/Odmr_pulsed'
                    self.odmr.save_line_plot(file_odmr + '.png')
                    self.odmr.save(file_odmr + '.pyd')
                    time.sleep(2.0)  
                    
                    if(len(self.odmr.fit_frequencies) > 3 or self.odmr.fit_contrast.max() < 8 or self.odmr.fit_contrast.max() > 40):
                        continue
                        
                    freq = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)
                    
                    if(np.isnan(freq)):
                        continue
                    
                    # Rabi #####################
                    
                    self.psawg.measurement.freq = freq
                    self.psawg.measurement.power = power
                    self.psawg.measurement.load()
                    time.sleep(2.0) 
                    self.psawg.fit = RabiFit_phase()
                    time.sleep(2.0) 
                    '''
                    rabi_flag = True
                    if np.isnan(power):
                        power = 6
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
                        
                        if(self.psawg.fit.contrast[0] < 18):
                            continue
                        half_pi = self.psawg.fit.t_pi2[0]
                        pi = self.psawg.fit.t_pi[0]   
                        if pi < 90 and pi > 72:
                            rabi_flag = False
                        else:
                            amp = 82.0/pi
                            #amp^2 = power/power_next
                            power = power - 10*np.log10(amp**2)
                            if power > 16 or power < 0:
                                rabi_flag=False
                        if np.isnan(power):
                            power = 9 
                            '''
                    while self.psawg.measurement.reload == True:
                            threading.currentThread().stop_request.wait(1.0)
                        
                    self.psawg.measurement.submit()
                       
                    while self.psawg.measurement.state != 'done':
                         threading.currentThread().stop_request.wait(1.0)
                         if threading.currentThread().stop_request.isSet():
                            break
                            
                    self.psawg.measurement.progress = 0
                    self.psawg.measurement.elapsed_sweeps = 0
                    
                    
                    half_pi = self.psawg.fit.t_pi2[0]
                    pi = self.psawg.fit.t_pi[0] 
                     
                    file_rabi = file_name + '/Rabi'
                    self.psawg.save_line_plot(file_rabi + '.png')
                    self.psawg.save(file_rabi + '.pyd')
                    time.sleep(10.0)
                    self.psawg.measurement.remove()
                    self.psawg.measurement.state = 'idle'            
                    time.sleep(5.0)
                    #half_pi = self.psawg.fit.t_pi2[0]
                    #pi = self.psawg.fit.t_pi[0]
                    
                    condition = np.isnan(half_pi) or self.psawg.fit.contrast[0] < 18
                    if condition:
                        continue
                    '''    
                    from measurements.shallow_NV import XY8_Ref
                    self.pdawg.measurement = XY8_Ref()
                    self.pdawg.measurement.power = power
                    self.pdawg.measurement.freq_center = freq_center
                    self.pdawg.measurement.freq = freq
                    self.pdawg.measurement.pi2_1 = half_pi
                    self.pdawg.measurement.pi_1 = pi
                    self.pdawg.measurement.pulse_num = 10
                    self.pdawg.measurement.tau_begin = 50
                    self.pdawg.measurement.tau_end = 85
                    self.pdawg.measurement.tau_delta = 1.5
                    self.pdawg.measurement.sweeps = 3.5e5
                    
                    self.pdawg.measurement.load()
                    time.sleep(100.0)
                    
                    while self.pdawg.measurement.reload == True:
                        threading.currentThread().stop_request.wait(1.0)
                    
                    self.pdawg.measurement.submit()
                    
                    while self.pdawg.measurement.state != 'done':
                         threading.currentThread().stop_request.wait(1.0)
                         if threading.currentThread().stop_request.isSet():
                            break
                            
                    file_hahn = file_name + '/dynamics_XY8_' + str(self.pdawg.measurement.pulse_num) + '_' + str(int(pi)) + 'ns_65_95ns'
                    self.pdawg.save_line_plot(file_hahn + '.png')
                    self.pdawg.save(file_hahn + '.pyd')
                    time.sleep(10.0)
                    self.pdawg.measurement.remove()
                    self.pdawg.measurement.state = 'idle'
                    time.sleep(20.0)       
                    '''
                    
                    from measurements.shallow_NV import Hahn
                    self.pdawg.measurement = Hahn()
                    self.pdawg.measurement.power = power
                    self.pdawg.measurement.freq_center = freq_center
                    self.pdawg.measurement.freq = freq
                    self.pdawg.measurement.pi2_1 = half_pi
                    self.pdawg.measurement.pi_1 = pi
                    
                    self.pdawg.measurement.tau_begin = 300
                    self.pdawg.measurement.tau_end = 10000
                    self.pdawg.measurement.tau_delta = 300
                    self.pdawg.measurement.sweeps = 3.0e5
                    
                    self.pdawg.measurement.load()
                    time.sleep(20.0)
                    
                    while self.pdawg.measurement.reload == True:
                        threading.currentThread().stop_request.wait(1.0)
                    
                    self.pdawg.measurement.submit()
                    
                    while self.pdawg.measurement.state != 'done':
                         threading.currentThread().stop_request.wait(1.0)
                         if threading.currentThread().stop_request.isSet():
                            break
                            
                    file_hahn = file_name + '/T2'
                    self.pdawg.save_line_plot(file_hahn + '.png')
                    self.pdawg.save(file_hahn + '.pyd')
                    time.sleep(10.0)
                    self.pdawg.measurement.remove()
                    self.pdawg.measurement.state = 'idle'
                    time.sleep(20.0)       
                    
                    from measurements.shallow_NV import T1
                    self.pdawg.measurement = T1()
                    self.pdawg.measurement.power = power
                    self.pdawg.measurement.freq_center = freq_center
                    self.pdawg.measurement.freq = freq
                    self.pdawg.measurement.pi_1 = pi
                    self.pdawg.measurement.tau_begin = 300
                    self.pdawg.measurement.tau_end = 600000
                    self.pdawg.measurement.tau_delta = 40000
                    self.pdawg.measurement.sweeps = 1.0e5
                    
                    self.pdawg.measurement.load()
                    time.sleep(15.0)
                    
                    while self.pdawg.measurement.reload == True:
                        threading.currentThread().stop_request.wait(1.0)
                    
                    self.pdawg.measurement.submit()
                    
                    while self.pdawg.measurement.state != 'done':
                         threading.currentThread().stop_request.wait(1.0)
                         if threading.currentThread().stop_request.isSet():
                            break
                            
                    file_hahn = file_name + '/T1'
                    self.pdawg.save_line_plot(file_hahn + '.png')
                    self.pdawg.save(file_hahn + '.pyd')
                    time.sleep(10.0)
                    self.pdawg.measurement.remove()
                    self.pdawg.measurement.state = 'idle'
                    time.sleep(20.0)       
                    
                    
                    x_last = self.confocal.x
                    y_last = self.confocal.y
                
                self.confocal.x1 = x1    
                self.confocal.x2 = x2
                self.confocal.y1 = y1
                self.confocal.y2 = y2
                
                self.auto_focus.periodic_focus = False 
                self.confocal.resolution = 150                
                self.confocal.submit()
                time.sleep(200)
                self.confocal.remove_all_labels()
                self.auto_focus.remove_all_targets()

                file_image = file_path + '/image1.png'
                self.sf.confocal.save_image(file_image)
                file_image1 = file_path + '/image1.pyd'
                self.sf.confocal.save(file_image1)
                
                if len(self.sf.Centroids)==0:
                    self.auto_focus.periodic_focus = False    
                    self.confocal.remove_all_labels()
                    self.auto_focus.remove_all_targets()
                    self.sf._ImportImage_fired()
                    time.sleep(2.0)
                    self.sf.SpotMinInt = 10
                    self.sf.SpotMaxInt = 180
                    self.sf._ProcessBWimage_fired()
                    time.sleep(15.0)
                        
                    self.sf._ExportButton_fired()
                    inde = -1
                    if len(self.sf.Centroids)>0:
                        for nscan in (range(len(self.sf.Centroids))):
                            self.confocal.y = self.sf.Centroids[inde][0]
                            self.confocal.x = self.sf.Centroids[inde][1]
                            x_last = self.confocal.x
                            y_last = self.confocal.y
                            
                            self.auto_focus.submit()
                            time.sleep(16.0)
                            self.auto_focus.submit()
                            time.sleep(16.0)
                            
                            z_axis1 = np.sort(self.auto_focus.Z)
                            zlen1=(z_axis1[-1]-z_axis1[0])/self.auto_focus.step_z
                            zpos1=(self.auto_focus.zfit-z_axis1[0])/self.auto_focus.step_z
                            zcondition1 = float(zpos1/zlen1)>0.4 and float(zpos1)/zlen1
                            for nfocus in range(3):
                                if zcondition1 and self.auto_focus.data_z.max()-self.auto_focus.data_z.min()>70:
                                    break
                                else:
                                    self.auto_focus.submit()
                                    time.sleep(16.0)
                                    z_axis1 = np.sort(self.auto_focus.Z)
                                    zlen1=(z_axis1[-1]-z_axis1[0])/self.auto_focus.step_z
                                    zpos1=(self.auto_focus.zfit-z_axis1[0])/self.auto_focus.step_z
                                    zcondition1 = float(zpos1/zlen1)>0.4 and  float(zpos1/zlen1)<0.6
                            if zcondition1 and self.auto_focus.data_z.max()-self.auto_focus.data_z.min()>70:
                                    self.auto_focus.save('defaults/auto_focus.pyd')
                                    break
                            inde = inde - 1        
                    else:
                        self.confocal.y = y_last
                        self.confocal.x = x_last
                    
                        
                
