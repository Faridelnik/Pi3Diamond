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

class auto_search(ManagedJob):
    def __init__(self,auto_focus, confocal,odmr=None,psawg=None,pair=None,sf=None,gs=None):
        super(auto_search, self).__init__()     
        self.auto_focus=auto_focus   
        self.confocal=confocal
        if odmr is not None:
            self.odmr = odmr
            
        if psawg is not None:
            self.psawg = psawg

        if pair is not None:
            self.pair = pair
            
        if sf is not None:
            self.sf = sf    
            
        if gs is not None:
            self.gs = gs      
            
    def _odmr(self,fst,fend):
        t_pi = 2700
        power_p = -30
        stop_time = 900
        
        self.odmr.pulsed = True
        self.odmr.perform_fit = True
        #self.odmr.number_of_resonances=2
        self.odmr.power_p = power_p
        self.odmr.t_pi = t_pi
        self.odmr.stop_time = stop_time 
        
        self.odmr.frequency_begin_p = fst
        self.odmr.frequency_end_p = fend
        self.odmr.frequency_delta_p = 1.5e5

    def _run(self):
        file_pos = 'D:/data/ProgGate/Images/pair_search/01-10(search)/x_4_y_4'
        os.path.exists(file_pos)
        
        for nk in range(10):
            for nl in range(10):
                if nk%2==1:
                    nl=9-nl
                x1 = (nk+0) * 5
                x2 = (nk+1) * 5
                y1 = (nl+0) * 5
                y2 = (nl+1) * 5
                
                file_path = file_pos + '/x_' + str(x1) + '_' + str(x2) + '_y_'  + str(y1) + '_' + str(y2)
                if not os.path.isdir(file_path):
                    os.mkdir(file_path)
                else:
                    continue
                    
                    
                self.confocal.x1 = x1    
                self.confocal.x2 = x2
                self.confocal.y1 = y1
                self.confocal.y2 = y2
                self.confocal.resolution = 150
                
                self.auto_focus.periodic_focus = False
                file_image1 = file_path + '/image.pyd'
                
                if os.path.isfile(file_image1):
                    self.confocal.load(file_image1)
                    time.sleep(1.0)
                else:
                    self.confocal.submit()
                    time.sleep(140)
                self.confocal.remove_all_labels()
                self.auto_focus.remove_all_targets()
                self.sf._ImportImage_fired()
                time.sleep(2.0)
                self.sf.SpotMinInt = 110
                self.sf.SpotMaxInt = 240
                self.sf._ProcessBWimage_fired()
                time.sleep(15.0)
                    
                self.sf._ExportButton_fired()
                file_image = file_path + '/image.png'
                self.sf.confocal.save_image(file_image)
                #file_image1 = file_path + '/image.pyd'
                self.sf.confocal.save(file_image1)
                
                
                for ncenter in range(len(self.sf.Centroids)):
                    self.auto_focus.periodic_focus = False
                    self.auto_focus.target_name = 'nv'+str(ncenter)
                    time.sleep(1.0)
                    self.confocal.y = self.sf.Centroids[ncenter][0]
                    self.confocal.x = self.sf.Centroids[ncenter][1]
                    self.auto_focus.submit()
                    time.sleep(8.0)
                    z_axis = np.sort(self.auto_focus.Z)
                    zlen=(z_axis[-1]-z_axis[0])/self.auto_focus.step_z
                    zpos=(self.auto_focus.zfit-z_axis[0])/self.auto_focus.step_z
                    zcondition = float(zpos/zlen)>0.4 and float(zpos)/zlen
                    for nfocus in range(25):
                        if zcondition and self.auto_focus.data_z.max()-self.auto_focus.data_z.min()>70:
                            break
                        else:
                            self.auto_focus.submit()
                            time.sleep(8.0)
                            z_axis = np.sort(self.auto_focus.Z)
                            zlen=(z_axis[-1]-z_axis[0])/self.auto_focus.step_z
                            zpos=(self.auto_focus.zfit-z_axis[0])/self.auto_focus.step_z
                            zcondition = float(zpos/zlen)>0.4 and  float(zpos/zlen)<0.6
                    self.auto_focus.fit_xy()
                    time.sleep(1.0)       

                    if self.auto_focus.xfit - 0.25<0:
                        self.confocal.x1=0
                        self.confocal.x2=2*self.auto_focus.xfit
                        self.confocal.y1=self.auto_focus.yfit - self.auto_focus.xfit
                        self.confocal.y2=self.auto_focus.yfit + self.auto_focus.xfit
                    elif self.auto_focus.yfit - 0.25<0:
                        self.auto_focus.y1=0
                        self.auto_focus.y2=2*self.auto_focus.yfit
                        self.confocal.x1=self.auto_focus.xfit - self.auto_focus.yfit
                        self.confocal.x2=self.auto_focus.xfit + self.auto_focus.yfit
                    elif self.auto_focus.xfit + 0.25>50:  
                        self.confocal.x2=50
                        self.confocal.x1=2*self.auto_focus.xfit-50
                        self.confocal.y1=self.auto_focus.yfit + self.auto_focus.xfit - 50
                        self.confocal.y2=self.auto_focus.yfit + 50 - self.auto_focus.xfit
                    elif  self.auto_focus.yfit + 0.25>50:    
                        self.confocal.y2=50
                        self.confocal.y1=2*self.auto_focus.yfit-50
                        self.confocal.x1=self.auto_focus.xfit + self.auto_focus.yfit - 50
                        self.confocal.x2=self.auto_focus.xfit + 50 - self.auto_focus.yfit
                    else:    
                        self.confocal.x1=self.auto_focus.xfit - 0.25
                        self.confocal.x2=self.auto_focus.xfit + 0.25
                        self.confocal.y1=self.auto_focus.yfit - 0.25
                        self.confocal.y2=self.auto_focus.yfit + 0.25
                    self.confocal.submit()
                    time.sleep(140)
                    
                    self.gs._ImportImage_fired()
                    time.sleep(2.0)
                    self.gs._Getfitimage_fired()
                    file_image = file_path + '/nv' + str(ncenter) + '.png'
                    self.gs.save_image(file_image)
                    fitx=abs(self.gs.fitparemeter[4])
                    fity=abs(self.gs.fitparemeter[3])
                    spotcondition = fitx>0.125 or fity>0.125 or fitx > 1.3*fity or fity>1.3*fitx
                    if spotcondition:
                        continue
                        
                    
                    file_name = file_path + '/nv' + str(ncenter)
                    if not os.path.isdir(file_name):
                        os.mkdir(file_name)
                    else:
                        continue
                    
                    self.auto_focus.submit()
                    time.sleep(8.0)
                    self.confocal.load(file_image1)
                    time.sleep(1.0)
                    self.confocal.x = self.auto_focus.xfit
                    self.confocal.y = self.auto_focus.yfit
                    file_nv = file_name + '/image.png'
                    self.sf.confocal.save_image(file_nv)
                    time.sleep(8.0)
                    self.auto_focus.periodic_focus = True
                    
                
                    
                    fstart= 2.951e9     
                    fend= 2.965e9
                    self._odmr(fstart,fend)
                    time.sleep(1.0)
                    self.odmr.submit()   
                    time.sleep(2.0)
                    
                    while self.odmr.state != 'done':
                        #print threading.currentThread().getName()

                        threading.currentThread().stop_request.wait(1.0)
                        if threading.currentThread().stop_request.isSet():
                             break
                             
                    file_odmr = file_name + '/odmr_nv1_high'
                    self.odmr.save_line_plot(file_odmr + '.png')
                    self.odmr.save(file_odmr + '.pyd')
                    time.sleep(5.0)  
                    
                    fhigh1 = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)
                    
                    # Rabi #####################
                    from measurements.pair_search import Rabi
                    self.psawg.measurement = Rabi()
                    from analysis.pulsedawgan import RabiFit_phase   
                    self.psawg.fit = RabiFit_phase()
                    
                    self.psawg.measurement.power = 16
                    self.psawg.measurement.freq_center = 2.71e9

                    self.psawg.measurement.tau_begin = 10
                    self.psawg.measurement.tau_end = 500
                    self.psawg.measurement.tau_delta = 5
                    self.psawg.measurement.sweeps = 2e5
                    
                    self.psawg.measurement.freq = fhigh1
                    self.psawg.measurement.load()
                    time.sleep(15.0)
                    while self.psawg.measurement.reload == True:
                        threading.currentThread().stop_request.wait(1.0)
                        
                    self.psawg.measurement.submit()
                    time.sleep(5.0)
                    self.psawg.fit = RabiFit_phase()
                    time.sleep(5.0)
                    self.psawg.fit = RabiFit_phase()
                       
                    while self.psawg.measurement.state != 'done':
                         threading.currentThread().stop_request.wait(1.0)
                         if threading.currentThread().stop_request.isSet():
                            break
                     
                    file_rabi = file_name + '/rabi_nv1_high'
                    self.psawg.save_line_plot(file_rabi + '.png')
                    self.psawg.save(file_rabi + '.pyd')
                    time.sleep(10.0)
                    self.psawg.measurement.remove()
                    self.psawg.measurement.state = 'idle'            
                    time.sleep(5.0)
                    half_pi_nv1 = self.psawg.fit.t_pi2[0]
                    
                    condition = np.isnan(half_pi_nv1) or self.psawg.fit.contrast[0] < 7 or self.psawg.fit.contrast[0] > 25
                    if condition:
                        continue
                        
                    fstart= 2.784e9    
                    fend= 2.798e9
                    self.odmr.frequency_begin_p = fstart
                    self.odmr.frequency_end_p = fend
                    time.sleep(1.0)
                    self.odmr.submit()   
                    time.sleep(2.0)
                    
                    while self.odmr.state != 'done':
                        #print threading.currentThread().getName()

                        threading.currentThread().stop_request.wait(1.0)
                        if threading.currentThread().stop_request.isSet():
                             break
                             
                    file_odmr = file_name + '/odmr_nv1_low'
                    self.odmr.save_line_plot(file_odmr + '.png')
                    self.odmr.save(file_odmr + '.pyd')
                    time.sleep(5.0)  
                    
                    flow1 = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)
                    
                    # Rabi #####################                   
                    self.psawg.measurement.freq = flow1
                    self.psawg.measurement.load()
                    time.sleep(15.0)
                    while self.psawg.measurement.reload == True:
                        threading.currentThread().stop_request.wait(1.0)
                        
                    self.psawg.measurement.submit()
                    time.sleep(5.0)
                    self.psawg.fit = RabiFit_phase()
                    time.sleep(5.0)
                    self.psawg.fit = RabiFit_phase()
                       
                    while self.psawg.measurement.state != 'done':
                         threading.currentThread().stop_request.wait(1.0)
                         if threading.currentThread().stop_request.isSet():
                            break
                     
                    file_rabi = file_name+ '/rabi_nv1_low'
                    self.psawg.save_line_plot(file_rabi + '.png')
                    self.psawg.save(file_rabi + '.pyd')
                    time.sleep(10.0)
                    self.psawg.measurement.remove()
                    self.psawg.measurement.state = 'idle'            
                    time.sleep(5.0)
                    pi_p_nv1 = self.psawg.fit.t_pi[0]
                    
                    condition = np.isnan(pi_p_nv1) or self.psawg.fit.contrast[0] < 7 or self.psawg.fit.contrast[0] > 25
                    if condition:
                        continue
                        
                    # odmr    
                    self.odmr.frequency_begin_p = 2.918e9
                    self.odmr.frequency_end_p = 2.934e9
                    self.odmr.submit()   
                    time.sleep(10)
            
                    while self.odmr.state != 'done':
                        #print threading.currentThread().getName()

                        threading.currentThread().stop_request.wait(1.0)
                        if threading.currentThread().stop_request.isSet():
                             break

                    file_odmr = file_name + '/odmr_nv2_high'
                    self.odmr.save_line_plot(file_odmr + '.png')
                    self.odmr.save(file_odmr + '.pyd')
                    time.sleep(10)
                    fhigh2 = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)    
                    
                    # Rabi #####################
                    self.psawg.measurement.freq = fhigh2
                    self.psawg.measurement.load()
                    time.sleep(15.0)
                    while self.psawg.measurement.reload == True:
                        threading.currentThread().stop_request.wait(1.0)
                        
                    self.psawg.measurement.submit()
                    time.sleep(5.0)
                    self.psawg.fit = RabiFit_phase()
                       
                    while self.psawg.measurement.state != 'done':
                         threading.currentThread().stop_request.wait(1.0)
                         if threading.currentThread().stop_request.isSet():
                            break
                     
                    file_rabi = file_name + '/rabi_nv2_high'
                    self.psawg.save_line_plot(file_rabi + '.png')
                    self.psawg.save(file_rabi + '.pyd')
                    time.sleep(10.0)
                    self.psawg.measurement.remove()
                    self.psawg.measurement.state = 'idle'            
                    time.sleep(5.0)
                    half_pi_nv2 = self.psawg.fit.t_pi2[0]
                    
                    condition = np.isnan(half_pi_nv2) or self.psawg.fit.contrast[0] < 7 or self.psawg.fit.contrast[0] > 25
                    if condition:
                        continue
                        
                    self.odmr.frequency_begin_p = 2.820e9
                    self.odmr.frequency_end_p = 2.837e9
                    time.sleep(1.0)
                    self.odmr.submit()   
                    time.sleep(2.0)
                    
                    while self.odmr.state != 'done':
                        #print threading.currentThread().getName()

                        threading.currentThread().stop_request.wait(1.0)
                        if threading.currentThread().stop_request.isSet():
                             break
                             
                    file_odmr = file_name + '/odmr_nv2_low'
                    self.odmr.save_line_plot(file_odmr + '.png')
                    self.odmr.save(file_odmr + '.pyd')
                    time.sleep(5.0)  
                    
                    flow2 = sum(self.odmr.fit_frequencies)/len(self.odmr.fit_frequencies)
                    
                    # Rabi #####################                   
                    self.psawg.measurement.freq = flow2
                    self.psawg.measurement.load()
                    time.sleep(15.0)
                    while self.psawg.measurement.reload == True:
                        threading.currentThread().stop_request.wait(1.0)
                        
                    self.psawg.measurement.submit()
                    time.sleep(5.0)
                    self.psawg.fit = RabiFit_phase()
                       
                    while self.psawg.measurement.state != 'done':
                         threading.currentThread().stop_request.wait(1.0)
                         if threading.currentThread().stop_request.isSet():
                            break
                     
                    file_rabi= file_name + '/rabi_nv2_low'
                    self.psawg.save_line_plot(file_rabi + '.png')
                    self.psawg.save(file_rabi + '.pyd')
                    time.sleep(10.0)
                    self.psawg.measurement.remove()
                    self.psawg.measurement.state = 'idle'            
                    time.sleep(5.0)
                    pi_p_nv2 = self.psawg.fit.t_pi[0]
                    
                    condition = np.isnan(pi_p_nv2) or self.psawg.fit.contrast[0] < 7 or self.psawg.fit.contrast[0] > 25
                    if condition:
                        continue

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
                    self.pair.measurement.tau_end = 40000
                    self.pair.measurement.tau_delta = 1000
                    self.pair.measurement.sweeps = 2.0e5
                    
                    self.pair.measurement.load()
                    time.sleep(35.0)
                    
                    while self.pair.measurement.reload == True:
                        threading.currentThread().stop_request.wait(1.0)
                    
                    self.pair.measurement.submit()
                    
                    while self.pair.measurement.state != 'done':
                         threading.currentThread().stop_request.wait(1.0)
                         if threading.currentThread().stop_request.isSet():
                            break
                            
                    file_hahn = file_name + '/hahnpair_double'
                    self.pair.save_line_plot(file_hahn + '.png')
                    self.pair.save(file_hahn + '.pyd')
                    time.sleep(10.0)
                    self.pair.measurement.remove()
                    self.pair.measurement.state = 'idle'
                    time.sleep(20.0)       
                
                if len(self.sf.Centroids)==0:
                    self.auto_focus.periodic_focus = False    
                    self.confocal.remove_all_labels()
                    self.auto_focus.remove_all_targets()
                    self.sf._ImportImage_fired()
                    time.sleep(2.0)
                    self.sf.SpotMinInt = 10
                    self.sf.SpotMaxInt = 250
                    self.sf._ProcessBWimage_fired()
                    time.sleep(15.0)
                        
                    self.sf._ExportButton_fired()
                    
                    if len(self.sf.Centroids)>0:
                        self.confocal.y = self.sf.Centroids[-1][0]
                        self.confocal.x = self.sf.Centroids[-1][1]
                        x_last = self.confocal.x
                        y_last = self.confocal.y
                    else:
                        self.confocal.y = y_last
                        self.confocal.x = x_last
                    self.auto_focus.submit()
                    time.sleep(8.0)
                    z_axis1 = np.sort(self.auto_focus.Z)
                    zlen1=(z_axis1[-1]-z_axis1[0])/self.auto_focus.step_z
                    zpos1=(self.auto_focus.zfit-z_axis1[0])/self.auto_focus.step_z
                    zcondition1 = float(zpos1/zlen1)>0.4 and float(zpos1)/zlen1
                    for nfocus in range(25):
                        if zcondition1 and self.auto_focus.data_z.max()-self.auto_focus.data_z.min()>70:
                            break
                        else:
                            self.auto_focus.submit()
                            time.sleep(8.0)
                            z_axis1 = np.sort(self.auto_focus.Z)
                            zlen1=(z_axis1[-1]-z_axis1[0])/self.auto_focus.step_z
                            zpos1=(self.auto_focus.zfit-z_axis1[0])/self.auto_focus.step_z
                            zcondition1 = float(zpos1/zlen1)>0.4 and  float(zpos1/zlen1)<0.6
                        
                
