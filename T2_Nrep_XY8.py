
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

class T2_Nrep(ManagedJob):

   def __init__(self,sensing):
        super(T2_Nrep, self).__init__()     
        self.sensing = sensing
       
   def _run(self):
        
        file_pos = 'D:/data/protonNMR/NV search/x_20_30_y_40_50/NV1/proton_signal_dynamics'
        os.path.exists(file_pos)
        
        #Nrep = [40, 44, 48, 52]
        Nrep = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
        tau_end = [12000,8000, 6000, 5000, 4000, 3000, 2500, 2000, 2000]
        tau_step = [600, 400,  300,  250, 200,  150,  130,  100,  100]
        
        power = 12
        wait_time = 8e3
        freq = 1.897312e9
        pi2_1 = 40.34
        pi_1 = 80.68
        tau_begin = 80
        tau_end = 120
        tau_delta = 1
        tau_ref_flag = True
        tau_ref_int = 20
        sweeps = 6.0e5
        
        for k in range(len(Nrep)):
            from measurements.shallow_NV import XY8
            self.sensing.measurement = XY8()
            time.sleep(2)
            self.sensing.measurement.power = power
            self.sensing.measurement.wait = wait_time
            self.sensing.measurement.freq = freq
            self.sensing.measurement.pi2_1 = pi2_1
            self.sensing.measurement.pi_1 = pi_1
            self.sensing.measurement.tau_begin = tau_begin
            self.sensing.measurement.tau_end = tau_end
            self.sensing.measurement.tau_delta = tau_delta
            self.sensing.measurement.tau_ref_flag = tau_ref_flag
            self.sensing.measurement.tau_ref_int = tau_ref_int
            self.sensing.measurement.sweeps = sweeps
            
            self.sensing.measurement.pulse_num = Nrep[k]
            #self.sensing.measurement.tau_end = tau_end[k]
            #self.sensing.measurement.tau_delta = tau_step[k]
            time.sleep(5)
            #self.sensing.measurement.sweeps = 1.0e4
            self.sensing.measurement.load()   
            time.sleep(120)
            self.sensing.measurement.submit()  
    
            while self.sensing.measurement.state != 'done':
                #print threading.currentThread().getName()
                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break

            #file_name = file_pos + '\odmr' + '_x_' + str(current_x) + '_y_' + str(current_y) + '_z_' + str(current_z)
            #file_name = file_pos + '/odmr_'  + str(t/2+1) + '_' + str(t%2+1)
            #file_name = file_pos + '/N='  + str(Nrep[k]) +'_proton'
            file_name = file_pos + '/dynamics_XY8_N='  + str(Nrep[k]) + '_pi_' + str(self.sensing.measurement.power)
            self.sensing.save_line_plot(file_name + '.png')
            self.sensing.save(file_name + '.pyd')
            time.sleep(90)
            
            '''
            from measurements.shallow_NV import XY4
            self.sensing.measurement = XY4()
            time.sleep(2)
            self.sensing.measurement.power = power
            self.sensing.measurement.wait = wait_time
            self.sensing.measurement.freq = freq
            self.sensing.measurement.pi2_1 = pi2_1
            self.sensing.measurement.pi_1 = pi_1
            self.sensing.measurement.tau_begin = tau_begin
            self.sensing.measurement.tau_end = tau_end
            self.sensing.measurement.tau_delta = tau_delta
            self.sensing.measurement.tau_ref_flag = tau_ref_flag
            self.sensing.measurement.tau_ref_int = tau_ref_int
            self.sensing.measurement.sweeps = sweeps
            
            self.sensing.measurement.pulse_num = Nrep[k]*2
            #self.sensing.measurement.tau_end = tau_end[k]
            #self.sensing.measurement.tau_delta = tau_step[k]
            time.sleep(8)
            #self.sensing.measurement.sweeps = 1.0e4
            self.sensing.measurement.load()   
            time.sleep(120)
            self.sensing.measurement.submit()  
    
            while self.sensing.measurement.state != 'done':
                #print threading.currentThread().getName()
                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break

            #file_name = file_pos + '\odmr' + '_x_' + str(current_x) + '_y_' + str(current_y) + '_z_' + str(current_z)
            #file_name = file_pos + '/odmr_'  + str(t/2+1) + '_' + str(t%2+1)
            #file_name = file_pos + '/N='  + str(Nrep[k]) +'_proton'
            file_name = file_pos + '/T2_XY4_N='  + str(Nrep[k]) + '_pi_' + str(self.sensing.measurement.power)
            self.sensing.save_line_plot(file_name + '.png')
            self.sensing.save(file_name + '.pyd')
            time.sleep(90)    
            
            from measurements.shallow_NV import CPMG4
            self.sensing.measurement = CPMG4()
            time.sleep(2)
            self.sensing.measurement.power = power
            self.sensing.measurement.wait = wait_time
            self.sensing.measurement.freq = freq
            self.sensing.measurement.pi2_1 = pi2_1
            self.sensing.measurement.pi_1 = pi_1
            self.sensing.measurement.tau_begin = tau_begin
            self.sensing.measurement.tau_end = tau_end
            self.sensing.measurement.tau_delta = tau_delta
            self.sensing.measurement.tau_ref_flag = tau_ref_flag
            self.sensing.measurement.tau_ref_int = tau_ref_int
            self.sensing.measurement.sweeps = sweeps
            
            self.sensing.measurement.pulse_num = Nrep[k]*2
            #self.sensing.measurement.tau_end = tau_end[k]
            #self.sensing.measurement.tau_delta = tau_step[k]
            time.sleep(5)
            #self.sensing.measurement.sweeps = 1.0e4
            self.sensing.measurement.load()   
            time.sleep(120)
            self.sensing.measurement.submit()  
    
            while self.sensing.measurement.state != 'done':
                #print threading.currentThread().getName()
                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break

            #file_name = file_pos + '\odmr' + '_x_' + str(current_x) + '_y_' + str(current_y) + '_z_' + str(current_z)
            #file_name = file_pos + '/odmr_'  + str(t/2+1) + '_' + str(t%2+1)
            #file_name = file_pos + '/N='  + str(Nrep[k]) +'_proton'
            file_name = file_pos + '/T2_CPMG4_N='  + str(Nrep[k]) + '_pi_' + str(self.sensing.measurement.power)
            self.sensing.save_line_plot(file_name + '.png')
            self.sensing.save(file_name + '.pyd')
            time.sleep(90)        
            '''
            
        
   def submit(self):
        """Submit the job to the JobManager."""
        ManagedJob.submit(self)    
