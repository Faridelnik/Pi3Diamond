
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

class HartmannHahnscan(ManagedJob):

   def __init__(self,sensing):
        super(HartmannHahnscan, self).__init__()     
        self.sensing = sensing
       
   def _run(self):
        
        file_pos = 'D:/data/protonNMR/NV search/x_20_30_y_40_50/NV1/HHscan' 
      
        os.path.exists(file_pos)
        
        from measurements.shallow_NV import HartmannHahnampOneway
        self.sensing.measurement = HartmannHahnampOneway()
        
        self.sensing.measurement.freq = 1.67309e9
        self.sensing.measurement.pi2_1 = 39.3
        self.sensing.measurement.freq_center = 1.86e9
        self.sensing.measurement.sweeps = 5.0e5
        self.sensing.measurement.power = 12
        self.sensing.measurement.tau_begin = 0.26
        self.sensing.measurement.tau_end = 0.36
        self.sensing.measurement.tau_delta = 0.004
        
        tpump=[60000, 30000]
        twait=[30000, 60000, 90000]
        
        for k in range(len(tpump)):
            for h in range(len(twait)):
                self.sensing.measurement.tpump = tpump[k]
                self.sensing.measurement.wait = twait[h]
                time.sleep(5)
            
                self.sensing.measurement.load()   
                time.sleep(30)
                self.sensing.measurement.submit()  
    
                while self.sensing.measurement.state != 'done':
                    #print threading.currentThread().getName()
                    threading.currentThread().stop_request.wait(1.0)
                    if threading.currentThread().stop_request.isSet():
                         break

                #file_name = file_pos + '\odmr' + '_x_' + str(current_x) + '_y_' + str(current_y) + '_z_' + str(current_z)
                #file_name = file_pos + '/odmr_'  + str(t/2+1) + '_' + str(t%2+1)
                #file_name = file_pos + '/N='  + str(Nrep[k]) +'_proton'
                file_name = file_pos + '/HHscan_one_way_amp_0.26_0.36'  + '_tpump=' + str( tpump[k]) + '_twait=' + str( twait[h]) 
                self.sensing.save_line_plot(file_name + '.png')
                self.sensing.save(file_name + '.pyd')
                time.sleep(30)
            
   def submit(self):
        """Submit the job to the JobManager."""
        ManagedJob.submit(self)    
