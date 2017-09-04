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

class Auto_correlation(ManagedJob):

    def __init__(self,sensing=None):
        super(Auto_correlation, self).__init__()     
        
        if sensing is not None:
            self.sensing = sensing
            
            
    def _run(self):
        
        file_path ='D:/data/protonNMR/membrane_2/micelles4/5keV/M22/NV5/correlation'
        
        os.path.exists(file_path)
        
        self.sensing.measurement.load()
        time.sleep(90.0)
        
        while self.sensing.measurement.reload == True:  
            threading.currentThread().stop_request.wait(1.0)
        
        self.sensing.measurement.submit()
        
        while self.sensing.measurement.state != 'done':
             threading.currentThread().stop_request.wait(1.0)
             if threading.currentThread().stop_request.isSet():
                break
                
             if  self.sensing.measurement.state == 'error':
                 print 'Error!!!'
                 time.sleep(4)
                 self.sensing.measurement.resubmit()
             

        self.sensing.measurement.progress = 0
        self.sensing.measurement.elapsed_sweeps = 0               
        file_XY = file_path + '/correlation_' + str(self.sensing.measurement.tau_begin/1000) + '_' + str(self.sensing.measurement.tau_end/1000) + 'us'
        self.sensing.save_line_plot(file_XY + '.png')
        self.sensing.save(file_XY + '.pyd')
        time.sleep(10.0)
        self.sensing.measurement.remove()
        self.sensing.measurement.state = 'idle'
        time.sleep(10.0)   
        
        self.auto_focus.periodic_focus = False
   
                        
                
