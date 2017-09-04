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

class Pair_scanning(ManagedJob):

    def __init__(self,confocal):
        #super(confocal, self).__init__()     
        self.confocal = confocal
        
    def _run(self):
        start_l = [0,10, 20, 30, 40]
        end_l = [10, 20, 30, 40 ,50]
        
        self.confocal.resolution = 150
        self.confocal.thresh_high = 280
        
        file_pos = 'D:/data/ProgGate/Images/pair_search/10_08/x_1_y_0'
        os.path.exists(file_pos)
        
        for t1 in range(len(start_l)):
            self.confocal.x1 = start_l[t1]
            self.confocal.x2 = end_l[t1]
            
            for t2 in range(len(start_l)):
                self.confocal.y1 = start_l[t2]
                self.confocal.y2 = end_l[t2]
                
                self.confocal.submit()   
                time.sleep(1)
    
                while self.confocal.state != 'idle':
                    #print threading.currentThread().getName()

                    threading.currentThread().stop_request.wait(1.0)
                    if threading.currentThread().stop_request.isSet():
                         break

                file_name = file_pos + '/x_' + str(start_l[t1]) + '_' + str(end_l[t1]) + '_y_' + str(start_l[t2]) + '_' + str(end_l[t2])
                self.confocal.save_image(file_name + '.png')
                self.confocal.save(file_name + '.pyd')
                time.sleep(10)
                
    def submit(self):
        """Submit the job to the JobManager."""
        ManagedJob.submit(self)                
        