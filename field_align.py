
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

class Odmr_field(ManagedJob):

   def __init__(self,odmr):
        super(Odmr_field, self).__init__()     
        self.odmr = odmr
        
   def _generate_current(self,xlim,ylim,zlim,stepx=1.0,stepz=0.1):

        lx = np.linspace(-xlim,0,round(xlim*1/stepx + 1)).tolist()
        ly = np.linspace(-ylim,0,round(ylim*1/stepx + 1)).tolist()
        lz = np.linspace(-zlim,0,round(zlim*1/stepz + 1)).tolist()

        flist = []
        ordy = True
        ordz = True
        for ix in lx:
            for iy in ly :
                for iz in lz:
                    flist.append([ix,iy,iz])
                lz.reverse()
            ly.reverse()
            
        print len(flist)
        lastf = flist[0]
        dlist = []
        for f in flist:
            if np.abs(f[0]-lastf[0]) > 1.0 or np.abs(f[1]-lastf[1]) > 1.0 or np.abs(f[2]-lastf[2]) >= 1.0:
                print 'error'
            lastf = f
            dict = {'x':f[0],'y':f[1],'z':f[2]}
    #         outtext = outtext + '{'+"'x':" + str(f[0])+','+"'y':" + str(f[1])+','+"'z':" + str(f[2])+','+'},'
            dlist.append(dict)
        return dlist
        
   def _run(self):
        # get ODMR under different conditions to help align the magnetic field
        #current = self._generate_current(1,1,0.0)
        #self.current = current
        #current = [{'x': 0.0, 'y': 0.0, 'z': 0.0},{'x': 0.05, 'y': 0.0, 'z': -0.0},{'x': 0.10, 'y': 0.0, 'z': -0.0},
                                              #    {'x': 0.0, 'y': 0.05, 'z': -0.0},{'x': 0.0, 'y': 0.10, 'z': -0.0},
                                                #  {'x': 0.0, 'y': 0.0, 'z': -0.05},{'x': 0.10, 'y': 0.0, 'z': -0.0},
       # {'x': 0.10, 'y': 0.0, 'z': -0.0},
                  # {'x': 0.1, 'y': 0.0, 'z': -0.0},{'x': 0.125, 'y': 0.0, 'z': -0.0},{'x': 0.15, 'y': 0.0, 'z': -0.0}]
        #current = [{'x': -0.2, 'y': 0.0, 'z': -0.35},{'x': -0.3, 'y': 0.0, 'z': -0.35},{'x': -0.4, 'y': 0.0, 'z': -0.35},{'x': -0.5, 'y': 0.0, 'z': -0.35},{'x': -0.6, 'y': 0.0, 'z': -0.35}]

        z_coil = ha.Coil()('z')
        y_coil = ha.Coil()('y')
        #z_coil = ha.Coil()('z')
        
        freq_low = [1.0e9, 2.9046e9, 2.9538e9, 2.9568e9]
        freq_high = [2.9027e9, 2.9059e9, 2.9551e9, 2.9582e9]
        
        #freq_low = [2.790e9, 2.842e9, 2.9067e9, 2.953e9  ] 
        #freq_high = [2.795e9, 2.847e9, 2.911e9, 2.957e9  ]



        self.odmr.pulsed = True
        self.odmr.perform_fit = True
        self.odmr.power_p = -16
        self.odmr.t_pi = 1250
        self.odmr.stop_time = 60 
        
        self.odmr.frequency_delta_p = 1.5e5
        self.odmr.frequency_begin_p = 1.86e9
        self.odmr.frequency_end_p = 1.873e9
        
        self.odmr.power = -20
        self.odmr.stop_time = 40 
        
        self.odmr.frequency_delta = 1.5e6
        self.odmr.frequency_begin = 1.55e9
        self.odmr.frequency_end = 1.75e9
        
        self.odmr.number_of_resonances = 2
        self.odmr.threshold = -40

        file_pos = 'D:/data/protonNMR/membrane_2/micelle/L11_22/NV2/odmr2'
        os.path.exists(file_pos)

        current_x = np.linspace(0.05,-0.05,5)
        current_y = np.linspace(-0.15,-0.05,5)
        for tx in range(len(current_x)):
            for ty in range(len(current_y)):
                current_x_ele = current_x[tx]
                current_y_ele = current_y[ty]
                print(current_x_ele,current_y_ele)
            #current_z = current[t]['z']
                z_coil._set_current(current_x_ele)
                time.sleep(1)
                y_coil._set_current(current_y_ele)
                time.sleep(1)
            #z_coil._set_current(current_z)
            
                file_name = file_pos + '/odmr_x_'  + str(current_x_ele) +'_y_' + str(current_y_ele)
                if os.path.isfile(file_name + '.png'):
                    continue
                
    
                self.odmr.submit()   
                time.sleep(1)
    
                while self.odmr.state != 'done':
                    #print threading.currentThread().getName()

                    threading.currentThread().stop_request.wait(1.0)
                    if threading.currentThread().stop_request.isSet():
                         break

                #file_name = file_pos + '\odmr' + '_x_' + str(current_x) + '_y_' + str(current_y) + '_z_' + str(current_z)
                #file_name = file_pos + '/odmr_'  + str(t/2+1) + '_' + str(t%2+1)
                self.odmr.remove()   
                self.odmr.state = 'idle'          
                
                self.odmr.save_line_plot(file_name + '.png')
                self.odmr.save(file_name + '.pyd')
                time.sleep(3)
        
   def submit(self):
        """Submit the job to the JobManager."""
        ManagedJob.submit(self)    
