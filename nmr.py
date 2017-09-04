from tools.utility import edit_singleton
from datetime import date

from traits.api import Bool
import imp
import math
import numpy as np
import threading, time, os, logging

from tools.emod import ManagedJob
from tools.utility import GetSetItemsHandler, GetSetItemsMixin

import hardware.api as ha

class NMR(ManagedJob):
    def __init__(self,odmr,psawg,pdawg):
        super(NMR, self).__init__()     
        self.odmr = odmr
        self.psawg = psawg
        self.pdawg = pdawg
        
    def _odmr(self):
        self.x_coil = ha.Coil()('x')
        self.y_coil = ha.Coil()('y')
        self.z_coil = ha.Coil()('z')

        current_z=self.z_coil.current
        current_x=self.x_coil.current
        current_y=self.y_coil.current
        file_pos = 'D:/data/QuantumSim/parallel_1/Property/nmr/endor'       
        os.path.exists(file_pos)

        self.odmr.pulsed = True
        self.odmr.perform_fit = True
        self.odmr.power_p = -17
        self.odmr.t_pi = 2000
        self.odmr.stop_time = 100 
        
        label_f = [[2.825e9,2.835e9],  [2.9e9,2.91e9]]
        self.e_freqs = []
        for t in range(len(label_f)):
            self.odmr.frequency_begin_p = label_f[t][0]
            self.odmr.frequency_end_p = label_f[t][1]
        
            self.odmr.submit()
        
            while self.odmr.state != 'done':
                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                   break
        
        
            file_name = file_pos + '/odmr' + str(t) + '_x_' + str(current_x) + '_y_' + str(current_y) + '_z_' + str(current_z)
            self.odmr.save_line_plot(file_name + '.png')
            self.odmr.save(file_name + '.pyd')
        
            self.e_freqs.append(self.odmr.fit_frequencies[0])
            self.e_freqs.append(self.odmr.fit_frequencies[1])
            self.e_freqs.append(self.odmr.fit_frequencies[2])
            if(len(self.odmr.fit_frequencies)<3):
                logging.getLogger().exception('frequency range of odmr is too small')
            
    def _rabiawg(self, freqs=None):
        file_pos = 'D:/data/QuantumSim/luck2/nmr'
        os.path.exists(file_pos)
        
        from measurements.pulsed_awg import Rabi
        self.psawg.measurement = Rabi()
        from analysis.pulsedawgan import RabiFit
        from analysis.pulsedawgan import PulsedFit
        time.sleep(1.0)
        self.psawg.fit = RabiFit()
        if(hasattr(self.psawg.fit,'t_pi')==False):
           logging.getLogger().exception('No RabiFit')   
        
        if(freqs is not None):
           self.e_freqs = freqs
        self.psawg.measurement.power = -22
        
        self.psawg.measurement.tau_begin = 300
        self.psawg.measurement.tau_end = 8000
        self.psawg.measurement.tau_delta = 80
        self.psawg.measurement.sweeps = 1e5
        
        self.e_t_pi = []
        for t in range(len(self.e_freqs)):
            self.psawg.measurement.freq = self.e_freqs[t]
            self.psawg.measurement.load()
            time.sleep(15.0)
            self.psawg.measurement.submit()
            self.psawg.fit = PulsedFit()
            time.sleep(1.0)
            self.psawg.fit = RabiFit()
            while self.psawg.measurement.state != 'done':
                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                   break
            if threading.currentThread().stop_request.isSet():
               break       
            
            if math.isnan(self.psawg.fit.t_pi[0]):
                self.psawg.measurement.load()
                time.sleep(15.0)
                self.psawg.measurement.submit()
                while self.psawg.measurement.state != 'done':
                   threading.currentThread().stop_request.wait(1.0)
                   if threading.currentThread().stop_request.isSet():
                      break
                if threading.currentThread().stop_request.isSet():
                      break      
               
            file_name = file_pos + '/Erabi' + '_freq_' + str(t)
            self.psawg.save_line_plot(file_name + '.png')
            self.odmr.save(file_name + '.pyd')
        
            self.e_t_pi.append(self.psawg.fit.t_pi[0]) 
        
        if(len(self.e_t_pi)<len(self.e_freqs)):
            logging.getLogger().exception('go to check e_rabi, it is not completly done') 

    def _endor_sawg(self, freqs=None, t_pis=None):
        file_pos = 'D:/data/QuantumSim/luck2/nmr' 
        os.path.exists(file_pos)
        
        if(freqs is not None):
            self.e_freqs = freqs
        if(t_pis is not None):
            self.e_t_pi = t_pis    
        
        label = [[0,1],[1,2]]
        label_f = [[4.95e6,4.965e6],[4.93e6,4.96e6]]
        label_p = [0.025, 0.025] # rf have different power in different frequency
        from measurements.pulsed_awg import N14_Ms0_RF_sweep
        self.psawg.measurement = N14_Ms0_RF_sweep()
        
        self.psawg.measurement.sweeps = 3e5
        #the power should be the same with power in rabi measurement
        self.psawg.measurement.power = -26
        
        self.psawg.measurement.rf_time = 135e3
        self.psawg.measurement.wait_time = 20e3
        
        for t in range(len(label)):
           self.psawg.measurement.freq = self.e_freqs[label[t][0]]
           self.psawg.measurement.freq_2 = self.e_freqs[label[t][1]]
           self.psawg.measurement.pi_1 = self.e_t_pi[label[t][0]]
           self.psawg.measurement.pi_2 = self.e_t_pi[label[t][1]]
           
           self.psawg.measurement.amp = label_p[t]
        
           self.psawg.measurement.tau_begin = label_f[t][0]  #2.767e6
           self.psawg.measurement.tau_end = label_f[t][1]    #2.794e6
           self.psawg.measurement.tau_delta = 0.8e3
           self.psawg.measurement.load()
           time.sleep(60.0)
           self.psawg.measurement.submit()
           
           while self.psawg.measurement.state != 'done':
                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                   break
            
           file_name = file_pos + '/nmr_spec_med' + '_freq_' + str(t)
           self.psawg.save_line_plot(file_name + '.png')
           self.psawg.save(file_name + '.pyd') 
           self.psawg.measurement.remove()
           time.sleep(10.0)
           self.psawg.measurement.state = 'idle'           

    def _endor_dawg(self, freqs=None, t_pis=None):
        file_pos = 'D:/data/QuantumSim/Luck2/nmr' 
        os.path.exists(file_pos)

        if(freqs is not None):
            self.e_freqs = freqs
        if(t_pis is not None):
            self.e_t_pi = t_pis    
        
        #label = [[1,0],[4,3],[1,2],[4,5]]
        #label_f = [[7.095e6 ,7.120e6 ],[7.09e6 ,7.128e6 ],[2.762e6,2.799e6],[2.762e6,2.799e6]]
        #label_p = [0.014, 0.014, 0.028, 0.028] # rf have different power in different frequency
        
        label = [[1,2],[4,5]]
        label_f = [[2.775e6,2.795e6],[2.775e6,2.795e6]]
        label_p = [0.025, 0.025] # rf have different power in different frequency
        from measurements.pulsed_awg import Double_RF_sweep
        self.pdawg.measurement = Double_RF_sweep()
        
        self.pdawg.measurement.sweeps = 1.5e5
        #the power should be the same with power in rabi measurement
        self.pdawg.measurement.power = -26
        
        self.pdawg.measurement.rf_time = 135e3
        self.pdawg.measurement.wait_time = 20e3
        
        for t in range(len(label)):
           self.pdawg.measurement.freq = self.e_freqs[label[t][0]]
           self.pdawg.measurement.freq_2 = self.e_freqs[label[t][1]]
           self.pdawg.measurement.pi_1 = self.e_t_pi[label[t][0]]
           self.pdawg.measurement.pi_2 = self.e_t_pi[label[t][1]]
           
           self.pdawg.measurement.amp = label_p[t]
        
           self.pdawg.measurement.tau_begin = label_f[t][0]  #2.767e6
           self.pdawg.measurement.tau_end = label_f[t][1]    #2.794e6
           self.pdawg.measurement.tau_delta = 0.8e3
           self.pdawg.measurement.load()
           time.sleep(60.0)
           self.pdawg.measurement.submit()
           
           while self.pdawg.measurement.state != 'done':
                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                   break
            
           file_name = file_pos + '/nmr_spec' + '_freq_' + str(t)
           self.pdawg.save_line_plot(file_name + '.png')
           self.pdawg.save(file_name + '.pyd')
          
           self.pdawg.measurement.remove()
           time.sleep(10.0)
           self.pdawg.measurement.state = 'idle' 
                
    def _run(self):
        #First get spectrum and frequency through odmr
        '''
        freqs = [2835925221.7911105,
                 2838132648.2115655,
                 2840230505.4995027,
                 2901577479.1636772,
                 2903766092.983696,
                 2905982999.3204355
                ]
        
        t_pis = [1530,
                 1545,
                 1582,
                 1845,
                 2018,
                 2140
                ]
        '''        
        #freqs = [2.840140e9,
                # 2.842345e9,
               #  2.844460e9,
               #  2.897200e9,
               #  2.899397e9,
               #  2.901562e9
               # ]
      #  t_pis = [3311,
               #  3268,
                # 3280,
               #  3290,
               #  3585,
               #  3636
                #]        
        freqs = [2.837093e9, # 2.837038e+09
                 2.839252e9, # 2.839200e+09
                 2.841430e9, # 2.841362e+09
                 2.900839e9, # 2.900764e9
                 2.903019e9, # 2.902930e+09
                 2.905179e9 # 2.905125e+09
                ]
        t_pis = [2026,
                 1938, #-26db,  (2442 -29db)
                 2400, #10dbm and 0.015
                 2165,
                 2207,
                 2390  #10dbm and 0.017          #2838
                ]               
        #self._odmr()
        #self._rabiawg(freqs)
        self._endor_sawg(freqs,t_pis)
        self._endor_dawg(freqs,t_pis)
       