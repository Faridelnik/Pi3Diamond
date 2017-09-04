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

from hardware.waveform import *
from hardware.api import AWG


class QuantumSim_FFT(ManagedJob):
     
    def __init__(self,odmr, psawg, pdawg):
        super(QuantumSim_FFT, self).__init__()
        self.odmr = odmr
        self.psawg = psawg
        self.pdawg = pdawg
        self.fit_parameter = [] 
        self.file_pos = 'D:/data/QuantumSim/Luck2/QuantumSim/excited state/R-28'
        self.p_opt_i = []
        self.p_opt_q = []
        self.phase = []
        self.phase_it = []
        self.awg = AWG()
            
    def _opt_pulse(self):
        sampling = 1.2e9

        f3 = (self.freqs[0] - self.psawg.measurement.freq_center) / sampling
        f4 = (self.freqs[5] - self.psawg.measurement.freq_center) / sampling
        zero = Idle(1)
        p_opt_sim_i = []
        p_opt_sim_q= []
        
        datfile = '/Users/nyxu/Downloads/pulse_18step_8_Ite-2_time-0.5.dat'
        fileHandle = open (datfile) 
        #read the cotend of the file
        datfilelines=fileHandle.read()
        exec datfilelines 
        fileHandle.close() 
        
        self.p_opt_i = p_opt_sim_i
        self.p_opt_q = p_opt_sim_q
            
    def _odmr_check(self):
    
        freq_check = [[2.8366e9,2.8375e9],[2.8388e9,2.8396e9],[2.8410e9,2.8418e9],[2.9004e9,2.9012e9],[2.9025e9,2.9033e9],[2.9047e9,2.9056e9]]
        self.odmr.pulsed = True
        self.odmr.perform_fit = True
        self.odmr.number_of_resonances = 1
        self.odmr.power_p = -50
        self.odmr.t_pi = 9600
        self.odmr.stop_time = 300 
        self.odmr.frequency_delta_p = 1e4
        
        self.freqs = []
        
        for t in range(len(freq_check)):
            self.odmr.frequency_begin_p = freq_check[t][0]
            self.odmr.frequency_end_p = freq_check[t][1]
            
            while self.odmr.state is not 'idle':
                threading.currentThread().stop_request.wait(1.0)
                
            self.odmr.submit()
            
            while self.odmr.state != 'done':
                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                     break
                     
            self.freqs.append(self.odmr.fit_frequencies[0])
            
            file_name = self.file_pos + '/freq_' + '%1.0f'%t
            self.odmr.save_line_plot(file_name + '.png')
            self.odmr.save(file_name + '.pyd')
            time.sleep(10.0)
            self.odmr.remove()
            self.odmr.state = 'idle'
            time.sleep(40.0)
  
    def _Do_sim(self):
    
        from measurements.pulsed_awg import QSim_FFT_ground
        self.psawg.measurement = QSim_FFT_ground()
        
        #from analysis.pulsedawgan import DoubleRabiFit_phase
        #self.pdawg.fit = DoubleRabiFit_phase()
        
        self.psawg.measurement.power = 10
        self.psawg.measurement.wait = 85e3
        self.psawg.measurement.wait_time_pol = 30e3
        self.psawg.measurement.polaring_time = 10e3
        self.psawg.measurement.wait_time_pump = 5e3
        self.psawg.measurement.pumping_time = 150
        self.psawg.measurement.repetitions = 2
        self.psawg.measurement.sweeps = 1.5e5
        
        '''
        self.psawg.measurement.rf_freq = 2.78e6
        self.psawg.measurement.rf_time = 7.9e3
        self.psawg.measurement.amp_rf = 0.2
        self.psawg.measurement.rf_freq2 = 7.113e6
        self.psawg.measurement.rf_time2 = 8.3e3
        self.psawg.measurement.amp_rf2 = 0.18
        self.psawg.measurement.rf_freq3 = 4.943e6
        self.psawg.measurement.amp_rf3 = 0.155
        '''
        
        #self.psawg.measurement.tau_begin = 0.0
        #self.psawg.measurement.tau_end = 440.0
        #self.psawg.measurement.tau_delta = 18
        #self.psawg.measurement.phase = 0.0
        
        # freqs[2] and [5] for DNP process
        self.psawg.measurement.freq = self.freqs[2]
        self.psawg.measurement.freq_2 = self.freqs[5]
        
        # freqs[1] and [4] for opt pi plus pulse
        self.psawg.measurement.freq_3 = self.freqs[1]
        self.psawg.measurement.freq_4 = self.freqs[4]
        
        # freqs[0] and [5] for opt sim pulse
        self.psawg.measurement.freq_5 = self.freqs[0]
        
        Rdistance = [80]
        iterations = [1,2,3,4,5]
        
        sampling = 1.2e9
        f3 = (self.freqs[0] - self.psawg.measurement.freq_center) / sampling
        f4 = (self.freqs[5] - self.psawg.measurement.freq_center) / sampling
        zero = Idle(1)
        
        
        #for r in range(len(Rdistance)):
        file = 'D:/data/QuantumSim/luck2/QuantumSim/groundstate/R-' + "%1.0f"%(Rdistance[0])
        for t in range(len(iterations)):
            p_opt_sim_i = []
            p_opt_sim_q= []
            #t = t+2
            datfile = file + '/pulse_' + "%1.0f"%(Rdistance[0]) + 'step_8_Ite-' +  "%1.0f"%(iterations[t]) + 'all.dat'
            fileHandle = open (datfile) 
            #read the cotend of the file
            datfilelines=fileHandle.read()
            exec datfilelines 
            fileHandle.close() 
        
            self.p_opt_i = p_opt_sim_i
            self.p_opt_q = p_opt_sim_q
            
            self.psawg.measurement.p_opt_i = self.p_opt_i
            self.psawg.measurement.p_opt_q = self.p_opt_q
       
            self.psawg.measurement.load()
            time.sleep(65.0)
            # make sure that load operation is done
            while self.psawg.measurement.reload == True:
                 threading.currentThread().stop_request.wait(1.0)
                    
            self.psawg.measurement.submit()
            
            while self.psawg.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
            if self.psawg.measurement.state == 'done':  
                time.sleep(1.0)

                file_name = file + '/1st_ite-' + "%1.0f"%(iterations[t])
                self.psawg.save_line_plot(file_name + '.png')
                self.psawg.save(file_name + '.pyd')
                time.sleep(10.0)
                self.psawg.measurement.remove()
                self.psawg.measurement.state = 'idle'            
                        
                time.sleep(60.0)
        
    def _run(self):
        #self._odmr_check()
        self.freqs = [2.837074e9, 2.839249e9, 2.841415e9, 2.900834e9, 2.902964e9, 2.905136e9]
        self._Do_sim()