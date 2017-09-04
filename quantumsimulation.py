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


class QuantumSim(ManagedJob):
     
    def __init__(self,odmr, psawg, pdawg):
        super(QuantumSim, self).__init__()
        self.odmr = odmr
        self.psawg = psawg
        self.pdawg = pdawg
        self.fit_parameter = [] 
        self.file_pos = 'D:/data/QuantumSim/Luck2/QuantumSim/groundstate/R-28'
        self.p_opt_i = []
        self.p_opt_q = []
        self.phase = []
        self.phase_it = []
        self.awg = AWG()
            
    def _opt_pulse(self):
        sampling = 1.2e9
        f3 = (self.freqs[0] - self.pdawg.measurement.freq_center) / sampling
        f4 = (self.freqs[5] - self.pdawg.measurement.freq_center) / sampling
        zero = Idle(1)
        p_opt_R_28_sim_i = []
        p_opt_R_28_sim_q = []
        p_opt_R_28_sim_i.append([zero,
                    Sin(168,f3,0.54333,0.0021135) + Sin(168,f4,2.2428,0.0051408), 
                    Sin(168,f3,0.85952,0.076092) + Sin(168,f4,1.333,0.087347), 
                    Sin(168,f3,-1.2307,0.0614) + Sin(168,f4,0.42206,0.095354), 
                    Sin(168,f3,-0.70544,0.13511) + Sin(168,f4,2.3434,0.15725), 
                    Sin(168,f3,-2.7594,0.10869) + Sin(168,f4,0.76589,0.15101), 
                    Sin(168,f3,-1.8126,0.10095) + Sin(168,f4,2.5275,0.13912), 
                    Sin(168,f3,2.3475,0.14498) + Sin(168,f4,1.4712,0.11389), 
                    Sin(168,f3,2.8111,0.017379) + Sin(168,f4,3.0716,0.053769), 
                    Sin(168,f3,0.82933,0.10246) + Sin(168,f4,2.3614,0.11445), 
                    Sin(168,f3,1.8535,0.00865) + Sin(168,f4,0.59439,0.016458), 
                    zero,
                    ])
        p_opt_R_28_sim_q.append([zero,
                    Sin(168,f3,2.1141,0.0021135) + Sin(168,f4,3.8136,0.0051408), 
                    Sin(168,f3,2.4303,0.076092) + Sin(168,f4,2.9038,0.087347), 
                    Sin(168,f3,0.34009,0.0614) + Sin(168,f4,1.9929,0.095354), 
                    Sin(168,f3,0.86536,0.13511) + Sin(168,f4,3.9142,0.15725), 
                    Sin(168,f3,-1.1886,0.10869) + Sin(168,f4,2.3367,0.15101), 
                    Sin(168,f3,-0.24177,0.10095) + Sin(168,f4,4.0983,0.13912), 
                    Sin(168,f3,3.9183,0.14498) + Sin(168,f4,3.042,0.11389), 
                    Sin(168,f3,4.3819,0.017379) + Sin(168,f4,4.6424,0.053769), 
                    Sin(168,f3,2.4001,0.10246) + Sin(168,f4,3.9322,0.11445), 
                    Sin(168,f3,3.4243,0.00865) + Sin(168,f4,2.1652,0.016458), 
                    zero,
                    ])
        p_opt_R_28_sim_i.append([zero,
                    Sin(168,f3,0.59069,-0.00021653) + Sin(168,f4,1.0381,-0.00084793), 
                    Sin(168,f3,1.6191,0.048946) + Sin(168,f4,-1.1364,0.091698), 
                    Sin(168,f3,2.3459,0.11062) + Sin(168,f4,-0.34923,0.045447), 
                    Sin(168,f3,-1.826,0.073413) + Sin(168,f4,0.76177,0.15683), 
                    Sin(168,f3,2.279,0.13651) + Sin(168,f4,-3.0772,0.10253), 
                    Sin(168,f3,-2.0106,0.11077) + Sin(168,f4,1.5777,0.10934), 
                    Sin(168,f3,1.9551,0.094446) + Sin(168,f4,-2.3403,0.15788), 
                    Sin(168,f3,-2.3868,0.10705) + Sin(168,f4,-1.3309,0.038084), 
                    Sin(168,f3,-2.2182,0.022185) + Sin(168,f4,-0.66417,0.10518), 
                    Sin(168,f3,1.5978,8.1483e-05) + Sin(168,f4,-1.0824,-0.00086169), 
                    zero,
                    ])
        p_opt_R_28_sim_q.append([zero,
                    Sin(168,f3,2.1615,-0.00021653) + Sin(168,f4,2.6089,-0.00084793), 
                    Sin(168,f3,3.1899,0.048946) + Sin(168,f4,0.43439,0.091698), 
                    Sin(168,f3,3.9167,0.11062) + Sin(168,f4,1.2216,0.045447), 
                    Sin(168,f3,-0.25517,0.073413) + Sin(168,f4,2.3326,0.15683), 
                    Sin(168,f3,3.8498,0.13651) + Sin(168,f4,-1.5064,0.10253), 
                    Sin(168,f3,-0.43983,0.11077) + Sin(168,f4,3.1485,0.10934), 
                    Sin(168,f3,3.5259,0.094446) + Sin(168,f4,-0.76955,0.15788), 
                    Sin(168,f3,-0.81601,0.10705) + Sin(168,f4,0.23986,0.038084), 
                    Sin(168,f3,-0.64741,0.022185) + Sin(168,f4,0.90663,0.10518), 
                    Sin(168,f3,3.1686,8.1483e-05) + Sin(168,f4,0.48838,-0.00086169), 
                    zero,
                    ])      
        p_opt_R_28_sim_i.append([zero,
                    Sin(168,f3,0.72221,0.0060384) + Sin(168,f4,-1.2199,0.011965), 
                    Sin(168,f3,-3.0899,0.013787) + Sin(168,f4,1.3078,0.098492), 
                    Sin(168,f3,-2.2147,0.052671) + Sin(168,f4,-0.053853,0.06051), 
                    Sin(168,f3,-1.1326,0.11331) + Sin(168,f4,2.3933,0.15554), 
                    Sin(168,f3,2.2793,0.13154) + Sin(168,f4,1.0211,0.13131), 
                    Sin(168,f3,-0.8508,0.13688) + Sin(168,f4,2.3089,0.13105), 
                    Sin(168,f3,2.3981,0.13047) + Sin(168,f4,0.77622,0.15642), 
                    Sin(168,f3,-1.7319,0.025595) + Sin(168,f4,2.6141,0.06699), 
                    Sin(168,f3,-0.79343,0.013609) + Sin(168,f4,1.765,0.084347), 
                    Sin(168,f3,2.9877,0.0049379) + Sin(168,f4,1.4673,0.022917), 
                    zero,
                    ])
        p_opt_R_28_sim_q.append([zero,                               
                    Sin(168,f3,2.293,0.0060384) + Sin(168,f4,0.35086,0.011965), 
                    Sin(168,f3,-1.5191,0.013787) + Sin(168,f4,2.8786,0.098492), 
                    Sin(168,f3,-0.64386,0.052671) + Sin(168,f4,1.5169,0.06051), 
                    Sin(168,f3,0.4382,0.11331) + Sin(168,f4,3.9641,0.15554), 
                    Sin(168,f3,3.8501,0.13154) + Sin(168,f4,2.5919,0.13131), 
                    Sin(168,f3,0.72,0.13688) + Sin(168,f4,3.8797,0.13105), 
                    Sin(168,f3,3.9689,0.13047) + Sin(168,f4,2.347,0.15642), 
                    Sin(168,f3,-0.1611,0.025595) + Sin(168,f4,4.1849,0.06699), 
                    Sin(168,f3,0.77737,0.013609) + Sin(168,f4,3.3358,0.084347), 
                    Sin(168,f3,4.5585,0.0049379) + Sin(168,f4,3.0381,0.022917), 
                    zero,
                    ])  
        p_opt_R_28_sim_i.append([zero,
                    Sin(168,f3,3.1354,0.027286) + Sin(168,f4,-2.9363,0.018704), 
                    Sin(168,f3,-2.5696,0.077803) + Sin(168,f4,0.88248,0.13482), 
                    Sin(168,f3,-2.6353,0.054154) + Sin(168,f4,-0.41149,0.060189), 
                    Sin(168,f3,-0.76675,0.14346) + Sin(168,f4,2.4121,0.13543), 
                    Sin(168,f3,2.496,0.12701) + Sin(168,f4,1.1943,0.12138), 
                    Sin(168,f3,-0.78217,0.14743) + Sin(168,f4,1.9439,0.12203), 
                    Sin(168,f3,2.0541,0.1185) + Sin(168,f4,0.785,0.16107), 
                    Sin(168,f3,-2.5036,0.051789) + Sin(168,f4,3.0667,0.086475), 
                    Sin(168,f3,1.9291,0.046228) + Sin(168,f4,1.9253,0.10254), 
                    Sin(168,f3,-3.1296,0.025447) + Sin(168,f4,1.6098,0.035384),  
                    zero,
                    ])
        p_opt_R_28_sim_q.append([zero,
                    Sin(168,f3,4.7062,0.027286) + Sin(168,f4,-1.3655,0.018704), 
                    Sin(168,f3,-0.99884,0.077803) + Sin(168,f4,2.4533,0.13482), 
                    Sin(168,f3,-1.0645,0.054154) + Sin(168,f4,1.1593,0.060189), 
                    Sin(168,f3,0.80404,0.14346) + Sin(168,f4,3.9829,0.13543), 
                    Sin(168,f3,4.0668,0.12701) + Sin(168,f4,2.7651,0.12138), 
                    Sin(168,f3,0.78862,0.14743) + Sin(168,f4,3.5147,0.12203), 
                    Sin(168,f3,3.6249,0.1185) + Sin(168,f4,2.3558,0.16107), 
                    Sin(168,f3,-0.93279,0.051789) + Sin(168,f4,4.6374,0.086475), 
                    Sin(168,f3,3.4999,0.046228) + Sin(168,f4,3.4961,0.10254), 
                    Sin(168,f3,-1.5588,0.025447) + Sin(168,f4,3.1806,0.035384),
                    zero,
                    ])      
        p_opt_R_28_sim_i.append([zero,
                    Sin(168,f3,0.58242,0.014973) + Sin(168,f4,-1.9778,0.025495), 
                    Sin(168,f3,2.2923,0.11182) + Sin(168,f4,2.3282,0.12324), 
                    Sin(168,f3,2.855,0.045033) + Sin(168,f4,0.73328,0.092636), 
                    Sin(168,f3,2.1384,0.12371) + Sin(168,f4,2.4033,0.15163), 
                    Sin(168,f3,-2.4305,0.1368) + Sin(168,f4,1.1732,0.11297), 
                    Sin(168,f3,0.32662,0.092831) + Sin(168,f4,-0.74199,0.098007), 
                    Sin(168,f3,-2.2212,0.1308) + Sin(168,f4,0.72815,0.15225), 
                    Sin(168,f3,-0.47692,0.10497) + Sin(168,f4,-0.8332,0.1337), 
                    Sin(168,f3,-1.0944,0.037171) + Sin(168,f4,1.2346,0.025351), 
                    Sin(168,f3,-2.2043,0.025643) + Sin(168,f4,-0.35702,0.026076),                             
                    zero,
                    ])
        p_opt_R_28_sim_q.append([zero,
                    Sin(168,f3,2.1532,0.014973) + Sin(168,f4,-0.40696,0.025495), 
                    Sin(168,f3,3.8631,0.11182) + Sin(168,f4,3.899,0.12324), 
                    Sin(168,f3,4.4258,0.045033) + Sin(168,f4,2.3041,0.092636), 
                    Sin(168,f3,3.7092,0.12371) + Sin(168,f4,3.9741,0.15163), 
                    Sin(168,f3,-0.85968,0.1368) + Sin(168,f4,2.744,0.11297), 
                    Sin(168,f3,1.8974,0.092831) + Sin(168,f4,0.8288,0.098007), 
                    Sin(168,f3,-0.65039,0.1308) + Sin(168,f4,2.2989,0.15225), 
                    Sin(168,f3,1.0939,0.10497) + Sin(168,f4,0.7376,0.1337), 
                    Sin(168,f3,0.47636,0.037171) + Sin(168,f4,2.8054,0.025351), 
                    Sin(168,f3,-0.63349,0.025643) + Sin(168,f4,1.2138,0.026076), 
                    zero,
                    ])                       
                    
        '''            
        
                    Sin(168,f3,3.1083,0.0067614) + Sin(168,f4,-1.6201,0.0010093), 
                    Sin(168,f3,0.20031,0.074036) + Sin(168,f4,-1.8981,0.066627), 
                    Sin(168,f3,-3.0457,0.058239) + Sin(168,f4,-0.50954,0.1069), 
                    Sin(168,f3,-0.80466,0.14588) + Sin(168,f4,-1.9712,0.12243), 
                    Sin(168,f3,2.948,0.061007) + Sin(168,f4,2.3747,0.15268), 
                    Sin(168,f3,1.705,0.069761) + Sin(168,f4,-2.3904,0.14073), 
                    Sin(168,f3,0.79876,0.14548) + Sin(168,f4,2.1897,0.13934), 
                    Sin(168,f3,-1.8886,0.064485) + Sin(168,f4,-0.10998,0.078904), 
                    Sin(168,f3,-0.47958,0.080846) + Sin(168,f4,2.7775,0.082521), 
                    Sin(168,f3,-1.5165,0.018114) + Sin(168,f4,0.6208,0.010683), 
                    zero,
                    ])

                    
        '''            
        self.p_opt_i = p_opt_R_28_sim_i
        self.p_opt_q = p_opt_R_28_sim_q
            
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
    
        from measurements.pulsed_awg import QSim_Ref
        self.pdawg.measurement = QSim_Ref()
        
        from analysis.pulsedawgan import DoubleRabiFit_phase
        self.pdawg.fit = DoubleRabiFit_phase()
        
        self.pdawg.measurement.power = 10
        self.pdawg.measurement.rf_freq = 2.78e6
        self.pdawg.measurement.rf_time = 7.9e3
        self.pdawg.measurement.amp_rf = 0.2
        self.pdawg.measurement.rf_time2 = 7.9e3
        self.pdawg.measurement.amp_rf2 = 0.2
        self.pdawg.measurement.wait = 85e3
        self.pdawg.measurement.wait_time_pol = 30e3
        self.pdawg.measurement.polaring_time = 10e3
        self.pdawg.measurement.wait_time_pump = 5e3
        self.pdawg.measurement.pumping_time = 150
        self.pdawg.measurement.repetitions = 2
        
        self.pdawg.measurement.tau_begin = 0.0
        self.pdawg.measurement.tau_end = 440.0
        self.pdawg.measurement.tau_delta = 18
        self.pdawg.measurement.sweeps = 5.0e5
        self.pdawg.measurement.phase = 0.0
        
        # freqs[2] and [5] for DNP process
        self.pdawg.measurement.freq = self.freqs[2]
        self.pdawg.measurement.freq_2 = self.freqs[5]
        
        # freqs[1] and [4] for opt pi plus pulse
        self.pdawg.measurement.freq_3 = self.freqs[1]
        self.pdawg.measurement.freq_4 = self.freqs[4]
        
        # freqs[0] and [5] for opt sim pulse
        self.pdawg.measurement.freq_5 = self.freqs[0]
        
        for t in range(len(self.p_opt_i)):
            #t = t+2
            self.pdawg.measurement.p_opt_i = self.p_opt_i[t]
            self.pdawg.measurement.p_opt_q = self.p_opt_q[t]
       
            self.pdawg.measurement.load()
            time.sleep(55.0)
            # make sure that load operation is done
            while self.pdawg.measurement.reload == True:
                 threading.currentThread().stop_request.wait(1.0)
                    
            self.pdawg.measurement.submit()
            
            while self.pdawg.measurement.state != 'done':
                 threading.currentThread().stop_request.wait(1.0)
                 if threading.currentThread().stop_request.isSet():
                    break
            if self.pdawg.measurement.state == 'done':  
                time.sleep(1.0)
                qph = (self.pdawg.fit.phase_2[0] - self.pdawg.fit.phase[0])%-360
                self.phase.append(qph)
                ph = self.phase[t]/-45.0
                ph = round(ph)%8

                if t == 0:
                    self.phase_it.append(self.phase[0]/8)
                else:
                   
                    self.phase_it.append((self.phase_it[t-1] - 45 * ph)/8)
                    
                self.pdawg.measurement.phase = self.phase_it[t]    
                    
                    
                self.fit_parameter.append([self.pdawg.fit.period, self.pdawg.fit.phase, self.pdawg.fit.period_2, self.pdawg.fit.phase_2])
                file_name = self.file_pos + '/iteration_' + "%1.0f"%(len(self.p_opt_i) - t)
                self.pdawg.save_line_plot(file_name + '.png')
                self.pdawg.save(file_name + '.pyd')
                time.sleep(10.0)
                self.pdawg.measurement.remove()
                self.pdawg.measurement.state = 'idle'
                
                #self.awg.stop()
                '''
                files = []
                files.append('DNP_I.WFM')
                files.append('DNP_Q.WFM')
                files.append('DNP_rf.WFM')
                files.append('Read_mw_I.WFM')
                files.append('Read_mw_Q.WFM')
                files.append('Read_mw_rf.WFM')
                for k in self.pdawg.measurement.tau:
                    name_i = 'ref_sup_i_%04i.WFM'%k
                    name_q = 'ref_sup_q_%04i.WFM'%k
                    name_rf = 'ref_sup_rf_%04i.WFM'%k
                    files.append(name_i)
                    files.append(name_q)
                    files.append(name_rf)
                    
                    name_i = 'qs_sup_i_%04i.WFM'%k
                    name_q = 'qs_sup_q_%04i.WFM'%k
                    name_rf = 'qs_sup_rf_%04i.WFM'%k
                    files.append(name_i)
                    files.append(name_q)
                    files.append(name_rf)
                self.awg.delete(files)  
                '''                
                        
                time.sleep(60.0)
        
    def _run(self):
        #self._odmr_check()
        self.freqs = [2.837074e9, 2.839249e9, 2.841415e9, 2.900834e9, 2.902964e9, 2.905136e9]
        self._opt_pulse()
        self._Do_sim()