__version__ = '12.12.04'
import numpy as np

from enthought.traits.api import SingletonHasTraits, Instance, Property, Range, Float, Int, Bool, Array, List, Enum, Trait,\
                                 Button, on_trait_change, cached_property, Code
from enthought.traits.ui.api import View, Item, HGroup, VGroup, Tabbed, EnumEditor, RangeEditor, TextEditor
from enthought.enable.api import Component, ComponentEditor
from enthought.chaco.api import ArrayDataSource, LinePlot, LinearMapper, ArrayPlotData, Plot, CMapImagePlot
from enthought.chaco.tools.api import PanTool, ZoomTool
from enthought.chaco.tools.cursor_tool import CursorTool, CursorTool1D

import time

import logging

from tools.emod import ManagedJob

from tools.utility import GetSetItemsMixin

from hardware.api import PulseGenerator 
from hardware.api import Microwave 
from hardware.api import AWG, FastComtec

from hardware.awg import *
from hardware.waveform import *
from measurements.pulsed_awg import Pulsed

PG = PulseGenerator()
MW = Microwave()
FC = FastComtec()
AWG = AWG()
#awg_device = AWG(('192.168.0.44', 4000))
#import SMIQ_2 as RF
def find_laser_pulses(sequence):
    n = 0
    prev = []
    for channels, t in sequence:
        if 'laser' in channels and not 'laser' in prev:
            n += 1
        prev = channels
        if 'sequence' in channels:
            break
    return n

                       
class DEERpair(Pulsed):                   

    
    freq_center = Range(low=1, high=20e9, value=2.61e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_2 = Range(low=1, high=20e9, value=2.845829e9, desc='frequency 2nd trans [Hz]', label='freq2 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_3 = Range(low=1, high=20e9, value=2.907644e9, desc='frequency 3nd trans [Hz]', label='freq3 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_4 = Range(low=1, high=20e9, value=2.955537e9, desc='frequency 4th trans [Hz]', label='freq4 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi nv1', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi nv1', mode='text', auto_set=False, enter_set=True)
    pi_2   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi nv2', mode='text', auto_set=False, enter_set=True)
    tau1 = Range(low=1, high=50e4, value=20e4, desc='first tau in hahn echo [ns]', label='tau1 [ns]', mode='text', auto_set=False, enter_set=True)
    tau_begin = Range(low=0., high=1e8, value=500., desc='tau begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau end [ns]', label='tau end < tau1 [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=1e8, value=50., desc='delta tau [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    
    reload = True
    
    def _init_(self):
        super(DEERpair, self).__init__()

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            pi2_1 = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            pi_2 = int(self.pi_2 * sampling/1.0e9)
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            # ms= 0 <> ms = +1
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 , self.amp/3) + Sin( pi2_1, (self.freq_2 - self.freq_center)/sampling, 0 ,self.amp/3)  + Sin( pi2_1, (self.freq_3 - self.freq_center)/sampling, 0 ,self.amp/3)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 , self.amp/3) + Sin( pi2_1, (self.freq_2 - self.freq_center)/sampling, np.pi/2 ,self.amp/3)  + Sin( pi2_1, (self.freq_3 - self.freq_center)/sampling, np.pi/2 ,self.amp/3)]
            
            
            p['pi_1 + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 , self.amp/3) + Sin( pi_1, (self.freq_2 - self.freq_center)/sampling, 0 ,self.amp/3)  + Sin( pi_1, (self.freq_3 - self.freq_center)/sampling, 0 ,self.amp/3)]
            p['pi_1 + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 , self.amp/3) + Sin( pi_1, (self.freq_2 - self.freq_center)/sampling, np.pi/2 ,self.amp/3)  + Sin( pi_1, (self.freq_3 - self.freq_center)/sampling, np.pi/2 ,self.amp/3)]
            
            p['pi_2 + 0']     = [zero, Sin( pi_2, (self.freq_4 - self.freq_center)/sampling, 0 ,1), zero]
            p['pi_2 + 90']    = [zero, Sin( pi_2, (self.freq_4 - self.freq_center)/sampling, np.pi/2 ,1), zero]
            

               

            
            f1 = (self.freq - self.freq_center)/sampling
            f2 = (self.freq_4 - self.freq_center)/sampling
            f3= (self.freq_2 - self.freq_center)/sampling
            f4 = (self.freq_3 - self.freq_center)/sampling
            
            
            p['initial + 0'] =[zero,
             Sin(144,f1,0.58502,0.0644) + Sin(144,f2,0.98288,0.2048) + Sin(144,f3,0.84418,0.0638) + Sin(144,f4,2.8056,0.0468), 
             Sin(144,f1,2.4556,0.1166) + Sin(144,f2,-2.4733,0.2602) + Sin(144,f3,2.2039,0.1053) + Sin(144,f4,-0.54015,0.1309), 
             Sin(144,f1,0.30188,0.0314) + Sin(144,f2,-2.4032,0.0408) + Sin(144,f3,1.1345,0.026) + Sin(144,f4,-1.7801,0.0844), 
             Sin(144,f1,-1.7929,0.1055) + Sin(144,f2,-0.70983,0.2369) + Sin(144,f3,-0.89037,0.1231) + Sin(144,f4,2.2407,0.143), 
             Sin(144,f1,-2.3232,0.1208) + Sin(144,f2,2.0975,0.2198) + Sin(144,f3,-0.72607,0.1339) + Sin(144,f4,0.63346,0.1307), 
             Sin(144,f1,2.8156,0.1091) + Sin(144,f2,2.2244,0.1981) + Sin(144,f3,-0.70008,0.078) + Sin(144,f4,0.40838,0.1244), 
             Sin(144,f1,1.124,0.0408) + Sin(144,f2,-2.1718,0.2391) + Sin(144,f3,-2.2472,0.0994) + Sin(144,f4,0.78618,0.1556), 
             Sin(144,f1,0.56336,0.0386) + Sin(144,f2,1.6896,0.1126) + Sin(144,f3,1.3494,0.0892) + Sin(144,f4,2.7348,0.0889), 
             Sin(144,f1,-1.5601,0.0926) + Sin(144,f2,0.53787,0.2381) + Sin(144,f3,-0.52609,0.0837) + Sin(144,f4,-2.3477,0.162), 
             Sin(144,f1,-0.22492,0.0845) + Sin(144,f2,2.4013,0.2521) + Sin(144,f3,0.93335,0.0376) + Sin(144,f4,2.1803,0.0874),
             zero,
             ]
             
            p['initial + 90'] =  [zero,
             Sin(144,f1,2.1558,0.0644) + Sin(144,f2,2.5537,0.2048) + Sin(144,f3,2.415,0.0638) + Sin(144,f4,4.3764,0.0468), 
             Sin(144,f1,4.0264,0.1166) + Sin(144,f2,-0.90253,0.2602) + Sin(144,f3,3.7747,0.1053) + Sin(144,f4,1.0306,0.1309), 
             Sin(144,f1,1.8727,0.0314) + Sin(144,f2,-0.83236,0.0408) + Sin(144,f3,2.7053,0.026) + Sin(144,f4,-0.20933,0.0844), 
             Sin(144,f1,-0.22209,0.1055) + Sin(144,f2,0.86097,0.2369) + Sin(144,f3,0.68043,0.1231) + Sin(144,f4,3.8115,0.143), 
             Sin(144,f1,-0.75245,0.1208) + Sin(144,f2,3.6683,0.2198) + Sin(144,f3,0.84473,0.1339) + Sin(144,f4,2.2043,0.1307), 
             Sin(144,f1,4.3864,0.1091) + Sin(144,f2,3.7952,0.1981) + Sin(144,f3,0.87071,0.078) + Sin(144,f4,1.9792,0.1244), 
             Sin(144,f1,2.6948,0.0408) + Sin(144,f2,-0.60103,0.2391) + Sin(144,f3,-0.6764,0.0994) + Sin(144,f4,2.357,0.1556), 
             Sin(144,f1,2.1342,0.0386) + Sin(144,f2,3.2603,0.1126) + Sin(144,f3,2.9202,0.0892) + Sin(144,f4,4.3056,0.0889), 
             Sin(144,f1,0.010662,0.0926) + Sin(144,f2,2.1087,0.2381) + Sin(144,f3,1.0447,0.0837) + Sin(144,f4,-0.77691,0.162), 
             Sin(144,f1,1.3459,0.0845) + Sin(144,f2,3.9721,0.2521) + Sin(144,f3,2.5041,0.0376) + Sin(144,f4,3.7511,0.0874),
             zero,
             ]
             
            p['pi_1_x + 0'] = [zero,
             Sin(144,f1,2.536,0.104) + Sin(144,f2,-2.8726,0.1849) + Sin(144,f3,-2.7689,0.061) + Sin(144,f4,0.062588,0.0728), 
             Sin(144,f1,0.9057,0.1024) + Sin(144,f2,-1.3458,0.1482) + Sin(144,f3,2.1642,0.0898) + Sin(144,f4,1.2709,0.0913), 
             Sin(144,f1,-0.86897,0.1142) + Sin(144,f2,-1.0426,0.2136) + Sin(144,f3,1.3822,0.0868) + Sin(144,f4,2.2478,0.0949), 
             Sin(144,f1,2.1466,0.1031) + Sin(144,f2,-0.8671,0.232) + Sin(144,f3,0.65342,0.0776) + Sin(144,f4,3.0354,0.0781), 
             Sin(144,f1,-3.0533,0.0456) + Sin(144,f2,-1.9972,0.0968) + Sin(144,f3,-0.98415,0.0392) + Sin(144,f4,-1.9129,0.0656), 
             Sin(144,f1,-1.2305,0.0502) + Sin(144,f2,-1.7998,0.1821) + Sin(144,f3,-2.9309,0.0549) + Sin(144,f4,-0.025253,0.0856), 
             Sin(144,f1,0.34389,0.1104) + Sin(144,f2,2.8645,0.1859) + Sin(144,f3,2.2385,0.0933) + Sin(144,f4,1.3062,0.1064), 
             Sin(144,f1,2.3007,0.1336) + Sin(144,f2,2.4223,0.2283) + Sin(144,f3,1.4123,0.0868) + Sin(144,f4,2.1331,0.0911), 
             Sin(144,f1,-0.66744,0.1106) + Sin(144,f2,2.1099,0.1941) + Sin(144,f3,0.68045,0.0637) + Sin(144,f4,2.8804,0.0841), 
             Sin(144,f1,-1.1191,0.0536) + Sin(144,f2,2.8869,0.056) + Sin(144,f3,-0.77742,0.0416) + Sin(144,f4,-1.7517,0.0645),
             zero,
             ]
             
            p['pi_1_x + 90'] = [zero,
             Sin(144,f1,4.1068,0.104) + Sin(144,f2,-1.3018,0.1849) + Sin(144,f3,-1.1981,0.061) + Sin(144,f4,1.6334,0.0728), 
             Sin(144,f1,2.4765,0.1024) + Sin(144,f2,0.22505,0.1482) + Sin(144,f3,3.735,0.0898) + Sin(144,f4,2.8417,0.0913), 
             Sin(144,f1,0.70183,0.1142) + Sin(144,f2,0.52822,0.2136) + Sin(144,f3,2.953,0.0868) + Sin(144,f4,3.8186,0.0949), 
             Sin(144,f1,3.7174,0.1031) + Sin(144,f2,0.7037,0.232) + Sin(144,f3,2.2242,0.0776) + Sin(144,f4,4.6062,0.0781), 
             Sin(144,f1,-1.4825,0.0456) + Sin(144,f2,-0.42639,0.0968) + Sin(144,f3,0.58665,0.0392) + Sin(144,f4,-0.34211,0.0656), 
             Sin(144,f1,0.34034,0.0502) + Sin(144,f2,-0.22895,0.1821) + Sin(144,f3,-1.3601,0.0549) + Sin(144,f4,1.5455,0.0856), 
             Sin(144,f1,1.9147,0.1104) + Sin(144,f2,4.4353,0.1859) + Sin(144,f3,3.8093,0.0933) + Sin(144,f4,2.877,0.1064), 
             Sin(144,f1,3.8715,0.1336) + Sin(144,f2,3.9931,0.2283) + Sin(144,f3,2.9831,0.0868) + Sin(144,f4,3.7039,0.0911), 
             Sin(144,f1,0.90336,0.1106) + Sin(144,f2,3.6807,0.1941) + Sin(144,f3,2.2512,0.0637) + Sin(144,f4,4.4512,0.0841), 
             Sin(144,f1,0.4517,0.0536) + Sin(144,f2,4.4577,0.056) + Sin(144,f3,0.79338,0.0416) + Sin(144,f4,-0.1809,0.0645),
             zero,
             ]
             
            p['pi_2_x + 0'] = [zero,
             Sin(144,f1,-3.0272,0.0906) + Sin(144,f2,1.2209,0.1381) + Sin(144,f3,-0.42223,0.0798) + Sin(144,f4,-1.3039,0.0986), 
             Sin(144,f1,-1.6724,0.0879) + Sin(144,f2,0.085045,0.1316) + Sin(144,f3,1.2122,0.0849) + Sin(144,f4,3.0422,0.0942), 
             Sin(144,f1,-0.74191,0.0924) + Sin(144,f2,-0.86147,0.1751) + Sin(144,f3,0.45337,0.0474) + Sin(144,f4,-2.4789,0.0253), 
             Sin(144,f1,-0.23816,0.0692) + Sin(144,f2,-1.1028,0.1199) + Sin(144,f3,-1.8609,0.0948) + Sin(144,f4,-0.060748,0.1075), 
             Sin(144,f1,1.094,0.1099) + Sin(144,f2,-2.5506,0.192) + Sin(144,f3,2.3547,0.1253) + Sin(144,f4,2.3039,0.1487), 
             Sin(144,f1,-2.9368,0.0931) + Sin(144,f2,1.1593,0.164) + Sin(144,f3,2.1845,0.1118) + Sin(144,f4,2.4336,0.1436), 
             Sin(144,f1,-1.5646,0.0839) + Sin(144,f2,-0.15412,0.1063) + Sin(144,f3,0.47073,0.098) + Sin(144,f4,-2.3465,0.1136), 
             Sin(144,f1,-0.92989,0.0864) + Sin(144,f2,-0.70524,0.1776) + Sin(144,f3,2.6941,0.0693) + Sin(144,f4,1.8458,0.0873), 
             Sin(144,f1,-0.092414,0.0859) + Sin(144,f2,-1.4111,0.1212) + Sin(144,f3,2.8217,0.0737) + Sin(144,f4,1.6342,0.0956), 
             Sin(144,f1,1.3031,0.0947) + Sin(144,f2,-2.7334,0.1481) + Sin(144,f3,-1.714,0.0893) + Sin(144,f4,-0.54677,0.1168),
             zero,
             ]
             
            p['pi_2_x + 90'] = [zero,
             Sin(144,f1,-1.4564,0.0906) + Sin(144,f2,2.7917,0.1381) + Sin(144,f3,1.1486,0.0798) + Sin(144,f4,0.26693,0.0986), 
             Sin(144,f1,-0.10164,0.0879) + Sin(144,f2,1.6558,0.1316) + Sin(144,f3,2.783,0.0849) + Sin(144,f4,4.613,0.0942), 
             Sin(144,f1,0.82889,0.0924) + Sin(144,f2,0.70933,0.1751) + Sin(144,f3,2.0242,0.0474) + Sin(144,f4,-0.90806,0.0253), 
             Sin(144,f1,1.3326,0.0692) + Sin(144,f2,0.46796,0.1199) + Sin(144,f3,-0.2901,0.0948) + Sin(144,f4,1.51,0.1075), 
             Sin(144,f1,2.6648,0.1099) + Sin(144,f2,-0.97978,0.192) + Sin(144,f3,3.9255,0.1253) + Sin(144,f4,3.8747,0.1487), 
             Sin(144,f1,-1.366,0.0931) + Sin(144,f2,2.7301,0.164) + Sin(144,f3,3.7553,0.1118) + Sin(144,f4,4.0044,0.1436), 
             Sin(144,f1,0.0062236,0.0839) + Sin(144,f2,1.4167,0.1063) + Sin(144,f3,2.0415,0.098) + Sin(144,f4,-0.77566,0.1136), 
             Sin(144,f1,0.64091,0.0864) + Sin(144,f2,0.86556,0.1776) + Sin(144,f3,4.2649,0.0693) + Sin(144,f4,3.4166,0.0873), 
             Sin(144,f1,1.4784,0.0859) + Sin(144,f2,0.15974,0.1212) + Sin(144,f3,4.3925,0.0737) + Sin(144,f4,3.205,0.0956), 
             Sin(144,f1,2.8739,0.0947) + Sin(144,f2,-1.1626,0.1481) + Sin(144,f3,-0.14324,0.0893) + Sin(144,f4,1.024,0.1168),
             zero,
             ]
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('DEER.SEQ')
            
            sup_x = Waveform('Sup1_x', p['pi2_1 + 0'])
            sup_y = Waveform('Sup1_y', p['pi2_1 + 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            t_1 = self.tau1*1.2 - sup_x.stub
            repeat_1 = int(t_1 / 256)
            mod.duration = int(t_1 % 256)
            t_0 = sup_x.duration + repeat_1 * 256
            
            name_x = 'REF3_X%04i.WFM' % 0
            name_y = 'REF3_Y%04i.WFM' % 0
            
            ref1_x = Waveform(name_x, [mod]+ p['pi_1 + 0'], t_0)
            ref1_y = Waveform(name_y, [mod]+ p['pi_1 + 90'], t_0)
            self.waves.append(ref1_x)
            self.waves.append(ref1_y)
                                 
            for i, t in enumerate(self.tau):
                #if(i < len(self.tau) - 5):
                t_0 = sup_x.duration + repeat_1 * 256
                t_2 = t*1.2 - ref1_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += ref1_x.duration + repeat_2 * 256
                
                name_x = 'REF_X%04i.WFM' % i
                name_y = 'REF_Y%04i.WFM' % i
                
                ref_x = Waveform(name_x, [mod] + p['pi_2 + 0'], t_0)
                ref_y = Waveform(name_y, [mod] + p['pi_2 + 90'] , t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                
                t_3 = self.tau1 * 1.2 - t * 1.2 - ref_x.stub
                repeat_3 = int(t_3 / 256)
                mod.duration = int(t_3 % 256)
                #print mod.duration
                t_0 += ref_x.duration + repeat_3 * 256
                
                name_x = 'MAP3_X%04i.WFM' % i
                name_y = 'MAP3_Y%04i.WFM' % i
            
                map_x = Waveform(name_x, [mod]+ p['pi2_1 + 0'], t_0)
                map_y = Waveform(name_y, [mod]+ p['pi2_1 + 90'], t_0)
                
                self.waves.append(map_x)
                self.waves.append(map_y)
               
                name = 'DQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x, sup_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(ref1_x, ref1_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(ref_x, ref_y)
                sub_seq.append(evo, evo,repeat=repeat_3)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)
                
                
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('DEER.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )       
        
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        sequence = []
        for i, t in enumerate(tau):
            #if(i < len(self.tau) - 5):
            sub = [ (['awgTrigger'], 100 ),
                    ([], 10 * 4 * 144 + self.tau1 *2 + 500 ),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)
           # else:
               # sub = [ (['laser', 'trigger' ], laser ),
                 #       ([], wait )
                #          ]
                #sequence.extend(sub)
                
        return sequence
        
    get_set_items = Pulsed.get_set_items + ['freq','freq_2','freq_3','freq_4','vpp','amp','tau1'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                     Item('tau1', width = 20),
                                     ),             
                              HGroup(
                                     Item('pi2_1', width = 20),
                                     Item('pi_1', width = 20),
                                     Item('pi_2', width = 20)
                                     ),                    
                              HGroup(Item('freq',  width=20),
                                     Item('freq_4',  width=20),
                                     Item('freq_2',  width=20),
                                     Item('freq_3',  width=20),
                                     ),        
                              HGroup(Item('laser', width=30),
                                     Item('wait', width=30),
                                     Item('bin_width', width= 30, enabled_when='state != "run"'),
                                     Item('record_length', width= 30, enabled_when='state != "run"'),
                                     ),
                                     
                              HGroup(Item('tau_begin', width=30),
                                     Item('tau_end', width=30),
                                     Item('tau_delta', width= 30),
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30),
                                     ),
                                                                         
                              ),
                       title='DEER_pair',
                       )        

class EspinTomo_diag(Pulsed):   

    pi_1 = Range(low=1., high=100000., value=51.7, desc='length of pi pulse 1st transition [ns]', label='pi [ns] freq1', mode='text', auto_set=False, enter_set=True)
    pi_2 = Range(low=1., high=100000., value=106.2, desc='length of pi pulse 2nd transition [ns]', label='pi [ns] freq4', mode='text', auto_set=False, enter_set=True)
    pi_3 = Range(low=1., high=100000., value=47.5, desc='length of pi pulse 3st transition [ns]', label='pi [ns] freq2', mode='text', auto_set=False, enter_set=True)
    pi_4 = Range(low=1., high=100000., value=55.4, desc='length of pi pulse 4nd transition [ns]', label='pi [ns] freq3', mode='text', auto_set=False, enter_set=True)
    
    freq_center = Range(low=1, high=20e9, value=2.61e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.784617e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_2 = Range(low=1, high=20e9, value=2.848743e9, desc='frequency 2nd trans [Hz]', label='freq2 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_3 = Range(low=1, high=20e9, value=2.906787e9, desc='frequency 3nd trans [Hz]', label='freq3 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_4 = Range(low=1, high=20e9, value=2.963224e9, desc='frequency 4th trans [Hz]', label='freq4 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=0.6, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    tau_begin = Range(low=0., high=1e8, value=1, desc='tau begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=10, desc='tau end [ns]', label='tau end < tau1 [ns]', mode='text', auto_set=False, enter_set=True)  #36
    tau_delta = Range(low=1., high=1e8, value=1, desc='delta tau [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    evolution = Range(low=0., high=100000., value=50500., desc='free evolution time between creation and tomography [ns]', label='evo [ns]', mode='text', auto_set=False, enter_set=True)
    tau1 = Range(low=0., high=100000., value=250., desc=' waiting time between optimal pulse[ns]', label='tau_waiting [ns]', mode='text', auto_set=False, enter_set=True)

    p_init_i = []
    p_init_q = []
    p_gate_i = []
    p_gate_q = []
    
    p_ref_i = []
    p_ref_q = []
    reload = True
    
    istate_pulse = 'pulse_name'
    

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            f1 = (self.freq - self.freq_center)/sampling
            f2 = (self.freq_4 - self.freq_center)/sampling
            f3= (self.freq_2 - self.freq_center)/sampling
            f4 = (self.freq_3 - self.freq_center)/sampling

            # Pulses
            p = {}
            
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            pi_2 = int(self.pi_2 * sampling/1.0e9)
            pi_3 = int(self.pi_3 * sampling/1.0e9)
            pi_4 = int(self.pi_4 * sampling/1.0e9)
            
            # ms= 0 <> ms = +1
            p['pi + 0']     = Sin( pi_1, f1, 0 ,self.amp)
            p['pi + 90']    = Sin( pi_1, f1, np.pi/2 ,self.amp)
            # ms= 0 <> ms = -1
            p['pi - 0']     = Sin( pi_2, f2, 0 ,self.amp)
            p['pi - 90']    = Sin( pi_2, f2, np.pi/2 ,self.amp)
           
            p['pi + 0 2']     = Sin( pi_3, f3, 0 ,self.amp)
            p['pi + 90 2']     = Sin( pi_3, f3, np.pi/2  ,self.amp)

            p['pi - 0 2']     = Sin( pi_4, f4, 0 ,self.amp)
            p['pi - 90 2']     = Sin( pi_4, f4, np.pi/2  ,self.amp)
            
            
            #datfile = 'D:/data/ProgGate/tomo/seq/diag/stpulse.py'
            #fileHandle = open (datfile) 
            #read the cotend of the file
           # datfilelines=fileHandle.read()
           # exec datfilelines 
           # fileHandle.close() 

            zero = Idle(1)
            mod = Idle(0)
            
            # ms= 0 <> ms = +1            
            datfile = 'D:/data/ProgGate/tomo/seq/diag/tomo_seq_diag.py'
            fileHandle = open (datfile) 
            #read the cotend of the file
            datfilelines=fileHandle.read()
            exec datfilelines 
            fileHandle.close() 
            
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('ESPIN_TOMO.SEQ') 
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            # Population Reference
            name_x = '01_I.WFM' 
            name_y = '01_Q.WFM'             
            sup_x = Waveform(name_x, Idle(self.tau1*1.2))
            sup_y = Waveform(name_y, Idle(self.tau1*1.2))
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            name = 'Tomo_01.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(sup_x, sup_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)
          
            name_x = '02_I.WFM' 
            name_y = '02_Q.WFM'             
            sup_x = Waveform(name_x, p['pi - 0 2'])
            sup_y = Waveform(name_y, p['pi - 90 2'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            name = 'Tomo_02.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(sup_x, sup_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)
            
            name_x = '03_I.WFM' 
            name_y = '03_Q.WFM'             
            sup_x = Waveform(name_x, p['pi - 0'])
            sup_y = Waveform(name_y, p['pi - 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            name = 'Tomo_03.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(sup_x, sup_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)
            
            name_x = '04_I.WFM' 
            name_y = '04_Q.WFM'             
            sup_x = Waveform(name_x, p['pi_p + 0 12'])
            sup_y = Waveform(name_y, p['pi_p + 90 12'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            name = 'Tomo_04.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(sup_x, sup_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)
            
            #self.p_init_i = p['-1_-1 + 0']
            #self.p_init_q = p['-1_-1 + 90']
            # Population Signal
            #self.p_init_i = p['Plus_+1 + 0']
            #self.p_init_q = p['Plus_+1 + 90']
            name_x = 'ref_state_i'
            name_y = 'ref_state_q'
            refstate_x = Waveform(name_x, self.p_ref_i)
            refstate_y = Waveform(name_y, self.p_ref_q)
            self.waves.append(refstate_x)
            self.waves.append(refstate_y)
            
            #seq_x = self.istate_pulse + ' + 0'
            #seq_y = self.istate_pulse + ' + 90'
            #self.p_init_i = p[seq_x]
            #self.p_init_q = p[seq_y]
            
            name_x = 'istate_2_i'
            name_y = 'istate_2_q'
            istate_x = Waveform(name_x, self.p_init_i)
            istate_y = Waveform(name_y, self.p_init_q)
            self.waves.append(istate_x)
            self.waves.append(istate_y)
            
            # gate
            #datfile1 = 'D:/data/ProgGate/tomo/seq/diag' + '/cnot.py'
            #fileHandle = open (datfile1) 
            #read the cotend of the file
            #datfilelines=fileHandle.read()
            #exec datfilelines 
            #fileHandle.close()
            
            #self.p_gate_i = va_vb_i + [Idle(12600 * 1.2)] + pi_i + [Idle(12600 * 1.2)] + ua_ub_i
            #self.p_gate_q = va_vb_q + [Idle(12600 * 1.2)] + pi_q + [Idle(12600 * 1.2)] + ua_ub_q
            #self.p_gate_i = [Idle(12600 * 1.2)] + pi_i + [Idle(12600 * 1.2)] 
            #self.p_gate_q = [Idle(12600 * 1.2)] + pi_q + [Idle(12600 * 1.2)]
            

            t_0 = istate_x.duration
            name_x = 'gate_I.WFM' 
            name_y = 'gate_Q.WFM'   
            gate_x = Waveform(name_x, [Idle(self.tau1*1.2)] + self.p_gate_i, t_0)
            gate_y = Waveform(name_y, [Idle(self.tau1*1.2)] + self.p_gate_q, t_0)
            self.waves.append(gate_x)
            self.waves.append(gate_y)
                      
            t_0 += gate_x.duration          
            name_x = '05_I.WFM' 
            name_y = '05_Q.WFM'             
            #ref_x = Waveform(name_x, [zero], t_0)
            #ref_y = Waveform(name_y, [zero], t_0)
            #self.waves.append(ref_x)
            #self.waves.append(ref_y)
            name = 'Tomo_05.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(istate_x, istate_y)
            sub_seq.append(gate_x, gate_y)
            #sub_seq.append(ref_x, ref_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)
            
            
            name_x = '06_I.WFM' 
            name_y = '06_Q.WFM'             
            ref_x = Waveform(name_x, [Idle(self.tau1*1.2)] + [p['pi - 0 2']],t_0 )
            ref_y = Waveform(name_y, [Idle(self.tau1*1.2)] + [p['pi - 90 2']],t_0 )
            self.waves.append(ref_x)
            self.waves.append(ref_y)
            name = 'Tomo_06.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(istate_x, istate_y)
            sub_seq.append(gate_x, gate_y)
            sub_seq.append(ref_x, ref_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)            
            
            name_x = '07_I.WFM' 
            name_y = '07_Q.WFM'             
            ref_x = Waveform(name_x, [Idle(self.tau1*1.2)] + [p['pi + 0 2']],t_0 )
            ref_y = Waveform(name_y, [Idle(self.tau1*1.2)] + [p['pi + 90 2']],t_0 )
            self.waves.append(ref_x)
            self.waves.append(ref_y)
            name = 'Tomo_07.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(istate_x, istate_y)
            sub_seq.append(gate_x, gate_y)
            sub_seq.append(ref_x, ref_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)  
            
            name_x = '08_I.WFM' 
            name_y = '08_Q.WFM'             
            ref_x = Waveform(name_x, [Idle(self.tau1*1.2)] + [p['pi - 0']],t_0 )
            ref_y = Waveform(name_y, [Idle(self.tau1*1.2)] + [p['pi - 90']],t_0 )
            self.waves.append(ref_x)
            self.waves.append(ref_y)
            name = 'Tomo_08.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(istate_x, istate_y)
            sub_seq.append(gate_x, gate_y)
            sub_seq.append(ref_x, ref_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)  

            name_x = '09_I.WFM' 
            name_y = '09_Q.WFM'             
            ref_x = Waveform(name_x, [Idle(self.tau1*1.2)] + [p['pi + 0']],t_0 )
            ref_y = Waveform(name_y, [Idle(self.tau1*1.2)] + [p['pi + 90']],t_0 )
            self.waves.append(ref_x)
            self.waves.append(ref_y)
            name = 'Tomo_09.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(istate_x, istate_y)
            sub_seq.append(gate_x, gate_y)
            sub_seq.append(ref_x, ref_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)        
            
            '''
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Pop_1_I%04i.WFM' %t
                name_y = 'Pop_1_Q%04i.WFM' %t 
                
                pi2_tr_x = Waveform(name_x, [Idle(int(330*1.2))] +  p['pi/2_dq + 0 2'],t_0 )
                pi2_tr_y = Waveform(name_y, [Idle(int(330*1.2))] +  p['pi/2_dq + 90 2'],t_0 )
                self.waves.append(pi2_tr_x)
                self.waves.append(pi2_tr_y)
                
                t_1 = (self.evolution + t)*1.2 - pi2_tr_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi2_tr_x.duration + repeat_1 * 256
                
                name_x = 'Pop_2_I%04i.WFM' %t
                name_y = 'Pop_2_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Pop_3_I%04i.WFM' %t
                name_y = 'Pop_3_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 2'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_diag_1_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(pi2_tr_x, pi2_tr_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)   
                
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration
                name_x = 'Pop_1_I%04i.WFM' %t
                name_y = 'Pop_1_Q%04i.WFM' %t 

                pi2_tr_x = Waveform(name_x, [Idle(12600 * 1.2)] + p['pi_all + 0 2'] + [Idle(12600 * 1.2)] +  p['pi/2_dq + 0 2'],t_0 )
                pi2_tr_y = Waveform(name_y, [Idle(12600 * 1.2)] + p['pi_all + 0 2'] + [Idle(12600 * 1.2)] +  p['pi/2_dq + 90 2'],t_0 )
                self.waves.append(pi2_tr_x)
                self.waves.append(pi2_tr_y)
                
                t_0 += pi2_tr_x.duration
                name_x = 'Pop_11_I%04i.WFM' %t
                name_y = 'Pop_11_Q%04i.WFM' %t 
                
                tvar = (self.tau1 + t) * 1.2
                pi_ref_x = Waveform(name_x, [Idle(tvar)] +  [p['pi - 0']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(tvar)] +  [p['pi - 90']],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'Pop_2_I%04i.WFM' %t
                name_y = 'Pop_2_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Pop_3_I%04i.WFM' %t
                name_y = 'Pop_3_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 2'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_diag_1_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                #sub_seq.append(gate_x, gate_y)
                sub_seq.append(pi2_tr_x, pi2_tr_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)  
                '''
            '''    
            for t in np.arange(100,601,40):
                t_0 = refstate_x.duration + gate_x.duration
                name_x = 'ref_1_I%04i.WFM' %t
                name_y = 'ref_1_Q%04i.WFM' %t 

                #pi2_tr_x = Waveform(name_x, [Idle(12600 * 1.2)] + p['pi_all + 0 12'] + [Idle(12600 * 1.2)] +  p['pi/2_dq + 0 2'],t_0 )
                #pi2_tr_y = Waveform(name_y, [Idle(12600 * 1.2)] + p['pi_all + 90 12'] + [Idle(12600 * 1.2)] +  p['pi/2_dq + 90 2'],t_0 )
                pi2_tr_x = Waveform(name_x, [Idle(int(330*1.2))] +  p['pi/2_dq + 0 2'],t_0 )
                pi2_tr_y = Waveform(name_y, [Idle(int(330*1.2))] +  p['pi/2_dq + 90 2'],t_0 )
                self.waves.append(pi2_tr_x)
                self.waves.append(pi2_tr_y)
                
                t_0 += pi2_tr_x.duration
                name_x = 'ref_11_I%04i.WFM' %t
                name_y = 'ref_11_Q%04i.WFM' %t 
                
                tvar = (self.tau1 + t) * 1.2
                pi_ref_x = Waveform(name_x, [Idle(tvar)] +  [p['pi - 0']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(tvar)] +  [p['pi - 90']],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'ref_2_I%04i.WFM' %t
                name_y = 'ref_2_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_pa + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_pa + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'ref3_I%04i.WFM' %t
                name_y = 'ref_3_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 2'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_diag_3_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(refstate_x, refstate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(pi2_tr_x, pi2_tr_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)    
                
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Pop_4_I%04i.WFM' %t
                name_y = 'Pop_4_Q%04i.WFM' %t 
                
                redeph = int(100000 / 256)
                t_0 += redeph * 256
                
                pi2_tr_x = Waveform(name_x, [Idle(int(330*1.2))] +  p['pi/2_dq + 0 2'],t_0 )
                pi2_tr_y = Waveform(name_y, [Idle(int(330*1.2))] +  p['pi/2_dq + 90 2'],t_0 )
                self.waves.append(pi2_tr_x)
                self.waves.append(pi2_tr_y)
                
                t_0 += pi2_tr_x.duration
                name_x = 'Pop_5_I%04i.WFM' %t
                name_y = 'Pop_5_Q%04i.WFM' %t 
                
                tvar = (self.tau1 + t) * 1.2
                pi_ref_x = Waveform(name_x, [Idle(tvar)] +  [p['pi - 0']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(tvar)] +  [p['pi - 90']],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'Pop_6_I%04i.WFM' %t
                name_y = 'Pop_6_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_pa + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_pa + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Pop_7_I%04i.WFM' %t
                name_y = 'Pop_7_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 2'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_diag_2_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(evo, evo,repeat=redeph)
                sub_seq.append(pi2_tr_x, pi2_tr_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)     
                '''
                
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('ESPIN_TOMO.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )       
        
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        sequence = []
        for t in tau:
            if(t<10):
                sub = [ (['awgTrigger'], 100 ),
                        ([], 1740 * 5 + 25000 + 1000 ),
                        (['laser', 'trigger' ], laser ),
                        ([], wait )
                      ]
                sequence.extend(sub)
            elif(9<t<23):
                sub = [ (['awgTrigger'], 100 ),
                        ([], 1740 * 8  + 25000 + self.evolution * 2 + 2000 ),
                        (['laser', 'trigger' ], laser ),
                        ([], wait )
                      ]
                sequence.extend(sub)    
            else:
                sub = [ (['awgTrigger'], 100 ),
                        ([], 1740 * 8  + 125000 + self.evolution * 2 + 2000 ),
                        (['laser', 'trigger' ], laser ),
                        ([], wait )
                      ]
                sequence.extend(sub)

        return sequence
        
    get_set_items = Pulsed.get_set_items + ['freq','freq_2','freq_3','freq_4','vpp','amp','evolution','pi_1','pi_2','pi_3','pi_4'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('evolution', width=20), 
                                     Item('tau1', width=20)
                                     ),             
                              HGroup(Item('freq',  width=20),
                                     Item('freq_4',  width=20),
                                     Item('freq_2',  width=20),
                                     Item('freq_3',  width=20),
                                     ),        
                              HGroup( Item('pi_1'),
                                      Item('pi_2'),
                                      Item('pi_3'),
                                      Item('pi_4'),
                                    ),       
                              HGroup(Item('laser', width=30),
                                     Item('wait', width=30),
                                     Item('bin_width', width= 30, enabled_when='state != "run"'),
                                     Item('record_length', width= 30, enabled_when='state != "run"'),
                                     ),
                                     
                              HGroup(Item('tau_begin', width=30),
                                     Item('tau_end', width=30),
                                     Item('tau_delta', width= 30),
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30),
                                     ),
                                                                         
                              ),
                       title='EspinTomo_diag',
                       )    
class EspinTomo_diag_Nloc(Pulsed):   

    pi_1 = Range(low=1., high=100000., value=51.7, desc='length of pi pulse 1st transition [ns]', label='pi [ns] freq1', mode='text', auto_set=False, enter_set=True)
    pi_2 = Range(low=1., high=100000., value=106.2, desc='length of pi pulse 2nd transition [ns]', label='pi [ns] freq4', mode='text', auto_set=False, enter_set=True)
    pi_3 = Range(low=1., high=100000., value=47.5, desc='length of pi pulse 3st transition [ns]', label='pi [ns] freq2', mode='text', auto_set=False, enter_set=True)
    pi_4 = Range(low=1., high=100000., value=55.4, desc='length of pi pulse 4nd transition [ns]', label='pi [ns] freq3', mode='text', auto_set=False, enter_set=True)
    
    freq_center = Range(low=1, high=20e9, value=2.61e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.784617e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_2 = Range(low=1, high=20e9, value=2.848743e9, desc='frequency 2nd trans [Hz]', label='freq2 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_3 = Range(low=1, high=20e9, value=2.906787e9, desc='frequency 3nd trans [Hz]', label='freq3 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_4 = Range(low=1, high=20e9, value=2.963224e9, desc='frequency 4th trans [Hz]', label='freq4 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=0.6, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    tau_begin = Range(low=0., high=1e8, value=1, desc='tau begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=10, desc='tau end [ns]', label='tau end < tau1 [ns]', mode='text', auto_set=False, enter_set=True)  #36
    tau_delta = Range(low=1., high=1e8, value=1, desc='delta tau [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    evolution = Range(low=0., high=100000., value=50500., desc='free evolution time between creation and tomography [ns]', label='evo [ns]', mode='text', auto_set=False, enter_set=True)
    tau1 = Range(low=0., high=100000., value=660., desc=' waiting time between optimal pulse[ns]', label='tau_waiting [ns]', mode='text', auto_set=False, enter_set=True)

    p_init_i = []
    p_init_q = []
    p_gate_i = []
    p_gate_q = []
    
    p_ref_i = []
    p_ref_q = []
    reload = True
    

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            f1 = (self.freq - self.freq_center)/sampling
            f2 = (self.freq_4 - self.freq_center)/sampling
            f3= (self.freq_2 - self.freq_center)/sampling
            f4 = (self.freq_3 - self.freq_center)/sampling

            # Pulses
            p = {}
            
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            pi_2 = int(self.pi_2 * sampling/1.0e9)
            pi_3 = int(self.pi_3 * sampling/1.0e9)
            pi_4 = int(self.pi_4 * sampling/1.0e9)
            
            # Pulses
            p = {}
            # ms= 0 <> ms = +1
            p['pi + 0']     = Sin( pi_1, f1, 0 ,self.amp)
            p['pi + 90']    = Sin( pi_1, f1, np.pi/2 ,self.amp)
            # ms= 0 <> ms = -1
            p['pi - 0']     = Sin( pi_2, f2, 0 ,self.amp)
            p['pi - 90']    = Sin( pi_2, f2, np.pi/2 ,self.amp)
           
            p['pi + 0 2']     = Sin( pi_3, f3, 0 ,self.amp)
            p['pi + 90 2']     = Sin( pi_3, f3, np.pi/2  ,self.amp)

            p['pi - 0 2']     = Sin( pi_4, f4, 0 ,self.amp)
            p['pi - 90 2']     = Sin( pi_4, f4, np.pi/2  ,self.amp)

            zero = Idle(1)
            mod = Idle(0)
            
            # ms= 0 <> ms = +1            
            datfile = 'D:/data/ProgGate/tomo/seq/diag/tomo_seq_diag.py'
            fileHandle = open (datfile) 
            #read the cotend of the file
            datfilelines=fileHandle.read()
            exec datfilelines 
            fileHandle.close() 
            
            datfile = 'D:/data/ProgGate/tomo/seq/diag/initial_seq.py'
            fileHandle = open (datfile) 
            #read the cotend of the file
            datfilelines=fileHandle.read()
            exec datfilelines 
            fileHandle.close() 
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('ESPIN_TOMO.SEQ') 
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            # Population Reference
            name_x = '01_I.WFM' 
            name_y = '01_Q.WFM'             
            sup_x = Waveform(name_x, Idle(self.tau1*1.2))
            sup_y = Waveform(name_y, Idle(self.tau1*1.2))
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            name = 'Tomo_01.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(sup_x, sup_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)
          
            name_x = '02_I.WFM' 
            name_y = '02_Q.WFM'             
            sup_x = Waveform(name_x, p['pi - 0 2'])
            sup_y = Waveform(name_y, p['pi - 90 2'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            name = 'Tomo_02.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(sup_x, sup_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)
            
            name_x = '03_I.WFM' 
            name_y = '03_Q.WFM'             
            sup_x = Waveform(name_x, p['pi - 0'])
            sup_y = Waveform(name_y, p['pi - 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            name = 'Tomo_03.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(sup_x, sup_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)
            
            name_x = '04_I.WFM' 
            name_y = '04_Q.WFM'             
            sup_x = Waveform(name_x, p['pi_p + 0 12'])
            sup_y = Waveform(name_y, p['pi_p + 90 12'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            name = 'Tomo_04.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(sup_x, sup_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)
            
            #self.p_init_i = p['-1_-1 + 0']
            #self.p_init_q = p['-1_-1 + 90']
            # Population Signal
            #self.p_init_i = p['Plus_+1 + 0']
            #self.p_init_q = p['Plus_+1 + 90']
            name_x = 'ref_state_i'
            name_y = 'ref_state_q'
            refstate_x = Waveform(name_x, self.p_ref_i)
            refstate_y = Waveform(name_y, self.p_ref_q)
            self.waves.append(refstate_x)
            self.waves.append(refstate_y)
            
            name_x = 'istate_2_i'
            name_y = 'istate_2_q'
            istate_x = Waveform(name_x, self.p_init_i)
            istate_y = Waveform(name_y, self.p_init_q)
            self.waves.append(istate_x)
            self.waves.append(istate_y)
            
            t_0 = istate_x.duration
            name_x = 'gate_I.WFM' 
            name_y = 'gate_Q.WFM'   
            gate_x = Waveform(name_x, [Idle(self.tau1*1.2)] + self.p_gate_i, t_0)
            gate_y = Waveform(name_y, [Idle(self.tau1*1.2)] + self.p_gate_q, t_0)
            self.waves.append(gate_x)
            self.waves.append(gate_y)
            
            # gate
            #datfile1 = 'D:/data/ProgGate/tomo/seq/diag' + '/cnot.py'
            #fileHandle = open (datfile1) 
            #read the cotend of the file
            #datfilelines=fileHandle.read()
            #exec datfilelines 
            #fileHandle.close()
            
            for t in np.arange(100,601,40):
                t_0 = refstate_x.duration + gate_x.duration
                name_x = 'ref_1_I%04i.WFM' %t
                name_y = 'ref_1_Q%04i.WFM' %t 

                #pi2_tr_x = Waveform(name_x, [Idle(12600 * 1.2)] + p['pi_all + 0 12'] + [Idle(12600 * 1.2)] +  p['pi/2_dq + 0 2'],t_0 )
                #pi2_tr_y = Waveform(name_y, [Idle(12600 * 1.2)] + p['pi_all + 90 12'] + [Idle(12600 * 1.2)] +  p['pi/2_dq + 90 2'],t_0 )
                pi2_tr_x = Waveform(name_x, [Idle(int(330*1.2))] +  p['pi/2_dq + 0 2'],t_0 )
                pi2_tr_y = Waveform(name_y, [Idle(int(330*1.2))] +  p['pi/2_dq + 90 2'],t_0 )
                self.waves.append(pi2_tr_x)
                self.waves.append(pi2_tr_y)
                
                t_0 += pi2_tr_x.duration
                name_x = 'ref_11_I%04i.WFM' %t
                name_y = 'ref_11_Q%04i.WFM' %t 
                
                tvar = (self.tau1 + t) * 1.2
                pi_ref_x = Waveform(name_x, [Idle(tvar)] +  [p['pi - 0']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(tvar)] +  [p['pi - 90']],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'ref_2_I%04i.WFM' %t
                name_y = 'ref_2_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_pa + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_pa + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'ref3_I%04i.WFM' %t
                name_y = 'ref_3_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 2'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_diag_3_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(refstate_x, refstate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(pi2_tr_x, pi2_tr_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)    
                
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Pop_4_I%04i.WFM' %t
                name_y = 'Pop_4_Q%04i.WFM' %t 
                
                redeph = int(100000 / 256)
                t_0 += redeph * 256
                
                pi2_tr_x = Waveform(name_x, [Idle(int(330*1.2))] +  p['pi/2_dq + 0 2'],t_0 )
                pi2_tr_y = Waveform(name_y, [Idle(int(330*1.2))] +  p['pi/2_dq + 90 2'],t_0 )
                self.waves.append(pi2_tr_x)
                self.waves.append(pi2_tr_y)
                
                t_0 += pi2_tr_x.duration
                name_x = 'Pop_5_I%04i.WFM' %t
                name_y = 'Pop_5_Q%04i.WFM' %t 
                
                tvar = (self.tau1 + t) * 1.2
                pi_ref_x = Waveform(name_x, [Idle(tvar)] +  [p['pi - 0']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(tvar)] +  [p['pi - 90']],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'Pop_6_I%04i.WFM' %t
                name_y = 'Pop_6_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_pa + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_pa + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Pop_7_I%04i.WFM' %t
                name_y = 'Pop_7_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 2'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_diag_2_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(evo, evo,repeat=redeph)
                sub_seq.append(pi2_tr_x, pi2_tr_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)     
                
                
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('ESPIN_TOMO.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )       
        
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        sequence = []
        for t in tau:
            if(t<10):
                sub = [ (['awgTrigger'], 100 ),
                        ([], 1740 * 5 + 25000 + 1000 ),
                        (['laser', 'trigger' ], laser ),
                        ([], wait )
                      ]
                sequence.extend(sub)
            elif(9<t<23):
                sub = [ (['awgTrigger'], 100 ),
                        ([], 1740 * 8  + 25000 + self.evolution * 2 + 2000 ),
                        (['laser', 'trigger' ], laser ),
                        ([], wait )
                      ]
                sequence.extend(sub)    
            else:
                sub = [ (['awgTrigger'], 100 ),
                        ([], 1740 * 8  + 125000 + self.evolution * 2 + 2000 ),
                        (['laser', 'trigger' ], laser ),
                        ([], wait )
                      ]
                sequence.extend(sub)

        return sequence
        
    get_set_items = Pulsed.get_set_items + ['freq','freq_2','freq_3','freq_4','vpp','amp','evolution','pi_1','pi_2','pi_3','pi_4'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('evolution', width=20), 
                                     Item('tau1', width=20)
                                     ),             
                              HGroup(Item('freq',  width=20),
                                     Item('freq_4',  width=20),
                                     Item('freq_2',  width=20),
                                     Item('freq_3',  width=20),
                                     ),        
                              HGroup( Item('pi_1'),
                                      Item('pi_2'),
                                      Item('pi_3'),
                                      Item('pi_4'),
                                    ),       
                              HGroup(Item('laser', width=30),
                                     Item('wait', width=30),
                                     Item('bin_width', width= 30, enabled_when='state != "run"'),
                                     Item('record_length', width= 30, enabled_when='state != "run"'),
                                     ),
                                     
                              HGroup(Item('tau_begin', width=30),
                                     Item('tau_end', width=30),
                                     Item('tau_delta', width= 30),
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30),
                                     ),
                                                                         
                              ),
                       title='EspinTomo_diag_Nloc',
                       )                           
class EspinTomo_NLoC(Pulsed):                   


    pi_1 = Range(low=1., high=100000., value=51.7, desc='length of pi pulse 1st transition [ns]', label='pi [ns] freq1', mode='text', auto_set=False, enter_set=True)
    pi_2 = Range(low=1., high=100000., value=106.2, desc='length of pi pulse 2nd transition [ns]', label='pi [ns] freq4', mode='text', auto_set=False, enter_set=True)
    pi_3 = Range(low=1., high=100000., value=47.5, desc='length of pi pulse 3st transition [ns]', label='pi [ns] freq2', mode='text', auto_set=False, enter_set=True)
    pi_4 = Range(low=1., high=100000., value=55.4, desc='length of pi pulse 4nd transition [ns]', label='pi [ns] freq3', mode='text', auto_set=False, enter_set=True)
    
    freq_center = Range(low=1, high=20e9, value=2.61e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.784617e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_2 = Range(low=1, high=20e9, value=2.848743e9, desc='frequency 2nd trans [Hz]', label='freq2 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_3 = Range(low=1, high=20e9, value=2.906787e9, desc='frequency 3nd trans [Hz]', label='freq3 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_4 = Range(low=1, high=20e9, value=2.963224e9, desc='frequency 4th trans [Hz]', label='freq4 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=0.6, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    tau_begin = Range(low=0., high=1e8, value=1, desc='tau begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=201, desc='tau end [ns]', label='tau end < tau1 [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=1e8, value=1, desc='delta tau [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    evolution = Range(low=0., high=100000., value=50500., desc='free evolution time between creation and tomography [ns]', label='evo [ns]', mode='text', auto_set=False, enter_set=True)
    evolution_2 = Range(low=0., high=100000., value=12600., desc='free evolution time between creation and tomography [ns]', label='evo_2 [ns]', mode='text', auto_set=False, enter_set=True)
    tau1 = Range(low=0., high=100000., value=50., desc=' waiting time between optimal pulse[ns]', label='tau_waiting [ns]', mode='text', auto_set=False, enter_set=True)
    
    p_init_i = []
    p_init_q = []
    
    p_gate_i = []
    p_gate_q = []
    
    p_ref_i = []
    p_ref_q = []
    
    reload = True
    

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            f1 = (self.freq - self.freq_center)/sampling
            f2 = (self.freq_4 - self.freq_center)/sampling
            f3= (self.freq_2 - self.freq_center)/sampling
            f4 = (self.freq_3 - self.freq_center)/sampling

            # Pulses
            p = {}
            
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            pi_2 = int(self.pi_2 * sampling/1.0e9)
            pi_3 = int(self.pi_3 * sampling/1.0e9)
            pi_4 = int(self.pi_4 * sampling/1.0e9)
            
            # Pulses
            p = {}
            # ms= 0 <> ms = +1
            p['pi + 0']     = Sin( pi_1, f1, 0 ,self.amp)
            p['pi + 90']    = Sin( pi_1, f1, np.pi/2 ,self.amp)
            # ms= 0 <> ms = -1
            p['pi - 0']     = Sin( pi_2, f2, 0 ,self.amp)
            p['pi - 90']    = Sin( pi_2, f2, np.pi/2 ,self.amp)
           
            p['pi + 0 2']     = Sin( pi_3, f3, 0 ,self.amp)
            p['pi + 90 2']     = Sin( pi_3, f3, np.pi/2  ,self.amp)

            p['pi - 0 2']     = Sin( pi_4, f4, 0 ,self.amp)
            p['pi - 90 2']     = Sin( pi_4, f4, np.pi/2  ,self.amp)

            zero = Idle(1)
            mod = Idle(0)

            datfile = 'D:/data/ProgGate/tomo/seq/diag/tomo_seq_diag.py'
            fileHandle = open (datfile) 
            #read the cotend of the file
            datfilelines=fileHandle.read()
            exec datfilelines 
            fileHandle.close() 
            
            datfile = 'D:/data/ProgGate/tomo/seq/diag/initial_seq.py'
            fileHandle = open (datfile) 
            #read the cotend of the file
            datfilelines=fileHandle.read()
            exec datfilelines 
            fileHandle.close() 
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('ESPIN_TOMO.SEQ') 
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)

            name_x = 'istate_2_i'
            name_y = 'istate_2_q'
            istate_x = Waveform(name_x, self.p_init_i)
            istate_y = Waveform(name_y, self.p_init_q)
            self.waves.append(istate_x)
            self.waves.append(istate_y)
            
            name_x = 'refstate_1_i'
            name_y = 'refstate_1_q'
            refstate_x = Waveform(name_x, self.p_ref_i)
            refstate_y = Waveform(name_y, self.p_ref_q)
            self.waves.append(refstate_x)
            self.waves.append(refstate_y)
            
            t_0 = istate_x.duration
            name_x = 'gate_I.WFM' 
            name_y = 'gate_Q.WFM'   
            gate_x = Waveform(name_x, [Idle(self.tau1*1.2)] + self.p_gate_i, t_0)
            gate_y = Waveform(name_y, [Idle(self.tau1*1.2)] + self.p_gate_q, t_0)
            self.waves.append(gate_x)
            self.waves.append(gate_y)
            
            
            # Reference for C14 and C23
            for t in np.arange(10, 1001,25):
                t_0 = refstate_x.duration
                name_i='Ref1_1_I_%04i.WFM' %t
                name_q='Ref1_1_Q_%04i.WFM' %t
                
                t_var = int((self.tau1+t)*1.2)
                pi2_1_x = Waveform(name_i, [Idle(t_var)] + p['pi/2_dq_y + 0 1'], t_0)
                pi2_1_y = Waveform(name_q, [Idle(t_var)] + p['pi/2_dq_y + 90 1'], t_0)
                self.waves.append(pi2_1_x)
                self.waves.append(pi2_1_y)
                
                t_1 = (self.evolution_2)*1.2 - pi2_1_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi2_1_x.duration + repeat_1 * 256
                
                name_x='Ref1_2_I_%04i.WFM' %t
                name_y='Ref1_2_Q_%04i.WFM' %t
                
                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution_2)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x='Ref1_3_I_%04i.WFM' %t
                name_y='Ref1_3_Q_%04i.WFM' %t
                
                map_x = Waveform(name_x, [mod] + p['read_y + 0 2'],t_0 )
                map_y = Waveform(name_y, [mod] + p['read_y + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C14_ref_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(refstate_x, refstate_y)
                #sub_seq.append(gate_x,gate_y)
                sub_seq.append(pi2_1_x, pi2_1_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                #sub_seq.append(pi2_2_x, pi2_2_y)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True) 
                
                
            # Signal for C14_x and C23  c14 + c23 _x
            for t in np.arange(10, 1001,25):
                #t_0 = refstate_x.duration
                t_0 = istate_x.duration + gate_x.duration
                name_i='Real1_1_I_%04i.WFM' %t
                name_q='Real1_1_Q_%04i.WFM' %t
                
                t_var = int((self.tau1+t)*1.2)
                pi2_1_x = Waveform(name_i, [Idle(t_var)] + p['pi/2_dq_y + 0 1'], t_0)
                pi2_1_y = Waveform(name_q, [Idle(t_var)] + p['pi/2_dq_y + 90 1'], t_0)
                self.waves.append(pi2_1_x)
                self.waves.append(pi2_1_y)
                
                t_1 = (self.evolution_2)*1.2 - pi2_1_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi2_1_x.duration + repeat_1 * 256
                
                name_x='Real1_2_I_%04i.WFM' %t
                name_y='Real1_2_Q_%04i.WFM' %t
                
                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution_2)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x='Real1_3_I_%04i.WFM' %t
                name_y='Real1_3_Q_%04i.WFM' %t
                
                map_x = Waveform(name_x,  [mod] + p['read_y + 0 2'],t_0 )
                map_y = Waveform(name_y, [mod] + p['read_y + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C14_r1_%04i.SEQ' %t
                sub_seq=Sequence(name)
                
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x,gate_y)
                #sub_seq.append(refstate_x, refstate_y)
                sub_seq.append(pi2_1_x, pi2_1_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                #sub_seq.append(pi2_2_x, pi2_2_y)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)  

            # Signal for C14_x and C23  c14 - c23  _x
            for t in np.arange(10, 1001,25):
                #t_0 = refstate_x.duration
                t_0 = istate_x.duration + gate_x.duration
                name_i='Real2_1_I_%04i.WFM' %t
                name_q='Real2_1_Q_%04i.WFM' %t
                
                t_var = int((self.tau1+t)*1.2)
                pi2_1_x = Waveform(name_i, [Idle(t_var)] + p['pi/2_dq_x + 0 1'], t_0)
                pi2_1_y = Waveform(name_q, [Idle(t_var)] + p['pi/2_dq_x + 90 1'], t_0)
                self.waves.append(pi2_1_x)
                self.waves.append(pi2_1_y)
                
                t_1 = (self.evolution_2)*1.2 - pi2_1_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi2_1_x.duration + repeat_1 * 256
                
                name_x='Real2_2_I_%04i.WFM' %t
                name_y='Real2_2_Q_%04i.WFM' %t
                
                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution_2)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x='Real2_3_I_%04i.WFM' %t
                name_y='Real2_3_Q_%04i.WFM' %t
                
                map_x = Waveform(name_x,  [mod] + p['read_x + 0 2'],t_0 )
                map_y = Waveform(name_y,  [mod] + p['read_x + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C14_r2_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x,gate_y)
                #sub_seq.append(refstate_x, refstate_y)
                sub_seq.append(pi2_1_x, pi2_1_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                #sub_seq.append(pi2_2_x, pi2_2_y)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)    
                
            
            # Signal for C14_y and C23  c14 - c23 _y
            for t in np.arange(10, 1001,25):
                #t_0 = refstate_x.duration
                t_0 = istate_x.duration + gate_x.duration
                name_i='Im1_1_I_%04i.WFM' %t
                name_q='Im1_1_Q_%04i.WFM' %t
                
                t_var = int((self.tau1+t)*1.2)
                pi2_1_x = Waveform(name_i, [Idle(t_var)] + p['pi/2_dq_y + 0 1'], t_0)
                pi2_1_y = Waveform(name_q, [Idle(t_var)] + p['pi/2_dq_y + 90 1'], t_0)
                self.waves.append(pi2_1_x)
                self.waves.append(pi2_1_y)
                
                t_1 = (self.evolution_2)*1.2 - pi2_1_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi2_1_x.duration + repeat_1 * 256
                
                name_x='Im1_2_I_%04i.WFM' %t
                name_y='Im1_2_Q_%04i.WFM' %t
                
                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution_2)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x='Im1_3_I_%04i.WFM' %t
                name_y='Im1_3_Q_%04i.WFM' %t
                
                map_x = Waveform(name_x,  [mod] + p['read_x + 0 2'],t_0 )
                map_y = Waveform(name_y,  [mod] + p['read_x + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C14_i1_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x,gate_y)
                #sub_seq.append(refstate_x, refstate_y)
                sub_seq.append(pi2_1_x, pi2_1_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                #sub_seq.append(pi2_2_x, pi2_2_y)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)  

            # Signal for C14_y and C23  c14 + c23  _x
            for t in np.arange(10, 1001,25):
                #t_0 = refstate_x.duration
                t_0 = istate_x.duration + gate_x.duration
                name_i='Im2_1_I_%04i.WFM' %t
                name_q='Im2_1_Q_%04i.WFM' %t
                
                t_var = int((self.tau1+t)*1.2)
                pi2_1_x = Waveform(name_i, [Idle(t_var)] + p['pi/2_dq_x + 0 1'], t_0)
                pi2_1_y = Waveform(name_q, [Idle(t_var)] + p['pi/2_dq_x + 90 1'], t_0)
                self.waves.append(pi2_1_x)
                self.waves.append(pi2_1_y)
                
                t_1 = (self.evolution_2)*1.2 - pi2_1_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi2_1_x.duration + repeat_1 * 256
                
                name_x='Im2_2_I_%04i.WFM' %t
                name_y='Im2_2_Q_%04i.WFM' %t
                
                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution_2)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x='Im2_3_I_%04i.WFM' %t
                name_y='Im2_3_Q_%04i.WFM' %t
                
                map_x = Waveform(name_x,  [mod] + p['read_y + 0 2'],t_0 )
                map_y = Waveform(name_y,  [mod] + p['read_y + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C14_i2_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x,gate_y)
                #sub_seq.append(refstate_x, refstate_y)
                sub_seq.append(pi2_1_x, pi2_1_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                #sub_seq.append(pi2_2_x, pi2_2_y)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)      
                
               
                          
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('ESPIN_TOMO.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )       
        
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        sequence = []
        # tau_end = 9 + 13 * 13 + 1
        for t in tau:
            
            if(t<41):
                sub = [ (['awgTrigger'], 100 ),
                    ([], 1440 * 4 + self.evolution_2 * 2  + 1000),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
                sequence.extend(sub)
            else:
            
            
                sub = [ (['awgTrigger'], 100 ),
                        ([], 1440 * 7 + 25200 + self.evolution_2 * 2  + 1000 ),
                        #([], 1440 * 4 + self.evolution_2 * 2  + 1000 ),
                        (['laser', 'trigger' ], laser ),
                        ([], wait )
                      ]
                sequence.extend(sub)
        return sequence
        
    get_set_items = Pulsed.get_set_items + ['freq','freq_2','freq_3','freq_4','vpp','amp','evolution','evolution_2','pi_1','pi_2','pi_3','pi_4'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq_center',  width=20),
                                     Item('power', width=-30),
                                     Item('vpp', width=-30),   
                                     Item('evolution', width=-50), 
                                     Item('evolution_2', width=-50), 
                                     Item('tau1', width=-40)
                                     ),             
                              HGroup(Item('freq',  width=20),
                                     Item('freq_4',  width=20),
                                     Item('freq_2',  width=20),
                                     Item('freq_3',  width=20),
                                     ),        
                              HGroup( Item('pi_1'),
                                      Item('pi_2'),
                                      Item('pi_3'),
                                      Item('pi_4'),
                                    ),       
                              HGroup(Item('laser', width=30),
                                     Item('wait', width=30),
                                     Item('bin_width', width= 30, enabled_when='state != "run"'),
                                     Item('record_length', width= 30, enabled_when='state != "run"'),
                                     ),
                                     
                              HGroup(Item('tau_begin', width=30),
                                     Item('tau_end', width=30),
                                     Item('tau_delta', width= 30),
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30),
                                     ),
                                                                         
                              ),
                       title='EspinTomo_NLoC',
                      )       
class EspinTomo_LoC1(Pulsed):                   


    pi_1 = Range(low=1., high=100000., value=51.7, desc='length of pi pulse 1st transition [ns]', label='pi [ns] freq1', mode='text', auto_set=False, enter_set=True)
    pi_2 = Range(low=1., high=100000., value=106.2, desc='length of pi pulse 2nd transition [ns]', label='pi [ns] freq4', mode='text', auto_set=False, enter_set=True)
    pi_3 = Range(low=1., high=100000., value=47.5, desc='length of pi pulse 3st transition [ns]', label='pi [ns] freq2', mode='text', auto_set=False, enter_set=True)
    pi_4 = Range(low=1., high=100000., value=55.4, desc='length of pi pulse 4nd transition [ns]', label='pi [ns] freq3', mode='text', auto_set=False, enter_set=True)
    
    freq_center = Range(low=1, high=20e9, value=2.61e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.784617e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_2 = Range(low=1, high=20e9, value=2.848743e9, desc='frequency 2nd trans [Hz]', label='freq2 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_3 = Range(low=1, high=20e9, value=2.906787e9, desc='frequency 3nd trans [Hz]', label='freq3 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_4 = Range(low=1, high=20e9, value=2.963224e9, desc='frequency 4th trans [Hz]', label='freq4 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=0.6, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    tau_begin = Range(low=0., high=1e8, value=1, desc='tau begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=89, desc='tau end [ns]', label='tau end < tau1 [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=1e8, value=1, desc='delta tau [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    evolution = Range(low=0., high=100000., value=50500., desc='free evolution time between creation and tomography [ns]', label='evo [ns]', mode='text', auto_set=False, enter_set=True)
    tau1 = Range(low=0., high=100000., value=10., desc=' waiting time between optimal pulse[ns]', label='tau_waiting [ns]', mode='text', auto_set=False, enter_set=True)
    #evolution2 = Range(low=0., high=100000., value=50500., desc='free evolution time between gate [ns]', label='evo_2 [ns]', mode='text', auto_set=False, enter_set=True)
    p_init_i = []
    p_init_q = []
    p_ref_i = []
    p_ref_q = []
    
    p_gate_i = []
    p_gate_q = []
    
    reload = True
    

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            f1 = (self.freq - self.freq_center)/sampling
            f2 = (self.freq_4 - self.freq_center)/sampling
            f3= (self.freq_2 - self.freq_center)/sampling
            f4 = (self.freq_3 - self.freq_center)/sampling

            # Pulses
            p = {}
            
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            pi_2 = int(self.pi_2 * sampling/1.0e9)
            pi_3 = int(self.pi_3 * sampling/1.0e9)
            pi_4 = int(self.pi_4 * sampling/1.0e9)
            
            # Pulses
            p = {}
            # ms= 0 <> ms = +1
            p['pi + 0']     = Sin( pi_1, f1, 0 ,self.amp)
            p['pi + 90']    = Sin( pi_1, f1, np.pi/2 ,self.amp)
            # ms= 0 <> ms = -1
            p['pi - 0']     = Sin( pi_2, f2, 0 ,self.amp)
            p['pi - 90']    = Sin( pi_2, f2, np.pi/2 ,self.amp)
           
            p['pi + 0 2']     = Sin( pi_3, f3, 0 ,self.amp)
            p['pi + 90 2']     = Sin( pi_3, f3, np.pi/2  ,self.amp)

            p['pi - 0 2']     = Sin( pi_4, f4, 0 ,self.amp)
            p['pi - 90 2']     = Sin( pi_4, f4, np.pi/2  ,self.amp)

            zero = Idle(1)
            mod = Idle(0)
 
            datfile = 'D:/data/ProgGate/tomo/seq/diag/tomo_seq_diag.py'
            fileHandle = open (datfile) 
            #read the cotend of the file
            datfilelines=fileHandle.read()
            exec datfilelines 
            fileHandle.close() 
            
            datfile = 'D:/data/ProgGate/tomo/seq/diag/initial_seq.py'
            fileHandle = open (datfile) 
            #read the cotend of the file
            datfilelines=fileHandle.read()
            exec datfilelines 
            fileHandle.close() 
            
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('ESPIN_TOMO.SEQ') 
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
           
            name_x = 'istate_2_i'
            name_y = 'istate_2_q'
            istate_x = Waveform(name_x, self.p_init_i)
            istate_y = Waveform(name_y, self.p_init_q)
            self.waves.append(istate_x)
            self.waves.append(istate_y)
            
            #self.p_ref_i = p['Plus_Plus + 0']
            #self.p_ref_q = p['Plus_Plus + 90']
            name_x = 'refstate_1_i'
            name_y = 'refstate_1_q'
            refstate_x = Waveform(name_x, self.p_ref_i)
            refstate_y = Waveform(name_y, self.p_ref_q)
            self.waves.append(refstate_x)
            self.waves.append(refstate_y)

            
            t_0 = istate_x.duration
            name_x = 'gate_I.WFM' 
            name_y = 'gate_Q.WFM'   
            gate_x = Waveform(name_x, [Idle(self.tau1*1.2)] + self.p_gate_i, t_0)
            gate_y = Waveform(name_y, [Idle(self.tau1*1.2)] + self.p_gate_q, t_0)
            self.waves.append(gate_x)
            self.waves.append(gate_y)
            
                        
            # Reference for Coherence(C12,C34) without decoherence
            for t in np.arange(100, 501,40):
                name_i='Ref1_I_%04i.WFM' %t
                name_q='Ref1_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_ref_i + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 0 1']) )
                self.waves.append(Waveform( name_q, self.p_ref_q + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 90 1']) )
                self.main_seq.append(*self.waves[-2:], wait=True)
                
            # Reference for Coherence(C12,C34) with decoherence
            for t in np.arange(100, 501,40):
                name_i='Ref2_I_%04i.WFM' %t
                name_q='Ref2_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_ref_i + [Idle(12600 * 1.2)] + p['pi_all + 0 1'] + [Idle(12600 * 1.2)] + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 0 1']) )
                self.waves.append(Waveform( name_q, self.p_ref_q + [Idle(12600 * 1.2)] + p['pi_all + 90 1']+ [Idle(12600 * 1.2)] + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 90 1']) )
                self.main_seq.append(*self.waves[-2:], wait=True)    
                
            # Signal for total_x C12 and C34
            for t in np.arange(100, 501,40):
                name_i='Real1_I_%04i.WFM' %t
                name_q='Real1_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_init_i + self.p_gate_i + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 0 1']) )
                self.waves.append(Waveform( name_q, self.p_init_q + self.p_gate_q + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 90 1']) )
                self.main_seq.append(*self.waves[-2:], wait=True) 
                
            # Signal for total_y C12 and C34    
            for t in np.arange(100,501,40):
                name_i='Im1_I_%04i.WFM' %t
                name_q='Im1_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_init_i + self.p_gate_i + [Idle(int((self.tau1+t) * 1.2))]+ p['read_y + 0 1']) )
                self.waves.append(Waveform( name_q, self.p_init_q + self.p_gate_q + [Idle(int((self.tau1+t) * 1.2))]+ p['read_y + 90 1']) )
                self.main_seq.append(*self.waves[-2:], wait=True)
                
            for t in np.arange(100,501,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Real1_1_I%04i.WFM' %t
                name_y = 'Real1_1_Q%04i.WFM' %t 
                
                t_1 = (self.evolution + t)*1.2 - gate_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 +=  repeat_1 * 256

                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Real1_2_I%04i.WFM' %t
                name_y = 'Real1_2_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 1'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 1'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C12_1_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)   

            # Signal for individual_y C12 and C34
            for t in np.arange(100,501,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Im1_1_I%04i.WFM' %t
                name_y = 'Im1_1_Q%04i.WFM' %t 
                
                t_1 =(self.evolution + t)*1.2 - gate_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += repeat_1 * 256

                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Im1_2_I%04i.WFM' %t
                name_y = 'Im1_2_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_y + 0 1'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_y + 90 1'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C12_3_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)          
              
            # Signal for individual_x C12 and C34
            for t in np.arange(100,501,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Real1_3_I%04i.WFM' %t
                name_y = 'Real1_3_Q%04i.WFM' %t 
                
                
                t_evol = (self.tau1 + t)*1.2
                pi_ref_x = Waveform(name_x, [Idle(t_evol)] + [p['pi - 0 2']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(t_evol)] + [p['pi - 90 2']],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'Real1_4_I%04i.WFM' %t
                name_y = 'Real1_4_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_ap + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_ap + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Real1_5_I%04i.WFM' %t
                name_y = 'Real1_5_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 1'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 1'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C12_2_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)     
                
            
            # Signal for individual_y C12 and C34
            for t in np.arange(100,501,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Im1_3_I%04i.WFM' %t
                name_y = 'Im1_3_Q%04i.WFM' %t 
                
                t_evol = (self.tau1 + t)*1.2
                pi_ref_x = Waveform(name_x, [Idle(t_evol)] +  [p['pi - 0 2']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(t_evol)] +  [p['pi - 90 2']],t_0 )
                #pi_ref_x = Waveform(name_x, [Idle(t_evol)] + p['pi_p + 0 1'],t_0 )
                #pi_ref_y = Waveform(name_y, [Idle(t_evol)] + p['pi_p + 90 1'],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'Im1_4_I%04i.WFM' %t
                name_y = 'Im1_4_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_ap + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_ap + 90 12'],t_0 )
                #pi_x = Waveform(name_x, [mod] + [p['pi - 0']] + [p['pi - 0 2']] + [p['pi + 0 2']]+  [p['pi - 0 2']] ,t_0 )
                #pi_y = Waveform(name_y, [mod] +  [p['pi - 90']] + [p['pi - 90 2']] + [p['pi + 90 2']]+  [p['pi - 90 2']],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Im1_5_I%04i.WFM' %t
                name_y = 'Im1_5_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_y + 0 1'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_y + 90 1'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C12_4_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)    
                   
            
            '''            
            # Signal for total_x C13 and C24
            for t in np.arange(100, 601,40):
                name_i='Real2_I_%04i.WFM' %t
                name_q='Real2_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_init_i + self.p_gate_i + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 0 1']) )
                self.waves.append(Waveform( name_q, self.p_init_q + self.p_gate_q + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 90 1']) )
                self.main_seq.append(*self.waves[-2:], wait=True)    
            # Signal for total_y C12 and C34    
            for t in np.arange(100,601,40):
                name_i='Im2_I_%04i.WFM' %t
                name_q='Im2_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_init_i + self.p_gate_i + [Idle(int((self.tau1+t) * 1.2))]+ p['read_y + 0 1']) )
                self.waves.append(Waveform( name_q, self.p_init_q + self.p_gate_q + [Idle(int((self.tau1+t) * 1.2))]+ p['read_y + 90 1']) )
                self.main_seq.append(*self.waves[-2:], wait=True)        
            
            # Signal for individual_x C12 and C34
            
                
                
            # Signal for individual_x C13 and C24
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Real2_1_I%04i.WFM' %t
                name_y = 'Real2_1_Q%04i.WFM' %t 
                
                t_1 = (self.evolution + t)*1.2 - gate_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += repeat_1 * 256

                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Real1_2_I%04i.WFM' %t
                name_y = 'Real1_2_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 1'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 1'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C13_1_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)   

            # Signal for individual_y C13 and C24
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Im2_1_I%04i.WFM' %t
                name_y = 'Im2_1_Q%04i.WFM' %t 
                
                t_1 =(self.evolution + t)*1.2 - gate_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += repeat_1 * 256

                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Im2_2_I%04i.WFM' %t
                name_y = 'Im2_2_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_y + 0 1'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_y + 90 1'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C13_3_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)          
              
            # Signal for individual_x C13 and C24
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Real2_3_I%04i.WFM' %t
                name_y = 'Real2_3_Q%04i.WFM' %t 
                
                t_evol = (self.tau1 + t)*1.2
                pi_ref_x = Waveform(name_x, [Idle(t_evol)] + [p['pi - 0']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(t_evol)] + [p['pi - 90']],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'Real2_4_I%04i.WFM' %t
                name_y = 'Real2_4_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_ap + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_ap + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Real2_5_I%04i.WFM' %t
                name_y = 'Real2_5_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 1'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 1'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C13_2_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)     
                
            
            # Signal for individual_y C13 and C24
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Im2_3_I%04i.WFM' %t
                name_y = 'Im2_3_Q%04i.WFM' %t 
                
                t_evol = (self.tau1 + t)*1.2
                pi_ref_x = Waveform(name_x, [Idle(t_evol)] +  [p['pi - 0']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(t_evol)] +  [p['pi - 90']],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'Im2_4_I%04i.WFM' %t
                name_y = 'Im2_4_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_ap + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_ap + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Im2_5_I%04i.WFM' %t
                name_y = 'Im2_5_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_y + 0 1'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_y + 90 1'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C13_4_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)       
                '''
                          
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('ESPIN_TOMO.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )       
        
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        sequence = []
        # tau_end = 9 + 13 * 13 + 1
        for t in tau:
            if(t<12):
                sub = [ (['awgTrigger'], 100 ),
                        ([], 1440 * 2 + 750 ),
                        (['laser', 'trigger' ], laser ),
                        ([], wait )
                      ]
                sequence.extend(sub)
            elif(11<t<45):
                sub = [ (['awgTrigger'], 100 ),
                        ([], 1440 * 5 + 26000 ),
                        (['laser', 'trigger' ], laser ),
                        ([], wait )
                      ]
                sequence.extend(sub)
            else:
                sub = [ (['awgTrigger'], 100 ),
                        ([], 1440 * 8 + 26000  + self.evolution * 2 ),
                        (['laser', 'trigger' ], laser ),
                        ([], wait )
                      ]
                sequence.extend(sub)
        

        return sequence
        
    get_set_items = Pulsed.get_set_items + ['freq','freq_2','freq_3','freq_4','vpp','amp','evolution','pi_1','pi_2','pi_3','pi_4'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq_center',  width=20),
                                     Item('power', width=-30),
                                     Item('vpp', width=-30),   
                                     Item('evolution', width=-50), 
                                     Item('tau1', width=-40),
                                     #Item('evolution2', width=20), 
                                     ),             
                              HGroup(Item('freq',  width=20),
                                     Item('freq_4',  width=20),
                                     Item('freq_2',  width=20),
                                     Item('freq_3',  width=20),
                                     ),        
                              HGroup( Item('pi_1'),
                                      Item('pi_2'),
                                      Item('pi_3'),
                                      Item('pi_4'),
                                    ),       
                              HGroup(Item('laser', width=30),
                                     Item('wait', width=30),
                                     Item('bin_width', width= 30, enabled_when='state != "run"'),
                                     Item('record_length', width= 30, enabled_when='state != "run"'),
                                     ),
                                     
                              HGroup(Item('tau_begin', width=30),
                                     Item('tau_end', width=30),
                                     Item('tau_delta', width= 30),
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30),
                                     ),
                                                                         
                              ),
                       title='EspinTomo_LoC1',
                       )                                                 
class EspinTomo_LoC2(Pulsed):                   


    pi_1 = Range(low=1., high=100000., value=51.7, desc='length of pi pulse 1st transition [ns]', label='pi [ns] freq1', mode='text', auto_set=False, enter_set=True)
    pi_2 = Range(low=1., high=100000., value=106.2, desc='length of pi pulse 2nd transition [ns]', label='pi [ns] freq4', mode='text', auto_set=False, enter_set=True)
    pi_3 = Range(low=1., high=100000., value=47.5, desc='length of pi pulse 3st transition [ns]', label='pi [ns] freq2', mode='text', auto_set=False, enter_set=True)
    pi_4 = Range(low=1., high=100000., value=55.4, desc='length of pi pulse 4nd transition [ns]', label='pi [ns] freq3', mode='text', auto_set=False, enter_set=True)
    
    freq_center = Range(low=1, high=20e9, value=2.61e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.784617e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_2 = Range(low=1, high=20e9, value=2.848743e9, desc='frequency 2nd trans [Hz]', label='freq2 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_3 = Range(low=1, high=20e9, value=2.906787e9, desc='frequency 3nd trans [Hz]', label='freq3 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_4 = Range(low=1, high=20e9, value=2.963224e9, desc='frequency 4th trans [Hz]', label='freq4 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=0.6, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    tau_begin = Range(low=0., high=1e8, value=1, desc='tau begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=89, desc='tau end [ns]', label='tau end < tau1 [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=1e8, value=1, desc='delta tau [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    evolution = Range(low=0., high=100000., value=50500., desc='free evolution time between creation and tomography [ns]', label='evo [ns]', mode='text', auto_set=False, enter_set=True)
    tau1 = Range(low=0., high=100000., value=660., desc=' waiting time between optimal pulse[ns]', label='tau_waiting [ns]', mode='text', auto_set=False, enter_set=True)
    #evolution2 = Range(low=0., high=100000., value=50500., desc='free evolution time between gate [ns]', label='evo_2 [ns]', mode='text', auto_set=False, enter_set=True)
    p_init_i = []
    p_init_q = []
    p_ref_i = []
    p_ref_q = []
    
    p_gate_i = []
    p_gate_q = []
    
    istate_pulse = '+1_+1'
    npulse = 100
    reload = True
    

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            f1 = (self.freq - self.freq_center)/sampling
            f2 = (self.freq_4 - self.freq_center)/sampling
            f3= (self.freq_2 - self.freq_center)/sampling
            f4 = (self.freq_3 - self.freq_center)/sampling

            # Pulses
            p = {}
            
            
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            pi_2 = int(self.pi_2 * sampling/1.0e9)
            pi_3 = int(self.pi_3 * sampling/1.0e9)
            pi_4 = int(self.pi_4 * sampling/1.0e9)
            
            # ms= 0 <> ms = +1
            p['pi + 0']     = Sin( pi_1, f1, 0 ,self.amp)
            p['pi + 90']    = Sin( pi_1, f1, np.pi/2 ,self.amp)
            # ms= 0 <> ms = -1
            p['pi - 0']     = Sin( pi_2, f2, 0 ,self.amp)
            p['pi - 90']    = Sin( pi_2, f2, np.pi/2 ,self.amp)
           
            p['pi + 0 2']     = Sin( pi_3, f3, 0 ,self.amp)
            p['pi + 90 2']     = Sin( pi_3, f3, np.pi/2  ,self.amp)

            p['pi - 0 2']     = Sin( pi_4, f4, 0 ,self.amp)
            p['pi - 90 2']     = Sin( pi_4, f4, np.pi/2  ,self.amp)
            
            
            #datfile = 'D:/data/ProgGate/tomo/seq/diag/stpulse.py'
            #fileHandle = open (datfile) 
            #read the cotend of the file
            #datfilelines=fileHandle.read()
            #exec datfilelines 
            #fileHandle.close() 

            zero = Idle(1)
            mod = Idle(0)
 
            datfile = 'D:/data/ProgGate/tomo/seq/diag/tomo_seq_diag.py'
            fileHandle = open (datfile) 
            #read the cotend of the file
            datfilelines=fileHandle.read()
            exec datfilelines 
            fileHandle.close() 
            
            #datfile = 'D:/data/ProgGate/tomo/seq/diag/initial_seq.py'
            #fileHandle = open (datfile) 
            #read the cotend of the file
            #datfilelines=fileHandle.read()
            #exec datfilelines 
            #fileHandle.close() 
            
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('ESPIN_TOMO.SEQ') 
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            #seq_x = self.istate_pulse + ' + 0'
            #seq_y = self.istate_pulse + ' + 90'
            #self.p_init_i = p[seq_x]
            #self.p_init_q = p[seq_y]
           
            name_x = 'istate_2_i'
            name_y = 'istate_2_q'
            istate_x = Waveform(name_x, self.p_init_i)
            istate_y = Waveform(name_y, self.p_init_q)
            self.waves.append(istate_x)
            self.waves.append(istate_y)
            
            #self.p_ref_i = p['Plus_Plus + 0']
            #self.p_ref_q = p['Plus_Plus + 90']
            name_x = 'refstate_1_i'
            name_y = 'refstate_1_q'
            refstate_x = Waveform(name_x, self.p_ref_i)
            refstate_y = Waveform(name_y, self.p_ref_q)
            self.waves.append(refstate_x)
            self.waves.append(refstate_y)
            
            #p_opt_sim_i = []
            #p_opt_sim_q = []
            #datfile1 ='D:/data/ProgGate/tomo/seq/diag/+1_plus_pulse_test_all.dat'
            #fileHandle = open (datfile1) 
            #read the cotend of the file
            #datfilelines=fileHandle.read()
            #exec datfilelines 
            #fileHandle.close()
            
            #p['read_x + 0 2'] = p_opt_sim_i[self.npulse]
            #p['read_x + 90 2'] = p_opt_sim_q[self.npulse]

            
            t_0 = istate_x.duration
            name_x = 'gate_I.WFM' 
            name_y = 'gate_Q.WFM'   
            gate_x = Waveform(name_x, [Idle(self.tau1*1.2)] + self.p_gate_i, t_0)
            gate_y = Waveform(name_y, [Idle(self.tau1*1.2)] + self.p_gate_q, t_0)
            self.waves.append(gate_x)
            self.waves.append(gate_y)
            
            
            # Reference for Coherence(C12,C34) without decoherence
            for t in np.arange(100, 501,40):
                name_i='Ref1_I_%04i.WFM' %t
                name_q='Ref1_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_ref_i + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 0 2']) )
                self.waves.append(Waveform( name_q, self.p_ref_q + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 90 2']) )
                self.main_seq.append(*self.waves[-2:], wait=True)
                
                
            # Reference for Coherence(C12,C34) with decoherence
            
            for t in np.arange(100, 501,40):
                name_i='Ref2_I_%04i.WFM' %t
                name_q='Ref2_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_ref_i + [Idle(12600 * 1.2)] + p['pi_all + 0 2'] + [Idle(12600 * 1.2)] + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 0 2']) )
                self.waves.append(Waveform( name_q, self.p_ref_q + [Idle(12600 * 1.2)] + p['pi_all + 90 2']+ [Idle(12600 * 1.2)] + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 90 2']) )
                self.main_seq.append(*self.waves[-2:], wait=True)    
              
            # Signal for total_x C12 and C34
            for t in np.arange(100, 501,40):
                name_i='Real1_I_%04i.WFM' %t
                name_q='Real1_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_init_i + self.p_gate_i + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 0 2']) )
                self.waves.append(Waveform( name_q, self.p_init_q + self.p_gate_q + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 90 2']) )
                self.main_seq.append(*self.waves[-2:], wait=True) 
                
            # Signal for total_y C12 and C34    
            for t in np.arange(100,501,40):
                name_i='Im1_I_%04i.WFM' %t
                name_q='Im1_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_init_i + self.p_gate_i + [Idle(int((self.tau1+t) * 1.2))]+ p['read_y + 0 2']) )
                self.waves.append(Waveform( name_q, self.p_init_q + self.p_gate_q + [Idle(int((self.tau1+t) * 1.2))]+ p['read_y + 90 2']) )
                self.main_seq.append(*self.waves[-2:], wait=True)
            
            for t in np.arange(100,501,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Real1_1_I%04i.WFM' %t
                name_y = 'Real1_1_Q%04i.WFM' %t 
                
                t_1 = (self.evolution + t)*1.2 - gate_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 +=  repeat_1 * 256

                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Real1_2_I%04i.WFM' %t
                name_y = 'Real1_2_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 2'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C12_1_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)   

            # Signal for individual_y C12 and C34
            for t in np.arange(100,501,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Im1_1_I%04i.WFM' %t
                name_y = 'Im1_1_Q%04i.WFM' %t 
                
                t_1 =(self.evolution + t)*1.2 - gate_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += repeat_1 * 256

                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Im1_2_I%04i.WFM' %t
                name_y = 'Im1_2_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_y + 0 2'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_y + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C12_3_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)   
                
              
            # Signal for individual_x C12 and C34
            for t in np.arange(100,501,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Real1_3_I%04i.WFM' %t
                name_y = 'Real1_3_Q%04i.WFM' %t 
                
                t_evol = (self.tau1 + t )*1.2 
                pi_ref_x = Waveform(name_x, [Idle(t_evol)] + [p['pi - 0']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(t_evol)] + [p['pi - 90']],t_0 )
                #pi_ref_x = Waveform(name_x, [Idle(t_evol)] + p['pi_p + 0 1'],t_0 )
                #pi_ref_y = Waveform(name_y, [Idle(t_evol)] + p['pi_p + 90 1'],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'Real1_4_I%04i.WFM' %t
                name_y = 'Real1_4_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_pa + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_pa + 90 12'],t_0 )
                #pi_x = Waveform(name_x, [mod] + [p['pi - 0']] + [p['pi - 0 2']] + [p['pi + 0 2']]+  [p['pi - 0 2']] ,t_0 )
                #pi_y = Waveform(name_y, [mod] +  [p['pi - 90']] + [p['pi - 90 2']] + [p['pi + 90 2']]+  [p['pi - 90 2']],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Real1_5_I%04i.WFM' %t
                name_y = 'Real1_5_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 2'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C12_2_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)     
                
            
            # Signal for individual_y C12 and C34
            for t in np.arange(100,501,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Im1_3_I%04i.WFM' %t
                name_y = 'Im1_3_Q%04i.WFM' %t 
                
                t_evol = (self.tau1 + t)*1.2
                pi_ref_x = Waveform(name_x, [Idle(t_evol)] +  [p['pi - 0']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(t_evol)] +  [p['pi - 90']],t_0 )
                #pi_ref_x = Waveform(name_x, [Idle(t_evol)] + p['pi_p + 0 1'],t_0 )
                #pi_ref_y = Waveform(name_y, [Idle(t_evol)] + p['pi_p + 90 1'],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'Im1_4_I%04i.WFM' %t
                name_y = 'Im1_4_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_pa + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_pa + 90 12'],t_0 )
                #pi_x = Waveform(name_x, [mod] + [p['pi - 0']] + [p['pi - 0 2']] + [p['pi + 0 2']]+  [p['pi - 0 2']] ,t_0 )
                #pi_y = Waveform(name_y, [mod] +  [p['pi - 90']] + [p['pi - 90 2']] + [p['pi + 90 2']]+  [p['pi - 90 2']],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Im1_5_I%04i.WFM' %t
                name_y = 'Im1_5_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_y + 0 2'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_y + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C12_4_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)    
                
                
            
            '''            
            # Signal for total_x C13 and C24
            for t in np.arange(100, 601,40):
                name_i='Real2_I_%04i.WFM' %t
                name_q='Real2_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_init_i + self.p_gate_i + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 0 1']) )
                self.waves.append(Waveform( name_q, self.p_init_q + self.p_gate_q + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 90 1']) )
                self.main_seq.append(*self.waves[-2:], wait=True)    
            # Signal for total_y C12 and C34    
            for t in np.arange(100,601,40):
                name_i='Im2_I_%04i.WFM' %t
                name_q='Im2_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_init_i + self.p_gate_i + [Idle(int((self.tau1+t) * 1.2))]+ p['read_y + 0 1']) )
                self.waves.append(Waveform( name_q, self.p_init_q + self.p_gate_q + [Idle(int((self.tau1+t) * 1.2))]+ p['read_y + 90 1']) )
                self.main_seq.append(*self.waves[-2:], wait=True)        
            
            # Signal for individual_x C12 and C34
            
                
                
            # Signal for individual_x C13 and C24
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Real2_1_I%04i.WFM' %t
                name_y = 'Real2_1_Q%04i.WFM' %t 
                
                t_1 = (self.evolution + t)*1.2 - gate_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += repeat_1 * 256

                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Real1_2_I%04i.WFM' %t
                name_y = 'Real1_2_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 1'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 1'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C13_1_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)   

            # Signal for individual_y C13 and C24
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Im2_1_I%04i.WFM' %t
                name_y = 'Im2_1_Q%04i.WFM' %t 
                
                t_1 =(self.evolution + t)*1.2 - gate_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += repeat_1 * 256

                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Im2_2_I%04i.WFM' %t
                name_y = 'Im2_2_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_y + 0 1'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_y + 90 1'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C13_3_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)          
              
            # Signal for individual_x C13 and C24
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Real2_3_I%04i.WFM' %t
                name_y = 'Real2_3_Q%04i.WFM' %t 
                
                t_evol = (self.tau1 + t)*1.2
                pi_ref_x = Waveform(name_x, [Idle(t_evol)] + [p['pi - 0']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(t_evol)] + [p['pi - 90']],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'Real2_4_I%04i.WFM' %t
                name_y = 'Real2_4_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_ap + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_ap + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Real2_5_I%04i.WFM' %t
                name_y = 'Real2_5_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 1'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 1'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C13_2_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)     
                
            
            # Signal for individual_y C13 and C24
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Im2_3_I%04i.WFM' %t
                name_y = 'Im2_3_Q%04i.WFM' %t 
                
                t_evol = (self.tau1 + t)*1.2
                pi_ref_x = Waveform(name_x, [Idle(t_evol)] +  [p['pi - 0']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(t_evol)] +  [p['pi - 90']],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'Im2_4_I%04i.WFM' %t
                name_y = 'Im2_4_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_ap + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_ap + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Im2_5_I%04i.WFM' %t
                name_y = 'Im2_5_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_y + 0 1'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_y + 90 1'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C13_4_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)       
                '''
                          
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('ESPIN_TOMO.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )       
        
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        sequence = []
        # tau_end = 9 + 13 * 13 + 1
        for t in tau:
            
            if(t<12):
                sub = [ (['awgTrigger'], 100 ),
                        ([], 1800 * 2 + 2750 ),
                        (['laser', 'trigger' ], laser ),
                        ([], wait )
                      ]
                sequence.extend(sub)
            elif(11<t<45):
                sub = [ (['awgTrigger'], 100 ),
                        ([], 1800 * 5 + 1*26000 ),
                        (['laser', 'trigger' ], laser ),
                        ([], wait )
                      ]
                sequence.extend(sub)
            else:
            
                sub = [ (['awgTrigger'], 100 ),
                        ([], 1800 * 8 + 1*26000  + self.evolution * 2 ),
                        (['laser', 'trigger' ], laser ),
                        ([], wait )
                      ]
                sequence.extend(sub)
        

        return sequence
        
    get_set_items = Pulsed.get_set_items + ['freq','freq_2','freq_3','freq_4','vpp','amp','evolution','pi_1','pi_2','pi_3','pi_4'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq_center',  width=20),
                                     Item('power', width=-30),
                                     Item('vpp', width=-30),   
                                     Item('evolution', width=-50), 
                                     Item('tau1', width=-40),
                                     #Item('evolution2', width=20), 
                                     ),             
                              HGroup(Item('freq',  width=20),
                                     Item('freq_4',  width=20),
                                     Item('freq_2',  width=20),
                                     Item('freq_3',  width=20),
                                     ),        
                              HGroup( Item('pi_1'),
                                      Item('pi_2'),
                                      Item('pi_3'),
                                      Item('pi_4'),
                                    ),       
                              HGroup(Item('laser', width=30),
                                     Item('wait', width=30),
                                     Item('bin_width', width= 30, enabled_when='state != "run"'),
                                     Item('record_length', width= 30, enabled_when='state != "run"'),
                                     ),
                                     
                              HGroup(Item('tau_begin', width=30),
                                     Item('tau_end', width=30),
                                     Item('tau_delta', width= 30),
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30),
                                     ),
                                                                         
                              ),
                       title='EspinTomo_LoC2',
                       )                                                      

class EspinTomo_LoC(Pulsed):                   


    pi_1 = Range(low=1., high=100000., value=51, desc='length of pi pulse 1st transition [ns]', label='pi [ns] freq1', mode='text', auto_set=False, enter_set=True)
    pi_2 = Range(low=1., high=100000., value=87, desc='length of pi pulse 2nd transition [ns]', label='pi [ns] freq4', mode='text', auto_set=False, enter_set=True)
    pi_3 = Range(low=1., high=100000., value=46, desc='length of pi pulse 3st transition [ns]', label='pi [ns] freq2', mode='text', auto_set=False, enter_set=True)
    pi_4 = Range(low=1., high=100000., value=49, desc='length of pi pulse 4nd transition [ns]', label='pi [ns] freq3', mode='text', auto_set=False, enter_set=True)
    
    freq_center = Range(low=1, high=20e9, value=2.61e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_2 = Range(low=1, high=20e9, value=2.845829e9, desc='frequency 2nd trans [Hz]', label='freq2 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_3 = Range(low=1, high=20e9, value=2.907644e9, desc='frequency 3nd trans [Hz]', label='freq3 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_4 = Range(low=1, high=20e9, value=2.955537e9, desc='frequency 4th trans [Hz]', label='freq4 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=0.6, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    tau_begin = Range(low=0., high=1e8, value=1, desc='tau begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=170, desc='tau end [ns]', label='tau end < tau1 [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=1e8, value=1, desc='delta tau [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    evolution = Range(low=0., high=100000., value=50500., desc='free evolution time between creation and tomography [ns]', label='evo [ns]', mode='text', auto_set=False, enter_set=True)
    tau1 = Range(low=0., high=100000., value=250., desc=' waiting time between optimal pulse[ns]', label='tau_waiting [ns]', mode='text', auto_set=False, enter_set=True)
    evolution2 = Range(low=0., high=100000., value=50500., desc='free evolution time between gate [ns]', label='evo_2 [ns]', mode='text', auto_set=False, enter_set=True)
    p_init_i = []
    p_init_q = []
    p_ref_i = []
    p_ref_q = []
    
    p_gate_i = []
    p_gate_q = []
    
    reload = True
    

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            f1 = (self.freq - self.freq_center)/sampling
            f2 = (self.freq_4 - self.freq_center)/sampling
            f3= (self.freq_2 - self.freq_center)/sampling
            f4 = (self.freq_3 - self.freq_center)/sampling

            # Pulses
            p = {}
            
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            pi_2 = int(self.pi_2 * sampling/1.0e9)
            pi_3 = int(self.pi_3 * sampling/1.0e9)
            pi_4 = int(self.pi_4 * sampling/1.0e9)
            
            # Pulses
            p = {}
            # ms= 0 <> ms = +1
            p['pi + 0']     = Sin( pi_1, f1, 0 ,self.amp)
            p['pi + 90']    = Sin( pi_1, f1, np.pi/2 ,self.amp)
            # ms= 0 <> ms = -1
            p['pi - 0']     = Sin( pi_2, f2, 0 ,self.amp)
            p['pi - 90']    = Sin( pi_2, f2, np.pi/2 ,self.amp)
           
            p['pi + 0 2']     = Sin( pi_3, f3, 0 ,self.amp)
            p['pi + 90 2']     = Sin( pi_3, f3, np.pi/2  ,self.amp)

            p['pi - 0 2']     = Sin( pi_4, f4, 0 ,self.amp)
            p['pi - 90 2']     = Sin( pi_4, f4, np.pi/2  ,self.amp)

            zero = Idle(1)
            mod = Idle(0)
 
            datfile = 'D:/data/ProgGate/tomo/seq/diag/tomo_seq_diag.py'
            fileHandle = open (datfile) 
            #read the cotend of the file
            datfilelines=fileHandle.read()
            exec datfilelines 
            fileHandle.close() 
            
            datfile = 'D:/data/ProgGate/tomo/seq/diag/initial_seq.py'
            fileHandle = open (datfile) 
            #read the cotend of the file
            datfilelines=fileHandle.read()
            exec datfilelines 
            fileHandle.close() 
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('ESPIN_TOMO.SEQ') 
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            '''
            # Population Reference
            name_x = '01_I.WFM' 
            name_y = '01_Q.WFM'             
            sup_x = Waveform(name_x, Idle(self.tau1*1.2))
            sup_y = Waveform(name_y, Idle(self.tau1*1.2))
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            name = 'Tomo_01.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(sup_x, sup_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)
          
            name_x = '02_I.WFM' 
            name_y = '02_Q.WFM'             
            sup_x = Waveform(name_x, p['pi - 0 2'])
            sup_y = Waveform(name_y, p['pi - 90 2'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            name = 'Tomo_02.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(sup_x, sup_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)
            
            name_x = '03_I.WFM' 
            name_y = '03_Q.WFM'             
            sup_x = Waveform(name_x, p['pi - 0'])
            sup_y = Waveform(name_y, p['pi - 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            name = 'Tomo_03.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(sup_x, sup_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)
            
            name_x = '04_I.WFM' 
            name_y = '04_Q.WFM'             
            dup_x = Waveform(name_x, p['pi_p + 0 12'])
            dup_y = Waveform(name_y, p['pi_p + 90 12'])
            self.waves.append(dup_x)
            self.waves.append(dup_y)
            name = 'Tomo_04.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(dup_x, dup_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)
            '''
            
            # Population Signal

            '''          
            name_x = '05_I.WFM' 
            name_y = '05_Q.WFM'             
            ref_x = Waveform(name_x, [zero])
            ref_y = Waveform(name_y, [zero])
            self.waves.append(ref_x)
            self.waves.append(ref_y)
            name = 'Tomo_05.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(istate_x, istate_y)
            sub_seq.append(ref_x, ref_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)
            
            name_x = '06_I.WFM' 
            name_y = '06_Q.WFM'             
            ref_x = Waveform(name_x, p['pi - 0 2'],t_0 )
            ref_y = Waveform(name_y, p['pi - 90 2'],t_0 )
            self.waves.append(ref_x)
            self.waves.append(ref_y)
            name = 'Tomo_06.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(istate_x, istate_y)
            sub_seq.append(ref_x, ref_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)            
            
            name_x = '07_I.WFM' 
            name_y = '07_Q.WFM'             
            ref_x = Waveform(name_x, p['pi + 0 2'],t_0 )
            ref_y = Waveform(name_y, p['pi + 90 2'],t_0 )
            self.waves.append(ref_x)
            self.waves.append(ref_y)
            name = 'Tomo_07.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(istate_x, istate_y)
            sub_seq.append(ref_x, ref_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)  
            
            name_x = '08_I.WFM' 
            name_y = '08_Q.WFM'             
            ref_x = Waveform(name_x, p['pi - 0'],t_0 )
            ref_y = Waveform(name_y, p['pi - 90'],t_0 )
            self.waves.append(ref_x)
            self.waves.append(ref_y)
            name = 'Tomo_08.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(istate_x, istate_y)
            sub_seq.append(ref_x, ref_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)  

            name_x = '09_I.WFM' 
            name_y = '09_Q.WFM'             
            ref_x = Waveform(name_x, p['pi + 0'],t_0 )
            ref_y = Waveform(name_y, p['pi + 90'],t_0 )
            self.waves.append(ref_x)
            self.waves.append(ref_y)
            name = 'Tomo_09.SEQ'
            sub_seq=Sequence(name)
            sub_seq.append(istate_x, istate_y)
            sub_seq.append(ref_x, ref_y)
            AWG.upload(sub_seq)
            self.main_seq.append(sub_seq,wait=True)  
            '''
            #self.p_init_i = p['-1_-1 + 0']
            #self.p_init_q = p['-1_-1 + 90']
            name_x = 'istate_2_i'
            name_y = 'istate_2_q'
            istate_x = Waveform(name_x, self.p_init_i)
            istate_y = Waveform(name_y, self.p_init_q)
            self.waves.append(istate_x)
            self.waves.append(istate_y)
            
            #self.p_ref_i = p['Plus_Plus + 0']
            #self.p_ref_q = p['Plus_Plus + 90']
            name_x = 'refstate_1_i'
            name_y = 'refstate_1_q'
            refstate_x = Waveform(name_x, self.p_ref_i)
            refstate_y = Waveform(name_y, self.p_ref_q)
            self.waves.append(refstate_x)
            self.waves.append(refstate_y)
            
                        # gate
            datfile1 = 'D:/data/ProgGate/tomo/seq/diag' + '/cnot.py'
            fileHandle = open (datfile1) 
            #read the cotend of the file
            datfilelines=fileHandle.read()
            exec datfilelines 
            fileHandle.close()
            
            #self.p_gate_i = va_vb_i + [Idle(self.evolution2 * 1.2)] + pi_i + [Idle(self.evolution2 * 1.2)] + ua_ub_i
            #self.p_gate_q = va_vb_q + [Idle(self.evolution2 * 1.2)] + pi_q + [Idle(self.evolution2 * 1.2)] + ua_ub_q
            #self.p_gate_i = va_vb_i + [Idle(12500 * 1.2)] + pi_i + [Idle(12500 * 1.2)]
            #self.p_gate_q = va_vb_q + [Idle(12500 * 1.2)] + pi_q + [Idle(12500 * 1.2)]
            
            t_0 = istate_x.duration
            name_x = 'gate_I.WFM' 
            name_y = 'gate_Q.WFM'   
            gate_x = Waveform(name_x, [Idle(self.tau1*1.2)] + self.p_gate_i, t_0)
            gate_y = Waveform(name_y, [Idle(self.tau1*1.2)] + self.p_gate_q, t_0)
            self.waves.append(gate_x)
            self.waves.append(gate_y)
            
            # Reference for C12 and C34
            for t in np.arange(100, 601,40):
                name_i='Ref1_I_%04i.WFM' %t
                name_q='Ref1_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_ref_i + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 0 2']) )
                self.waves.append(Waveform( name_q, self.p_ref_q + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 90 2']) )
                self.main_seq.append(*self.waves[-2:], wait=True)
                
            # Signal for total_x C12 and C34
            for t in np.arange(100, 601,40):
                name_i='Real1_I_%04i.WFM' %t
                name_q='Real1_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_init_i + self.p_gate_i + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 0 2']) )
                self.waves.append(Waveform( name_q, self.p_init_q + self.p_gate_q + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 90 2']) )
                self.main_seq.append(*self.waves[-2:], wait=True)    
            # Signal for total_y C12 and C34    
            for t in np.arange(100,601,40):
                name_i='Im1_I_%04i.WFM' %t
                name_q='Im1_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_init_i + self.p_gate_i + [Idle(int((self.tau1+t) * 1.2))]+ p['read_y + 0 2']) )
                self.waves.append(Waveform( name_q, self.p_init_q + self.p_gate_q + [Idle(int((self.tau1+t) * 1.2))]+ p['read_y + 90 2']) )
                self.main_seq.append(*self.waves[-2:], wait=True)
                
            # Signal for total_x C13 and C24
            for t in np.arange(100, 601,40):
                name_i='Real2_I_%04i.WFM' %t
                name_q='Real2_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_init_i + self.p_gate_i + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 0 1']) )
                self.waves.append(Waveform( name_q, self.p_init_q + self.p_gate_q + [Idle(int((self.tau1+t) * 1.2))]+ p['read_x + 90 1']) )
                self.main_seq.append(*self.waves[-2:], wait=True)    
            # Signal for total_y C12 and C34    
            for t in np.arange(100,601,40):
                name_i='Im2_I_%04i.WFM' %t
                name_q='Im2_Q_%04i.WFM' %t
                self.waves.append(Waveform( name_i, self.p_init_i + self.p_gate_i + [Idle(int((self.tau1+t) * 1.2))]+ p['read_y + 0 1']) )
                self.waves.append(Waveform( name_q, self.p_init_q + self.p_gate_q + [Idle(int((self.tau1+t) * 1.2))]+ p['read_y + 90 1']) )
                self.main_seq.append(*self.waves[-2:], wait=True)        
            '''    
            # Signal for individual_x C12 and C34
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Real1_1_I%04i.WFM' %t
                name_y = 'Real1_1_Q%04i.WFM' %t 
                
                t_1 = (self.evolution + t)*1.2 - gate_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 +=  repeat_1 * 256

                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Real1_2_I%04i.WFM' %t
                name_y = 'Real1_2_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 2'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C12_1_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)   

            # Signal for individual_y C12 and C34
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Im1_1_I%04i.WFM' %t
                name_y = 'Im1_1_Q%04i.WFM' %t 
                
                t_1 =(self.evolution + t)*1.2 - gate_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += repeat_1 * 256

                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Im1_2_I%04i.WFM' %t
                name_y = 'Im1_2_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_y + 0 2'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_y + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C12_3_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)          
              
            # Signal for individual_x C12 and C34
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Real1_3_I%04i.WFM' %t
                name_y = 'Real1_3_Q%04i.WFM' %t 
                
                t_evol = (self.tau1 + t)*1.2
                pi_ref_x = Waveform(name_x, [Idle(t_evol)] + [p['pi - 0']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(t_evol)] + [p['pi - 90']],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'Real1_4_I%04i.WFM' %t
                name_y = 'Real1_4_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_pa + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_pa + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Real1_5_I%04i.WFM' %t
                name_y = 'Real1_5_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 2'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C12_2_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)     
                
            
            # Signal for individual_y C12 and C34
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Im1_3_I%04i.WFM' %t
                name_y = 'Im1_3_Q%04i.WFM' %t 
                
                t_evol = (self.tau1 + t)*1.2
                pi_ref_x = Waveform(name_x, [Idle(t_evol)] +  [p['pi - 0']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(t_evol)] +  [p['pi - 90']],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'Im1_4_I%04i.WFM' %t
                name_y = 'Im1_4_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_pa + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_pa + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Im1_5_I%04i.WFM' %t
                name_y = 'Im1_5_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_y + 0 2'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_y + 90 2'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C12_4_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)     
                
                
            # Signal for individual_x C13 and C24
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Real2_1_I%04i.WFM' %t
                name_y = 'Real2_1_Q%04i.WFM' %t 
                
                t_1 = (self.evolution + t)*1.2 - gate_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += repeat_1 * 256

                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Real1_2_I%04i.WFM' %t
                name_y = 'Real1_2_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 1'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 1'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C13_1_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)   

            # Signal for individual_y C13 and C24
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Im2_1_I%04i.WFM' %t
                name_y = 'Im2_1_Q%04i.WFM' %t 
                
                t_1 =(self.evolution + t)*1.2 - gate_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += repeat_1 * 256

                pi_x = Waveform(name_x, [mod] + p['pi_all + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_all + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Im2_2_I%04i.WFM' %t
                name_y = 'Im2_2_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_y + 0 1'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_y + 90 1'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C13_3_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)          
              
            # Signal for individual_x C13 and C24
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Real2_3_I%04i.WFM' %t
                name_y = 'Real2_3_Q%04i.WFM' %t 
                
                t_evol = (self.tau1 + t)*1.2
                pi_ref_x = Waveform(name_x, [Idle(t_evol)] + [p['pi - 0']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(t_evol)] + [p['pi - 90']],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'Real2_4_I%04i.WFM' %t
                name_y = 'Real2_4_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_ap + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_ap + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Real2_5_I%04i.WFM' %t
                name_y = 'Real2_5_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_x + 0 1'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_x + 90 1'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C13_2_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)     
                
            
            # Signal for individual_y C13 and C24
            for t in np.arange(100,601,40):
                t_0 = istate_x.duration + gate_x.duration
                name_x = 'Im2_3_I%04i.WFM' %t
                name_y = 'Im2_3_Q%04i.WFM' %t 
                
                t_evol = (self.tau1 + t)*1.2
                pi_ref_x = Waveform(name_x, [Idle(t_evol)] +  [p['pi - 0']],t_0 )
                pi_ref_y = Waveform(name_y, [Idle(t_evol)] +  [p['pi - 90']],t_0 )
                self.waves.append(pi_ref_x)
                self.waves.append(pi_ref_y)
                
                t_1 = (self.evolution)*1.2 - pi_ref_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 += pi_ref_x.duration + repeat_1 * 256
                
                name_x = 'Im2_4_I%04i.WFM' %t
                name_y = 'Im2_4_Q%04i.WFM' %t 

                pi_x = Waveform(name_x, [mod] + p['pi_ap + 0 12'],t_0 )
                pi_y = Waveform(name_y, [mod] + p['pi_ap + 90 12'],t_0 )
                self.waves.append(pi_x)
                self.waves.append(pi_y)
                
                t_2 = (self.evolution)*1.2 - pi_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += pi_x.duration + repeat_2 * 256
                
                name_x = 'Im2_5_I%04i.WFM' %t
                name_y = 'Im2_5_Q%04i.WFM' %t 
                
                map_x = Waveform(name_x, [mod]+p['read_y + 0 1'],t_0 )
                map_y = Waveform(name_y, [mod]+p['read_y + 90 1'],t_0 )
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'Tomo_C13_4_%04i.SEQ' %t
                sub_seq=Sequence(name)
                sub_seq.append(istate_x, istate_y)
                sub_seq.append(gate_x, gate_y)
                sub_seq.append(pi_ref_x, pi_ref_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(pi_x, pi_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(map_x, map_y)
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)       
                '''
                          
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('ESPIN_TOMO.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )       
        
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        sequence = []
        # tau_end = 9 + 13 * 13 + 1
        for t in tau:
            if(t<66):
                sub = [ (['awgTrigger'], 100 ),
                        ([], 1440 * 5 + 25000 + 1000 ),
                        (['laser', 'trigger' ], laser ),
                        ([], wait )
                      ]
                sequence.extend(sub)
            else:
                sub = [ (['awgTrigger'], 100 ),
                        ([], 1440 * 7 + 25000  + self.evolution * 2 + 1000 ),
                        (['laser', 'trigger' ], laser ),
                        ([], wait )
                      ]
                sequence.extend(sub)
        

        return sequence
        
    get_set_items = Pulsed.get_set_items + ['freq','freq_2','freq_3','freq_4','vpp','amp','evolution','pi_1','pi_2','pi_3','pi_4'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq_center',  width=20),
                                     Item('power', width=-30),
                                     Item('vpp', width=-30),   
                                     Item('evolution', width=-50), 
                                     Item('tau1', width=-40),
                                     Item('evolution2', width=20), 
                                     ),             
                              HGroup(Item('freq',  width=20),
                                     Item('freq_4',  width=20),
                                     Item('freq_2',  width=20),
                                     Item('freq_3',  width=20),
                                     ),        
                              HGroup( Item('pi_1'),
                                      Item('pi_2'),
                                      Item('pi_3'),
                                      Item('pi_4'),
                                    ),       
                              HGroup(Item('laser', width=30),
                                     Item('wait', width=30),
                                     Item('bin_width', width= 30, enabled_when='state != "run"'),
                                     Item('record_length', width= 30, enabled_when='state != "run"'),
                                     ),
                                     
                              HGroup(Item('tau_begin', width=30),
                                     Item('tau_end', width=30),
                                     Item('tau_delta', width= 30),
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30),
                                     ),
                                                                         
                              ),
                       title='EspinTomo_LoC',
                       )                                
