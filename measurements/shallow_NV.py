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
import threading
import logging

from tools.emod import ManagedJob

from tools.utility import GetSetItemsMixin

from hardware.api import PulseGenerator 
from hardware.api import Microwave_HMC,Microwave 
from hardware.api import AWG, FastComtec

from hardware.awg import *
from hardware.waveform import *
from measurements.pulsed_awg import Pulsed
from measurements.odmr import ODMR

PG = PulseGenerator()
#MW = Microwave_HMC()
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
    
class Rabi( Pulsed ):
    """Rabi measurement.
    """ 
    #def _init_(self):
        #super(Rabi, self).__init__()
        
    reload = True
  
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency [Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_center = Range(low=1, high=20e9, value=1.96e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)
    
    
    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            AWG.delete_all()

            drive_x = Sin(0, (self.freq - self.freq_center)/sampling, 0 ,self.amp)
            drive_y = Sin(0, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)
            zero = Idle(1)
            self.waves = []
            self.main_seq = Sequence('RABI.SEQ')
            for i,t in enumerate(self.tau):
                t = int(t * sampling / 1.0e9)
                drive_x.duration = t
                drive_y.duration = t
                x_name = 'X_RA_%03i.WFM' % i
                y_name = 'Y_RA_%03i.WFM' % i
                self.waves.append(Waveform(x_name, [zero, drive_x, zero]))
                self.waves.append(Waveform(y_name, [zero, drive_y, zero]))
                self.main_seq.append(*self.waves[-2:], wait=True)
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('RABI.SEQ')
        AWG.set_vpp(self.vpp)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        #AWG.set_output( 0b0011 )
        AWG.set_output( 0b0011 )  
    
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        sequence = []
        for t in tau:
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , t+600) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
        return sequence
        
    get_set_items = Pulsed.get_set_items + ['freq','amp','vpp']    
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),   
                             HGroup( Item('freq',  width=20),                                    
                                     Item('freq_center',  width=20),
                                     Item('power', width=20),
                                     Item('amp', width=20),
                                     Item('vpp', width=20), 
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
                       title='Rabi Measurement',
                       )    
                       
class FID( Pulsed ):
    #FID 
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half plus pi', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency plus [Hz]', label='freq1 p [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_center = Range(low=1, high=20e9, value=1.96e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )   
            pi2_1 = int(self.pi2_1 * sampling/1.0e9) 
            # Pulses
            p = {}
            p['pi2_1 + 0']     = Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,1)
            p['pi2_1 + 90']    = Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,1)
                  
            zero = Idle(1)
            mod = Idle(0)

            pi2_p_i = [zero,p['pi2_1 + 0'],zero]
            pi2_p_q = [zero,p['pi2_1 + 90'],zero]
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('FID.SEQ')
            
            sup_x = Waveform('Sup1_x', pi2_p_i)
            sup_y = Waveform('Sup1_y', pi2_p_q)
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
                                 
            for i, t in enumerate(self.tau):
                t_1 = t*1.2 - sup_x.stub
                repeat_1 = int(t_1 / 256)
                 
                mod.duration = int(t_1 % 256)
                t_0 = sup_x.duration + repeat_1 * 256
                
                name_x = 'REF_X%04i.WFM' % i
                name_y = 'REF_Y%04i.WFM' % i
                name_empt = 'Empt_map%04i.WFM' % i
                
                map_x = Waveform(name_x, [mod]+pi2_p_i, t_0)
                map_y = Waveform(name_y, [mod]+pi2_p_q , t_0)
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                if(repeat_1 == 0):
                    sub_seq.append(sup_x, sup_y)
                    sub_seq.append(map_x, map_y)
                    AWG.upload(sub_seq)
                else:
                    sub_seq.append(sup_x, sup_y)
                    sub_seq.append(evo, evo, repeat=repeat_1)
                    sub_seq.append(map_x, map_y)
                    AWG.upload(sub_seq)
      
                self.main_seq.append(sub_seq,wait=True)

                
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('FID.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1=self.pi2_1
        sequence = []
        for t in tau:
            sub = [ (['awgTrigger'], 100 ),
                    ([], t + pi2_1 * 2 + 200 ),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)
        return sequence
        
    get_set_items = Pulsed.get_set_items + ['freq','vpp','pi2_1','amp'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ), 
                              HGroup(Item('freq',  width=-60),      
                                     Item('pi2_1', width=-40),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
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
                       title='FID',
                       )     

class FID_Db( Pulsed ):
    #FID 
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half plus pi', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency plus [Hz]', label='freq1 p [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_center = Range(low=1, high=20e9, value=1.96e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
     
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )   
            pi2_1 = int(self.pi2_1 * sampling/1.0e9) 
 
            # Pulses
            p = {}
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,1)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,1)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,1)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,1)]
                  
            zero = Idle(1)
            mod = Idle(0)

            #pi2_p_i = [zero,p['pi2_1 + 0'],zero]
            #pi2_p_q = [zero,p['pi2_1 + 90'],zero]
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('FID.SEQ')
            
            sup_x = Waveform('Sup1_x', p['pi2_1 + 0'] + [Idle(20)])
            sup_y = Waveform('Sup1_y', p['pi2_1 + 90'] + [Idle(20)])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
                                 
            for i, t in enumerate(self.tau):
                
                t_1 = t * 1.2 - sup_x.stub
                
                repeat_1 = int(t_1/256)
                mod.duration = int(t_1%256)
                
                t_0 = sup_x.duration + repeat_1 * 256
                
                name_x = 'REF_X%04i.WFM' % i
                name_y = 'REF_Y%04i.WFM' % i
                
                map_x = Waveform(name_x, [mod]+p['pi2_1 + 0'], t_0)
                map_y = Waveform(name_y, [mod]+p['pi2_1 + 90'], t_0)
                self.waves.append(map_x)
                self.waves.append(map_y)

                
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                if(repeat_1 == 0):
                    sub_seq.append(sup_x, sup_y)
                    sub_seq.append(map_x, map_y)
                    AWG.upload(sub_seq)
                else:
                    sub_seq.append(sup_x, sup_y)
                    sub_seq.append(evo, evo, repeat = repeat_1)
                    sub_seq.append(map_x, map_y)
                    AWG.upload(sub_seq)
      
                self.main_seq.append(sub_seq,wait=True)
                
                name_x = 'REF1_X%04i.WFM' % i
                name_y = 'REF1_Y%04i.WFM' % i

                map1_x = Waveform(name_x, [mod]+p['pi2_1 - 0'], t_0)
                map1_y = Waveform(name_y, [mod]+p['pi2_1 - 90'], t_0)
                self.waves.append(map1_x)
                self.waves.append(map1_y)

                
                name = 'BSQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                if(repeat_1 == 0):
                    sub_seq.append(sup_x, sup_y)
                    sub_seq.append(map1_x, map1_y)
                    AWG.upload(sub_seq)
                else:
                    sub_seq.append(sup_x, sup_y)
                    sub_seq.append(evo, evo, repeat = repeat_1)
                    sub_seq.append(map1_x, map1_y)
                    AWG.upload(sub_seq)
      
                self.main_seq.append(sub_seq,wait=True)

                
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('FID.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 ) 
        
    def _get_sequence_points(self):
        return 2 * len(self.tau)    
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1=self.pi2_1
        sequence = []
        for t in tau:
            sub = [ (['awgTrigger'], 100 ),
                    ([], t + pi2_1 * 2 + 200 ),
                    (['laser', 'trigger' ], laser ),
                    ([], wait ),
                    
                    (['awgTrigger'], 100 ),
                    ([], t + pi2_1 * 2 + 200 ),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)
        return sequence
        
    get_set_items = Pulsed.get_set_items + ['freq','vpp','pi2_1','amp'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ), 
                              HGroup(Item('freq',  width=-60),      
                                     Item('pi2_1', width=-40),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
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
                       title='FID_Double',
                       )                         
class Hahn( Pulsed ):
    #FID 
    
    rabi_contrast=Range(low=1., high=100, value=30.0, desc='Rabi contrast [%]', label='contrast', mode='text', auto_set=False, enter_set=True)
    pi2_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='pi plus', mode='text', auto_set=False, enter_set=True)
    
    freq_center = Range(low=1, high=20e9, value=1.96e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)   
    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            p = {}
            
            pi2_1 = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            
            p['pi_1 + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1 + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
           
            zero = Idle(1)
            mod = Idle(0)
            
            
            p['initial + 0'] =  p['pi2_1 + 0']
            p['initial + 90'] = p['pi2_1 + 90']
            
            p['pi + 0'] = p['pi_1 + 0']
            p['pi + 90'] = p['pi_1 + 90']
            
            p['read + 0'] = p['pi2_1 + 0']
            p['read + 90'] = p['pi2_1 + 90'] 
            
            p['read - 0'] = p['pi2_1 - 0']
            p['read - 90'] = p['pi2_1 - 90'] 
                
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('Hahn.SEQ')
            
            sup_x = Waveform('Sup1_x', p['initial + 0'])
            sup_y = Waveform('Sup1_y', p['initial + 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
          
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
           
            for i, t in  enumerate(self.tau):
                t_1 = t*1.2 - sup_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 = sup_x.duration + repeat_1 * 256
                
                name_x = 'MAP3_X%04i.WFM' % i
                name_y = 'MAP3_Y%04i.WFM' % i
            
                map_x = Waveform(name_x, [mod]+ p['pi + 0'], t_0)
                map_y = Waveform(name_y, [mod]+ p['pi + 90'], t_0)
                
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                t_2 = t * 1.2 - map_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += map_x.duration + repeat_2 * 256
                
                name_x = 'ref3_X%04i.WFM' % i
                name_y = 'ref3_Y%04i.WFM' % i
                
                ref_x = Waveform(name_x, [mod]+ p['read + 0'], t_0)
                ref_y = Waveform(name_y, [mod]+ p['read + 90'], t_0)
                
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                if(repeat_1 == 0):
                    if(repeat_2 == 0):
                        sub_seq.append(sup_x, sup_y)
                        sub_seq.append(map_x, map_y)
                        sub_seq.append(ref_x, ref_y)
                    else:
                        sub_seq.append(sup_x, sup_y)
                        sub_seq.append(map_x, map_y)
                        sub_seq.append(evo, evo,repeat=repeat_2)
                        sub_seq.append(ref_x, ref_y)
                else:
                    if(repeat_2 == 0):
                        sub_seq.append(sup_x, sup_y)
                        sub_seq.append(evo, evo,repeat=repeat_1)
                        sub_seq.append(map_x, map_y)
                        sub_seq.append(ref_x, ref_y)
                    else:
                        sub_seq.append(sup_x, sup_y)
                        sub_seq.append(evo, evo,repeat=repeat_1)
                        sub_seq.append(map_x, map_y)
                        sub_seq.append(evo, evo,repeat=repeat_2)
                        sub_seq.append(ref_x, ref_y)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq,wait=True)
                
                
                name_x = 'Bref3_X%04i.WFM' % i
                name_y = 'Bref3_Y%04i.WFM' % i
                
                ref1_x = Waveform(name_x, [mod]+ p['read - 0'], t_0)
                ref1_y = Waveform(name_y, [mod]+ p['read - 90'], t_0)
                
                self.waves.append(ref1_x)
                self.waves.append(ref1_y)
                
                name = 'BSQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                if(repeat_1 == 0):
                    if(repeat_2 == 0):
                        sub_seq.append(sup_x, sup_y)
                        sub_seq.append(map_x, map_y)
                        sub_seq.append(ref1_x, ref1_y)
                    else:
                        sub_seq.append(sup_x, sup_y)
                        sub_seq.append(map_x, map_y)
                        sub_seq.append(evo, evo,repeat=repeat_2)
                        sub_seq.append(ref1_x, ref1_y)
                else:
                    if(repeat_2 == 0):
                        sub_seq.append(sup_x, sup_y)
                        sub_seq.append(evo, evo,repeat=repeat_1)
                        sub_seq.append(map_x, map_y)
                        sub_seq.append(ref1_x, ref1_y)
                    else:
                        sub_seq.append(sup_x, sup_y)
                        sub_seq.append(evo, evo,repeat=repeat_1)
                        sub_seq.append(map_x, map_y)
                        sub_seq.append(evo, evo,repeat=repeat_2)
                        sub_seq.append(ref1_x, ref1_y)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq,wait=True)

            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('Hahn.SEQ')
        AWG.set_vpp(self.vpp)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def _get_sequence_points(self):
        return 2 * len(self.tau)       
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1 = self.pi2_1
        pi_1 = self.pi_1
        sequence = []
        
        for t in tau:
            sub = [ (['awgTrigger'], 100 ),
                    ([], 2 * t + pi_1 + pi2_1*2 + 100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait ),
                    
                    (['awgTrigger'], 100 ),
                    ([], 2 * t + pi_1 + pi2_1*2 + 100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)

        return sequence
            
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','pi_1', 'rabi_contrast'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-65),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     ),       
                             HGroup(
                                    Item('pi2_1', width=20), 
                                    Item('pi_1', width=20), 
                                    Item('rabi_contrast', width=20)
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
                       title='Hahn',
                       )        

class T1( Pulsed ):
    #FID 
    
    pi_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='pi plus', mode='text', auto_set=False, enter_set=True)
    rabi_contrast=Range(low=1., high=100, value=30.0, desc='Rabi contrast [%]', label='contrast', mode='text', auto_set=False, enter_set=True)
    
    freq_center = Range(low=1, high=20e9, value=1.96e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)   
    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            p = {}
            
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            
            p['pi_1 + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1 + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
           
            zero = Idle(1)
            mod = Idle(0)
            
            p['pi + 0'] = p['pi_1 + 0']
            p['pi + 90'] = p['pi_1 + 90']
                
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('T1.SEQ')
            
            flip_x = Waveform('pi_x', p['pi + 0'])
            flip_y = Waveform('pi_y', p['pi + 90'])
            self.waves.append(flip_x)
            self.waves.append(flip_y)
          
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
           
            for i, t in  enumerate(self.tau):
                t_1 = t*1.2 - flip_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 = flip_x.duration + repeat_1 * 256
                
                name_x = 'MAP3_X%04i.WFM' % i
                name_y = 'MAP3_Y%04i.WFM' % i
            
                map_x = Waveform(name_x, [mod]+ p['pi + 0'], t_0)
                map_y = Waveform(name_y, [mod]+ p['pi + 90'], t_0)
                
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                if(repeat_1 == 0):
                    sub_seq.append(flip_x, flip_y)
                    sub_seq.append(map_x, map_y)
                else:
                    sub_seq.append(flip_x, flip_y)
                    sub_seq.append(evo, evo,repeat=repeat_1)
                    sub_seq.append(map_x, map_y)
                    
                AWG.upload(sub_seq)    
                self.main_seq.append(sub_seq,wait=True)    
                
                name_x = 'BMAP3_X%04i.WFM' % i
                name_y = 'BMAP3_Y%04i.WFM' % i
                map1_x = Waveform(name_x, [mod] , t_0)
                map1_y = Waveform(name_y, [mod], t_0)
                
                self.waves.append(map1_x)
                self.waves.append(map1_y)
                
                name = 'BSQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                if(repeat_1 == 0):
                    sub_seq.append(flip_x, flip_y)
                    sub_seq.append(map1_x, map1_y)
                else:
                    sub_seq.append(flip_x, flip_y)
                    sub_seq.append(evo, evo,repeat=repeat_1)
                    sub_seq.append(map1_x, map1_y)    

                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq,wait=True)

            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('T1.SEQ')
        AWG.set_vpp(self.vpp)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def _get_sequence_points(self):
        return 2 * len(self.tau)      
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi_1 = self.pi_1
        sequence = []
        
        for t in tau:
            sub = [ (['awgTrigger'], 100 ),
                    ([], t + pi_1 * 2 + 100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait ),
                    
                    (['awgTrigger'], 100 ),
                    ([], t + pi_1 * 2 + 100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)

        return sequence
            
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi_1'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-65),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('pi_1', width=20), 
                                     ),         
                              HGroup(Item('laser', width=30),
                                     Item('wait', width=30),
                                     Item('bin_width', width= -40, enabled_when='state != "run"'),
                                     Item('record_length', width=-40, enabled_when='state != "run"'),
                                     ),
                                     
                              HGroup(Item('tau_begin', width=30),
                                     Item('tau_end', width=30),
                                     Item('tau_delta', width= 30),
                                     Item('rabi_contrast', width=-40),
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30),
                                     ),
                                                                         
                              ),
                       title='T1',
                       )   
                       
          

class T1_Ms0( Pulsed ):
    #FID 
    
    pi_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='pi plus', mode='text', auto_set=False, enter_set=True)
    
    freq_center = Range(low=1, high=20e9, value=1.96e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)   
    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            p = {}
            
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            
            p['pi_1 + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1 + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
           
            zero = Idle(1)
            mod = Idle(0)
            
            p['pi + 0'] = p['pi_1 + 0']
            p['pi + 90'] = p['pi_1 + 90']
                
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('T1.SEQ')
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
           
            for i, t in  enumerate(self.tau):
                t_1 = t*1.2
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 = repeat_1 * 256
                
                name_x = 'BMAP3_X%04i.WFM' % i
                name_y = 'BMAP3_Y%04i.WFM' % i
                map1_x = Waveform(name_x, [mod], t_0)
                map1_y = Waveform(name_y, [mod], t_0)
                
                self.waves.append(map1_x)
                self.waves.append(map1_y)
                
                name_x = 'MAP3_X%04i.WFM' % i
                name_y = 'MAP3_Y%04i.WFM' % i
            
                map_x = Waveform(name_x, [mod]+ p['pi + 0'], t_0)
                map_y = Waveform(name_y, [mod]+ p['pi + 90'], t_0)
                
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                if(repeat_1 == 0):
                    sub_seq.append(map1_x, map1_y)
                else:
                    sub_seq.append(evo, evo,repeat=repeat_1)
                    sub_seq.append(map1_x, map1_y)
                    
                AWG.upload(sub_seq)    
                self.main_seq.append(sub_seq,wait=True)    
  
                name = 'BSQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                if(repeat_1 == 0):
                    sub_seq.append(map_x, map_y)
                else:
                    sub_seq.append(evo, evo,repeat=repeat_1)
                    sub_seq.append(map_x, map_y)    

                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq,wait=True)

            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('T1.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def _get_sequence_points(self):
        return 2 * len(self.tau)      
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi_1 = self.pi_1
        sequence = []
        
        for t in tau:
            sub = [ (['awgTrigger'], 100 ),
                    ([], t + pi_1  + 100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait ),
                    
                    (['awgTrigger'], 100 ),
                    ([], t + pi_1 + 100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)

        return sequence
            
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi_1'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-65),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('pi_1', width=20), 
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
                       title='T1_Ms=0',
                       )                          
                          
class XY8(Pulsed):                   

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
   
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    amp1 = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp1', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=0., high=1e8, value=500., desc='tau/2 begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau/2 end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau/2 [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    
    tau_ref_int = Range(low=1., high=1e5, value=30., desc='Reference tau initial [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_ref_flag = Bool(False, label='tau_ref_flag')
    
    pulse_num = Range(low=1, high=1000, value=1, desc='repetetion of XY8 pulses', label='repetitions', mode='text', auto_set=False, enter_set=True)
    reload = True
    
    #def _init_(self):
        #super(XY8, self).__init__()

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp1)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp1)]
            
            p['pi_1_x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1_x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('XY8.SEQ')
            
            sup_x_ref = Waveform('Sup_x_ref', p['pi2_1 + 0'])
            sup_y_ref = Waveform('Sup_y_ref', p['pi2_1 + 90'])
            t_sup_ref = sup_x_ref.stub
            
            sup_x = Waveform('Sup_x', [Idle(t_sup_ref)]+p['pi2_1 + 0'])
            sup_y = Waveform('Sup_y', [Idle(t_sup_ref)]+p['pi2_1 + 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            # Reference signal_initial points
            
            for i, t in enumerate(self.tau):
                if(self.tau_ref_flag and i < 4):
                    name = 'RefQH_12_%04i.SEQ' % i
                    sub_seq=Sequence(name)
                    sub_seq.append(sup_x,sup_y)
                    t_0 = sup_x.duration
                    t_1 = self.tau_ref_int * 1.2
                    t_tau = self.tau_ref_int * 1.2 * 2
                    for k in range(self.pulse_num):
                        x_name = 'RefX_XY4_%03i' % i + '_%03i.WFM' % k
                        y_name = 'RefY_XY4_%03i' % i + '_%03i.WFM' % k
                       
                        map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                    +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                        map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                    +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                                                    
                        self.waves.append(map_x_1)
                        self.waves.append(map_y_1)
                        sub_seq.append(map_x_1,map_y_1)
                        
                        t_0 += map_x_1.duration
                        t_1 = t_tau - map_x_1.stub
                        '''
                        map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                    +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0'],t_0)
                        map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                    +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90'],t_0)                              
                        self.waves.append(map_x_1)
                        self.waves.append(map_y_1)
                        sub_seq.append(map_x_1,map_y_1)
                        
                        t_0 += map_x_1.duration
                        t_1 = t_tau - map_x_1.stub
                        
                        x_name = 'RefX_YX4_%03i' % i + '_%03i.WFM' % k
                        y_name = 'RefY_YX4_%03i' % i + '_%03i.WFM' % k
                        map_x_2 = Waveform(x_name, [Idle(t_1)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                        map_y_2 = Waveform(y_name, [Idle(t_1)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                        self.waves.append(map_x_2)
                        self.waves.append(map_y_2)
                        sub_seq.append(map_x_2,map_y_2)
                        
                        t_0 += map_x_2.duration
                        t_1 = t_tau - map_x_2.stub
                        '''
                       

                    mod.duration = self.tau_ref_int * 1.2 - map_x_1.stub
                    name_x = 'Ref_Read_x_%04i.WFM' % i
                    name_y = 'Ref_Read_y_%04i.WFM' % i
                    ref_x = Waveform(name_x, [mod] + p['pi2_1 + 0'],t_0)
                    ref_y = Waveform(name_y, [mod] + p['pi2_1 + 90'],t_0)

                    self.waves.append(ref_x)
                    self.waves.append(ref_y)
                    sub_seq.append(ref_x,ref_y)
                    AWG.upload(sub_seq)
                    
                    self.main_seq.append(sub_seq, wait=True)
                    
                else:    
                    t_tau = t*1.2*2
                    
                    name = 'SQH_12_%04i.SEQ' % i
                    sub_seq=Sequence(name)
                    sub_seq.append(sup_x,sup_y)
                    t_0 = sup_x.duration
                    t_1 = t * 1.2
                    for k in range(self.pulse_num):
                        x_name = 'X_XY4_%03i' % i + '_%03i.WFM' % k
                        y_name = 'Y_XY4_%03i' % i + '_%03i.WFM' % k
                        map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                    +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                        map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                    +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                        self.waves.append(map_x_1)
                        self.waves.append(map_y_1)
                        sub_seq.append(map_x_1,map_y_1)
                        
                        t_0 += map_x_1.duration
                        t_1 = t_tau - map_x_1.stub
                        '''
                        x_name = 'X_YX4_%03i' % i + '_%03i.WFM' % k
                        y_name = 'Y_YX4_%03i' % i + '_%03i.WFM' % k
                        map_x_2 = Waveform(x_name, [Idle(t_1)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0'],t_0)
                        map_y_2 = Waveform(y_name, [Idle(t_1)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90'],t_0) 
                        #map_x_2 = Waveform(x_name, [Idle(t_1)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                        #map_y_2 = Waveform(y_name, [Idle(t_1)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0) 
                        self.waves.append(map_x_2)
                        self.waves.append(map_y_2)
                        sub_seq.append(map_x_2,map_y_2)
                        
                        t_0 += map_x_2.duration
                        t_1 = t_tau - map_x_2.stub
                        '''

                    mod.duration = t * 1.2 - map_x_1.stub
                    name_x = 'Read_x_%04i.WFM' % i
                    name_y = 'Read_y_%04i.WFM' % i
                    ref_x = Waveform(name_x, [mod] + p['pi2_1 + 0'],t_0)
                    ref_y = Waveform(name_y, [mod] + p['pi2_1 + 90'],t_0)

                    self.waves.append(ref_x)
                    self.waves.append(ref_y)
                    sub_seq.append(ref_x,ref_y)
                    AWG.upload(sub_seq)
                    
                    self.main_seq.append(sub_seq, wait=True)

            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            #time.sleep(70)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('XY8.SEQ')
            
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def load(self):
        self.reload = True
        #make sure tau is updated
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta) 
        if(self.tau_ref_flag):
            npoints = len(self.tau)
            a = np.arange(npoints+4)
            b = np.arange(self.tau[0]-4,self.tau[0],1)
            a[:4]= b
            a[-npoints:]=self.tau
            self.tau = a
        #self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta) 
        self.prepare_awg()
        self.reload = False        
        
    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.previous_time = 0
            #self.previous_count_data = FC.GetData()
            self.prepare_awg()
            MW.setFrequency(self.freq_center)
            MW.setPower(self.power)
            AWG.run()
            time.sleep(40.0)
            FC.Start()
            time.sleep(0.1)
            PG.Sequence(self.sequence, loop=True)

            start_time = time.time()

            while True:
               self.thread.stop_request.wait(5.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               self.elapsed_time = self.previous_elapsed_time + time.time() - start_time
               self.run_time += self.elapsed_time
               runtime, cycles = FC.GetState()
               sweeps = cycles / FC.GetData().shape[0]
               self.elapsed_sweeps = self.previous_sweeps + sweeps
               self.progress = int( 100 * self.elapsed_sweeps / self.sweeps ) 
               self.count_data = self.old_count_data + FC.GetData()
               if self.elapsed_sweeps > self.sweeps:
                  break

            FC.Halt()
            MW.Off()
            PG.High(['laser', 'mw'])
            AWG.stop()
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  
            
        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'    

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        tau_1 = self.tau_ref_int
        sequence = []
        
        for i, t in enumerate(tau):
            if(self.tau_ref_flag and i < 4):
                sequence.append( (['awgTrigger']      , 100) )
                sequence.append( ([ ]                 , tau_1*16 * self.pulse_num + 2*pi2_1 + 8*self.pulse_num*pi_1 + 2000) )
                sequence.append( (['laser', 'trigger'], laser) )
                sequence.append( ([ ]                 , wait) )
            else:
                sequence.append( (['awgTrigger']      , 100) )
                sequence.append( ([ ]                 , t*16 * self.pulse_num + 2*pi2_1 + 8*self.pulse_num*pi_1 + 2000) )
                sequence.append( (['laser', 'trigger'], laser) )
                sequence.append( ([ ]                 , wait) )
        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','amp1','pi2_1','pi_1','tau_ref_int','tau_ref_flag','pulse_num']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                     Item('amp1', width=-40), 
                                     ),                         
                              HGroup(
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40),
                                     Item('tau_ref_flag', enabled_when='state != "run"'),
                                     Item('tau_ref_int', width = -40),
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
                       title='XY8',
                       )  
                       
class XY4_Ref(Pulsed):                   

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
   
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    amp1 = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp1', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    rabi_contrast=Range(low=1., high=100, value=30.0, desc='Rabi contrast [%]', label='contrast', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=0., high=1e8, value=500., desc='tau/2 begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau/2 end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau/2 [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
  
    pulse_num = Range(low=1, high=1000, value=1, desc='repetetion of XY8 pulses', label='repetitions', mode='text', auto_set=False, enter_set=True)
    reload = True
    
    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp1)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp1)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp1)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp1)]
            
            p['pi_1_x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1_x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('XY4.SEQ')
            
            sup_x_ref = Waveform('Sup_x_ref', p['pi2_1 + 0'])
            sup_y_ref = Waveform('Sup_y_ref', p['pi2_1 + 90'])
            t_sup_ref = sup_x_ref.stub
            
            sup_x = Waveform('Sup_x', [Idle(t_sup_ref)]+p['pi2_1 + 0'])
            sup_y = Waveform('Sup_y', [Idle(t_sup_ref)]+p['pi2_1 + 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            # Reference signal_initial points
            
            for i, t in enumerate(self.tau):
                t_tau = t*1.2*2
                
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x,sup_y)
                t_0 = sup_x.duration
                t_1 = t * 1.2
                
                for k in range(int(self.pulse_num)/2):
                    x_name = 'X_XY4_%03i' % i + '_%03i.WFM' % k
                    y_name = 'Y_XY4_%03i' % i + '_%03i.WFM' % k
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                for k in range(self.pulse_num%2):
                    x_name = 'X_XY4_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    y_name = 'Y_XY4_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub        
                    
                   
                    
                mod.duration = t * 1.2 - map_x_1.stub
                name_x = 'Read_x_%04i.WFM' % i
                name_y = 'Read_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 + 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 + 90'],t_0)

                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)
                
                t_tau = t*1.2*2
                name = 'BSQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x,sup_y)
                t_0 = sup_x.duration
                t_1 = t * 1.2
                for k in range(int(self.pulse_num)/2):
                    x_name = 'BX_XY4_%03i' % i + '_%03i.WFM' % k
                    y_name = 'BY_XY4_%03i' % i + '_%03i.WFM' % k
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                for k in range(self.pulse_num%2):
                    x_name = 'BX_XY4_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    y_name = 'BY_XY4_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub   
                    
                mod.duration = t * 1.2 - map_x_1.stub
                name_x = 'BRead_x_%04i.WFM' % i
                name_y = 'BRead_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 - 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 - 90'],t_0)

                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)

            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            #time.sleep(70)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('XY4.SEQ')
            
        AWG.set_vpp(self.vpp)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def load(self):
        self.reload = True
        #make sure tau is updated
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta) 
        self.prepare_awg()
        self.reload = False        
        
    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.previous_time = 0
            #self.previous_count_data = FC.GetData()
            self.prepare_awg()
            MW.setFrequency(self.freq_center)
            MW.setPower(self.power)
            AWG.run()
            time.sleep(40.0)
            FC.Start()
            time.sleep(0.1)
            PG.Sequence(self.sequence, loop=True)

            start_time = time.time()

            while True:
               self.thread.stop_request.wait(5.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               self.elapsed_time = self.previous_elapsed_time + time.time() - start_time
               self.run_time += self.elapsed_time
               runtime, cycles = FC.GetState()
               sweeps = cycles / FC.GetData().shape[0]
               self.elapsed_sweeps = self.previous_sweeps + sweeps
               self.progress = int( 100 * self.elapsed_sweeps / self.sweeps ) 
               self.count_data = self.old_count_data + FC.GetData()
               if self.elapsed_sweeps > self.sweeps:
                  break

            FC.Halt()
            MW.Off()
            PG.High(['laser', 'mw'])
            AWG.stop()
            #time.sleep(4.0)
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  
            
        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'    
            
    def _get_sequence_points(self):
        return 2 * len(self.tau)    
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        sequence = []
        
        for i, t in enumerate(tau):
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , t*8 * self.pulse_num + 2*pi2_1 + 4*self.pulse_num*pi_1 + 2000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
            
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , t*8 * self.pulse_num + 2*pi2_1 + 4*self.pulse_num*pi_1 + 2000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','amp1','pi2_1','pi_1','pulse_num']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                     Item('amp1', width=-40), 
                                     ),                         
                              HGroup(
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40),
                                     Item('rabi_contrast', width=-40),
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
                       title='XY4_Ref',
                       )     

                       
class XY8_Ref(Pulsed):                   

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
   
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    amp1 = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp1', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    rabi_contrast=Range(low=1., high=100, value=30.0, desc='Rabi contrast [%]', label='contrast', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=0., high=1e8, value=500., desc='tau/2 begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau/2 end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau/2 [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
  
    pulse_num = Range(low=1, high=1000, value=1, desc='repetetion of XY8 pulses', label='repetitions', mode='text', auto_set=False, enter_set=True)
    reload = True
    
    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp1)]                       #X
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp1)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp1)]                   #-X
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp1)]
            
            p['pi_1_x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1_x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('XY8.SEQ')
            
            sup_x_ref = Waveform('Sup_x_ref', p['pi2_1 + 0'])
            sup_y_ref = Waveform('Sup_y_ref', p['pi2_1 + 90'])
            t_sup_ref = sup_x_ref.stub
            
            sup_x = Waveform('Sup_x', [Idle(t_sup_ref)]+p['pi2_1 + 0'])
            sup_y = Waveform('Sup_y', [Idle(t_sup_ref)]+p['pi2_1 + 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            # Reference signal_initial points
            
            for i, t in enumerate(self.tau):
                t_tau = t*1.2*2
                
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x,sup_y)
                t_0 = sup_x.duration
                t_1 = t * 1.2
                
                for k in range(int(self.pulse_num)/2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % k
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % k
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                for k in range(self.pulse_num%2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub        
                    
                   
                    
                mod.duration = t * 1.2 - map_x_1.stub
                name_x = 'Read_x_%04i.WFM' % i
                name_y = 'Read_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 + 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 + 90'],t_0)

                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)
                
                t_tau = t*1.2*2
                name = 'BSQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x,sup_y)
                t_0 = sup_x.duration
                t_1 = t * 1.2
                for k in range(int(self.pulse_num)/2):
                    x_name = 'BX_XY8_%03i' % i + '_%03i.WFM' % k
                    y_name = 'BY_XY8_%03i' % i + '_%03i.WFM' % k
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                for k in range(self.pulse_num%2):
                    x_name = 'BX_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    y_name = 'BY_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub   
                    
                mod.duration = t * 1.2 - map_x_1.stub
                name_x = 'BRead_x_%04i.WFM' % i
                name_y = 'BRead_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 - 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 - 90'],t_0)

                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)

            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            #time.sleep(70)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('XY8.SEQ')
            
        AWG.set_vpp(self.vpp)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def load(self):
        self.reload = True
        #make sure tau is updated
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta) 
        self.prepare_awg()
        self.reload = False        
        
    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.previous_time = 0
            #self.previous_count_data = FC.GetData()
            self.prepare_awg()
            MW.setFrequency(self.freq_center)
            MW.setPower(self.power)
            AWG.run()
            time.sleep(40.0)
            FC.Start()
            time.sleep(0.1)
            PG.Sequence(self.sequence, loop=True)

            start_time = time.time()

            while True:
               self.thread.stop_request.wait(5.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               self.elapsed_time = self.previous_elapsed_time + time.time() - start_time
               self.run_time += self.elapsed_time
               runtime, cycles = FC.GetState()
               sweeps = cycles / FC.GetData().shape[0]
               self.elapsed_sweeps = self.previous_sweeps + sweeps
               self.progress = int( 100 * self.elapsed_sweeps / self.sweeps ) 
               self.count_data = self.old_count_data + FC.GetData()
               if self.elapsed_sweeps > self.sweeps:
                  break

            FC.Halt()
            time.sleep(0.1)
            MW.Off()
            time.sleep(0.1)
            PG.High(['laser', 'mw'])
            time.sleep(0.1)
            AWG.stop()
            time.sleep(5.0)
            #time.sleep(4.0)
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  
            
        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'    
            
    def _get_sequence_points(self):
        return 2 * len(self.tau)    
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        sequence = []
        
        for i, t in enumerate(tau):
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , t*16 * self.pulse_num + 2*pi2_1 + 8*self.pulse_num*pi_1 + 2000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
            
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , t*16 * self.pulse_num + 2*pi2_1 + 8*self.pulse_num*pi_1 + 2000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','amp1','pi2_1','pi_1','pulse_num', 'rabi_contrast']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                     Item('amp1', width=-40), 
                                     ),                         
                              HGroup(
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40),
                                     Item('rabi_contrast', width = -40),
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
                       title='XY8_Ref',
                       )          

class XY16_Ref(Pulsed):                   

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
   
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    amp1 = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp1', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    rabi_contrast=Range(low=1., high=100, value=30.0, desc='Rabi contrast [%]', label='contrast', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=0., high=1e8, value=500., desc='tau/2 begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau/2 end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau/2 [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
  
    pulse_num = Range(low=1, high=1000, value=1, desc='repetetion of XY8 pulses', label='repetetions', mode='text', auto_set=False, enter_set=True)
    reload = True
    
    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp1)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp1)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp1)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp1)]
            
            p['pi_1_x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1_x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            p['pi_1_x - 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            p['pi_1_x - 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            
            p['pi_1_y - 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            p['pi_1_y - 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi*2 ,self.amp)]
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('XY16.SEQ')
            
            sup_x_ref = Waveform('Sup_x_ref', p['pi2_1 + 0'])
            sup_y_ref = Waveform('Sup_y_ref', p['pi2_1 + 90'])
            t_sup_ref = sup_x_ref.stub
            
            sup_x = Waveform('Sup_x', [Idle(t_sup_ref)]+p['pi2_1 + 0'])
            sup_y = Waveform('Sup_y', [Idle(t_sup_ref)]+p['pi2_1 + 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            # Reference signal_initial points
            
            for i, t in enumerate(self.tau):
                t_tau = t*1.2*2
                
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x,sup_y)
                t_0 = sup_x.duration
                t_1 = t * 1.2
                
                for k in range(int(self.pulse_num)):
                    x_name = 'X_XY16_%03i' % i + '_%03i.WFM' % k
                    y_name = 'Y_XY16_%03i' % i + '_%03i.WFM' % k
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x - 0']+[Idle(t_tau)]+p['pi_1_y - 0']+[Idle(t_tau)]+p['pi_1_x - 0']+[Idle(t_tau)]+p['pi_1_y - 0']\
                                                +[Idle(t_tau)]+p['pi_1_y - 0']+[Idle(t_tau)]+p['pi_1_x - 0']+[Idle(t_tau)]+p['pi_1_y - 0']+[Idle(t_tau)]+p['pi_1_x - 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x - 90']+[Idle(t_tau)]+p['pi_1_y - 90']+[Idle(t_tau)]+p['pi_1_x - 90']+[Idle(t_tau)]+p['pi_1_y - 90']\
                                                +[Idle(t_tau)]+p['pi_1_y - 90']+[Idle(t_tau)]+p['pi_1_x - 90']+[Idle(t_tau)]+p['pi_1_y - 90']+[Idle(t_tau)]+p['pi_1_x - 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1, map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub

                    
                mod.duration = t * 1.2 - map_x_1.stub
                name_x = 'Read_x_%04i.WFM' % i
                name_y = 'Read_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 + 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 + 90'],t_0)

                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)
                
                t_tau = t*1.2*2
                name = 'BSQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x,sup_y)
                t_0 = sup_x.duration
                t_1 = t * 1.2
                for k in range(int(self.pulse_num)):
                    x_name = 'BX_XY16_%03i' % i + '_%03i.WFM' % k
                    y_name = 'BY_XY16_%03i' % i + '_%03i.WFM' % k
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x - 0']+[Idle(t_tau)]+p['pi_1_y - 0']+[Idle(t_tau)]+p['pi_1_x - 0']+[Idle(t_tau)]+p['pi_1_y - 0']\
                                                +[Idle(t_tau)]+p['pi_1_y - 0']+[Idle(t_tau)]+p['pi_1_x - 0']+[Idle(t_tau)]+p['pi_1_y - 0']+[Idle(t_tau)]+p['pi_1_x - 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x - 90']+[Idle(t_tau)]+p['pi_1_y - 90']+[Idle(t_tau)]+p['pi_1_x - 90']+[Idle(t_tau)]+p['pi_1_y - 90']\
                                                +[Idle(t_tau)]+p['pi_1_y - 90']+[Idle(t_tau)]+p['pi_1_x - 90']+[Idle(t_tau)]+p['pi_1_y - 90']+[Idle(t_tau)]+p['pi_1_x - 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
              
                mod.duration = t * 1.2 - map_x_1.stub
                name_x = 'BRead_x_%04i.WFM' % i
                name_y = 'BRead_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 - 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 - 90'],t_0)

                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)

            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            #time.sleep(70)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('XY16.SEQ')
            
        AWG.set_vpp(self.vpp)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def load(self):
        self.reload = True
        self.freq_center=self.freq-0.1e+9
        #make sure tau is updated
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta) 
        self.prepare_awg()
        self.reload = False        
        
    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.previous_time = 0
            #self.previous_count_data = FC.GetData()
            self.prepare_awg()
            MW.setFrequency(self.freq_center)
            MW.setPower(self.power)
            AWG.run()
            time.sleep(40.0)
            FC.Start()
            time.sleep(0.1)
            PG.Sequence(self.sequence, loop=True)

            start_time = time.time()

            while True:
               self.thread.stop_request.wait(5.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               self.elapsed_time = self.previous_elapsed_time + time.time() - start_time
               self.run_time += self.elapsed_time
               runtime, cycles = FC.GetState()
               sweeps = cycles / FC.GetData().shape[0]
               self.elapsed_sweeps = self.previous_sweeps + sweeps
               self.progress = int( 100 * self.elapsed_sweeps / self.sweeps ) 
               self.count_data = self.old_count_data + FC.GetData()
               if self.elapsed_sweeps > self.sweeps:
                  break

            FC.Halt()
            time.sleep(0.1)
            MW.Off()
            time.sleep(0.1)
            PG.High(['laser', 'mw'])
            time.sleep(0.1)
            AWG.stop()
            time.sleep(1.0)
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  
            
        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'    
            
    def _get_sequence_points(self):
        return 2 * len(self.tau)    
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        sequence = []
        
        for i, t in enumerate(tau):
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , t*32 * self.pulse_num + 2*pi2_1 + 16*self.pulse_num*pi_1 + 2000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
            
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , t*32 * self.pulse_num + 2*pi2_1 + 16*self.pulse_num*pi_1 + 2000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','amp1','pi2_1','pi_1','pulse_num', 'rabi_contrast']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                     Item('amp1', width=-40), 
                                     ),                         
                              HGroup(
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40),
                                     Item('rabi_contrast', width = -40),
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
                       title='XY16_Ref',
                       )                                                

class XY4(Pulsed):                   

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
   
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=0., high=1e8, value=500., desc='tau/2 begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau/2 end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau/2 [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    
    tau_ref_int = Range(low=1., high=1e5, value=30., desc='Reference tau initial [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_ref_flag = Bool(False, label='tau_ref_flag')
    
    pulse_num = Range(low=1, high=1000, value=1, desc='repetetion of XY8 pulses', label='repetetions', mode='text', auto_set=False, enter_set=True)
    reload = True
    
    #def _init_(self):
        #super(XY8, self).__init__()

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1_x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('XY4.SEQ')
            
            sup_x_ref = Waveform('Sup_x_ref', p['pi2_1 + 0'])
            sup_y_ref = Waveform('Sup_y_ref', p['pi2_1 + 90'])
            t_sup_ref = sup_x_ref.stub
            
            sup_x = Waveform('Sup_x', [Idle(t_sup_ref)]+p['pi2_1 + 0'])
            sup_y = Waveform('Sup_y', [Idle(t_sup_ref)]+p['pi2_1 + 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            # Reference signal_initial points
            
            for i, t in enumerate(self.tau):
                if(self.tau_ref_flag and i < 4):
                    name = 'RefQH_12_%04i.SEQ' % i
                    sub_seq=Sequence(name)
                    sub_seq.append(sup_x,sup_y)
                    t_0 = sup_x.duration
                    t_1 = self.tau_ref_int * 1.2
                    t_tau = self.tau_ref_int * 1.2 * 2
                    for k in range(self.pulse_num/2):
                        x_name = 'RefX_XY4_%03i' % i + '_%03i.WFM' % k
                        y_name = 'RefY_XY4_%03i' % i + '_%03i.WFM' % k
                       
                        map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                    +[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0'],t_0)
                        map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                    +[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90'],t_0)  
                                                    
                        self.waves.append(map_x_1)
                        self.waves.append(map_y_1)
                        sub_seq.append(map_x_1,map_y_1)
                        
                        t_0 += map_x_1.duration
                        t_1 = t_tau - map_x_1.stub
                       

                    mod.duration = self.tau_ref_int * 1.2 - map_x_1.stub
                    name_x = 'Ref_Read_x_%04i.WFM' % i
                    name_y = 'Ref_Read_y_%04i.WFM' % i
                    ref_x = Waveform(name_x, [mod] + p['pi2_1 + 0'],t_0)
                    ref_y = Waveform(name_y, [mod] + p['pi2_1 + 90'],t_0)

                    self.waves.append(ref_x)
                    self.waves.append(ref_y)
                    sub_seq.append(ref_x,ref_y)
                    AWG.upload(sub_seq)
                    
                    self.main_seq.append(sub_seq, wait=True)
                    
                else:    
                    t_tau = t*1.2*2
                    
                    name = 'SQH_12_%04i.SEQ' % i
                    sub_seq=Sequence(name)
                    sub_seq.append(sup_x,sup_y)
                    t_0 = sup_x.duration
                    t_1 = t * 1.2
                    for k in range(self.pulse_num/2):
                        x_name = 'X_XY4_%03i' % i + '_%03i.WFM' % k
                        y_name = 'Y_XY4_%03i' % i + '_%03i.WFM' % k
                        map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                    +[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0'],t_0)
                        map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                    +[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90'],t_0)  
                        self.waves.append(map_x_1)
                        self.waves.append(map_y_1)
                        sub_seq.append(map_x_1,map_y_1)
                        
                        t_0 += map_x_1.duration
                        t_1 = t_tau - map_x_1.stub

                    mod.duration = t * 1.2 - map_x_1.stub
                    name_x = 'Read_x_%04i.WFM' % i
                    name_y = 'Read_y_%04i.WFM' % i
                    ref_x = Waveform(name_x, [mod] + p['pi2_1 + 0'],t_0)
                    ref_y = Waveform(name_y, [mod] + p['pi2_1 + 90'],t_0)

                    self.waves.append(ref_x)
                    self.waves.append(ref_y)
                    sub_seq.append(ref_x,ref_y)
                    AWG.upload(sub_seq)
                    
                    self.main_seq.append(sub_seq, wait=True)

            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            #time.sleep(70)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('XY4.SEQ')
            
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def load(self):
        self.reload = True
        #make sure tau is updated
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta) 
        if(self.tau_ref_flag):
            npoints = len(self.tau)
            a = np.arange(npoints+4)
            b = np.arange(self.tau[0]-4,self.tau[0],1)
            a[:4]= b
            a[-npoints:]=self.tau
            self.tau = a
        #self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta) 
        self.prepare_awg()
        self.reload = False        
        
    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.previous_time = 0
            #self.previous_count_data = FC.GetData()
            self.prepare_awg()
            MW.setFrequency(self.freq_center)
            MW.setPower(self.power)
            AWG.run()
            time.sleep(40.0)
            FC.Start()
            time.sleep(0.1)
            PG.Sequence(self.sequence, loop=True)

            start_time = time.time()

            while True:
               self.thread.stop_request.wait(5.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               self.elapsed_time = self.previous_elapsed_time + time.time() - start_time
               self.run_time += self.elapsed_time
               runtime, cycles = FC.GetState()
               sweeps = cycles / FC.GetData().shape[0]
               self.elapsed_sweeps = self.previous_sweeps + sweeps
               self.progress = int( 100 * self.elapsed_sweeps / self.sweeps ) 
               self.count_data = self.old_count_data + FC.GetData()
               if self.elapsed_sweeps > self.sweeps:
                  break

            FC.Halt()
            MW.Off()
            PG.High(['laser', 'mw'])
            AWG.stop()
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  
            
        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'    

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        tau_1 = self.tau_ref_int
        sequence = []
        
        for i, t in enumerate(tau):
            if(self.tau_ref_flag and i < 4):
                sequence.append( (['awgTrigger']      , 100) )
                sequence.append( ([ ]                 , tau_1*8 * self.pulse_num + 2*pi2_1 + 4*self.pulse_num*pi_1 + 2000) )
                sequence.append( (['laser', 'trigger'], laser) )
                sequence.append( ([ ]                 , wait) )
            else:
                sequence.append( (['awgTrigger']      , 100) )
                sequence.append( ([ ]                 , t*8 * self.pulse_num + 2*pi2_1 + 4*self.pulse_num*pi_1 + 2000) )
                sequence.append( (['laser', 'trigger'], laser) )
                sequence.append( ([ ]                 , wait) )
        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','pi_1','tau_ref_int','tau_ref_flag','pulse_num']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                     ),                         
                              HGroup(
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40),
                                     Item('tau_ref_flag', enabled_when='state != "run"'),
                                     Item('tau_ref_int', width = -40),
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
                       title='XY4',
                       )     

class CPMG4(Pulsed):                   

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
   
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=0., high=1e8, value=500., desc='tau/2 begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau/2 end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau/2 [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    
    tau_ref_int = Range(low=1., high=1e5, value=30., desc='Reference tau initial [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_ref_flag = Bool(False, label='tau_ref_flag')
    
    pulse_num = Range(low=1, high=1000, value=1, desc='repetetion of XY8 pulses', label='repetetions', mode='text', auto_set=False, enter_set=True)
    reload = True
    
    #def _init_(self):
        #super(XY8, self).__init__()

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1_x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('CPMG4.SEQ')
            
            sup_x_ref = Waveform('Sup_x_ref', p['pi2_1 + 0'])
            sup_y_ref = Waveform('Sup_y_ref', p['pi2_1 + 90'])
            t_sup_ref = sup_x_ref.stub
            
            sup_x = Waveform('Sup_x', [Idle(t_sup_ref)]+p['pi2_1 + 0'])
            sup_y = Waveform('Sup_y', [Idle(t_sup_ref)]+p['pi2_1 + 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            # Reference signal_initial points
            
            for i, t in enumerate(self.tau):
                if(self.tau_ref_flag and i < 4):
                    name = 'RefQH_12_%04i.SEQ' % i
                    sub_seq=Sequence(name)
                    sub_seq.append(sup_x,sup_y)
                    t_0 = sup_x.duration
                    t_1 = self.tau_ref_int * 1.2
                    t_tau = self.tau_ref_int * 1.2 * 2
                    for k in range(self.pulse_num/2):
                        x_name = 'RefX_XY4_%03i' % i + '_%03i.WFM' % k
                        y_name = 'RefY_XY4_%03i' % i + '_%03i.WFM' % k
                       
                        map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                    +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0'],t_0)
                        map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                    +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90'],t_0)  
                                                    
                        self.waves.append(map_x_1)
                        self.waves.append(map_y_1)
                        sub_seq.append(map_x_1,map_y_1)
                        
                        t_0 += map_x_1.duration
                        t_1 = t_tau - map_x_1.stub
                       

                    mod.duration = self.tau_ref_int * 1.2 - map_x_1.stub
                    name_x = 'Ref_Read_x_%04i.WFM' % i
                    name_y = 'Ref_Read_y_%04i.WFM' % i
                    ref_x = Waveform(name_x, [mod] + p['pi2_1 + 0'],t_0)
                    ref_y = Waveform(name_y, [mod] + p['pi2_1 + 90'],t_0)

                    self.waves.append(ref_x)
                    self.waves.append(ref_y)
                    sub_seq.append(ref_x,ref_y)
                    AWG.upload(sub_seq)
                    
                    self.main_seq.append(sub_seq, wait=True)
                    
                else:    
                    t_tau = t*1.2*2
                    
                    name = 'SQH_12_%04i.SEQ' % i
                    sub_seq=Sequence(name)
                    sub_seq.append(sup_x,sup_y)
                    t_0 = sup_x.duration
                    t_1 = t * 1.2
                    for k in range(self.pulse_num/2):
                        x_name = 'X_CPMG4_%03i' % i + '_%03i.WFM' % k
                        y_name = 'Y_CPMG4_%03i' % i + '_%03i.WFM' % k
                        map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                    +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0'],t_0)
                        map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                    +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90'],t_0)  
                        self.waves.append(map_x_1)
                        self.waves.append(map_y_1)
                        sub_seq.append(map_x_1,map_y_1)
                        
                        t_0 += map_x_1.duration
                        t_1 = t_tau - map_x_1.stub

                    mod.duration = t * 1.2 - map_x_1.stub
                    name_x = 'Read_x_%04i.WFM' % i
                    name_y = 'Read_y_%04i.WFM' % i
                    ref_x = Waveform(name_x, [mod] + p['pi2_1 + 0'],t_0)
                    ref_y = Waveform(name_y, [mod] + p['pi2_1 + 90'],t_0)

                    self.waves.append(ref_x)
                    self.waves.append(ref_y)
                    sub_seq.append(ref_x,ref_y)
                    AWG.upload(sub_seq)
                    
                    self.main_seq.append(sub_seq, wait=True)

            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            #time.sleep(70)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('XY4.SEQ')
            
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def load(self):
        self.reload = True
        #make sure tau is updated
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta) 
        if(self.tau_ref_flag):
            npoints = len(self.tau)
            a = np.arange(npoints+4)
            b = np.arange(self.tau[0]-4,self.tau[0],1)
            a[:4]= b
            a[-npoints:]=self.tau
            self.tau = a
        #self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta) 
        self.prepare_awg()
        self.reload = False        
        
    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.previous_time = 0
            #self.previous_count_data = FC.GetData()
            self.prepare_awg()
            MW.setFrequency(self.freq_center)
            MW.setPower(self.power)
            AWG.run()
            time.sleep(40.0)
            FC.Start()
            time.sleep(0.1)
            PG.Sequence(self.sequence, loop=True)

            start_time = time.time()

            while True:
               self.thread.stop_request.wait(5.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               self.elapsed_time = self.previous_elapsed_time + time.time() - start_time
               self.run_time += self.elapsed_time
               runtime, cycles = FC.GetState()
               sweeps = cycles / FC.GetData().shape[0]
               self.elapsed_sweeps = self.previous_sweeps + sweeps
               self.progress = int( 100 * self.elapsed_sweeps / self.sweeps ) 
               self.count_data = self.old_count_data + FC.GetData()
               if self.elapsed_sweeps > self.sweeps:
                  break

            FC.Halt()
            MW.Off()
            PG.High(['laser', 'mw'])
            AWG.stop()
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  
            
        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'    

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        tau_1 = self.tau_ref_int
        sequence = []
        
        for i, t in enumerate(tau):
            if(self.tau_ref_flag and i < 4):
                sequence.append( (['awgTrigger']      , 100) )
                sequence.append( ([ ]                 , tau_1*8 * self.pulse_num + 2*pi2_1 + 4*self.pulse_num*pi_1 + 2000) )
                sequence.append( (['laser', 'trigger'], laser) )
                sequence.append( ([ ]                 , wait) )
            else:
                sequence.append( (['awgTrigger']      , 100) )
                sequence.append( ([ ]                 , t*8 * self.pulse_num + 2*pi2_1 + 4*self.pulse_num*pi_1 + 2000) )
                sequence.append( (['laser', 'trigger'], laser) )
                sequence.append( ([ ]                 , wait) )
        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','pi_1','tau_ref_int','tau_ref_flag','pulse_num']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                     ),                         
                              HGroup(
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40),
                                     Item('tau_ref_flag', enabled_when='state != "run"'),
                                     Item('tau_ref_int', width = -40),
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
                       title='CPMG4',
                       )                            

class SpinLocking_decay( Pulsed ):
    #FID 
    
    pi2_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)

    freq_center = Range(low=1, high=20e9, value=1.96e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)   
    amp1 = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='locking amp', mode='text', auto_set=False, enter_set=True)
    
    phase = Range(low=0., high=360, value=0.0, desc='phase of locking pulse', label='locking phase', mode='text', auto_set=False, enter_set=True)
    
    rabi_contrast=Range(low=1., high=100, value=30.0, desc='Rabi contrast [%]', label='contrast', mode='text', auto_set=False, enter_set=True)
    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            p = {}
            
            pi2_1 = int(self.pi2_1 * sampling/1.0e9)
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]                  #x
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]            #y
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]              #-x
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]          #-y
            

            zero = Idle(1)
            mod = Idle(0)

            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('SLD.SEQ')
            
            for i, t in  enumerate(self.tau):

                t_1 = int(t*1.2)
                phase = self.phase/360 * np.pi
                name_x = 'lock1_X%04i.WFM' % i
                name_y = 'lock1_Y%04i.WFM' % i
                
                map_x = Waveform(name_x, p['pi2_1 + 0'] + [Sin( t_1 , (self.freq - self.freq_center)/sampling, phase + np.pi/2, self.amp1)] + p['pi2_1 + 0'])
                map_y = Waveform(name_y, p['pi2_1 + 90'] + [Sin( t_1, (self.freq - self.freq_center)/sampling, phase + np.pi, self.amp1)] + p['pi2_1 + 90'])
                
                #map_x = Waveform(name_x, p['pi2_1 + 0'] + [Idle(t_1)] + p['pi2_1 + 0'])
                #map_y = Waveform(name_y, p['pi2_1 + 90'] + [Idle(t_1)] + p['pi2_1 + 90'])
                
                self.waves.append(map_x)
                self.waves.append(map_y)
                self.main_seq.append(*self.waves[-2:],wait=True)
                

                name_x = 'Block1_X%04i.WFM' % i
                name_y = 'Block1_Y%04i.WFM' % i

                map1_x = Waveform(name_x, p['pi2_1 + 0'] + [Sin( t_1 , (self.freq - self.freq_center)/sampling, phase + np.pi/2, self.amp1)] + p['pi2_1 - 0'])
                map1_y = Waveform(name_y, p['pi2_1 + 90'] + [Sin( t_1, (self.freq - self.freq_center)/sampling, phase + np.pi, self.amp1)] + p['pi2_1 - 90'])
                
                #map1_x = Waveform(name_x, p['pi2_1 + 0'] + [Idle(t_1)] + p['pi2_1 - 0'])
                #map1_y = Waveform(name_y, p['pi2_1 + 90'] + [Idle(t_1)] + p['pi2_1 - 90'])
                
                self.waves.append(map1_x)
                self.waves.append(map1_y)
                self.main_seq.append(*self.waves[-2:],wait=True)
                
                
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('SLD.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def _get_sequence_points(self):
        return 2 * len(self.tau)    
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1 = self.pi2_1
        sequence = []
        
        for t in tau:
            sub = [ (['awgTrigger'], 100 ),
                    ([], t + pi2_1*2 + 100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait ),
                    
                    (['awgTrigger'], 100 ),
                    ([], t + pi2_1*2 + 100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)

        return sequence
            
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','amp1','phase', 'rabi_contrast'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-65),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     ),       
                             HGroup(Item('amp1', width=-40),
                                    Item('pi2_1', width=20), 
                                    Item('phase', width=20),
                                    Item('rabi_contrast', width=-40),
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
                       title='Spin Locking',
                       ) 

class Spin_Locking_without_polarization( Pulsed ):
    #FID 
    
    pi2_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)

    freq_center = Range(low=1, high=20e9, value=1.96e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)   
    amp1 = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='locking amp', mode='text', auto_set=False, enter_set=True)
    
    phase = Range(low=0., high=360, value=0.0, desc='phase of locking pulse', label='locking phase', mode='text', auto_set=False, enter_set=True)
    
    rabi_contrast=Range(low=1., high=100, value=30.0, desc='Rabi contrast [%]', label='contrast', mode='text', auto_set=False, enter_set=True)
    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            p = {}
            
            pi2_1 = int(self.pi2_1 * sampling/1.0e9)
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]                  #x
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2, self.amp)]            #y
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi, self.amp)]              #-x
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2, self.amp)]          #-y
            

            zero = Idle(1)
            mod = Idle(0)

            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('SLD.SEQ')
            
            # for i, t in  enumerate(self.tau):

                # t_1 = int(t*1.2)
                # phase = self.phase/360 * np.pi
                # name_x = 'lock1_X%04i.WFM' % i
                # name_y = 'lock1_Y%04i.WFM' % i
                
                # map_x = Waveform(name_x, p['pi2_1 + 0'] + [Sin( t_1 , (self.freq - self.freq_center)/sampling, phase + np.pi/2, self.amp1)] + p['pi2_1 + 0'])
                # map_y = Waveform(name_y, p['pi2_1 + 90'] + [Sin( t_1, (self.freq - self.freq_center)/sampling, phase + np.pi, self.amp1)] + p['pi2_1 + 90'])
                                
                # self.waves.append(map_x)
                # self.waves.append(map_y)
                # self.main_seq.append(*self.waves[-2:],wait=True)
                

                # name_x = 'Block1_X%04i.WFM' % i
                # name_y = 'Block1_Y%04i.WFM' % i

                # map1_x = Waveform(name_x, p['pi2_1 + 0'] + [Sin( t_1 , (self.freq - self.freq_center)/sampling, phase + np.pi*3/2, self.amp1)] + p['pi2_1 + 0'])
                # map1_y = Waveform(name_y, p['pi2_1 + 90'] + [Sin( t_1, (self.freq - self.freq_center)/sampling, phase + np.pi*2, self.amp1)] + p['pi2_1 + 90'])
                
                # self.waves.append(map1_x)
                # self.waves.append(map1_y)
                # self.main_seq.append(*self.waves[-2:],wait=True)
                #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
            for i, t in  enumerate(self.tau):

                t_1 = int(t*1.2)
                phase = self.phase/360 * np.pi
                name_x = 'lock1_X%04i.WFM' % i
                name_y = 'lock1_Y%04i.WFM' % i
                
                map_x = Waveform(name_x, p['pi2_1 + 0'] + [Sin( t_1 , (self.freq - self.freq_center)/sampling, phase + np.pi/2, self.amp1)] + p['pi2_1 - 0'])
                map_y = Waveform(name_y, p['pi2_1 + 90'] + [Sin( t_1, (self.freq - self.freq_center)/sampling, phase + np.pi, self.amp1)] + p['pi2_1 - 90'])
                                
                self.waves.append(map_x)
                self.waves.append(map_y)
                self.main_seq.append(*self.waves[-2:],wait=True)
                

                name_x = 'Block1_X%04i.WFM' % i
                name_y = 'Block1_Y%04i.WFM' % i

                map1_x = Waveform(name_x, p['pi2_1 + 0'] + [Sin( t_1 , (self.freq - self.freq_center)/sampling, phase + np.pi*3/2, self.amp1)] + p['pi2_1 - 0'])
                map1_y = Waveform(name_y, p['pi2_1 + 90'] + [Sin( t_1, (self.freq - self.freq_center)/sampling, phase + np.pi*2, self.amp1)] + p['pi2_1 - 90'])
                
                self.waves.append(map1_x)
                self.waves.append(map1_y)
                self.main_seq.append(*self.waves[-2:],wait=True)
                
                
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('SLD.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def _get_sequence_points(self):
        return 2 * len(self.tau)    
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1 = self.pi2_1
        sequence = []
        
        for t in tau:
            sub = [ (['awgTrigger'], 100 ),
                    ([], t + pi2_1*2 + 100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait ),
                    
                    (['awgTrigger'], 100 ),
                    ([], t + pi2_1*2 + 100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)
           
        return sequence
            
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','amp1','phase', 'rabi_contrast'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-65),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     ),       
                             HGroup(Item('amp1', width=-40),
                                    Item('pi2_1', width=20), 
                                    Item('phase', width=20),
                                    Item('rabi_contrast', width=-40),
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
                       title='Spin_Locking_without_polarization',
                       )                        

class pulse_phase_check( Pulsed ):
    #FID 
    
    pi2_1 = Range(low=1., high=100000., value=25.75, desc='length of half pi pulse 1st transition [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='pi', mode='text', auto_set=False, enter_set=True)

    freq_center = Range(low=1, high=20e9, value=1.96e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)   
    amp1 = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='locking amp', mode='text', auto_set=False, enter_set=True)
    
    tau_evo = Range(low=0.0, high=100000, value=100, desc='tau between pi pulses', label='tau_evo', mode='text', auto_set=False, enter_set=True)
    phase = Range(low=0., high=360, value=0.0, desc='phase of locking pulse', label='locking phase', mode='text', auto_set=False, enter_set=True)
    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            p = {}
            
            pi2_1 = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            
            phase = self.phase/360 * np.pi
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1_x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, phase + np.pi/2 ,self.amp)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, phase + np.pi ,self.amp)]

            zero = Idle(1)
            mod = Idle(0)

            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('PPC.SEQ')
            
            sup_x_ref = Waveform('Sup_x_ref', p['pi2_1 + 0'])
            sup_y_ref = Waveform('Sup_y_ref', p['pi2_1 + 90'])
            t_sup_ref = sup_x_ref.stub
            
            sup_x = Waveform('Sup_x', [Idle(t_sup_ref)]+p['pi2_1 + 0'])
            sup_y = Waveform('Sup_y', [Idle(t_sup_ref)]+p['pi2_1 + 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            t_tau = self.tau_evo * 1.2 * 2
            for i, t in enumerate(self.tau):

                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x,sup_y)
                t_0 = sup_x.duration
                t_1 = self.tau_evo * 1.2
                for k in range(int(t)):
                    x_name = 'X_XY4_%03i' % i + '_%03i.WFM' % k
                    y_name = 'Y_XY4_%03i' % i + '_%03i.WFM' % k
                    map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub

                mod.duration = self.tau_evo * 1.2 - map_x_1.stub
                name_x = 'Read_x_%04i.WFM' % i
                name_y = 'Read_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 + 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 + 90'],t_0)

                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)

                
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('PPC.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1 = self.pi2_1
        pi_1 = self.pi_1
        tau_evo = self.tau_evo
        sequence = []
        
        for t in tau:
            sub = [ (['awgTrigger'], 100 ),
                    ([], 4 * tau_evo * t + pi2_1*2 + 2 * pi_1 * t + 2000),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)

        return sequence
            
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','amp1','phase','pi_1','tau_evo'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-65),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     ),       
                             HGroup(Item('amp1', width=-40),
                                    Item('pi2_1', width=20), 
                                    Item('pi_1', width=20), 
                                    Item('phase', width=20), 
                                    Item('tau_evo', width=20), 
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
                       title='phase-checking',
                       )                              
                       
class HartmannHahn( Pulsed ):
    #FID 
    
    pi2_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)   
    amp1 = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='locking amp', mode='text', auto_set=False, enter_set=True)
    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            p = {}
            
            pi2_1 = int(self.pi2_1 * sampling/1.0e9)
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            

            zero = Idle(1)
            mod = Idle(0)

            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('HHahn.SEQ')
            
            for i, t in  enumerate(self.tau):
                t_1 = t*1.2
                name_x = 'lock1_X%04i.WFM' % i
                name_y = 'lock1_Y%04i.WFM' % i
            
                map_x = Waveform(name_x, p['pi2_1 + 0'] + [Idle(10)] + [Sin( t_1 , (self.freq - self.freq_center)/sampling, np.pi/2, self.amp1)]  + [Idle(10)] + p['pi2_1 + 0'])
                map_y = Waveform(name_y, p['pi2_1 + 90'] +  [Idle(10)] + [Sin( t_1, (self.freq - self.freq_center)/sampling, np.pi, self.amp1)] +  [Idle(10)] +p['pi2_1 + 90'])
                
                self.waves.append(map_x)
                self.waves.append(map_y)
                self.main_seq.append(*self.waves[-2:],wait=True)
                
                name_x = 'lock2_X%04i.WFM' % i
                name_y = 'lock2_Y%04i.WFM' % i
                
                ref_x = Waveform(name_x, p['pi2_1 - 0'] + [Idle(10)] +[Sin( t_1 , (self.freq - self.freq_center)/sampling, np.pi/2, self.amp1)] + [Idle(10)] + p['pi2_1 - 0'])
                ref_y = Waveform(name_y, p['pi2_1 - 90'] +[Idle(10)] + [Sin( t_1, (self.freq - self.freq_center)/sampling, np.pi, self.amp1)] + [Idle(10)] + p['pi2_1 - 90'])
                
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                
                self.main_seq.append(*self.waves[-2:],wait=True)
                
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('HHahn.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1 = self.pi2_1
        sequence = []
        
        for t in tau:
            sub = [ (['awgTrigger'], 100 ),
                    ([], t + pi2_1*2 + 2100),
                    (['laser' ], laser ),
                    ([], wait ),
                    (['awgTrigger'], 100 ),
                    ([], t + pi2_1*2 + 2100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)

        return sequence
            
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','amp1'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-65),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     ),       
                             HGroup(Item('amp1', width=-40),
                                    Item('pi2_1', width=20), 
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
                       title='Hartmann-hahn',
                       )      

class HartmannHahnOneway( Pulsed ):
    #FID 
    
    pi2_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)   
    amp1 = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='locking amp', mode='text', auto_set=False, enter_set=True)
    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            p = {}
            
            pi2_1 = int(self.pi2_1 * sampling/1.0e9)
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            

            zero = Idle(1)
            mod = Idle(0)

            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('HHahn.SEQ')
            
            for i, t in  enumerate(self.tau):
                t_1 = int(t*1.2/1.0)
                name_x = 'lock1_X%04i.WFM' % i
                name_y = 'lock1_Y%04i.WFM' % i
            
                map_x = Waveform(name_x, p['pi2_1 + 0'] + [Idle(10)] + [Sin( t_1 , (self.freq - self.freq_center)/sampling, np.pi/2, self.amp1)]\
                                                                     #+ [Sin( t_1 , (self.freq - self.freq_center)/sampling, np.pi*3/2, self.amp1)]\
                                                                     #+ [Sin( t_1 , (self.freq - self.freq_center)/sampling, np.pi/2, self.amp1)]\
                                                                     #+ [Sin( t_1 , (self.freq - self.freq_center)/sampling, np.pi*3/2, self.amp1)]\
                                                         + [Idle(10)] + p['pi2_1 + 0'])
                map_y = Waveform(name_y, p['pi2_1 + 90'] +  [Idle(10)] + [Sin( t_1, (self.freq - self.freq_center)/sampling, np.pi, self.amp1)]\
                                                                      # + [Sin( t_1, (self.freq - self.freq_center)/sampling, np.pi*2, self.amp1)]\
                                                                      # + [Sin( t_1, (self.freq - self.freq_center)/sampling, np.pi, self.amp1)]\
                                                                      # + [Sin( t_1, (self.freq - self.freq_center)/sampling, np.pi*2, self.amp1)]\
                                                        +  [Idle(10)] +p['pi2_1 + 90'])
                
                self.waves.append(map_x)
                self.waves.append(map_y)
                self.main_seq.append(*self.waves[-2:],wait=True)
                
                name_x = 'lock2_X%04i.WFM' % i
                name_y = 'lock2_Y%04i.WFM' % i
            
                map_x = Waveform(name_x, p['pi2_1 + 0'] + [Idle(10)] + [Sin( t_1 , (self.freq - self.freq_center)/sampling, np.pi/2, self.amp1)]\
                                                                     #+ [Sin( t_1 , (self.freq - self.freq_center)/sampling, np.pi*3/2, self.amp1)]\
                                                                     #+ [Sin( t_1 , (self.freq - self.freq_center)/sampling, np.pi/2, self.amp1)]\
                                                                     #+ [Sin( t_1 , (self.freq - self.freq_center)/sampling, np.pi*3/2, self.amp1)]\
                                                         + [Idle(10)] + p['pi2_1 - 0'])
                map_y = Waveform(name_y, p['pi2_1 + 90'] +  [Idle(10)] + [Sin( t_1, (self.freq - self.freq_center)/sampling, np.pi, self.amp1)]\
                                                                      # + [Sin( t_1, (self.freq - self.freq_center)/sampling, np.pi*2, self.amp1)]\
                                                                      # + [Sin( t_1, (self.freq - self.freq_center)/sampling, np.pi, self.amp1)]\
                                                                      # + [Sin( t_1, (self.freq - self.freq_center)/sampling, np.pi*2, self.amp1)]\
                                                        +  [Idle(10)] +p['pi2_1 - 90'])
                
                self.waves.append(map_x)
                self.waves.append(map_y)
                self.main_seq.append(*self.waves[-2:],wait=True)
                
                
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('HHahn.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def _get_sequence_points(self):
        return 2 * len(self.tau)     
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1 = self.pi2_1
        sequence = []
        
        for t in tau:
            sub = [ (['awgTrigger'], 100 ),
                    ([], t + pi2_1*2 + 2100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                    
                    (['awgTrigger'], 100 ),
                    ([], t + pi2_1*2 + 2100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)

        return sequence
            
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','amp1'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-65),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     ),       
                             HGroup(Item('amp1', width=-40),
                                    Item('pi2_1', width=20), 
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
                       title='Hartmann-hahn-oneway',
                       )   

class HartmannHahnOnewayfsweep( Pulsed ):
    #FID 
    
    pi2_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)   
    amp1 = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='locking amp', mode='text', auto_set=False, enter_set=True)
    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            p = {}
            
            pi2_1 = int(self.pi2_1 * sampling/1.0e9)
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            

            zero = Idle(1)
            mod = Idle(0)

            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('HHahn.SEQ')
            
            for i, t in  enumerate(self.tau):
                t_1 = int(t*1.2/11.0)
                name_x = 'lock1_X%04i.WFM' % i
                name_y = 'lock1_Y%04i.WFM' % i
                
                f1 = (self.freq - self.freq_center)/sampling
                df = 3e4/sampling
                map_x = Waveform(name_x, p['pi2_1 + 0'] + [Idle(10)] + [SinRamp( t_1 , f1, np.pi/2, self.amp1,self.amp1*1.1)]\
                                                         + [Idle(10)] + p['pi2_1 + 0'])
                map_y = Waveform(name_y, p['pi2_1 + 90'] +  [Idle(10)] + [SinRamp( t_1 , f1, np.pi, self.amp1,self.amp1*1.1)]\
                                                        +  [Idle(10)] +p['pi2_1 + 90'])
                
                self.waves.append(map_x)
                self.waves.append(map_y)
                self.main_seq.append(*self.waves[-2:],wait=True)
                
                
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('HHahn.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1 = self.pi2_1
        sequence = []
        
        for t in tau:
            sub = [ (['awgTrigger'], 100 ),
                    ([], t + pi2_1*2 + 2100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)

        return sequence
            
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','amp1'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-65),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     ),       
                             HGroup(Item('amp1', width=-40),
                                    Item('pi2_1', width=20), 
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
                       title='Hartmann-hahn-oneway-fsweep',
                       )                                   

class HartmannHahnamp( Pulsed ):
    #FID 
    
    pi2_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)   
    
    tpump = Range(low=1., high=300000., value=10000, desc='nuclear spin polarization time [ns]', label='pumptime', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=0., high=1.0, value=0., desc='AWG amp begin', label='amp begin', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=0., high=1.0, value=1.0, desc='AWG amp end', label='amp end', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0., high=1., value=0.01, desc='delta AWG amp', label='delta amp', mode='text', auto_set=False, enter_set=True)
    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            p = {}
            
            pi2_1 = int(self.pi2_1 * sampling/1.0e9)
            tpump = int(self.tpump * sampling/1.0e9)
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            

            zero = Idle(1)
            mod = Idle(0)

            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('HHahnscan.SEQ')
            
            for i, t in  enumerate(self.tau):
                name_x = 'lock1_X%04i.WFM' % i
                name_y = 'lock1_Y%04i.WFM' % i
            
                map_x = Waveform(name_x, p['pi2_1 + 0'] + [Idle(10)] + [Sin( tpump , (self.freq - self.freq_center)/sampling, np.pi/2, t)]  + [Idle(10)] + p['pi2_1 + 0'])
                map_y = Waveform(name_y, p['pi2_1 + 90'] +  [Idle(10)] + [Sin( tpump, (self.freq - self.freq_center)/sampling, np.pi, t)] +  [Idle(10)] +p['pi2_1 + 90'])
                
                self.waves.append(map_x)
                self.waves.append(map_y)
                self.main_seq.append(*self.waves[-2:],wait=True)
                
                name_x = 'lock2_X%04i.WFM' % i
                name_y = 'lock2_Y%04i.WFM' % i
                
                ref_x = Waveform(name_x, p['pi2_1 - 0'] + [Idle(10)] +[Sin( tpump , (self.freq - self.freq_center)/sampling, np.pi/2, t)] + [Idle(10)] + p['pi2_1 - 0'])
                ref_y = Waveform(name_y, p['pi2_1 - 90'] +[Idle(10)] + [Sin( tpump, (self.freq - self.freq_center)/sampling, np.pi, t)] + [Idle(10)] + p['pi2_1 - 90'])
                
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                
                self.main_seq.append(*self.waves[-2:],wait=True)
                
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('HHahnscan.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1 = self.pi2_1
        tpump = self.tpump
        sequence = []
        
        for t in tau:
            sub = [ (['awgTrigger'], 100 ),
                    ([], tpump + pi2_1*2 + 2100),
                    (['laser' ], laser ),
                    ([], wait ),
                    (['awgTrigger'], 100 ),
                    ([], tpump + pi2_1*2 + 2100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)

        return sequence
            
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','tpump'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-65),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     ),       
                             HGroup(Item('tpump', width=-40),
                                    Item('pi2_1', width=20), 
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
                       title='Hartmann-hahn_amp_scan',
                       )              

class HartmannHahnampOneway( Pulsed ):
    #FID 
    
    pi2_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)   
    
    tpump = Range(low=1., high=300000., value=10000, desc='nuclear spin polarization time [ns]', label='pumptime', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=0., high=1.0, value=0., desc='AWG amp begin', label='amp begin', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=0., high=1.0, value=1.0, desc='AWG amp end', label='amp end', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0., high=1., value=0.01, desc='delta AWG amp', label='delta amp', mode='text', auto_set=False, enter_set=True)
    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            p = {}
            
            pi2_1 = int(self.pi2_1 * sampling/1.0e9)
            tpump = int(self.tpump * sampling/1.0e9/1.0)
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            
            zero = Idle(1)
            mod = Idle(0)

            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('HHahnscan.SEQ')
            
            for i, t in  enumerate(self.tau):
                name_x = 'lock1_X%04i.WFM' % i
                name_y = 'lock1_Y%04i.WFM' % i
            
                map_x = Waveform(name_x, p['pi2_1 + 0'] + [Idle(10)] + [Sin( tpump , (self.freq - self.freq_center)/sampling, np.pi/2, t)]\
                                                                     #+ [Sin( tpump , (self.freq - self.freq_center)/sampling, np.pi*3/2, t)]\
                                                                     #+ [Sin( tpump , (self.freq - self.freq_center)/sampling, np.pi/2, t)]\
                                                                     #+ [Sin( tpump , (self.freq - self.freq_center)/sampling, np.pi*3/2, t)]\
                                                         + [Idle(10)] + p['pi2_1 + 0'])
                map_y = Waveform(name_y, p['pi2_1 + 90'] +  [Idle(10)] + [Sin( tpump, (self.freq - self.freq_center)/sampling, np.pi, t)]\
                                                                      # + [Sin( tpump, (self.freq - self.freq_center)/sampling, np.pi*2, t)]\
                                                                       #+ [Sin( tpump, (self.freq - self.freq_center)/sampling, np.pi, t)]\
                                                                       #+ [Sin( tpump, (self.freq - self.freq_center)/sampling, np.pi*2, t)]\
                                                        +  [Idle(10)] +p['pi2_1 + 90'])
                
                self.waves.append(map_x)
                self.waves.append(map_y)
                self.main_seq.append(*self.waves[-2:],wait=True)
                
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('HHahnscan.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1 = self.pi2_1
        tpump = self.tpump
        sequence = []
        
        for t in tau:
            sub = [ (['awgTrigger'], 100 ),
                    ([], tpump + pi2_1*2 + 2100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)

        return sequence
            
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','tpump'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-65),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     ),       
                             HGroup(Item('tpump', width=-40),
                                    Item('pi2_1', width=20), 
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
                       title='Hartmann-hahn_amp_scan_oneway',
                       )                

class HartmannHahnampOnewayfsweep( Pulsed ):
    #FID 
    
    pi2_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)   
    
    tpump = Range(low=1., high=300000., value=10000, desc='nuclear spin polarization time [ns]', label='pumptime', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=0., high=1.0, value=0., desc='AWG amp begin', label='amp begin', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=0., high=1.0, value=1.0, desc='AWG amp end', label='amp end', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0., high=1., value=0.01, desc='delta AWG amp', label='delta amp', mode='text', auto_set=False, enter_set=True)
    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            p = {}
            
            pi2_1 = int(self.pi2_1 * sampling/1.0e9)
            tpump = int(self.tpump * sampling/1.0e9)
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            
            zero = Idle(1)
            mod = Idle(0)

            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('HHahnscan.SEQ')
            
            f1 = (self.freq - self.freq_center)/sampling
            #df = 6e4/sampling
            for i, t in  enumerate(self.tau):
                name_x = 'lock1_X%04i.WFM' % i
                name_y = 'lock1_Y%04i.WFM' % i

                map_x = Waveform(name_x, p['pi2_1 + 0'] + [Idle(10)] + [SinRamp( tpump , f1, np.pi/2, t, t*1.1)]\
                                                         + [Idle(10)] + p['pi2_1 + 0'])
                map_y = Waveform(name_y, p['pi2_1 + 90'] +[Idle(10)] + [SinRamp( tpump , f1, np.pi, t, t*1.1)]\
                                                        +  [Idle(10)] +p['pi2_1 + 90'])
                
                self.waves.append(map_x)
                self.waves.append(map_y)
                self.main_seq.append(*self.waves[-2:],wait=True)
                
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('HHahnscan.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1 = self.pi2_1
        tpump = self.tpump
        sequence = []
        
        for t in tau:
            sub = [ (['awgTrigger'], 100 ),
                    ([], tpump + pi2_1*2 + 2100),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)

        return sequence
            
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','tpump'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-65),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     ),       
                             HGroup(Item('tpump', width=-40),
                                    Item('pi2_1', width=20), 
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
                       title='Hartmann-hahn_amp_scan_oneway_fsweep',
                       )                                   

class PulsePol(Pulsed):      # M. Plenio sequence              

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
   
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=800., high=1e8, value=500., desc='tau_evo begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau_evo end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau_evo [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
           
    pulse_num = Range(low=1, high=1000, value=1, desc='How many times repeat the sequence', label='repetitions', mode='text', auto_set=False, enter_set=True)
    reload = True
    
    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]                          #pi/2  X
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 + 180']   = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]                      #pi/2 -X
            p['pi2_1 + 270']   = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]                    #pi/2  Y
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            p['pi2_1 - 180']   = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]                  #pi/2  -Y
            p['pi2_1 - 270']   = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*2 ,self.amp)]
            
            p['pi_1_x + 0']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]                           #X
            p['pi_1_x + 90']   = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_y + 0']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]                     #Y
            p['pi_1_y + 90']   = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            p['pi_-x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]                       #pi -X
            p['pi_-x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
                        
            # Waveforms
            sub_seq = []

            main_seq = Sequence('PulsePol.SEQ')
            waves = []

            for i, t in enumerate(self.tau):
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)

                # signal integration 
                t_tau = t*1.2

                max_threads = 10

                for k in range(self.pulse_num):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % k
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % k
                    map_x_1 = Waveform(x_name,   p['pi2_1 - 0']+[Idle(t_tau)] + p['pi_-x + 0']+ [Idle(t_tau)]+p['pi2_1 - 0']+ p['pi2_1 + 0']+[Idle(t_tau)]+ p['pi_1_y + 0']+ [Idle(t_tau)]+p['pi2_1 + 0']\
                                                 +p['pi2_1 - 0']+[Idle(t_tau)] + p['pi_-x + 0']+ [Idle(t_tau)]+p['pi2_1 - 0']+ p['pi2_1 + 0']+[Idle(t_tau)]+ p['pi_1_y + 0']+ [Idle(t_tau)]+p['pi2_1 + 0']        
                                       )
                    map_x_1.join()                            
                    map_y_1 = Waveform(y_name,   p['pi2_1 - 90']+[Idle(t_tau)] +p['pi_-x + 90']+[Idle(t_tau)]+p['pi2_1 - 90']+p['pi2_1 + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi2_1 + 90']\
                                                 +p['pi2_1 - 90']+[Idle(t_tau)] +p['pi_-x + 90']+[Idle(t_tau)]+p['pi2_1 - 90']+p['pi2_1 + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi2_1 + 90']
                                       )
                    map_y_1.join()
                    waves.append(map_x_1)
                    waves.append(map_y_1)
                    sub_seq.append(map_x_1, map_y_1)
                    main_seq.append(*waves[-2:],wait=True)

            AWG.upload(waves, 1)
            AWG.upload(main_seq, 1)
            AWG.tell('*WAI')
            AWG.load('PulsePol.SEQ')
            
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )

# class PulsePol(Pulsed):  # M. Plenio sequence
#
#     freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]',
#                         mode='text', auto_set=False, enter_set=True)
#     power = Range(low=-100., high=25., value=12, desc='power [dBm]', label='power [dBm]', mode='text',
#                   auto_set=False, enter_set=True)
#     freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text',
#                  auto_set=False, enter_set=True)
#
#     amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text',
#                 auto_set=False, enter_set=True)
#     vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text',
#                 auto_set=False, enter_set=True)
#
#     pi2_1 = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi',
#                   mode='text', auto_set=False, enter_set=True)
#     pi_1 = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text',
#                  auto_set=False, enter_set=True)
#
#     tau_begin = Range(low=800., high=1e8, value=500., desc='tau_evo begin [ns]', label='tau begin [ns]',
#                       mode='text', auto_set=False, enter_set=True)
#     tau_end = Range(low=1., high=1e8, value=4000., desc='tau_evo end [ns]', label='tau end [ns]', mode='text',
#                     auto_set=False, enter_set=True)
#     tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau_evo [ns]', label='delta tau [ns]', mode='text',
#                       auto_set=False, enter_set=True)
#
#     pulse_num = Range(low=1, high=1000, value=1, desc='How many times repeat the sequence', label='repetitions',
#                       mode='text', auto_set=False, enter_set=True)
#     reload = True
#
#     def prepare_awg(self):
#         sampling = 1.2e9
#         if self.reload:
#             AWG.stop()
#             AWG.set_output(0b0000)
#
#             pi2_1 = int(self.pi2_1 * sampling / 1.0e9)
#             pi_1 = int(self.pi_1 * sampling / 1.0e9)
#
#             zero = Idle(1)
#             mod = Idle(0)
#
#             # Pulses
#             p = {}
#
#             p['pi2_1 + 0'] = [Sin(pi2_1, (self.freq - self.freq_center) / sampling, 0, self.amp)]  # pi/2  X
#             p['pi2_1 + 90'] = [Sin(pi2_1, (self.freq - self.freq_center) / sampling, np.pi / 2, self.amp)]
#
#             p['pi2_1 + 180'] = [Sin(pi2_1, (self.freq - self.freq_center) / sampling, np.pi, self.amp)]  # pi/2 -X
#             p['pi2_1 + 270'] = [Sin(pi2_1, (self.freq - self.freq_center) / sampling, np.pi * 3 / 2, self.amp)]
#
#             p['pi2_1 - 0'] = [Sin(pi2_1, (self.freq - self.freq_center) / sampling, np.pi / 2, self.amp)]  # pi/2  Y
#             p['pi2_1 - 90'] = [Sin(pi2_1, (self.freq - self.freq_center) / sampling, np.pi, self.amp)]
#
#             p['pi2_1 - 180'] = [
#                 Sin(pi2_1, (self.freq - self.freq_center) / sampling, np.pi * 3 / 2, self.amp)]  # pi/2  -Y
#             p['pi2_1 - 270'] = [Sin(pi2_1, (self.freq - self.freq_center) / sampling, np.pi * 2, self.amp)]
#
#             p['pi_1_x + 0'] = [Sin(pi_1, (self.freq - self.freq_center) / sampling, 0, self.amp)]  # X
#             p['pi_1_x + 90'] = [Sin(pi_1, (self.freq - self.freq_center) / sampling, np.pi / 2, self.amp)]
#
#             p['pi_1_y + 0'] = [Sin(pi_1, (self.freq - self.freq_center) / sampling, np.pi / 2, self.amp)]  # Y
#             p['pi_1_y + 90'] = [Sin(pi_1, (self.freq - self.freq_center) / sampling, np.pi, self.amp)]
#
#             p['pi_-x + 0'] = [Sin(pi_1, (self.freq - self.freq_center) / sampling, np.pi, self.amp)]  # pi -X
#             p['pi_-x + 90'] = [Sin(pi_1, (self.freq - self.freq_center) / sampling, np.pi * 3 / 2, self.amp)]
#
#             # Waveforms
#             self.waves = []
#             sub_seq = []
#
#             self.main_seq = Sequence('PulsePol.SEQ')
#
#             for i, t in enumerate(self.tau):
#                 name = 'SQH_12_%04i.SEQ' % i
#                 sub_seq = Sequence(name)
#
#                 # signal integration
#                 t_tau = t * 1.2
#
#                 max_threads = 10
#
#                 for k in range(self.pulse_num):
#                     x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % k
#                     y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % k
#                     map_x_1 = Waveform(x_name, p['pi2_1 - 0'] + [Idle(t_tau)] + p['pi_-x + 0'] + [Idle(t_tau)] + p[
#                         'pi2_1 - 0'] + p['pi2_1 + 0'] + [Idle(t_tau)] + p['pi_1_y + 0'] + [Idle(t_tau)] + p[
#                                            'pi2_1 + 0'] \
#                                        + p['pi2_1 - 0'] + [Idle(t_tau)] + p['pi_-x + 0'] + [Idle(t_tau)] + p[
#                                            'pi2_1 - 0'] + p['pi2_1 + 0'] + [Idle(t_tau)] + p['pi_1_y + 0'] + [
#                                            Idle(t_tau)] + p['pi2_1 + 0']
#                                        )
#
#                     map_y_1 = Waveform(y_name,
#                                        p['pi2_1 - 90'] + [Idle(t_tau)] + p['pi_-x + 90'] + [Idle(t_tau)] + p[
#                                            'pi2_1 - 90'] + p['pi2_1 + 90'] + [Idle(t_tau)] + p['pi_1_y + 90'] + [
#                                            Idle(t_tau)] + p['pi2_1 + 90'] \
#                                        + p['pi2_1 - 90'] + [Idle(t_tau)] + p['pi_-x + 90'] + [Idle(t_tau)] + p[
#                                            'pi2_1 - 90'] + p['pi2_1 + 90'] + [Idle(t_tau)] + p['pi_1_y + 90'] + [
#                                            Idle(t_tau)] + p['pi2_1 + 90']
#                                        )
#                     self.waves.append(map_x_1)
#                     self.waves.append(map_y_1)
#                     sub_seq.append(map_x_1, map_y_1)
#                     self.main_seq.append(*self.waves[-2:], wait=True)
#
#             AWG.upload(self.waves)
#             AWG.upload(self.main_seq)
#             AWG.tell('*WAI')
#             AWG.load('PulsePol.SEQ')
#
#         AWG.set_vpp(0.6)
#         AWG.set_sample(sampling / 1.0e9)
#         AWG.set_mode('S')
#         AWG.set_output(0b0011)


    def load(self):
        self.reload = True
        
        c=[] 
        begin = self.tau_begin
        end   = self.tau_end            
        self.tau=np.arange(begin, end+self.tau_delta, self.tau_delta) 
        self.freq_center=self.freq-0.1e+9
        self.prepare_awg()
        self.reload = False       
    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.previous_time = 0
            #self.previous_count_data = FC.GetData()
            self.prepare_awg()
            MW.setFrequency(self.freq_center)
            MW.setPower(self.power)
            AWG.run()
            time.sleep(30.0)
            FC.Start()
            time.sleep(0.1)
            PG.Sequence(self.sequence, loop=True)

            start_time = time.time()

            while True:
               self.thread.stop_request.wait(5.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               self.elapsed_time = self.previous_elapsed_time + time.time() - start_time
               self.run_time += self.elapsed_time
               runtime, cycles = FC.GetState()
               sweeps = cycles / FC.GetData().shape[0]
               self.elapsed_sweeps = self.previous_sweeps + sweeps
               self.progress = int( 100 * self.elapsed_sweeps / self.sweeps ) 
               self.count_data = self.old_count_data + FC.GetData()
               if self.elapsed_sweeps > self.sweeps:
                  break

            FC.Halt()
            time.sleep(0.1)
            MW.Off()
            time.sleep(0.2)
            PG.High(['laser', 'mw'])
            time.sleep(0.2)
            AWG.stop()
            time.sleep(1)
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  
            
        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'    
            
    #def _get_sequence_points(self):
        #return 2 * len(self.tau) 
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        sequence = []
        
        for i, t in enumerate(tau):
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , (8*t + 8 * pi2_1 + 4*pi_1)*self.pulse_num + 4000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
            
        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','pi_1','pulse_num']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority')
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40) 
                                     ),                         
                              HGroup(
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40)
                                     ),                        
                              HGroup(Item('laser', width=30),
                                     Item('wait', width=30),
                                     Item('bin_width', width= 30, enabled_when='state != "run"'),
                                     Item('record_length', width= 30, enabled_when='state != "run"')
                                     ),
                                     
                              HGroup(Item('tau_begin', width=-40),
                                     Item('tau_end', width=-40),
                                     Item('tau_delta', width=-40)
                                    ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30)
                                     ),
                                                                         
                              ),
                       title='PulsePol',
                       )     
                       
class Correlation_Spec_XY8_phase(Pulsed):                   

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
   
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=800., high=1e8, value=500., desc='tau_evo begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau_evo end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau_evo [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    interval= Range(low=1., high=1e8, value=4000., desc='interval [ns]', label='interval [ns]', mode='text', auto_set=False, enter_set=True)
    M = Range(low=0, high=1000, value=0, desc='number of intervals', label='number of intervals', mode='text', auto_set=False, enter_set=True)
    
    tau_inte = Range(low=1., high=1e5, value=30., desc='tau for signal integration [ns]', label='tau integrate[ns]', mode='text', auto_set=False, enter_set=True)
    phase_flag = Bool(False, label='phase_flag')
    
    pulse_num = Range(low=1, high=1000, value=1, desc='repetetion of XY8 pulses', label='repetetions', mode='text', auto_set=False, enter_set=True)
    reload = True
    
    #def _init_(self):
        #super(XY8, self).__init__()

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]                            #pi/2  X
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 + 180']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]                      #pi/2 -X
            p['pi2_1 + 270']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]                      #pi/2  Y
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            p['pi2_1 - 180']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]                  #pi/2  -Y
            p['pi2_1 - 270']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*2 ,self.amp)]
            
            p['pi_1_x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]                             #X
            p['pi_1_x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]                       #Y
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            # if self.phase_flag:
                # p['read + 0'] = p['pi2_1 + 0']
                # p['read + 90'] = p['pi2_1 + 90']
                
                # p['read + 90'] = p['pi2_1 - 0']
                # p['read + 180'] = p['pi2_1 - 90']
            # else:
                # p['read + 0'] = p['pi2_1 - 0']
                # p['read + 90'] = p['pi2_1 - 90']
                
                # p['read + 90'] = p['pi2_1 + 0']
                # p['read + 180'] = p['pi2_1 + 90']
            
            # Waveforms
            self.waves = []
            sub_seq = []

            self.main_seq = Sequence('Correlation_Spec.SEQ')
            
            sup_x_ref = Waveform('Sup_x_ref', p['pi2_1 + 0'])
            sup_y_ref = Waveform('Sup_y_ref', p['pi2_1 + 90'])
            t_sup_ref = sup_x_ref.stub
            
            sup_x = Waveform('Sup_x', [Idle(t_sup_ref)]+p['pi2_1 + 0'])
            sup_y = Waveform('Sup_y', [Idle(t_sup_ref)]+p['pi2_1 + 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            
            for i, t in enumerate(self.tau):
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                
                
                # signal integration 
                t_tau = self.tau_inte * 1.2 * 2
                sub_seq.append(sup_x,sup_y)
                
                t_0 = sup_x.duration
                t_1 = self.tau_inte * 1.2
                
                for k in range(self.pulse_num/2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % k
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % k
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'], t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'], t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                for k in range(self.pulse_num%2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'], t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'], t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub    
                    
                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                name_x = 'Read_x_%04i.WFM' % i
                name_y = 'Read_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 - 0'], t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 - 90'], t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x, ref_y)
                
                
                # evolution
                t_evo = 1.2 * t
                t_2 = t_evo - ref_x.stub - 256
                repeat_1 = int(t_2 / 256)
                 
                t_1 = int(t_2 % 256)
                t_0 += ref_x.duration + repeat_1 * 256
                
                if repeat_1 > 0:
                    sub_seq.append(evo,evo,repeat=repeat_1)
                    
                
                # signal integration again
                x_name = 'super_x_%03i' % i
                y_name = 'super_y_%03i' % i
                
                sup_x_2 = Waveform(x_name, [Idle(256 + t_1)]+p['pi2_1 + 0'], t_0)
                sup_y_2 = Waveform(y_name, [Idle(256 + t_1)]+p['pi2_1 + 90'], t_0)
                self.waves.append(sup_x_2)
                self.waves.append(sup_y_2)
                sub_seq.append(sup_x_2,sup_y_2)

                t_0 += sup_x_2.duration
                t_1 = self.tau_inte * 1.2 - sup_x_2.stub
                for k in range(self.pulse_num/2):
                    x_name = 'X_XY4_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    y_name = 'Y_XY4_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    
                    map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                for k in range(self.pulse_num%2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % self.pulse_num*2
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % self.pulse_num*2
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub        

                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                name_x = 'Read1_x_%04i.WFM' % i
                name_y = 'Read1_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 - 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 - 90'],t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                
                
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)


            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            #time.sleep(70)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('Correlation_Spec.SEQ')
            
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def load(self):
        self.reload = True
        
        c=[] 
        begin = self.tau_begin
        end   = self.tau_end
        d=end-begin

        for i in range(self.M+1):        
            a = np.arange(begin, end+self.tau_delta, self.tau_delta) 
            begin=end+self.interval
            end=begin+d
            c = np.concatenate((c, a), axis=0)
            
        self.tau=c    
        self.freq_center=self.freq-0.1e+9
        self.prepare_awg()
        self.reload = False       
    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.previous_time = 0
            #self.previous_count_data = FC.GetData()
            self.prepare_awg()
            MW.setFrequency(self.freq_center)
            MW.setPower(self.power)
            AWG.run()
            time.sleep(30.0)
            FC.Start()
            time.sleep(0.1)
            PG.Sequence(self.sequence, loop=True)

            start_time = time.time()

            while True:
               self.thread.stop_request.wait(5.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               self.elapsed_time = self.previous_elapsed_time + time.time() - start_time
               self.run_time += self.elapsed_time
               runtime, cycles = FC.GetState()
               sweeps = cycles / FC.GetData().shape[0]
               self.elapsed_sweeps = self.previous_sweeps + sweeps
               self.progress = int( 100 * self.elapsed_sweeps / self.sweeps ) 
               self.count_data = self.old_count_data + FC.GetData()
               if self.elapsed_sweeps > self.sweeps:
                  break

            FC.Halt()
            time.sleep(0.1)
            MW.Off()
            time.sleep(0.2)
            PG.High(['laser', 'mw'])
            time.sleep(0.2)
            AWG.stop()
            time.sleep(1)
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  
            
        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'    
            
    #def _get_sequence_points(self):
        #return 2 * len(self.tau) 
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        tau_1 = self.tau_inte
        sequence = []
        
        for i, t in enumerate(tau):
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , t + 2 * tau_1 * 16 * self.pulse_num + 4 * pi2_1 + 2*8*self.pulse_num*pi_1 + 4000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
            
        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','pi_1','tau_inte','pulse_num','phase_flag']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                     ),                         
                              HGroup(Item('phase_flag', width = -40),
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40),
                                     Item('tau_inte', width = -40),
                                     ),                        
                              HGroup(Item('laser', width=30),
                                     Item('wait', width=30),
                                     Item('bin_width', width= 30, enabled_when='state != "run"'),
                                     Item('record_length', width= 30, enabled_when='state != "run"'),
                                     ),
                                     
                              HGroup(Item('tau_begin', width=-40),
                                     Item('tau_end', width=-40),
                                     Item('tau_delta', width=-40),
                                     Item('interval'),
                                     Item('M')
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30),
                                     ),
                                                                         
                              ),
                       title='Correlation_Spec_XY8_phase',
                       )     
                       
class Correlation_Spec_XY16_phase(Pulsed):                   

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
   
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=800., high=1e8, value=500., desc='tau_evo begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau_evo end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau_evo [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    
    tau_inte = Range(low=1., high=1e5, value=30., desc='tau for signal integration [ns]', label='tau integrate[ns]', mode='text', auto_set=False, enter_set=True)
    phase_flag = Bool(False, label='phase_flag')
    
    pulse_num = Range(low=1, high=1000, value=1, desc='repetetion of XY8 pulses', label='repetetions', mode='text', auto_set=False, enter_set=True)
    reload = True
    
    #def _init_(self):
        #super(XY8, self).__init__()

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 + 180']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            p['pi2_1 + 270']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            p['pi2_1 - 180']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            p['pi2_1 - 270']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*2 ,self.amp)]
            
            p['pi_1_x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]                 #X
            p['pi_1_x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]           #Y
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            p['pi_1_x - 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]             #-X
            p['pi_1_x - 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            
            p['pi_1_y - 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]         #-Y
            p['pi_1_y - 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi*2 ,self.amp)]
            
            if self.phase_flag:
                p['read + 0'] = p['pi2_1 + 0']
                p['read + 90'] = p['pi2_1 + 90']
                
                p['read + 90'] = p['pi2_1 - 0']
                p['read + 180'] = p['pi2_1 - 90']
            else:
                p['read + 0'] = p['pi2_1 - 0']
                p['read + 90'] = p['pi2_1 - 90']
                
                p['read + 90'] = p['pi2_1 + 0']
                p['read + 180'] = p['pi2_1 + 90']
            
            # Waveforms
            self.waves = []
            sub_seq = []

            self.main_seq = Sequence('Correlation_Spec.SEQ')
            
            sup_x_ref = Waveform('Sup_x_ref', p['pi2_1 + 0'])
            sup_y_ref = Waveform('Sup_y_ref', p['pi2_1 + 90'])
            t_sup_ref = sup_x_ref.stub
            
            sup_x = Waveform('Sup_x', [Idle(t_sup_ref)]+p['pi2_1 + 0'])
            sup_y = Waveform('Sup_y', [Idle(t_sup_ref)]+p['pi2_1 + 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            
            for i, t in enumerate(self.tau):
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                
                
                # signal integration 
                t_tau = self.tau_inte * 1.2 * 2
                sub_seq.append(sup_x,sup_y)
                
                t_0 = sup_x.duration
                t_1 = self.tau_inte * 1.2
                
                for k in range(self.pulse_num):
                    x_name = 'X_XY16_%03i' % i + '_%03i.WFM' % k
                    y_name = 'Y_XY16_%03i' % i + '_%03i.WFM' % k
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x - 0']+[Idle(t_tau)]+p['pi_1_y - 0']+[Idle(t_tau)]+p['pi_1_x - 0']+[Idle(t_tau)]+p['pi_1_y - 0']\
                                                +[Idle(t_tau)]+p['pi_1_y - 0']+[Idle(t_tau)]+p['pi_1_x - 0']+[Idle(t_tau)]+p['pi_1_y - 0']+[Idle(t_tau)]+p['pi_1_x - 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x - 90']+[Idle(t_tau)]+p['pi_1_y - 90']+[Idle(t_tau)]+p['pi_1_x - 90']+[Idle(t_tau)]+p['pi_1_y - 90']\
                                                +[Idle(t_tau)]+p['pi_1_y - 90']+[Idle(t_tau)]+p['pi_1_x - 90']+[Idle(t_tau)]+p['pi_1_y - 90']+[Idle(t_tau)]+p['pi_1_x - 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
             
                    
                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                name_x = 'Read_x_%04i.WFM' % i
                name_y = 'Read_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['read + 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['read + 90'],t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                
                
                # evolution
                t_evo = 1.2 * t
                t_2 = t_evo - ref_x.stub - 256
                repeat_1 = int(t_2 / 256)
                 
                t_1 = int(t_2 % 256)
                t_0 += ref_x.duration + repeat_1 * 256
                
                if repeat_1 > 0:
                    sub_seq.append(evo,evo,repeat=repeat_1)
                    
                
                # signal integration again
                x_name = 'super_x_%03i' % i
                y_name = 'super_y_%03i' % i
                
                sup_x_2 = Waveform(x_name, [Idle(256 + t_1)]+p['read + 90'], t_0)
                sup_y_2 = Waveform(y_name, [Idle(256 + t_1)]+p['read + 180'], t_0)
                self.waves.append(sup_x_2)
                self.waves.append(sup_y_2)
                sub_seq.append(sup_x_2,sup_y_2)

                t_0 += sup_x_2.duration
                t_1 = self.tau_inte * 1.2 - sup_x_2.stub
                for k in range(self.pulse_num):
                    x_name = 'BX_XY16_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    y_name = 'BY_XY16_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    
                    map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x - 0']+[Idle(t_tau)]+p['pi_1_y - 0']+[Idle(t_tau)]+p['pi_1_x - 0']+[Idle(t_tau)]+p['pi_1_y - 0']\
                                                +[Idle(t_tau)]+p['pi_1_y - 0']+[Idle(t_tau)]+p['pi_1_x - 0']+[Idle(t_tau)]+p['pi_1_y - 0']+[Idle(t_tau)]+p['pi_1_x - 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x - 90']+[Idle(t_tau)]+p['pi_1_y - 90']+[Idle(t_tau)]+p['pi_1_x - 90']+[Idle(t_tau)]+p['pi_1_y - 90']\
                                                +[Idle(t_tau)]+p['pi_1_y - 90']+[Idle(t_tau)]+p['pi_1_x - 90']+[Idle(t_tau)]+p['pi_1_y - 90']+[Idle(t_tau)]+p['pi_1_x - 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                name_x = 'Read1_x_%04i.WFM' % i
                name_y = 'Read1_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['read + 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['read + 90'],t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                
                
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)


            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            #time.sleep(70)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('Correlation_Spec.SEQ')
            
        AWG.set_vpp(self.vpp)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
             
    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.previous_time = 0
            #self.previous_count_data = FC.GetData()
            self.prepare_awg()
            MW.setFrequency(self.freq_center)
            MW.setPower(self.power)
            AWG.run()
            time.sleep(30.0)
            FC.Start()
            time.sleep(0.1)
            PG.Sequence(self.sequence, loop=True)

            start_time = time.time()

            while True:
               self.thread.stop_request.wait(5.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               self.elapsed_time = self.previous_elapsed_time + time.time() - start_time
               self.run_time += self.elapsed_time
               runtime, cycles = FC.GetState()
               sweeps = cycles / FC.GetData().shape[0]
               self.elapsed_sweeps = self.previous_sweeps + sweeps
               self.progress = int( 100 * self.elapsed_sweeps / self.sweeps ) 
               self.count_data = self.old_count_data + FC.GetData()
               if self.elapsed_sweeps > self.sweeps:
                  break

            FC.Halt()
            MW.Off()
            PG.High(['laser', 'mw'])
            AWG.stop()
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  
            
        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'    
            
    #def _get_sequence_points(self):
        #return 2 * len(self.tau) 
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        tau_1 = self.tau_inte
        sequence = []
        
        for i, t in enumerate(tau):
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , t + tau_1 * 2 * 16 * self.pulse_num * 2 + 4 * pi2_1 + 32*self.pulse_num*pi_1 + 4000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
            
        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','pi_1','tau_inte','pulse_num','phase_flag']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                     ),                         
                              HGroup(Item('phase_flag', width = -40),
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40),
                                     Item('tau_inte', width = -40),
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
                       title='Correlation_Spec_XY16_phase',
                       )                       
                       
class Correlation_Spec_XY8(Pulsed):


    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    rabi_contrast=Range(low=1., high=100, value=30.0, desc='Rabi contrast [%]', label='contrast', mode='text', auto_set=False, enter_set=True)
   
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=800., high=1e8, value=500., desc='tau_evo begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau_evo end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau_evo [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    interval= Range(low=1., high=1e8, value=4000., desc='interval [ns]', label='interval [ns]', mode='text', auto_set=False, enter_set=True)
    M = Range(low=0, high=1000, value=0, desc='number of intervals', label='number of intervals', mode='text', auto_set=False, enter_set=True)
    
    tau_inte = Range(low=1., high=1e5, value=30., desc='tau for signal integration [ns]', label='tau integrate[ns]', mode='text', auto_set=False, enter_set=True)
        
    pulse_num = Range(low=1, high=1000, value=1, desc='repetetion of XY8 pulses', label='repetetions', mode='text', auto_set=False, enter_set=True)
    reload = True
    
    #def _init_(self):
        #super(XY8, self).__init__()

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 + 180']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            p['pi2_1 + 270']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            p['pi2_1 - 180']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            p['pi2_1 - 270']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*2 ,self.amp)]
            
            p['pi_1_x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1_x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            # if self.phase_flag:
                # p['read + 0'] = p['pi2_1 + 0']
                # p['read + 90'] = p['pi2_1 + 90']
                
                # p['read + 90'] = p['pi2_1 - 0']
                # p['read + 180'] = p['pi2_1 - 90']
            # else:
                # p['read + 0'] = p['pi2_1 - 0']
                # p['read + 90'] = p['pi2_1 - 90']
                
                # p['read + 90'] = p['pi2_1 + 0']
                # p['read + 180'] = p['pi2_1 + 90']
            
            # Waveforms
            self.waves = []
            sub_seq = []

            self.main_seq = Sequence('Correlation_Spec.SEQ')
            
            sup_x_ref = Waveform('Sup_x_ref', p['pi2_1 + 0'])
            sup_y_ref = Waveform('Sup_y_ref', p['pi2_1 + 90'])
            t_sup_ref = sup_x_ref.stub
            
            sup_x = Waveform('Sup_x', [Idle(t_sup_ref)]+p['pi2_1 + 0'])
            sup_y = Waveform('Sup_y', [Idle(t_sup_ref)]+p['pi2_1 + 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            
            for i, t in enumerate(self.tau):
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                
                
                # signal integration 
                t_tau = self.tau_inte * 1.2 * 2
                sub_seq.append(sup_x,sup_y)
                
                t_0 = sup_x.duration
                t_1 = self.tau_inte * 1.2
                
                for k in range(self.pulse_num/2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % k
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % k
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                for k in range(self.pulse_num%2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub    
                    
                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                name_x = 'Read_x_%04i.WFM' % i
                name_y = 'Read_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 - 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 - 90'],t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                
                
                # evolution
                t_evo = 1.2 * t
                t_2 = t_evo - ref_x.stub - 256
                repeat_1 = int(t_2 / 256)
                 
                t_1 = int(t_2 % 256)
                t_0 += ref_x.duration + repeat_1 * 256
                
                if repeat_1 > 0:
                    sub_seq.append(evo,evo,repeat=repeat_1)
                    
                
                # signal integration again
                x_name = 'super_x_%03i' % i
                y_name = 'super_y_%03i' % i
                
                sup_x_2 = Waveform(x_name, [Idle(256 + t_1)]+p['pi2_1 + 0'], t_0)
                sup_y_2 = Waveform(y_name, [Idle(256 + t_1)]+p['pi2_1 + 0'], t_0)
                self.waves.append(sup_x_2)
                self.waves.append(sup_y_2)
                sub_seq.append(sup_x_2,sup_y_2)

                t_0 += sup_x_2.duration
                t_1 = self.tau_inte * 1.2 - sup_x_2.stub
                for k in range(self.pulse_num/2):
                    x_name = 'X_XY4_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    y_name = 'Y_XY4_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    
                    map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                for k in range(self.pulse_num%2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % self.pulse_num*2
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % self.pulse_num*2
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub        

                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                name_x = 'Read1_x_%04i.WFM' % i
                name_y = 'Read1_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 - 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 - 90'],t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                
                
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)
                #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                name = 'BSQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                
                
                # signal integration 
                t_tau = self.tau_inte * 1.2 * 2
                sub_seq.append(sup_x,sup_y)
                
                t_0 = sup_x.duration
                t_1 = self.tau_inte * 1.2
                
                for k in range(self.pulse_num/2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % k
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % k
                    map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                for k in range(self.pulse_num%2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub    
                    
                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                name_x = 'Read_x_%04i.WFM' % i
                name_y = 'Read_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 - 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 - 90'],t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                
                
                # evolution
                t_evo = 1.2 * t
                t_2 = t_evo - ref_x.stub - 256
                repeat_1 = int(t_2 / 256)
                 
                t_1 = int(t_2 % 256)
                t_0 += ref_x.duration + repeat_1 * 256
                
                if repeat_1 > 0:
                    sub_seq.append(evo,evo,repeat=repeat_1)
                    
                
                # signal integration again
                x_name = 'super_x_%03i' % i
                y_name = 'super_y_%03i' % i
                
                sup_x_2 = Waveform(x_name, [Idle(256 + t_1)]+p['pi2_1 + 0'], t_0)
                sup_y_2 = Waveform(y_name, [Idle(256 + t_1)]+p['pi2_1 + 0'], t_0)
                self.waves.append(sup_x_2)
                self.waves.append(sup_y_2)
                sub_seq.append(sup_x_2,sup_y_2)

                t_0 += sup_x_2.duration
                t_1 = self.tau_inte * 1.2 - sup_x_2.stub
                for k in range(self.pulse_num/2):
                    x_name = 'X_XY4_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    y_name = 'Y_XY4_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    
                    map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                for k in range(self.pulse_num%2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % self.pulse_num*2
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % self.pulse_num*2
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub        

                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                name_x = 'Read1_x_%04i.WFM' % i
                name_y = 'Read1_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 - 180'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 - 270'],t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                
                
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)
                
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            #time.sleep(70)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('Correlation_Spec.SEQ')
            
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
        
    def load(self):
        self.reload = True
        
        c=[] 
        begin = self.tau_begin
        end   = self.tau_end
        d=end-begin

        for i in range(self.M+1):        
            a = np.arange(begin, end+self.tau_delta, self.tau_delta) 
            begin=end+self.interval
            end=begin+d
            c = np.concatenate((c, a), axis=0)
            
        self.tau=c    
        self.freq_center=self.freq-0.1e+9
        self.prepare_awg()
        self.reload = False       
    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.previous_time = 0
            #self.previous_count_data = FC.GetData()
            self.prepare_awg()
            MW.setFrequency(self.freq_center)
            MW.setPower(self.power)
            AWG.run()
            time.sleep(30.0)
            FC.Start()
            time.sleep(0.1)
            PG.Sequence(self.sequence, loop=True)

            start_time = time.time()

            while True:
               self.thread.stop_request.wait(5.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               self.elapsed_time = self.previous_elapsed_time + time.time() - start_time
               self.run_time += self.elapsed_time
               runtime, cycles = FC.GetState()
               sweeps = cycles / FC.GetData().shape[0]
               self.elapsed_sweeps = self.previous_sweeps + sweeps
               self.progress = int( 100 * self.elapsed_sweeps / self.sweeps ) 
               self.count_data = self.old_count_data + FC.GetData()
               if self.elapsed_sweeps > self.sweeps:
                  break

            FC.Halt()
            time.sleep(0.1)
            MW.Off()
            time.sleep(0.2)
            PG.High(['laser', 'mw'])
            time.sleep(0.2)
            AWG.stop()
            time.sleep(1)
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  
            
        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'    
            
    def _get_sequence_points(self):
        return 2 * len(self.tau) 
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        tau_1 = self.tau_inte
        sequence = []
        
        for i, t in enumerate(tau):
                        
            sub = [ (['awgTrigger'], 100 ),
                    ([], t + 2 * tau_1 * 16 * self.pulse_num + 4 * pi2_1 + 2*8*self.pulse_num*pi_1 + 4000),
                    (['laser', 'trigger' ], laser ),
                    ([], wait ),
                    
                    (['awgTrigger'], 100 ),
                    ([], t + 2 * tau_1 * 16 * self.pulse_num + 4 * pi2_1 + 2*8*self.pulse_num*pi_1 + 4000),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)
            
        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','pi_1','tau_inte','pulse_num','phase_flag']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                     ),                         
                              HGroup(
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40),
                                     Item('tau_inte', width = -40),
                                     Item('rabi_contrast', width = -40)
                                     ),                        
                              HGroup(Item('laser', width=30),
                                     Item('wait', width=30),
                                     Item('bin_width', width= 30, enabled_when='state != "run"'),
                                     Item('record_length', width= 30, enabled_when='state != "run"'),
                                     ),
                                     
                              HGroup(Item('tau_begin', width=-40),
                                     Item('tau_end', width=-40),
                                     Item('tau_delta', width=-40),
                                     Item('interval'),
                                     Item('M')
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30),
                                     ),
                                                                         
                              ),
                       title='Correlation_Spec_XY8',
                       )                          
                       
class Spec_XY8_phase_check(Pulsed):                   

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
   
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=800., high=1e8, value=500., desc='tau_evo begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau_evo end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau_evo [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    
    tau_inte = Range(low=1., high=1e5, value=30., desc='tau for signal integration [ns]', label='tau integrate[ns]', mode='text', auto_set=False, enter_set=True)
    #tau_ref_flag = Bool(False, label='tau_ref_flag')
    
    pulse_num = Range(low=1, high=1000, value=1, desc='repetetion of XY8 pulses', label='repetetions', mode='text', auto_set=False, enter_set=True)
    reload = True
    
    #def _init_(self):
        #super(XY8, self).__init__()

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            p['pi_1_x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1_x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            # Waveforms
            self.waves = []
            sub_seq_1 = []
            sub_seq_2 = []
            self.main_seq = Sequence('Correlation_Spec.SEQ')
            
            sup_x_ref = Waveform('Sup_x_ref', p['pi2_1 + 0'])
            sup_y_ref = Waveform('Sup_y_ref', p['pi2_1 + 90'])
            t_sup_ref = sup_x_ref.stub
            
            sup_x = Waveform('Sup_x', [Idle(t_sup_ref)]+p['pi2_1 + 0'])
            sup_y = Waveform('Sup_y', [Idle(t_sup_ref)]+p['pi2_1 + 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            
            for i, t in enumerate(self.tau):
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq_1 = Sequence(name)
                
                name = 'BSQH_12_%04i.SEQ' % i
                sub_seq_2 = Sequence(name)
                
                # signal integration 
                t_tau = self.tau_inte * 1.2 * 2
                sub_seq_1.append(sup_x,sup_y)
                sub_seq_2.append(sup_x,sup_y)
                t_0 = sup_x.duration
                t_1 = self.tau_inte * 1.2
                t_evo = 1.2 * t
                for k in range(self.pulse_num):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % k
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % k
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq_1.append(map_x_1,map_y_1)
                    sub_seq_2.append(map_x_1,map_y_1)
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                name_x = 'Read_x_%04i.WFM' % i
                name_y = 'Read_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 - 0'] + [Idle(t_evo)],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 - 90'] + [Idle(t_evo)],t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq_1.append(ref_x,ref_y)
                
                AWG.upload(sub_seq_1)
                
                self.main_seq.append(sub_seq_1, wait=True)
                
                name_x = 'BRead_x_%04i.WFM' % i
                name_y = 'BRead_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 + 0'] + [Idle(t_evo)],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 + 90'] + [Idle(t_evo)],t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq_2.append(ref_x,ref_y)
                
                AWG.upload(sub_seq_2)
                
                self.main_seq.append(sub_seq_2, wait=True)
                
                

            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            #time.sleep(70)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('Correlation_Spec.SEQ')
            
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
             
    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.previous_time = 0
            #self.previous_count_data = FC.GetData()
            self.prepare_awg()
            MW.setFrequency(self.freq_center)
            MW.setPower(self.power)
            AWG.run()
            time.sleep(30.0)
            FC.Start()
            time.sleep(0.1)
            PG.Sequence(self.sequence, loop=True)

            start_time = time.time()

            while True:
               self.thread.stop_request.wait(5.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               self.elapsed_time = self.previous_elapsed_time + time.time() - start_time
               self.run_time += self.elapsed_time
               runtime, cycles = FC.GetState()
               sweeps = cycles / FC.GetData().shape[0]
               self.elapsed_sweeps = self.previous_sweeps + sweeps
               self.progress = int( 100 * self.elapsed_sweeps / self.sweeps ) 
               self.count_data = self.old_count_data + FC.GetData()
               if self.elapsed_sweeps > self.sweeps:
                  break

            FC.Halt()
            MW.Off()
            PG.High(['laser', 'mw'])
            AWG.stop()
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  
            
        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            
            self.state = 'error'    
    def _get_sequence_points(self):
        return 2 * len(self.tau)  
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        tau_1 = self.tau_inte
        sequence = []
        
        for i, t in enumerate(tau):
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , t + tau_1 * 2 * 8 * self.pulse_num  + 2 * pi2_1 + 8*self.pulse_num*pi_1 + 4000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
            
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , t + tau_1 * 2 * 8 * self.pulse_num  + 2 * pi2_1 + 8*self.pulse_num*pi_1 + 4000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','pi_1','tau_inte','pulse_num']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                     ),                         
                              HGroup(
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40),
                                     Item('tau_inte', width = -40),
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
                       title='Spec_XY8_phase_check',
                       )    
                       


class Correlation_Spec_XY4(Pulsed):                   

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
   
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=800., high=1e8, value=500., desc='tau_evo begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau_evo end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau_evo [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    
    tau_inte = Range(low=1., high=1e5, value=30., desc='tau for signal integration [ns]', label='tau integrate[ns]', mode='text', auto_set=False, enter_set=True)
    #tau_ref_flag = Bool(False, label='tau_ref_flag')
    
    pulse_num = Range(low=1, high=1000, value=1, desc='repetetion of XY8 pulses', label='repetetions', mode='text', auto_set=False, enter_set=True)
    reload = True
    
    #def _init_(self):
        #super(XY8, self).__init__()

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            p['pi_1_x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1_x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('Correlation_Spec.SEQ')
            
            sup_x_ref = Waveform('Sup_x_ref', p['pi2_1 + 0'])
            sup_y_ref = Waveform('Sup_y_ref', p['pi2_1 + 90'])
            t_sup_ref = sup_x_ref.stub
            
            sup_x = Waveform('Sup_x', [Idle(t_sup_ref)]+p['pi2_1 + 0'])
            sup_y = Waveform('Sup_y', [Idle(t_sup_ref)]+p['pi2_1 + 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            
            for i, t in enumerate(self.tau):
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                
                # signal integration 
                t_tau = self.tau_inte * 1.2 * 2
                sub_seq.append(sup_x,sup_y)
                t_0 = sup_x.duration
                t_1 = self.tau_inte * 1.2
                for k in range(self.pulse_num/2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % k
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % k
                    map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90'],t_0)   
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    

                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                name_x = 'Read_x_%04i.WFM' % i
                name_y = 'Read_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 - 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 - 90'],t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                
                # evolution
                t_evo = 1.2 * t
                t_2 = t_evo - ref_x.stub - 256
                repeat_1 = int(t_2 / 256)
                 
                t_1 = int(t_2 % 256)
                t_0 += ref_x.duration + repeat_1 * 256
                
                sub_seq.append(evo,evo,repeat=repeat_1)
                
                # signal integration again
                x_name = 'super_x_%03i' % i
                y_name = 'super_y_%03i' % i
                
                sup_x_2 = Waveform(x_name, [Idle(256 + t_1)]+p['pi2_1 + 0'], t_0)
                sup_y_2 = Waveform(y_name, [Idle(256 + t_1)]+p['pi2_1 + 90'], t_0)
                self.waves.append(sup_x_2)
                self.waves.append(sup_y_2)
                sub_seq.append(sup_x_2,sup_y_2)
                t_0 += sup_x_2.duration
                t_1 = self.tau_inte * 1.2 - sup_x_2.stub
                for k in range(self.pulse_num/2):
                    x_name = 'X_XY4_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    y_name = 'Y_XY4_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    
                    map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub

                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                name_x = 'Read1_x_%04i.WFM' % i
                name_y = 'Read1_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 - 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 - 90'],t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)

            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            #time.sleep(70)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('Correlation_Spec.SEQ')
            
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
             
    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.previous_time = 0
            #self.previous_count_data = FC.GetData()
            self.prepare_awg()
            MW.setFrequency(self.freq_center)
            MW.setPower(self.power)
            AWG.run()
            time.sleep(30.0)
            FC.Start()
            time.sleep(0.1)
            PG.Sequence(self.sequence, loop=True)

            start_time = time.time()

            while True:
               self.thread.stop_request.wait(5.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               self.elapsed_time = self.previous_elapsed_time + time.time() - start_time
               self.run_time += self.elapsed_time
               runtime, cycles = FC.GetState()
               sweeps = cycles / FC.GetData().shape[0]
               self.elapsed_sweeps = self.previous_sweeps + sweeps
               self.progress = int( 100 * self.elapsed_sweeps / self.sweeps ) 
               self.count_data = self.old_count_data + FC.GetData()
               if self.elapsed_sweeps > self.sweeps:
                  break

            FC.Halt()
            MW.Off()
            PG.High(['laser', 'mw'])
            AWG.stop()
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  
            
        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'    

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        tau_1 = self.tau_inte
        sequence = []
        
        for i, t in enumerate(tau):
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , t + tau_1 * 2 * 8 * self.pulse_num * 2 + 4 * pi2_1 + 16*self.pulse_num*pi_1 + 4000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','pi_1','tau_inte','pulse_num']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                     ),                         
                              HGroup(
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40),
                                     Item('tau_inte', width = -40),
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
                       title='Correlation_Spec_XY4',
                       )   

class Correlation_Spec_CPMG4(Pulsed):                   

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
   
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=800., high=1e8, value=500., desc='tau_evo begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau_evo end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau_evo [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    
    tau_inte = Range(low=1., high=1e5, value=30., desc='tau for signal integration [ns]', label='tau integrate[ns]', mode='text', auto_set=False, enter_set=True)
    #tau_ref_flag = Bool(False, label='tau_ref_flag')
    
    pulse_num = Range(low=1, high=1000, value=1, desc='repetetion of XY8 pulses', label='repetetions', mode='text', auto_set=False, enter_set=True)
    reload = True
    
    #def _init_(self):
        #super(XY8, self).__init__()

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            p['pi_1_x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1_x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('Correlation_Spec.SEQ')
            
            sup_x_ref = Waveform('Sup_x_ref', p['pi2_1 + 0'])
            sup_y_ref = Waveform('Sup_y_ref', p['pi2_1 + 90'])
            t_sup_ref = sup_x_ref.stub
            
            sup_x = Waveform('Sup_x', [Idle(t_sup_ref)]+p['pi2_1 + 0'])
            sup_y = Waveform('Sup_y', [Idle(t_sup_ref)]+p['pi2_1 + 90'])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            
            for i, t in enumerate(self.tau):
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                
                # signal integration 
                t_tau = self.tau_inte * 1.2 * 2
                sub_seq.append(sup_x,sup_y)
                t_0 = sup_x.duration
                t_1 = self.tau_inte * 1.2
                for k in range(self.pulse_num/2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % k
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % k
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)] +p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)] +p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    

                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                name_x = 'Read_x_%04i.WFM' % i
                name_y = 'Read_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 - 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 - 90'],t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                
                # evolution
                t_evo = 1.2 * t
                t_2 = t_evo - ref_x.stub - 256
                repeat_1 = int(t_2 / 256)
                 
                t_1 = int(t_2 % 256)
                t_0 += ref_x.duration + repeat_1 * 256
                
                sub_seq.append(evo,evo,repeat=repeat_1)
                
                # signal integration again
                x_name = 'super_x_%03i' % i
                y_name = 'super_y_%03i' % i
                
                sup_x_2 = Waveform(x_name, [Idle(256 + t_1)]+p['pi2_1 + 0'], t_0)
                sup_y_2 = Waveform(y_name, [Idle(256 + t_1)]+p['pi2_1 + 90'], t_0)
                self.waves.append(sup_x_2)
                self.waves.append(sup_y_2)
                sub_seq.append(sup_x_2,sup_y_2)
                t_0 += sup_x_2.duration
                t_1 = self.tau_inte * 1.2 - sup_x_2.stub
                for k in range(self.pulse_num/2):
                    x_name = 'X_XY4_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    y_name = 'Y_XY4_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    
                    map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)] +p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_y + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)] +p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_y + 90'],t_0)  
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    sub_seq.append(map_x_1,map_y_1)
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub

                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                name_x = 'Read1_x_%04i.WFM' % i
                name_y = 'Read1_y_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 - 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 - 90'],t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                sub_seq.append(ref_x,ref_y)
                
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)

            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            #time.sleep(70)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('Correlation_Spec.SEQ')
            
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )  
             
    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.previous_time = 0
            #self.previous_count_data = FC.GetData()
            self.prepare_awg()
            MW.setFrequency(self.freq_center)
            MW.setPower(self.power)
            AWG.run()
            time.sleep(30.0)
            FC.Start()
            time.sleep(0.1)
            PG.Sequence(self.sequence, loop=True)

            start_time = time.time()

            while True:
               self.thread.stop_request.wait(5.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               self.elapsed_time = self.previous_elapsed_time + time.time() - start_time
               self.run_time += self.elapsed_time
               runtime, cycles = FC.GetState()
               sweeps = cycles / FC.GetData().shape[0]
               self.elapsed_sweeps = self.previous_sweeps + sweeps
               self.progress = int( 100 * self.elapsed_sweeps / self.sweeps ) 
               self.count_data = self.old_count_data + FC.GetData()
               if self.elapsed_sweeps > self.sweeps:
                  break

            FC.Halt()
            MW.Off()
            PG.High(['laser', 'mw'])
            AWG.stop()
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  
            
        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'    

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        tau_1 = self.tau_inte
        sequence = []
        
        for i, t in enumerate(tau):
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , t + tau_1 * 2 * 8 * self.pulse_num * 2 + 4 * pi2_1 + 16*self.pulse_num*pi_1 + 4000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','pi_1','tau_inte','pulse_num']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                     ),                         
                              HGroup(
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40),
                                     Item('tau_inte', width = -40),
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
                       title='Correlation_Spec_CPMG4',
                       )     

class RF_sweep( Pulsed ):
    """Rabi measurement.
    """ 
    #def _init_(self):
        #super(Rabi, self).__init__()
        
    reload = True
  
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency [Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_center = Range(low=1, high=20e9, value=1.96e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)
    
    pi_1   = Range(low=1., high=100000., value=49, desc='length of first pi pulse [ns]', label='pi [ns] 1.trans', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=0., high=1.0e9, value=1.e6, desc='start freq [Hz]', label='start freq [Hz]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1.0e9, value=3.e6, desc='end freq [Hz]', label='end freq [Hz]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=1.0e7, value=1.e5, desc='freq step [Hz]', label='freq step [Hz]', mode='text', auto_set=False, enter_set=True)
    
    rf_time = Range(low=1., high=300000., value=75, desc='rf_time', label='rf time', mode='text', auto_set=False, enter_set=True)
    amp_rf = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM rf amp', mode='text', auto_set=False, enter_set=True)

    wait_time = Range(low=1.e3, high=300.e3, value=1.e3, desc='time between rf pulse and pi', label='wait_time', mode='text', auto_set=False, enter_set=True)  
    mn_flag = Bool(False, label='Transition_flag',desc='True for Ms=-1, False for Ms=0')
    
    def prepare_awg(self):
        sampling = 1.2e9
        
 
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            rf_time = int(self.rf_time * sampling/1.0e9)
            
            # Pulses
            p = {}
            
            p['pi + 0']     = Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)
            p['pi + 90']    = Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)
            
            zero = Idle(1)
            
            pi_1_i = [zero,p['pi + 0'],zero]
            pi_1_q = [zero,p['pi + 90'],zero]
            
            self.waves = []
            self.main_seq = Sequence('RF_Sweep.SEQ')
            for i,t in enumerate(self.tau):
                name_i='A_I_%04i.WFM' %i
                name_q='A_Q_%04i.WFM' %i
                name_rf='A_RF_%04i.WFM' %i
                if self.mn_flag:
                
                    self.waves.append(Waveform( name_i,pi_1_i) )
                    self.waves.append(Waveform( name_q,pi_1_q) )
                    self.waves.append(Waveform( name_rf, Idle(pi_1+2) ) )
                    
                else:
                    self.waves.append(Waveform( name_i,Idle(10)) )
                    self.waves.append(Waveform( name_q,Idle(10)) )
                    self.waves.append(Waveform( name_rf, Idle(10) ) )
                    
                self.main_seq.append(*self.waves[-3:], wait=True)
                name_i='A2_I_%04i.WFM' %i
                name_q='A2_Q_%04i.WFM' %i
                name_rf='A2_RF_%04i.WFM' %i
                self.waves.append(Waveform( name_i, Idle(rf_time )) )
                self.waves.append(Waveform( name_q,Idle(rf_time )) )
                self.waves.append(Waveform( name_rf,Sin( rf_time, t/sampling, 0 ,self.amp_rf) ) )
                self.main_seq.append(*self.waves[-3:], wait=True)
                name_i='A3_I_%04i.WFM' %i
                name_q='A3_Q_%04i.WFM' %i
                name_rf='A3_RF_%04i.WFM' %i
                self.waves.append(Waveform( name_i,pi_1_i) )
                self.waves.append(Waveform( name_q,pi_1_q) )
                self.waves.append(Waveform( name_rf, Idle(pi_1+2) ) )
                self.main_seq.append(*self.waves[-3:], wait=True)
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('RF_Sweep.SEQ')
        AWG.set_vpp(self.vpp)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )  
    
    def generate_sequence(self):
            tau = self.tau
            laser = self.laser
            wait = self.wait
            pi_1=self.pi_1

            rf_time = self.rf_time
            wait_time = self.wait_time
            sequence = []

            for t in tau:
                sequence.append( (['laser', ], laser ))
                sequence.append( ([], wait ) )
                sequence.append( (['awgTrigger'], 100 ))
                sequence.append(        ([], pi_1 +1000))
                sequence.append(        (['awgTrigger'], 100 ))
                sequence.append(        ([], rf_time + 1000))
                sequence.append(        (['awgTrigger'], 100 ))
                sequence.append(        ([], pi_1 ))
                sequence.append(        (['laser', 'trigger' ], laser ))
                sequence.append(        ([], wait_time ))
                      
            return sequence
        
    get_set_items = Rabi.get_set_items + ['freq','pi_1','rf_time','wait_time','amp_rf','mn_flag']  
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),   
                             HGroup( Item('freq',  width=20),                                    
                                     Item('freq_center',  width=20),
                                     Item('power', width=20),
                                     Item('vpp', width=40),             
                                     Item('amp', width=40),
                                     ),             
                              HGroup(Item('laser', width=30),
                                     Item('wait', width=30),
                                     Item('bin_width', width= 30, enabled_when='state != "run"'),
                                     Item('record_length', width= 30, enabled_when='state != "run"'),
                                     ),                                   
                               HGroup(Item('mn_flag', width=25),
                                      Item('pi_1', width=25),
                                      Item('rf_time', width=25),
                                      Item('amp_rf', width=25),
                                       Item('wait_time', width=25),
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
                       title='N15 nuclear resonance',
                       )       

class Single_RF_Rabi( Pulsed ):

    #def __init__(self):
        #super(Double_RF_Rabi,self).__init__()
        
    reload = True
    
    freq = Range(low=1, high=20e9, value=2.71e9, desc='frequency first transition [Hz]', label='freq 1. tran [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=49, desc='length of first pi pulse [ns]', label='pi [ns] 1.trans', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=1., high=1e7, value=100., desc='start time [ns]', label='start time [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=2e8, value=300., desc='end time [ns]', label='end time [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=1e6, value=5., desc='time step [ns]', label='time step [ns]', mode='text', auto_set=False, enter_set=True)
    
    rf_freq = Range(low=1, high=20e9, value=7.2e6, desc='frequency rf [Hz]', label='rf freq [Hz]', mode='text', auto_set=False, enter_set=True)    
    amp_rf = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM rf amp', mode='text', auto_set=False, enter_set=True)

    wait_time = Range(low=1.e3, high=300.e3, value=1.e3, desc='time between rf pulse and pi_2', label='wait_time', mode='text', auto_set=False, enter_set=True)     
     
    mn_flag = Bool(False, label='Transition_flag',desc='True for Ms=-1, False for Ms=0')
    
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi_1=self.pi_1
        rf_freq = self.rf_freq
        wait_time = self.wait_time
        sequence = []
        for t in tau:
            sub = [ (['laser', ], laser ),
                    ([], wait ),
                    (['awgTrigger'], 100  ),
                    ([], pi_1 + 1000 ),
                    (['awgTrigger'], 100 ),
                    ([], t+1000),
                    (['awgTrigger'], 100  ),
                    ([], pi_1 + 1000),
                    (['laser', 'trigger' ], laser ),
                    ([], wait_time )
                   ]
            sequence.extend(sub)
            
        return sequence
        
    
    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            rf_freq = self.rf_freq
            
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            # Pulses
            p = {}
            # ms= 0 <> ms = +1
            p['pi + 0']     = Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)
            p['pi + 90']    = Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)
            
            zero = Idle(1)
            
            pi_1_i = [zero,p['pi + 0'],zero]
            pi_1_q = [zero,p['pi + 90'],zero]
            
            # Waveforms
            self.waves = []
            self.main_seq = Sequence('RFrabi.SEQ')
                 
            for t in self.tau:
                name_i='A_I_%04i.WFM' %t
                name_q='A_Q_%04i.WFM' %t
                name_rf='A_RF_%04i.WFM' %t
                if self.mn_flag:
                
                    self.waves.append(Waveform( name_i,pi_1_i) )
                    self.waves.append(Waveform( name_q,pi_1_q) )
                    self.waves.append(Waveform( name_rf, Idle(pi_1+2) ) )
                    
                else:
                    self.waves.append(Waveform( name_i,Idle(10)) )
                    self.waves.append(Waveform( name_q,Idle(10)) )
                    self.waves.append(Waveform( name_rf, Idle(10) ) )
                
                self.main_seq.append(*self.waves[-3:], wait=True)
                name_i='A2_I_%04i.WFM' %t
                name_q='A2_Q_%04i.WFM' %t
                name_rf='A2_RF_%04i.WFM' %t
                self.waves.append(Waveform( name_i, Idle(t*1.2)) )
                self.waves.append(Waveform( name_q, Idle(t*1.2)) )
                self.waves.append(Waveform( name_rf,Sin( t*1.2, rf_freq/sampling, 0 ,self.amp_rf)) )
                self.main_seq.append(*self.waves[-3:], wait=True)
                name_i='A3_I_%04i.WFM' %t
                name_q='A3_Q_%04i.WFM' %t
                name_rf='A3_RF_%04i.WFM' %t
                self.waves.append(Waveform( name_i,pi_1_i) )
                self.waves.append(Waveform( name_q,pi_1_q) )
                self.waves.append(Waveform( name_rf, Idle(pi_1+2) ) )
                self.main_seq.append(*self.waves[-3:], wait=True)
                
            for w in self.waves:
                w.join()
              
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('RFrabi.SEQ')
        AWG.set_vpp(self.vpp)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )       

    get_set_items = Rabi.get_set_items + ['freq','vpp','pi_1','rf_freq','wait_time','amp_rf','mn_flag']  
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=40),
                                     Item('vpp', width=40),             
                                     Item('amp', width=40),          
                                     Item('freq_center', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=40),
                                     Item('power', width=40),
                                     ), 
                              HGroup( Item('mn_flag', width=25),
                                      Item('pi_1', width=25),
                                      Item('rf_freq', width=25),
                                      Item('amp_rf', width=25),
                                      Item('wait_time', width=25),
                                    ),                                     
                              HGroup(Item('laser', width=40),
                                     Item('wait', width=40),
                                     Item('bin_width', width= -80, enabled_when='state != "run"'),
                                     Item('record_length', width= -80, enabled_when='state != "run"'),
                                     ),
                                     
                              HGroup(Item('tau_begin', width=40),
                                     Item('tau_end', width=40),
                                     Item('tau_delta', width= 40),
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=40),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=40),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=40),
                                     ),
                                                                         
                              ),
                       title='Single RF Rabi Measurement',
                       )     

class Nuclea_Hahn( Pulsed ):

    #def __init__(self):
        #super(Double_RF_Rabi,self).__init__()
        
    reload = True
    
    freq = Range(low=1, high=20e9, value=2.71e9, desc='frequency first transition [Hz]', label='freq 1. tran [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=49, desc='length of first pi pulse [ns]', label='pi [ns] 1.trans', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=300., high=1e7, value=300., desc='start time [ns]', label='start time [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=2e8, value=300., desc='end time [ns]', label='end time [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=1e6, value=5., desc='time step [ns]', label='time step [ns]', mode='text', auto_set=False, enter_set=True)
    
    rf_freq = Range(low=1, high=20e9, value=7.2e6, desc='frequency rf [Hz]', label='rf freq [Hz]', mode='text', auto_set=False, enter_set=True)    
    amp_rf = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM rf amp', mode='text', auto_set=False, enter_set=True)
    half_pi_rf = Range(low=1., high=100000., value=1000, desc='length of pi half pulse [ns]', label='pi2_rf', mode='text', auto_set=False, enter_set=True)
    pi_rf = Range(low=1., high=100000., value=1000, desc='length of pi rf pulse [ns]', label='pi_rf', mode='text', auto_set=False, enter_set=True)
    wait_time = Range(low=1.e3, high=100.e3, value=10.e3, desc='time between rf pulse and pi_2', label='wait_time', mode='text', auto_set=False, enter_set=True)     
     
    def _get_sequence_points(self):
        return 2 * len(self.tau)    
     
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi_1 = self.pi_1
        pi2_rf = self.half_pi_rf
        pi_rf = self.pi_rf
        wait_time = self.wait_time
        sequence = []
        for t in tau:
            sub = [ (['laser'], laser ),
                    ([], wait ),
                    (['awgTrigger'], 100  ),
                    ([], pi_1 * 2 + 2 * pi2_rf + pi_rf + t*2 + 1000 ),
                    (['laser', 'trigger' ], laser ),
                    ([], wait_time )
                   ]
            sequence.extend(sub)
            
        return sequence
        
    
    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            rf_freq = self.rf_freq
            
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            pi2_rf = int(self.half_pi_rf * sampling/1.0e9)
            pi_rf = int(self.pi_rf * sampling/1.0e9)
            # Pulses
            p = {}
            # ms= 0 <> ms = +1
            p['pi + 0']     = Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)
            p['pi + 90']    = Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)
            
            zero = Idle(1)
            mod = Idle(0)
            
            pi_1_i = [Idle(100),p['pi + 0'],Idle(100)]
            pi_1_q = [Idle(100),p['pi + 90'],Idle(100)]
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('NucleaHahn.SEQ')
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            

            init_x = Waveform('init_x', pi_1_i, 0)
            init_y = Waveform('init_y', pi_1_q, 0)
            init_rf = Waveform('init_rf', Idle(pi_1+200), 0)
            self.waves.append(init_x)
            self.waves.append(init_y)
            self.waves.append(init_rf)
            
            for i, t in enumerate(self.tau):
                t0 = init_x.duration
                name_i='A1_I_%04i.WFM' %t
                name_q='A1_Q_%04i.WFM' %t
                name_rf='A1_RF_%04i.WFM' %t
                sup_x = Waveform( name_i, Idle(pi2_rf), t0)
                sup_y = Waveform( name_q, Idle(pi2_rf), t0)
                sup_rf = Waveform( name_rf, Sin( pi2_rf, rf_freq/sampling, 0 ,self.amp_rf), t0)
                
                self.waves.append(sup_x)
                self.waves.append(sup_y)
                self.waves.append(sup_rf)
                
                t_1 = t*1.2 - sup_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                #t0 += sup_x.duration + repeat_1 * 256
                t0 += sup_x.duration
                name_i='test1_I_%04i.WFM' %t
                name_q='test1_Q_%04i.WFM' %t
                name_rf='test1_RF_%04i.WFM' %t
                Idle1_x = Waveform( name_i, Idle(repeat_1*256), t0)
                Idle1_y = Waveform( name_q, Idle(repeat_1*256), t0)
                Idle1_rf = Waveform( name_rf, Sin(repeat_1*256), t0)
                self.waves.append(Idle1_x)
                self.waves.append(Idle1_y)
                self.waves.append(Idle1_rf)
                t0 += repeat_1 * 256
                
                name_i='A2_I_%04i.WFM' %t
                name_q='A2_Q_%04i.WFM' %t
                name_rf='A2_RF_%04i.WFM' %t
                flip_x = Waveform( name_i, [mod, Idle(pi_rf)], t0)
                flip_y = Waveform( name_q, [mod, Idle(pi_rf)], t0)
                flip_rf = Waveform( name_rf, [mod, Sin( pi_rf, rf_freq/sampling, 0 ,self.amp_rf)], t0)
                self.waves.append(flip_x)
                self.waves.append(flip_y)
                self.waves.append(flip_rf)
                
                t_1 = t*1.2 - flip_x.stub
                repeat_2 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                #t0 += flip_x.duration + repeat_2 * 256
                t0 += flip_x.duration
                name_i='test2_I_%04i.WFM' %t
                name_q='test2_Q_%04i.WFM' %t
                name_rf='test2_RF_%04i.WFM' %t
                Idle2_x = Waveform( name_i, Idle(repeat_2*256), t0)
                Idle2_y = Waveform( name_q, Idle(repeat_2*256), t0)
                Idle2_rf = Waveform( name_rf, Sin(repeat_2*256), t0)
                self.waves.append(Idle2_x)
                self.waves.append(Idle2_y)
                self.waves.append(Idle2_rf)
                t0 += repeat_2 * 256
                
                name_i='A3_I_%04i.WFM' %t
                name_q='A3_Q_%04i.WFM' %t
                name_rf='A3_RF_%04i.WFM' %t
                sup1_x = Waveform( name_i, [mod, Idle(pi2_rf)], t0)
                sup1_y = Waveform( name_q, [mod, Idle(pi2_rf)], t0)
                sup1_rf = Waveform( name_rf, [mod, Sin( pi2_rf, rf_freq/sampling, 0 ,self.amp_rf)], t0)
                
                self.waves.append(sup1_x)
                self.waves.append(sup1_y)
                self.waves.append(sup1_rf)
                
                t2 = t0 +  sup1_x.duration 
                
                name_i='A4_I_%04i.WFM' %t
                name_q='A4_Q_%04i.WFM' %t
                name_rf='A4_RF_%04i.WFM' %t
                read_x = Waveform(name_i, pi_1_i, t2)
                read_y = Waveform(name_q, pi_1_q, t2)
                read_rf = Waveform(name_rf, Idle(pi_1+200), t2)
                self.waves.append(read_x)
                self.waves.append(read_y)
                self.waves.append(read_rf)
                
                name = 'DQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(init_x, init_y, init_rf)
                
                sub_seq.append(sup_x, sup_y, sup_rf)
                #sub_seq.append(evo, evo, evo, repeat=repeat_1)
                sub_seq.append(Idle1_x, Idle1_y, Idle1_rf)
                sub_seq.append(flip_x, flip_y,flip_rf)
                #sub_seq.append(evo, evo, evo, repeat=repeat_2)
                sub_seq.append(Idle2_x, Idle2_y, Idle2_rf)
                sub_seq.append(sup1_x, sup1_y, sup1_rf)
                sub_seq.append(read_x, read_y, read_rf)
                
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)
                
                name_i='A5_I_%04i.WFM' %t
                name_q='A5_Q_%04i.WFM' %t
                name_rf='A5_RF_%04i.WFM' %t
                sup2_x = Waveform( name_i, [mod, Idle(pi2_rf)], t0)
                sup2_y = Waveform( name_q, [mod, Idle(pi2_rf)], t0)
                sup2_rf = Waveform( name_rf, [mod, Sin( pi2_rf, rf_freq/sampling, np.pi ,self.amp_rf)], t0)
                
                self.waves.append(sup2_x)
                self.waves.append(sup2_y)
                self.waves.append(sup2_rf)
                
                
                name = 'DQH_22_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(init_x, init_y, init_rf)
                
                sub_seq.append(sup_x, sup_y, sup_rf)
                sub_seq.append(evo, evo, evo, repeat=repeat_1)
                sub_seq.append(flip_x, flip_y,flip_rf)
                sub_seq.append(evo, evo, evo, repeat=repeat_2)
                sub_seq.append(sup2_x, sup2_y, sup2_rf)
                sub_seq.append(read_x, read_y, read_rf)
                
                AWG.upload(sub_seq)
                self.main_seq.append(sub_seq,wait=True)
                
                
                
            for w in self.waves:
                w.join()
              
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('NucleaHahn.SEQ')
        AWG.set_vpp(self.vpp)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )       

    get_set_items = Rabi.get_set_items + ['freq','vpp','amp','pi_1','rf_freq','wait_time','amp_rf','half_pi_rf','pi_rf']  
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=40),
                                     Item('vpp', width=-40),             
                                     Item('amp', width=-40),          
                                     Item('freq_center', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=40),
                                     Item('power', width=-40),
                                     ), 
                              HGroup( Item('pi_1', width=-35),
                                      Item('rf_freq', width=-65),
                                      Item('amp_rf', width=-45),
                                      Item('half_pi_rf', width=-45),
                                      Item('pi_rf', width=-45),
                                      Item('wait_time', width=-45),
                                    ),                                     
                              HGroup(Item('laser', width=40),
                                     Item('wait', width=40),
                                     Item('bin_width', width= -80, enabled_when='state != "run"'),
                                     Item('record_length', width= -80, enabled_when='state != "run"'),
                                     ),
                                     
                              HGroup(Item('tau_begin', width=40),
                                     Item('tau_end', width=40),
                                     Item('tau_delta', width= 40),
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=40),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=40),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=40),
                                     ),
                                                                         
                              ),
                       title='Nuclea T2 Measurement',
                       )                                                
                       
class Proton_longmagzationdet_freq(Pulsed):                   

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
   
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    
    tau_echo = Range(low=1., high=1e6, value=30e3, desc='tau for hahn echo [ns]', label='tau echo [ns]', mode='text', auto_set=False, enter_set=True)
    
    #pi_rf = Range(low=1., high=100000., value=1000, desc='length of rf pi pulse < tau_echo[ns]', label='pi_rf', mode='text', auto_set=False, enter_set=True)
    N_period = Range(low=1., high=1000., value=10, desc='the number of cycles [ns]', label='Ncycle_rf', mode='text', auto_set=False, enter_set=True)
    amp_rf = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='Rf amp', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=800., high=1e8, value=1.8e6, desc='rf freq begin [Hz]', label='freq begin [Hz]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=2.3e6, desc='rf freq end [Hz]', label='freq end [Hz]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=2e4, desc='rf freq delta [Hz]', label='delta freq [Hz]', mode='text', auto_set=False, enter_set=True)
    
    wait_time = Range(low=1.e3, high=100.e3, value=30e3, desc='Duty time', label='wait_time', mode='text', auto_set=False, enter_set=True)     
    
    reload = True
    
    #def _init_(self):
        #super(XY8, self).__init__()

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            tau_echo = int(self.tau_echo * sampling/1.0e9)
            #pi_rf = int(self.pi_rf * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            
            # Waveforms
            # Waveforms
            self.waves = []
            self.main_seq = Sequence('lmz_dec.SEQ')
            sub_seq = []
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
                 
            for i,t in enumerate(self.tau):
            
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                
                name_i='A_I_%04i.WFM' %i
                name_q='A_Q_%04i.WFM' %i
                name_rf='A_RF_%04i.WFM' %i
                sup_x = Waveform( name_i,[zero] + p['pi2_1 + 0'] + [Idle(200)] ) 
                sup_y = Waveform( name_q,[zero] + p['pi2_1 + 90'] + [Idle(200)] ) 
                sup_rf = Waveform( name_rf,Idle(pi2_1 + 201) ) 
                
                self.waves.append(sup_x )
                self.waves.append(sup_y )
                self.waves.append(sup_rf)
                t0 = sup_x.duration
                sub_seq.append(sup_x,sup_y,sup_rf)
                
                
                name_i='A2_I_%04i.WFM' %i
                name_q='A2_Q_%04i.WFM' %i
                name_rf='A2_RF_%04i.WFM' %i
                
                #nuflip_x = Waveform( name_i,Idle(pi_rf+2), t0 ) 
                #nuflip_y = Waveform( name_q,Idle(pi_rf+2), t0 ) 
                #nuflip_rf = Waveform( name_rf,[zero,Sin( pi_rf, t/sampling, 0 ,self.amp_rf),zero], t0) 
                
                length = int(self.N_period * sampling / t)
                phase_1 = np.pi *2 *(1 - (t/sampling * t0) %1)
                nuflip_x = Waveform( name_i,Idle(length+2), t0 ) 
                nuflip_y = Waveform( name_q,Idle(length+2), t0 ) 
                nuflip_rf = Waveform( name_rf,[zero,Sin( length, t/sampling, phase_1 ,self.amp_rf),zero], t0) 
                
                self.waves.append(nuflip_x )
                self.waves.append(nuflip_y )
                self.waves.append(nuflip_rf)
                t0 += nuflip_x.duration
                sub_seq.append(nuflip_x, nuflip_y, nuflip_rf)
                
                t1 = tau_echo - nuflip_x.duration - 200 - sup_x.stub
                
                repeat_1 = int(t1/256)
                t_2 = int(t1%256)
                t0 += repeat_1 * 256
                
                if repeat_1 > 0:
                    sub_seq.append(evo, evo, evo, repeat = repeat_1)
                    
                name_i='A3_I_%04i.WFM' %i
                name_q='A3_Q_%04i.WFM' %i
                name_rf='A3_RF_%04i.WFM' %i
                
                pi_echo_x = Waveform( name_i,[Idle(t_2)] + p['pi_1_y + 0'] + [Idle(200)], t0)
                pi_echo_y = Waveform( name_q,[Idle(t_2)] + p['pi_1_y + 90'] + [Idle(200)], t0)
                pi_echo_rf = Waveform( name_rf,[Idle(t_2 + pi_1 + 200)], t0)
                
                self.waves.append(pi_echo_x )
                self.waves.append(pi_echo_y )
                self.waves.append(pi_echo_rf)
                sub_seq.append(pi_echo_x, pi_echo_y, pi_echo_rf)
                
                t0 += pi_echo_x.duration
                
                name_i='A4_I_%04i.WFM' %i
                name_q='A4_Q_%04i.WFM' %i
                name_rf='A4_RF_%04i.WFM' %i
                
                #nuflip_x_2 = Waveform( name_i,Idle(pi_rf+2), t0 ) 
                #nuflip_y_2 = Waveform( name_q,Idle(pi_rf+2), t0 ) 
                #nuflip_rf_2 = Waveform( name_rf,[zero,Sin( pi_rf, t/sampling, 0 ,self.amp_rf),zero], t0) 
                
                phase = np.pi *2 *(1 - (t/sampling * (t0 - sup_x.duration)) %1) + phase_1
                nuflip_x_2 = Waveform( name_i,Idle(length+2), t0 ) 
                nuflip_y_2 = Waveform( name_q,Idle(length+2), t0 ) 
                nuflip_rf_2 = Waveform( name_rf,[zero,Sin( length, t/sampling, phase ,self.amp_rf),zero], t0) 
                
                self.waves.append(nuflip_x_2)
                self.waves.append(nuflip_y_2)
                self.waves.append(nuflip_rf_2)
                
                t0 += nuflip_x_2.duration
                sub_seq.append(nuflip_x_2, nuflip_y_2, nuflip_rf_2)
                
                t1 = tau_echo - nuflip_x_2.duration - 200 - pi_echo_x.stub
                
                repeat_2 = int(t1/256)
                t_2 = int(t1%256)
                t0 += repeat_2 * 256
                
                if repeat_2 > 0:
                    sub_seq.append(evo, evo, evo, repeat = repeat_2)
                    
                    
                name_i='A5_I_%04i.WFM' %i
                name_q='A5_Q_%04i.WFM' %i
                name_rf='A5_RF_%04i.WFM' %i
                
                read_x = Waveform( name_i,[Idle(t_2)] + p['pi2_1 + 0'] + [Idle(200)], t0)
                read_y = Waveform( name_q,[Idle(t_2)] + p['pi2_1 + 90'] + [Idle(200)], t0)
                read_rf = Waveform( name_rf,[Idle(t_2 + pi2_1 + 200)], t0)
                self.waves.append(read_x )
                self.waves.append(read_y )
                self.waves.append(read_rf)
                sub_seq.append(read_x, read_y, read_rf)
                
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)
                
                name = 'BSQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                
                name_i='B_I_%04i.WFM' %i
                name_q='B_Q_%04i.WFM' %i
                name_rf='B_RF_%04i.WFM' %i
                sup_x = Waveform( name_i,[zero] + p['pi2_1 - 0'] + [Idle(200)] ) 
                sup_y = Waveform( name_q,[zero] + p['pi2_1 - 90'] + [Idle(200)] ) 
                sup_rf = Waveform( name_rf,Idle(pi2_1 + 201) ) 
                
                self.waves.append(sup_x )
                self.waves.append(sup_y )
                self.waves.append(sup_rf)
                t0 = sup_x.duration
                sub_seq.append(sup_x,sup_y,sup_rf)
                
                
                name_i='B2_I_%04i.WFM' %i
                name_q='B2_Q_%04i.WFM' %i
                name_rf='B2_RF_%04i.WFM' %i
                
                #nuflip_x = Waveform( name_i,Idle(pi_rf+2), t0 ) 
                #nuflip_y = Waveform( name_q,Idle(pi_rf+2), t0 ) 
                #nuflip_rf = Waveform( name_rf,[zero,Sin( pi_rf, t/sampling, 0 ,self.amp_rf),zero], t0) 
                nuflip_x = Waveform( name_i,Idle(length+2), t0 ) 
                nuflip_y = Waveform( name_q,Idle(length+2), t0 ) 
                nuflip_rf = Waveform( name_rf,[zero,Sin( length, t/sampling, phase_1 ,self.amp_rf),zero], t0) 
                
                self.waves.append(nuflip_x )
                self.waves.append(nuflip_y )
                self.waves.append(nuflip_rf)
                t0 += nuflip_x.duration
                sub_seq.append(nuflip_x, nuflip_y, nuflip_rf)
                
                t1 = tau_echo - nuflip_x.duration - 200 - sup_x.stub
                
                repeat_1 = int(t1/256)
                t_2 = int(t1%256)
                t0 += repeat_1 * 256
                
                if repeat_1 > 0:
                    sub_seq.append(evo, evo, evo, repeat = repeat_1)
                    
                name_i='B3_I_%04i.WFM' %i
                name_q='B3_Q_%04i.WFM' %i
                name_rf='B3_RF_%04i.WFM' %i
                
                pi_echo_x = Waveform( name_i,[Idle(t_2)] + p['pi_1_y + 0'] + [Idle(200)], t0)
                pi_echo_y = Waveform( name_q,[Idle(t_2)] + p['pi_1_y + 90'] + [Idle(200)], t0)
                pi_echo_rf = Waveform( name_rf,[Idle(t_2 + pi_1 + 200)], t0)
                
                self.waves.append(pi_echo_x )
                self.waves.append(pi_echo_y )
                self.waves.append(pi_echo_rf)
                sub_seq.append(pi_echo_x, pi_echo_y, pi_echo_rf)
                
                t0 += pi_echo_x.duration
                
                name_i='B4_I_%04i.WFM' %i
                name_q='B4_Q_%04i.WFM' %i
                name_rf='B4_RF_%04i.WFM' %i
                
                #nuflip_x_2 = Waveform( name_i,Idle(pi_rf+2), t0 ) 
                #nuflip_y_2 = Waveform( name_q,Idle(pi_rf+2), t0 ) 
                #nuflip_rf_2 = Waveform( name_rf,[zero,Sin( pi_rf, t/sampling, 0 ,self.amp_rf),zero], t0) 
                
                phase = np.pi *2 *(1 - (t/sampling * (t0 - sup_x.duration)) %1) + phase_1
                
                nuflip_x_2 = Waveform( name_i,Idle(length+2), t0 ) 
                nuflip_y_2 = Waveform( name_q,Idle(length+2), t0 ) 
                nuflip_rf_2 = Waveform( name_rf,[zero,Sin( length, t/sampling, phase ,self.amp_rf),zero], t0) 
                
                self.waves.append(nuflip_x_2)
                self.waves.append(nuflip_y_2)
                self.waves.append(nuflip_rf_2)
                
                t0 += nuflip_x_2.duration
                sub_seq.append(nuflip_x_2, nuflip_y_2, nuflip_rf_2)
                
                t1 = tau_echo - nuflip_x_2.duration - 200 - pi_echo_x.stub
                
                repeat_2 = int(t1/256)
                t_2 = int(t1%256)
                t0 += repeat_2 * 256
                
                if repeat_2 > 0:
                    sub_seq.append(evo, evo, evo, repeat = repeat_2)
                    
                    
                name_i='B5_I_%04i.WFM' %i
                name_q='B5_Q_%04i.WFM' %i
                name_rf='B5_RF_%04i.WFM' %i
                
                read_x = Waveform( name_i,[Idle(t_2)] + p['pi2_1 + 0'] + [Idle(200)], t0)
                read_y = Waveform( name_q,[Idle(t_2)] + p['pi2_1 + 90'] + [Idle(200)], t0)
                read_rf = Waveform( name_rf,[Idle(t_2 + pi2_1 + 200)], t0)
                self.waves.append(read_x )
                self.waves.append(read_y )
                self.waves.append(read_rf)
                sub_seq.append(read_x, read_y, read_rf)
                
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)


            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('lmz_dec.SEQ')
            
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )  
        
        
    def _get_sequence_points(self):
        return 2 * len(self.tau)       

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        wait_time = self.wait_time
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        tau_echo = self.tau_echo
        sequence = []
        
        for i, t in enumerate(tau):
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , pi2_1 * 2 + pi_1 + 2 * tau_echo + 2000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait+wait_time) )
            
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , pi2_1 * 2 + pi_1 + 2 * tau_echo + 2000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait+wait_time) )

        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','pi_1','tau_echo','wait_time','amp_rf','N_period']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                  
                                     ),                         
                              HGroup(Item('pi2_1', width = -60),
                                     Item('pi_1', width = -60),
                                     Item('tau_echo', width = -60),
                                     ),      
                               HGroup(Item('N_period', width = -60),
                                     Item('amp_rf', width = -60),
                                     Item('wait_time', width = -60),
                                     ),             
                              HGroup(Item('laser', width=30),
                                     Item('wait', width=30),
                                     Item('bin_width', width= 30, enabled_when='state != "run"'),
                                     Item('record_length', width= 30, enabled_when='state != "run"'),
                                     ),
                                     
                              HGroup(Item('tau_begin', width=-60),
                                     Item('tau_end', width=-60),
                                     Item('tau_delta', width= -60),
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30),
                                     ),
                                                                         
                              ),
                       title='Proton_lmzdet_freq',
                       )                         
                       
                       
class Proton_longmagzationdet_time(Pulsed):                   

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
   
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    
    tau_echo = Range(low=1., high=1e6, value=30e3, desc='tau for hahn echo [ns]', label='tau echo [ns]', mode='text', auto_set=False, enter_set=True)
    
    freq_rf = Range(low=1., high=1e8, value=2.0e6, desc='rf freq', label='freq_rf', mode='text', auto_set=False, enter_set=True)
    amp_rf = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='Rf amp', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=100., high=1e8, value=100, desc='rf rabi begin [Hz]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=1000, desc='rf rabi end [Hz]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=10, desc='rf rabi delta [Hz]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    
    wait_time = Range(low=1.e3, high=100.e3, value=30e3, desc='Duty time', label='wait_time', mode='text', auto_set=False, enter_set=True)     
    
    reload = True
    
    #def _init_(self):
        #super(XY8, self).__init__()

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            tau_echo = int(self.tau_echo * sampling/1.0e9)
            freqrf = self.freq_rf/sampling
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            
            # Waveforms
            # Waveforms
            self.waves = []
            self.main_seq = Sequence('lmz_dec.SEQ')
            sub_seq = []
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
                 
            for i,t in enumerate(self.tau):
            
                name_i='A_I_%04i.WFM' %i
                name_q='A_Q_%04i.WFM' %i
                name_rf='A_RF_%04i.WFM' %i
                sup_x = Waveform( name_i,[zero] + p['pi2_1 + 0'] + [Idle(200)] ) 
                sup_y = Waveform( name_q,[zero] + p['pi2_1 + 90'] + [Idle(200)] ) 
                sup_rf = Waveform( name_rf,Idle(pi2_1 + 201) ) 
                
                self.waves.append(sup_x )
                self.waves.append(sup_y )
                self.waves.append(sup_rf)
                t0 = sup_x.duration
                
                name_i='A2_I_%04i.WFM' %i
                name_q='A2_Q_%04i.WFM' %i
                name_rf='A2_RF_%04i.WFM' %i
                
                nuflip_x = Waveform( name_i,Idle(t*1.2+2), t0 ) 
                nuflip_y = Waveform( name_q,Idle(t*1.2+2), t0 ) 
                nuflip_rf = Waveform( name_rf,[zero,Sin( t*1.2,freqrf, 0 ,self.amp_rf),zero], t0) 
                
                self.waves.append(nuflip_x )
                self.waves.append(nuflip_y )
                self.waves.append(nuflip_rf)
                t0 += nuflip_x.duration
                
                t1 = tau_echo - nuflip_x.duration - 200 - sup_x.stub
                repeat_1 = int(t1/256)
                t_2 = int(t1%256)
                t0 += repeat_1 * 256
                
                name_i='A3_I_%04i.WFM' %i
                name_q='A3_Q_%04i.WFM' %i
                name_rf='A3_RF_%04i.WFM' %i
                
                pi_echo_x = Waveform( name_i,[Idle(t_2)] + p['pi_1_y + 0'] + [Idle(200)], t0)
                pi_echo_y = Waveform( name_q,[Idle(t_2)] + p['pi_1_y + 90'] + [Idle(200)], t0)
                pi_echo_rf = Waveform( name_rf,[Idle(t_2 + pi_1 + 200)], t0)
                
                self.waves.append(pi_echo_x )
                self.waves.append(pi_echo_y )
                self.waves.append(pi_echo_rf)
                
                
                t0 += pi_echo_x.duration
                
                name_i='A4_I_%04i.WFM' %i
                name_q='A4_Q_%04i.WFM' %i
                name_rf='A4_RF_%04i.WFM' %i
                
                nuflip_x_2 = Waveform( name_i,Idle(t*1.2+2), t0 ) 
                nuflip_y_2 = Waveform( name_q,Idle(t*1.2+2), t0 ) 
                nuflip_rf_2 = Waveform( name_rf,[zero,Sin( t*1.2, freqrf, 0 ,self.amp_rf),zero], t0) 
                
                self.waves.append(nuflip_x_2)
                self.waves.append(nuflip_y_2)
                self.waves.append(nuflip_rf_2)
                
                t0 += nuflip_x_2.duration
                
                
                t1 = tau_echo - nuflip_x_2.duration - 200 - pi_echo_x.stub
                
                repeat_2 = int(t1/256)
                t_2 = int(t1%256)
                t0 += repeat_2 * 256
                
                name_i='A5_I_%04i.WFM' %i
                name_q='A5_Q_%04i.WFM' %i
                name_rf='A5_RF_%04i.WFM' %i
                
                read_x = Waveform( name_i,[Idle(t_2)] + p['pi2_1 + 0'] + [Idle(200)], t0)
                read_y = Waveform( name_q,[Idle(t_2)] + p['pi2_1 + 90'] + [Idle(200)], t0)
                read_rf = Waveform( name_rf,[Idle(t_2 + pi2_1 + 200)], t0)
                self.waves.append(read_x )
                self.waves.append(read_y )
                self.waves.append(read_rf)
                
                
                
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x,sup_y,sup_rf)
                sub_seq.append(nuflip_x, nuflip_y, nuflip_rf)
                if repeat_1 > 0:
                    sub_seq.append(evo, evo, evo, repeat = repeat_1)
                sub_seq.append(pi_echo_x, pi_echo_y, pi_echo_rf)
                sub_seq.append(nuflip_x_2, nuflip_y_2, nuflip_rf_2)
                if repeat_2 > 0:
                    sub_seq.append(evo, evo, evo, repeat = repeat_2)
                sub_seq.append(read_x, read_y, read_rf)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)
                
               
                name_i='B5_I_%04i.WFM' %i
                name_q='B5_Q_%04i.WFM' %i
                name_rf='B5_RF_%04i.WFM' %i
                
                read_x1 = Waveform( name_i,[Idle(t_2)] + p['pi2_1 - 0'] + [Idle(200)], t0)
                read_y1 = Waveform( name_q,[Idle(t_2)] + p['pi2_1 - 90'] + [Idle(200)], t0)
                read_rf1 = Waveform( name_rf,[Idle(t_2 + pi2_1 + 200)], t0)
                self.waves.append(read_x1 )
                self.waves.append(read_y1 )
                self.waves.append(read_rf1)
                
                
                name = 'BSQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x,sup_y,sup_rf)
                sub_seq.append(nuflip_x, nuflip_y, nuflip_rf)
                if repeat_1 > 0:
                    sub_seq.append(evo, evo, evo, repeat = repeat_1)
                sub_seq.append(pi_echo_x, pi_echo_y, pi_echo_rf)
                sub_seq.append(nuflip_x_2, nuflip_y_2, nuflip_rf_2)
                if repeat_2 > 0:
                    sub_seq.append(evo, evo, evo, repeat = repeat_2)
                sub_seq.append(read_x1, read_y1, read_rf1)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)


            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('lmz_dec.SEQ')
            
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )  
        
        
    def _get_sequence_points(self):
        return 2 * len(self.tau)       

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        wait_time = self.wait_time
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        tau_echo = self.tau_echo
        sequence = []
        
        for i, t in enumerate(tau):
            sequence.append( (['laser'], laser) )
            sequence.append( ([ ]                 , wait) )
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , pi2_1 * 2 + pi_1 + 2 * tau_echo + 2000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait_time) )
            
            sequence.append( (['laser'], laser) )
            sequence.append( ([ ]                 , wait) )
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , pi2_1 * 2 + pi_1 + 2 * tau_echo + 2000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait_time) )

        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','pi_1','tau_echo','wait_time','amp_rf','freq_rf']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                  
                                     ),                         
                              HGroup(Item('pi2_1', width = -60),
                                     Item('pi_1', width = -60),
                                     Item('tau_echo', width = -60),
                                     ),      
                               HGroup(Item('freq_rf', width = -60),
                                     Item('amp_rf', width = -60),
                                     Item('wait_time', width = -60),
                                     ),             
                              HGroup(Item('laser', width=30),
                                     Item('wait', width=30),
                                     Item('bin_width', width= 30, enabled_when='state != "run"'),
                                     Item('record_length', width= 30, enabled_when='state != "run"'),
                                     ),
                                     
                              HGroup(Item('tau_begin', width=-60),
                                     Item('tau_end', width=-60),
                                     Item('tau_delta', width= -60),
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30),
                                     ),
                                                                         
                              ),
                       title='Proton_lmzdet_time',
                       )                         
                       
class Hahn_AC( Pulsed ):
    #FID 
    
    pi2_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='pi plus', mode='text', auto_set=False, enter_set=True)
    
    freq_center = Range(low=1, high=20e9, value=1.96e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)  

    N_period = Range(low=0., high=1000., value=10, desc='the number of cycles [ns]', label='Ncycle_rf', mode='text', auto_set=False, enter_set=True)
    freq_rf = Range(low=1., high=1e8, value=2.0e6, desc='rf freq', label='freq_rf', mode='text', auto_set=False, enter_set=True)
    amp_rf = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='Rf amp', mode='text', auto_set=False, enter_set=True)    
    phi = Range(low=0., high=6.29, value=0.1, desc='relative phase', label='Rf phase', mode='text', auto_set=False, enter_set=True)    
    wait_time = Range(low=1.e3, high=100.e3, value=30e3, desc='Duty time', label='wait_time', mode='text', auto_set=False, enter_set=True)    
    time_space = Range(low=1, high=10e3, value=300, desc='time space', label='time_space', mode='text', auto_set=False, enter_set=True)    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            p = {}
            
            pi2_1 = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            freq_rf = self.freq_rf / sampling
            time_space = int(self.time_space * sampling/1.0e9)

            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            
            p['pi_1 + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1 + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
           
            zero = Idle(1)
            mod = Idle(0)
            

            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('Hahn_AC.SEQ')
            
            length = int(self.N_period /freq_rf)
            
            sup_x = Waveform('Sup1_x', [zero] + p['pi2_1 + 0'] + [Idle(time_space)] + [Idle(length+2)])
            sup_y = Waveform('Sup1_y', [zero] + p['pi2_1 + 90'] + [Idle(time_space)] + [Idle(length+2)])
            sup_rf = Waveform('Sup1_rf', [Idle(pi2_1 + 1 + time_space)] + [zero,Sin( length,freq_rf, 0, self.amp_rf),zero])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            self.waves.append(sup_rf)
          
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            for i,t in enumerate(self.tau):
            
                t0 = sup_x.duration
                '''
                name_i='A2_I_%04i.WFM' %i
                name_q='A2_Q_%04i.WFM' %i
                name_rf='A2_RF_%04i.WFM' %i
                
                
                
                nuflip_x = Waveform( name_i,Idle(length+2), t0 ) 
                nuflip_y = Waveform( name_q,Idle(length+2), t0 ) 
                nuflip_rf = Waveform( name_rf,[zero,Sin( length,freq_rf, phase_1, self.amp_rf),zero], t0) 
                
                self.waves.append(nuflip_x )
                self.waves.append(nuflip_y )
                self.waves.append(nuflip_rf)
                
                
                t1 = t * 1.2  - nuflip_x.duration - 200 - sup_x.stub
                
                repeat_1 = int(t1/256)
                mod.duration = int(t1%256)
                t0 += nuflip_x.duration + repeat_1 * 256
                '''
                
                t1 = t * 1.2 - length - time_space - sup_x.stub
                repeat_1 = int(t1/256)
                mod.duration = int(t1%256)
                t0 += repeat_1 * 256
                
                name_i='A3_I_%04i.WFM' %i
                name_q='A3_Q_%04i.WFM' %i
                name_rf='A3_RF_%04i.WFM' %i
                
                #phase_2 = np.pi *2 *(1 - (freq_rf * (t0 - sup_x.duration)) %1)
                pi_echo_x = Waveform( name_i,[mod] + p['pi_1 + 0'] + [Idle(time_space)] + [Idle(length+2)], t0)
                pi_echo_y = Waveform( name_q,[mod] + p['pi_1 + 90'] +[Idle(time_space)] + [Idle(length+2)], t0)
                pi_echo_rf = Waveform( name_rf,[mod] + [Idle(pi_1 + time_space)] + [zero,Sin(length, freq_rf, 0 ,self.amp_rf),zero] , t0)
                
                self.waves.append(pi_echo_x )
                self.waves.append(pi_echo_y )
                self.waves.append(pi_echo_rf)
                
                t0 += pi_echo_x.duration
                
                '''
                name_i='A4_I_%04i.WFM' %i
                name_q='A4_Q_%04i.WFM' %i
                name_rf='A4_RF_%04i.WFM' %i
                
                phase_2 = np.pi *2 *(1 - (freq_rf * (t0 - sup_x.duration)) %1) + phase_1
                nuflip_x_2 = Waveform( name_i,Idle(length+2), t0 ) 
                nuflip_y_2 = Waveform( name_q,Idle(length+2), t0 ) 
                nuflip_rf_2 = Waveform( name_rf,[zero,Sin(length, freq_rf, phase_2 ,self.amp_rf),zero], t0) 
                
                self.waves.append(nuflip_x_2)
                self.waves.append(nuflip_y_2)
                self.waves.append(nuflip_rf_2)
                
                t0 += nuflip_x_2.duration
                
                
                
                t1 = t * 1.2  - nuflip_x_2.duration - 200 - pi_echo_x.stub
                
                repeat_2 = int(t1/256)
                mod.duration = int(t1%256)
                t0 += repeat_2 * 256
                '''
                t1 = t * 1.2 - length - time_space - pi_echo_x.stub
                
                repeat_2 = int(t1/256)
                mod.duration = int(t1%256)
                t0 += repeat_2 * 256
                name_i='A5_I_%04i.WFM' %i
                name_q='A5_Q_%04i.WFM' %i
                name_rf='A5_RF_%04i.WFM' %i
                
                read_x = Waveform( name_i,[mod] + p['pi2_1 + 0'] + [Idle(200)], t0)
                read_y = Waveform( name_q,[mod] + p['pi2_1 + 90'] + [Idle(200)], t0)
                read_rf = Waveform( name_rf,[mod] + [Idle( pi2_1 + 200)], t0)
                self.waves.append(read_x )
                self.waves.append(read_y )
                self.waves.append(read_rf)
                
                
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x,sup_y,sup_rf)
                #sub_seq.append(nuflip_x, nuflip_y, nuflip_rf)
                if repeat_1 > 0:
                    sub_seq.append(evo, evo, evo, repeat = repeat_1)
                sub_seq.append(pi_echo_x, pi_echo_y, pi_echo_rf)
                #sub_seq.append(nuflip_x_2, nuflip_y_2, nuflip_rf_2)
                if repeat_2 > 0:
                    sub_seq.append(evo, evo, evo, repeat = repeat_2)
                sub_seq.append(read_x, read_y, read_rf)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)
                
               
                name_i='B5_I_%04i.WFM' %i
                name_q='B5_Q_%04i.WFM' %i
                name_rf='B5_RF_%04i.WFM' %i
                
                read_x1 = Waveform( name_i,[mod] + p['pi2_1 - 0'] + [Idle(200)], t0)
                read_y1 = Waveform( name_q,[mod] + p['pi2_1 - 90'] + [Idle(200)], t0)
                read_rf1 = Waveform( name_rf,[mod] + [Idle(pi2_1 + 200)], t0)
                self.waves.append(read_x1 )
                self.waves.append(read_y1 )
                self.waves.append(read_rf1)
                
                
                name = 'BSQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x,sup_y,sup_rf)
                #sub_seq.append(nuflip_x, nuflip_y, nuflip_rf)
                if repeat_1 > 0:
                    sub_seq.append(evo, evo, evo, repeat = repeat_1)
                sub_seq.append(pi_echo_x, pi_echo_y, pi_echo_rf)
                #sub_seq.append(nuflip_x_2, nuflip_y_2, nuflip_rf_2)
                if repeat_2 > 0:
                    sub_seq.append(evo, evo, evo, repeat = repeat_2)
                sub_seq.append(read_x1, read_y1, read_rf1)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)
           
           
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('Hahn_AC.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )  
        
    def _get_sequence_points(self):
        return 2 * len(self.tau)       
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        wait_time = self.wait_time
        pi2_1 = self.pi2_1
        pi_1 = self.pi_1
        time_space = self.time_space
        sequence = []
        
        for t in tau:
            sub = [ (['laser' ], laser ),
                    ([], wait ),
                    (['awgTrigger'], 100 ),
                    ([], 2 * t + pi_1 + pi2_1*2 + 600 + time_space * 2),
                    (['laser', 'trigger' ], laser ),
                    ([], wait_time ),
                    
                    (['laser' ], laser ),
                    ([], wait ),
                    (['awgTrigger'], 100 ),
                    ([], 2 * t + pi_1 + pi2_1*2 + 600 + time_space * 2),
                    (['laser', 'trigger' ], laser ),
                    ([], wait_time )
                  ]
            sequence.extend(sub)

        return sequence
            
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','pi_1','freq_rf','N_period','amp_rf', 'wait_time','phi','time_space'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-65),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     ),       
                             HGroup(
                                    Item('pi2_1', width=20), 
                                    Item('pi_1', width=20), 
                                    ),       
                              HGroup(Item('freq_rf', width = -60),
                                     Item('amp_rf', width = -60),
                                     Item('N_period', width = -60),
                                     Item('wait_time', width = -60),
                                     Item('time_space', width = -60)
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
                       title='Hahn_AC',
                       )        
                       
class FID_AC( Pulsed ):
    #FID 
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half plus pi', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency plus [Hz]', label='freq1 p [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_center = Range(low=1, high=20e9, value=1.96e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    N_period = Range(low=1., high=1000., value=10, desc='the number of cycles [ns]', label='Ncycle_rf', mode='text', auto_set=False, enter_set=True)
    freq_rf = Range(low=1., high=1e8, value=2.0e6, desc='rf freq', label='freq_rf', mode='text', auto_set=False, enter_set=True)
    amp_rf = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='Rf amp', mode='text', auto_set=False, enter_set=True)    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )   
            pi2_1 = int(self.pi2_1 * sampling/1.0e9) 
            freq_rf = self.freq_rf/sampling
            # Pulses
            p = {}
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,1)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,1)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,1)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,1)]
                  
            zero = Idle(1)
            mod = Idle(0)

            #pi2_p_i = [zero,p['pi2_1 + 0'],zero]
            #pi2_p_q = [zero,p['pi2_1 + 90'],zero]
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('FID_AC.SEQ')
            
            sup_x = Waveform('Sup1_x', p['pi2_1 + 0'] + [Idle(20)])
            sup_y = Waveform('Sup1_y', p['pi2_1 + 90'] + [Idle(20)])
            sup_rf = Waveform('Sup1_rf', Idle(pi2_1 + 20))
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            self.waves.append(sup_rf)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
                                 
            for i, t in enumerate(self.tau):
            
                t_0 = sup_x.duration
                
                name_x = 'AC_RF1_X%04i.WFM' % i
                name_y = 'AC_RF1_Y%04i.WFM' % i
                name_rf = 'AC_RF1_RF%04i.WFM' % i
                
                length = int(self.N_period /freq_rf)
                phase_1 = np.pi *2 *(1 - (freq_rf * t_0) %1)
                AC_x = Waveform( name_x,Idle(length+2), t_0 ) 
                AC_y = Waveform( name_y,Idle(length+2), t_0 ) 
                AC_rf = Waveform( name_rf,[zero,Sin( length, freq_rf, phase_1 ,self.amp_rf),zero], t_0) 
                
                self.waves.append(AC_x )
                self.waves.append(AC_y )
                self.waves.append(AC_rf)
                
                t_1 = t * 1.2 - AC_x.duration - 50 - sup_x.stub
                
                repeat_1 = int(t_1/256)
                mod.duration = int(t_1%256)
                
                t_0 += AC_x.duration + repeat_1 * 256
                
                name_x = 'REF_X%04i.WFM' % i
                name_y = 'REF_Y%04i.WFM' % i
                name_rf = 'REF_RF%04i.WFM' % i
                
                map_x = Waveform(name_x, [mod]+p['pi2_1 + 0'], t_0)
                map_y = Waveform(name_y, [mod]+p['pi2_1 + 90'], t_0)
                map_rf = Waveform(name_rf, [mod]+[Idle(pi2_1)] , t_0)
                self.waves.append(map_x)
                self.waves.append(map_y)
                self.waves.append(map_rf)
                
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                if(repeat_1 == 0):
                    sub_seq.append(sup_x, sup_y, sup_rf)
                    sub_seq.append(AC_x, AC_y, AC_rf)
                    sub_seq.append(map_x, map_y, map_rf)
                    AWG.upload(sub_seq)
                else:
                    sub_seq.append(sup_x, sup_y, sup_rf)
                    sub_seq.append(AC_x, AC_y, AC_rf)
                    sub_seq.append(evo, evo, evo, repeat = repeat_1)
                    sub_seq.append(map_x, map_y, map_rf)
                    AWG.upload(sub_seq)
      
                self.main_seq.append(sub_seq,wait=True)
                
                name_x = 'REF1_X%04i.WFM' % i
                name_y = 'REF1_Y%04i.WFM' % i
                name_rf = 'REF1_RF%04i.WFM' % i
                map1_x = Waveform(name_x, [mod]+p['pi2_1 - 0'], t_0)
                map1_y = Waveform(name_y, [mod]+p['pi2_1 - 90'], t_0)
                map1_rf = Waveform(name_rf, [mod]+[Idle(pi2_1)] , t_0)
                self.waves.append(map1_x)
                self.waves.append(map1_y)
                self.waves.append(map1_rf)
                
                name = 'BSQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                if(repeat_1 == 0):
                    sub_seq.append(sup_x, sup_y, sup_rf)
                    sub_seq.append(AC_x, AC_y, AC_rf)
                    sub_seq.append(map1_x, map1_y, map1_rf)
                    AWG.upload(sub_seq)
                else:
                    sub_seq.append(sup_x, sup_y, sup_rf)
                    sub_seq.append(AC_x, AC_y, AC_rf)
                    sub_seq.append(evo, evo, evo, repeat = repeat_1)
                    sub_seq.append(map1_x, map1_y, map1_rf)
                    AWG.upload(sub_seq)
      
                self.main_seq.append(sub_seq,wait=True)

                
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('FID_AC.SEQ')
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 ) 
        
    def _get_sequence_points(self):
        return 2 * len(self.tau)    
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1=self.pi2_1
        sequence = []
        for t in tau:
            sub = [ (['awgTrigger'], 100 ),
                    ([], t + pi2_1 * 2 + 200 ),
                    (['laser', 'trigger' ], laser ),
                    ([], wait ),
                    
                    (['awgTrigger'], 100 ),
                    ([], t + pi2_1 * 2 + 200 ),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)
        return sequence
        
    get_set_items = Pulsed.get_set_items + ['freq','vpp','pi2_1','amp', 'amp_rf', 'N_period', 'freq_rf'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ), 
                              HGroup(Item('freq',  width=-60),      
                                     Item('pi2_1', width=-40),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     ),                 
                              HGroup(Item('laser', width=30),
                                     Item('wait', width=30),
                                     Item('bin_width', width= 30, enabled_when='state != "run"'),
                                     Item('record_length', width= 30, enabled_when='state != "run"'),
                                     ),
                              HGroup(Item('freq_rf', width = -60),
                                     Item('amp_rf', width = -60),
                                     Item('N_period', width = -60),
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
                       title='FID_AC',
                       )           

class Hahn_AC_phase(Pulsed):                   

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
   
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    
    tau_echo = Range(low=1., high=1e6, value=30e3, desc='tau for hahn echo [ns]', label='tau echo [ns]', mode='text', auto_set=False, enter_set=True)
    
    #pi_rf = Range(low=1., high=100000., value=1000, desc='length of rf pi pulse < tau_echo[ns]', label='pi_rf', mode='text', auto_set=False, enter_set=True)
    N_period = Range(low=1., high=1000., value=10, desc='the number of cycles [ns]', label='Ncycle_rf', mode='text', auto_set=False, enter_set=True)
    freq_rf = Range(low=1., high=1e8, value=2.0e6, desc='rf freq', label='freq_rf', mode='text', auto_set=False, enter_set=True)
    amp_rf = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='Rf amp', mode='text', auto_set=False, enter_set=True)    
    #phi = Range(low=0., high=6.29, value=0.1, desc='relative phase', label='Rf phase', mode='text', auto_set=False, enter_set=True)    
    
    wait_time = Range(low=1.e3, high=100.e3, value=30e3, desc='Duty time', label='wait_time', mode='text', auto_set=False, enter_set=True)     
    tau_begin = Range(low=0., high=7., value=0., desc='rf phase begin [Hz]', label='phase begin [rad]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=0., high=10., value=6.3, desc='rf phase end [Hz]', label='phase end [rad]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=7., value=0.2, desc='rf phase delta [Hz]', label='delta phase [rad]', mode='text', auto_set=False, enter_set=True)
    
    wait_time = Range(low=1.e3, high=100.e3, value=30e3, desc='Duty time', label='wait_time', mode='text', auto_set=False, enter_set=True)     
    
    reload = True
    
    #def _init_(self):
        #super(XY8, self).__init__()

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            tau_echo = int(self.tau_echo * sampling/1.0e9)
            freq_rf = self.freq_rf / sampling
            #pi_rf = int(self.pi_rf * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            
            # Waveforms
            # Waveforms
            self.waves = []
            self.main_seq = Sequence('Hahn_AC_phase.SEQ')
            sub_seq = []
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
                 
            for i,t in enumerate(self.tau):
            
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                
                name_i='A_I_%04i.WFM' %i
                name_q='A_Q_%04i.WFM' %i
                name_rf='A_RF_%04i.WFM' %i
                sup_x = Waveform( name_i,[zero] + p['pi2_1 + 0'] + [Idle(200)] ) 
                sup_y = Waveform( name_q,[zero] + p['pi2_1 + 90'] + [Idle(200)] ) 
                sup_rf = Waveform( name_rf,Idle(pi2_1 + 201) ) 
                
                self.waves.append(sup_x )
                self.waves.append(sup_y )
                self.waves.append(sup_rf)
                t0 = sup_x.duration

                name_i='A2_I_%04i.WFM' %i
                name_q='A2_Q_%04i.WFM' %i
                name_rf='A2_RF_%04i.WFM' %i
                
                length = int(self.N_period /freq_rf)
                phase_1 = np.pi *2 *(1 - (t/sampling * t0) %1)
                nuflip_x = Waveform( name_i,Idle(length+2), t0 ) 
                nuflip_y = Waveform( name_q,Idle(length+2), t0 ) 
                nuflip_rf = Waveform( name_rf,[zero,Sin( length, freq_rf, phase_1 ,self.amp_rf),zero], t0) 
                
                self.waves.append(nuflip_x )
                self.waves.append(nuflip_y )
                self.waves.append(nuflip_rf)
                t0 += nuflip_x.duration
                
                t1 = tau_echo - nuflip_x.duration - 200 - sup_x.stub
                
                repeat_1 = int(t1/256)
                t_2 = int(t1%256)
                t0 += repeat_1 * 256
                  
                name_i='A3_I_%04i.WFM' %i
                name_q='A3_Q_%04i.WFM' %i
                name_rf='A3_RF_%04i.WFM' %i
                
                pi_echo_x = Waveform( name_i,[Idle(t_2)] + p['pi_1_y + 0'] + [Idle(200)], t0)
                pi_echo_y = Waveform( name_q,[Idle(t_2)] + p['pi_1_y + 90'] + [Idle(200)], t0)
                pi_echo_rf = Waveform( name_rf,[Idle(t_2 + pi_1 + 200)], t0)
                
                self.waves.append(pi_echo_x )
                self.waves.append(pi_echo_y )
                self.waves.append(pi_echo_rf)
                
                t0 += pi_echo_x.duration
                
                name_i='A4_I_%04i.WFM' %i
                name_q='A4_Q_%04i.WFM' %i
                name_rf='A4_RF_%04i.WFM' %i
                
                phase = np.pi *2 *(1 - (freq_rf * (t0 - sup_x.duration)) %1) + phase_1 + t
                nuflip_x_2 = Waveform( name_i,Idle(length+2), t0 ) 
                nuflip_y_2 = Waveform( name_q,Idle(length+2), t0 ) 
                nuflip_rf_2 = Waveform( name_rf,[zero,Sin( length, freq_rf, phase ,self.amp_rf),zero], t0) 
                
                self.waves.append(nuflip_x_2)
                self.waves.append(nuflip_y_2)
                self.waves.append(nuflip_rf_2)
                
                t0 += nuflip_x_2.duration
                
                t1 = tau_echo - nuflip_x_2.duration - 200 - pi_echo_x.stub
                
                repeat_2 = int(t1/256)
                t_2 = int(t1%256)
                t0 += repeat_2 * 256
                
                name_i='A5_I_%04i.WFM' %i
                name_q='A5_Q_%04i.WFM' %i
                name_rf='A5_RF_%04i.WFM' %i
                
                read_x = Waveform( name_i,[Idle(t_2)] + p['pi2_1 + 0'] + [Idle(200)], t0)
                read_y = Waveform( name_q,[Idle(t_2)] + p['pi2_1 + 90'] + [Idle(200)], t0)
                read_rf = Waveform( name_rf,[Idle(t_2 + pi2_1 + 200)], t0)
                self.waves.append(read_x )
                self.waves.append(read_y )
                self.waves.append(read_rf)
                
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x,sup_y,sup_rf)
                sub_seq.append(nuflip_x, nuflip_y, nuflip_rf)
                if repeat_1 > 0:
                    sub_seq.append(evo, evo, evo, repeat = repeat_1)
                sub_seq.append(pi_echo_x, pi_echo_y, pi_echo_rf)
                sub_seq.append(nuflip_x_2, nuflip_y_2, nuflip_rf_2)
                if repeat_2 > 0:
                    sub_seq.append(evo, evo, evo, repeat = repeat_2)
                sub_seq.append(read_x, read_y, read_rf)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)
                
                name_i='B5_I_%04i.WFM' %i
                name_q='B5_Q_%04i.WFM' %i
                name_rf='B5_RF_%04i.WFM' %i
                
                read_x1 = Waveform( name_i,[Idle(t_2)] + p['pi2_1 - 0'] + [Idle(200)], t0)
                read_y1 = Waveform( name_q,[Idle(t_2)] + p['pi2_1 - 90'] + [Idle(200)], t0)
                read_rf1 = Waveform( name_rf,[Idle(t_2 + pi2_1 + 200)], t0)
                self.waves.append(read_x1 )
                self.waves.append(read_y1 )
                self.waves.append(read_rf1)

                
                name = 'BSQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x,sup_y,sup_rf)
                sub_seq.append(nuflip_x, nuflip_y, nuflip_rf)
                if repeat_1 > 0:
                    sub_seq.append(evo, evo, evo, repeat = repeat_1)
                sub_seq.append(pi_echo_x, pi_echo_y, pi_echo_rf)
                sub_seq.append(nuflip_x_2, nuflip_y_2, nuflip_rf_2)
                if repeat_2 > 0:
                    sub_seq.append(evo, evo, evo, repeat = repeat_2)
                sub_seq.append(read_x1, read_y1, read_rf1)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)


            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('Hahn_AC_phase.SEQ')
            
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )  
        
        
    def _get_sequence_points(self):
        return 2 * len(self.tau)       

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        wait_time = self.wait_time
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        tau_echo = self.tau_echo
        sequence = []
        
        for i, t in enumerate(tau):
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , pi2_1 * 2 + pi_1 + 2 * tau_echo + 2000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait+wait_time) )
            
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , pi2_1 * 2 + pi_1 + 2 * tau_echo + 2000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait+wait_time) )

        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','pi_1','tau_echo','wait_time','amp_rf','N_period','freq_rf']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                  
                                     ),                         
                              HGroup(Item('pi2_1', width = -60),
                                     Item('pi_1', width = -60),
                                     Item('tau_echo', width = -60),
                                     ),      
                               HGroup(Item('N_period', width = -60),
                                     Item('amp_rf', width = -60),
                                     Item('freq_rf', width = -60),
                                     Item('wait_time', width = -60),
                                     ),             
                              HGroup(Item('laser', width=30),
                                     Item('wait', width=30),
                                     Item('bin_width', width= 30, enabled_when='state != "run"'),
                                     Item('record_length', width= 30, enabled_when='state != "run"'),
                                     ),
                                     
                              HGroup(Item('tau_begin', width=-60),
                                     Item('tau_end', width=-60),
                                     Item('tau_delta', width= -60),
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30),
                                     ),
                                                                         
                              ),
                       title='Hahn_AC_phase',
                       )                          
                       
class Correlation_Nuclear_Rabi_2(Pulsed):                   

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
        
    rf_freq = Range(low=1, high=20e9, value=7.2e6, desc='frequency rf [Hz]', label='rf freq [Hz]', mode='text', auto_set=False, enter_set=True)    
    amp_rf = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM rf amp', mode='text', auto_set=False, enter_set=True)
    
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    
    # tau range for nuclear Rabi oscillations
    tau_begin = Range(low=800., high=1e8, value=1000., desc='tau_evo begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau_evo end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau_evo [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    
    tau_inte = Range(low=1., high=1e5, value=30., desc='tau for signal integration [ns]', label='tau integrate[ns]', mode='text', auto_set=False, enter_set=True)
    phase_flag = Bool(False, label='phase_flag')
       
    pulse_num = Range(low=1, high=1000, value=1, desc='repetition of XY8 pulses', label='repetitions', mode='text', auto_set=False, enter_set=True)
    reload = True       
  
    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            rf_freq = self.rf_freq
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)
            
            # Pulses
            p = {}
                        
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 + 180']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            p['pi2_1 + 270']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            p['pi2_1 - 180']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            p['pi2_1 - 270']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*2 ,self.amp)]
            
            p['pi_1_x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1_x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            if self.phase_flag:
                p['read + 0'] = p['pi2_1 + 0']
                p['read + 90'] = p['pi2_1 + 90']
                
                p['read + 90'] = p['pi2_1 - 0']
                p['read + 180'] = p['pi2_1 - 90']
            else:
                p['read + 0'] = p['pi2_1 - 0']
                p['read + 90'] = p['pi2_1 - 90']
                
                p['read + 90'] = p['pi2_1 + 0']
                p['read + 180'] = p['pi2_1 + 90']
            
            # Waveforms
            self.waves = []
            sub_seq = []

            self.main_seq = Sequence('Correlation_Nucl_Rabi.SEQ')
            
            sup_x_ref = Waveform('Sup_x_ref', p['pi2_1 + 0'])
            sup_y_ref = Waveform('Sup_y_ref', p['pi2_1 + 90'])
            t_sup_ref = sup_x_ref.stub
            
            sup_x = Waveform('Sup_x', [Idle(t_sup_ref)]+p['pi2_1 + 0'])
            sup_y = Waveform('Sup_y', [Idle(t_sup_ref)]+p['pi2_1 + 90'])
            sup_rf= Waveform('Sup_rf', [Idle(sup_x.duration)])
            
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            self.waves.append(sup_rf)
                        
            for i, t in enumerate(self.tau):
           
                name = 'SQH_12_%04i.SEQ' % i
                
                sub_seq=Sequence(name)       
                
                # signal integration 
                t_tau = self.tau_inte * 1.2 * 2
                sub_seq.append(sup_x,sup_y, sup_rf)
                
                t_0 = sup_x.duration #duration of two pi/2 pulses
                t_1 = self.tau_inte * 1.2 # half signal integration
                
                for k in range(self.pulse_num/2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % k
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % k
                    rf_name= 'RF_XY8_%03i' % i + '_%03i.WFM' % k
                    
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                                                
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                                                
                    map_rf_1 = Waveform(rf_name, [Idle(map_x_1.duration)],t_0)
                                                
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    self.waves.append(map_rf_1)
                    sub_seq.append(map_x_1,map_y_1, map_rf_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                for k in range(self.pulse_num%2):
                    x_name =  'X_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    y_name =  'Y_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    rf_name = 'RF_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                                                
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0) 
                                                
                    map_rf_1 = Waveform(rf_name, [Idle(map_x_1.duration)],t_0)                            
                                                
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    self.waves.append(map_rf_1)
                    sub_seq.append(map_x_1,map_y_1, map_rf_1)
                                      
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub    
                    
                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                
                name_x = 'Read_x_%04i.WFM' % i
                name_y = 'Read_y_%04i.WFM' % i
                name_rf = 'Read_rf_%04i.WFM' % i
                
                ref_x = Waveform(name_x, [mod] + p['read + 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['read + 90'],t_0)
                ref_rf = Waveform(name_rf, [Idle(ref_x.duration)], t_0)
                
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                self.waves.append(ref_rf)
                sub_seq.append(ref_x, ref_y, ref_rf)
                
                #evolution & RF Rabi pulses
                
                name_rf='A2_RF_%04i.WFM' %i
                name_x='EVO_1_%04i.WFM' %i
                name_y='EVO_2_%04i.WFM' %i                
                                
                t_evo = 1.2 * t
                t_2 = t_evo - ref_x.stub - 256
                repeat_1 = int(t_2 / 256)
                 
                t_1 = int(t_2 % 256)
                t_0 += ref_x.duration + repeat_1 * 256
                
                if repeat_1 > 0:
                      evo_1 = Waveform(name_x, Idle(256*repeat_1))
                      evo_2 = Waveform(name_y, Idle(256*repeat_1))
                      rf = Waveform( name_rf, Sin(256*repeat_1, self.rf_freq/sampling, 0, self.amp_rf))
                    
                      self.waves.append(evo_1)
                      self.waves.append(evo_2)
                      self.waves.append(rf)
                      sub_seq.append(evo_1, evo_2, rf)  #repeat evoluiton repeat_1 times
                                 
                # signal integration again
                x_name = 'super_x_%03i' % i
                y_name = 'super_y_%03i' % i
                rf_name = 'super_rf_%03i' % i
                
                sup_x_2 = Waveform(x_name, [Idle(256 + t_1)]+p['read + 90'], t_0)
                sup_y_2 = Waveform(y_name, [Idle(256 + t_1)]+p['read + 180'], t_0)
                
                sup_rf_2 = Waveform(rf_name, [Idle(sup_x_2.duration)], t_0)
                self.waves.append(sup_x_2)
                self.waves.append(sup_y_2)
                self.waves.append(sup_rf_2)
                sub_seq.append(sup_x_2, sup_y_2, sup_rf_2)

                t_0 += sup_x_2.duration
                t_1 = self.tau_inte * 1.2 - sup_x_2.stub
                for k in range(self.pulse_num/2):
                    x_name = 'X_XY4_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    y_name = 'Y_XY4_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    rf_name = 'rf_XY4_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    
                    map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    map_rf_1=Waveform(rf_name, [Idle(map_x_1.duration)], t_0)
                    
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    self.waves.append(map_rf_1)
                    sub_seq.append(map_x_1,map_y_1, map_rf_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                for k in range(self.pulse_num%2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % self.pulse_num*2
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % self.pulse_num*2
                    rf_name = 'RF_XY8_%03i' % i + '_%03i.WFM' % self.pulse_num*2
                    
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0) 
                    mep_rf_1= Waveform(rf_name, [Idle(map_x_1.duration)], t_0)
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    self.waves.append(map_rf_1)
                    sub_seq.append(map_x_1,map_y_1, map_rf_1)
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub        

                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                name_x = 'Read1_x_%04i.WFM' % i
                name_y = 'Read1_y_%04i.WFM' % i
                name_rf = 'Read1_rf_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['read + 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['read + 90'],t_0)
                ref_rf = Waveform(name_rf, [Idle(ref_x.duration)], t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                self.waves.append(ref_rf)
                sub_seq.append(ref_x,ref_y, ref_rf)
                
                
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)


            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            #time.sleep(70)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('Correlation_Nucl_Rabi.SEQ')
            
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )  
             
    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.previous_time = 0
            #self.previous_count_data = FC.GetData()
            self.prepare_awg()
            
            MW.setFrequency(self.freq)
            MW.setPower(self.power)
           
            AWG.run()
            time.sleep(30.0)
            FC.Start()
            time.sleep(0.1)
            PG.Sequence(self.sequence, loop=True)

            start_time = time.time()

            while True:
               self.thread.stop_request.wait(5.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               self.elapsed_time = self.previous_elapsed_time + time.time() - start_time
               self.run_time += self.elapsed_time
               runtime, cycles = FC.GetState()
               sweeps = cycles / FC.GetData().shape[0]
               self.elapsed_sweeps = self.previous_sweeps + sweeps
               self.progress = int( 100 * self.elapsed_sweeps / self.sweeps ) 
               self.count_data = self.old_count_data + FC.GetData()
               if self.elapsed_sweeps > self.sweeps:
                  break

            FC.Halt()
            time.sleep(0.1)
            MW.Off()
            #ha.RFSource().setOutput(None, self.rf_frequency)
            time.sleep(0.2)
            PG.High(['laser', 'mw'])
            time.sleep(0.2)
            AWG.stop()
            time.sleep(1)
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  
            
        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'    
        
    def generate_sequence(self):
    
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        tau_1 = self.tau_inte
        rf_freq = self.rf_freq
        sequence = []
        
        for i, t in enumerate(tau):
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , t + tau_1 * 2 * 8 * self.pulse_num * 2 + 4 * pi2_1 + 16*self.pulse_num*pi_1 + 4000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
          
        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','pi_1','tau_inte','pulse_num','phase_flag', 'rf_freq','amp_rf',]   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                     ),                         
                              HGroup(Item('phase_flag', width = -40),
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40),
                                     Item('tau_inte', width = -40),
                                     ),   
                              HGroup(Item('rf_freq', width=-60),
                                     Item('amp_rf', width=-40)),                                     
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
                       title='Correlation_Nuclear_Rabi',
                       ) 
                       
class Correlation_Nuclear_Rabi(Pulsed):                   

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 12, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
        
    rf_freq = Range(low=1, high=20e9, value=7.2e6, desc='frequency rf [Hz]', label='rf freq [Hz]', mode='text', auto_set=False, enter_set=True)    
    amp_rf = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM rf amp', mode='text', auto_set=False, enter_set=True)
    
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    
    # tau range for nuclear Rabi oscillations
    tau_begin = Range(low=800., high=1e8, value=1000., desc='tau_evo begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau_evo end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau_evo [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    
    tau_inte = Range(low=1., high=1e5, value=30., desc='tau for signal integration [ns]', label='tau integrate[ns]', mode='text', auto_set=False, enter_set=True)
    phase_flag = Bool(False, label='phase_flag')
       
    pulse_num = Range(low=1, high=1000, value=1, desc='repetition of XY8 pulses', label='repetitions', mode='text', auto_set=False, enter_set=True)
    reload = True       
  
    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            rf_freq = self.rf_freq
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)
            
            # Pulses
            p = {}
                        
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi2_1 + 180']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            p['pi2_1 + 270']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            p['pi2_1 - 180']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp)]
            p['pi2_1 - 270']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*2 ,self.amp)]
            
            p['pi_1_x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
            p['pi_1_x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp)]
            
            if self.phase_flag:
                p['read + 0'] = p['pi2_1 + 0']
                p['read + 90'] = p['pi2_1 + 90']
                
                p['read + 90'] = p['pi2_1 - 0']
                p['read + 180'] = p['pi2_1 - 90']
            else:
                p['read + 0'] = p['pi2_1 - 0']
                p['read + 90'] = p['pi2_1 - 90']
                
                p['read + 90'] = p['pi2_1 + 0']
                p['read + 180'] = p['pi2_1 + 90']
            
            # Waveforms
            self.waves = []
            sub_seq = []

            self.main_seq = Sequence('Correlation_Nucl_Rabi.SEQ')
            
            sup_x_ref = Waveform('Sup_x_ref', p['pi2_1 + 0'])
            sup_y_ref = Waveform('Sup_y_ref', p['pi2_1 + 90'])
            t_sup_ref = sup_x_ref.stub
            
            sup_x = Waveform('Sup_x', [Idle(t_sup_ref)]+p['pi2_1 + 0'])
            sup_y = Waveform('Sup_y', [Idle(t_sup_ref)]+p['pi2_1 + 90'])
            sup_rf= Waveform('Sup_rf', [Idle(sup_x.duration)])
            
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            self.waves.append(sup_rf)
                        
            for i, t in enumerate(self.tau):
           
                name = 'SQH_12_%04i.SEQ' % i
                
                sub_seq=Sequence(name)       
                
                # signal integration 
                t_tau = self.tau_inte * 1.2 * 2
                sub_seq.append(sup_x,sup_y, sup_rf)
                
                t_0 = sup_x.duration #duration of two pi/2 pulses
                t_1 = self.tau_inte * 1.2 # half signal integration
                
                for k in range(self.pulse_num/2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % k
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % k
                    rf_name= 'RF_XY8_%03i' % i + '_%03i.WFM' % k
                    
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                                                
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                                                
                    map_rf_1 = Waveform(rf_name, [Idle(map_x_1.duration)],t_0)
                                                
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    self.waves.append(map_rf_1)
                    sub_seq.append(map_x_1,map_y_1, map_rf_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                for k in range(self.pulse_num%2):
                    x_name =  'X_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    y_name =  'Y_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    rf_name = 'RF_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                                                
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0) 
                                                
                    map_rf_1 = Waveform(rf_name, [Idle(map_x_1.duration)],t_0)                            
                                                
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    self.waves.append(map_rf_1)
                    sub_seq.append(map_x_1,map_y_1, map_rf_1)
                                      
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub    
                    
                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                
                name_x = 'Read_x_%04i.WFM' % i
                name_y = 'Read_y_%04i.WFM' % i
                name_rf = 'Read_rf_%04i.WFM' % i
                
                ref_x = Waveform(name_x, [mod] + p['read + 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['read + 90'],t_0)
                ref_rf = Waveform(name_rf, [Idle(ref_x.duration)], t_0)
                
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                self.waves.append(ref_rf)
                sub_seq.append(ref_x, ref_y, ref_rf)
                
                #evolution & RF Rabi pulses
                
                name_rf='A2_RF_%04i.WFM' %i
                name_x='EVO_1_%04i.WFM' %i
                name_y='EVO_2_%04i.WFM' %i                
                                
                t_evo = 1.2 * t
                # t_2 = t_evo - ref_x.stub - 256
                # repeat_1 = int(t_2 / 256)
                 
                # t_1 = int(t_2 % 256)
                # t_0 += ref_x.duration + repeat_1 * 256 + 256 + t_1
                
                # if repeat_1 > 0:
                      # evo_1 = Waveform(name_x, Idle(256*repeat_1+256 + t_1))
                      # evo_2 = Waveform(name_y, Idle(256*repeat_1+256 + t_1))
                      # rf = Waveform( name_rf, Sin(256*repeat_1+256 + t_1, self.rf_freq/sampling, 0, self.amp_rf))
                    
                      # self.waves.append(evo_1)
                      # self.waves.append(evo_2)
                      # self.waves.append(rf)
                      # sub_seq.append(evo_1, evo_2, rf)  #repeat evoluiton repeat_1 times
                      
                t_0 += ref_x.duration
                                 
                evo_1 = Waveform(name_x, [Idle(t_evo)], t_0)
                evo_2 = Waveform(name_y, [Idle(t_evo)], t_0)
                rf = Waveform( name_rf, [Sin(t_evo, self.rf_freq/sampling, 0, self.amp_rf)], t_0)
                    
                self.waves.append(evo_1)
                self.waves.append(evo_2)
                self.waves.append(rf)
                sub_seq.append(evo_1, evo_2, rf)  #repeat evoluiton repeat_1 times      
                t_0 += rf.duration
                
                # signal integration again
                x_name = 'super_x_%03i' % i
                y_name = 'super_y_%03i' % i
                rf_name = 'super_rf_%03i' % i
                
                sup_x_2 = Waveform(x_name, p['read + 90'], t_0)
                sup_y_2 = Waveform(y_name, p['read + 180'], t_0)
                sup_rf_2 = Waveform(rf_name, Idle(sup_x_2.duration), t_0)
                
                self.waves.append(sup_x_2)
                self.waves.append(sup_y_2)
                self.waves.append(sup_rf_2)
                sub_seq.append(sup_x_2, sup_y_2, sup_rf_2)

                t_0 += sup_x_2.duration
                t_1 = self.tau_inte * 1.2 - sup_x_2.stub
                for k in range(self.pulse_num/2):
                    x_name = 'X_XY4_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    y_name = 'Y_XY4_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    rf_name = 'rf_XY4_%03i' % i + '_%03i.WFM' % (k+self.pulse_num)
                    
                    map_x_1 = Waveform(x_name, [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    map_rf_1=Waveform(rf_name, [Idle(map_x_1.duration)], t_0)
                    
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    self.waves.append(map_rf_1)
                    sub_seq.append(map_x_1,map_y_1, map_rf_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                for k in range(self.pulse_num%2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % self.pulse_num*2
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % self.pulse_num*2
                    rf_name = 'RF_XY8_%03i' % i + '_%03i.WFM' % self.pulse_num*2
                    
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0) 
                    mep_rf_1= Waveform(rf_name, [Idle(map_x_1.duration)], t_0)
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    self.waves.append(map_rf_1)
                    sub_seq.append(map_x_1,map_y_1, map_rf_1)
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub        

                mod.duration = self.tau_inte * 1.2 - map_x_1.stub
                name_x = 'Read1_x_%04i.WFM' % i
                name_y = 'Read1_y_%04i.WFM' % i
                name_rf = 'Read1_rf_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['read + 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['read + 90'],t_0)
                ref_rf = Waveform(name_rf, [Idle(ref_x.duration)], t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                self.waves.append(ref_rf)
                sub_seq.append(ref_x,ref_y, ref_rf)
                
                
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)


            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            #time.sleep(70)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('Correlation_Nucl_Rabi.SEQ')
            
        AWG.set_vpp(0.6)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )  
             
    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.previous_time = 0
            #self.previous_count_data = FC.GetData()
            self.prepare_awg()
            
            MW.setFrequency(self.freq)
            MW.setPower(self.power)
           
            AWG.run()
            time.sleep(30.0)
            FC.Start()
            time.sleep(0.1)
            PG.Sequence(self.sequence, loop=True)

            start_time = time.time()

            while True:
               self.thread.stop_request.wait(5.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               self.elapsed_time = self.previous_elapsed_time + time.time() - start_time
               self.run_time += self.elapsed_time
               runtime, cycles = FC.GetState()
               sweeps = cycles / FC.GetData().shape[0]
               self.elapsed_sweeps = self.previous_sweeps + sweeps
               self.progress = int( 100 * self.elapsed_sweeps / self.sweeps ) 
               self.count_data = self.old_count_data + FC.GetData()
               if self.elapsed_sweeps > self.sweeps:
                  break

            FC.Halt()
            time.sleep(0.1)
            MW.Off()
            #ha.RFSource().setOutput(None, self.rf_frequency)
            time.sleep(0.2)
            PG.High(['laser', 'mw'])
            time.sleep(0.2)
            AWG.stop()
            time.sleep(1)
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  
            
        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'    
        
    def generate_sequence(self):
    
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        tau_1 = self.tau_inte
        rf_freq = self.rf_freq
        sequence = []
        
        for i, t in enumerate(tau):
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , t + tau_1 * 2 * 8 * self.pulse_num * 2 + 4 * pi2_1 + 16*self.pulse_num*pi_1 + 4000) )
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
          
        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['freq','vpp','amp','pi2_1','pi_1','tau_inte','pulse_num','phase_flag', 'rf_freq','amp_rf',]   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp', width=-40),   
                                     ),                         
                              HGroup(Item('phase_flag', width = -40),
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40),
                                     Item('tau_inte', width = -40),
                                     ),   
                              HGroup(Item('rf_freq', width=-60),
                                     Item('amp_rf', width=-40)),                                     
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
                       title='Correlation_Nuclear_Rabi_2',
                       ) 