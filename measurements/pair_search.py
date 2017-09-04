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
    
class Rabi( Pulsed ):
    """Rabi measurement.
    """ 
    #def _init_(self):
        #super(Rabi, self).__init__()
        
    reload = True
  
    freq = Range(low=1, high=20e9, value=2.71e9, desc='frequency [Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_2 = Range(low=1, high=20e9, value=2.71e9, desc='frequency [Hz]', label='freq2 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_3 = Range(low=1, high=20e9, value=2.71e9, desc='frequency [Hz]', label='freq3 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)
    multirabi = Bool(False, label='single or multi freq rabi',desc = 'false, single freq rabi')
    nfreq = Range(low=2, high=3, value=3, desc='number of freq', label='nfreq', mode='text', auto_set=False, enter_set=True)
    
    
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
                if self.multirabi:
                    if self.nfreq == 3:
                        drive_x = Sin(t, (self.freq - self.freq_center)/sampling, 0 ,self.amp/3) + Sin(t, (self.freq_2 - self.freq_center)/sampling, 0 ,self.amp/3) + Sin(t, (self.freq_3 - self.freq_center)/sampling, 0 ,self.amp/3)
                        drive_y = Sin(t, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp/3) + Sin(t, (self.freq_2 - self.freq_center)/sampling, np.pi/2 ,self.amp/3) + Sin(t, (self.freq_3 - self.freq_center)/sampling, np.pi/2 ,self.amp/3)
                    else:
                        drive_x = Sin(t, (self.freq - self.freq_center)/sampling, 0 ,self.amp/2) + Sin(t, (self.freq_2 - self.freq_center)/sampling, 0 ,self.amp/2) 
                        drive_y = Sin(t, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp/2) + Sin(t, (self.freq_2 - self.freq_center)/sampling, np.pi/2 ,self.amp/2)
                else:
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
        #sequence.append(  ([                   ] , 12.5  )  )
        return sequence
        
    get_set_items = Pulsed.get_set_items + ['freq','freq_2','freq_3','amp','vpp','nfreq','multirabi']    
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=20),
                                     Item('freq_2',  width=20),
                                     Item('freq_3',  width=20),
                                     Item('multirabi',  width=20),
                                     Item('nfreq',  width=20),
                                     ),       
                             HGroup( Item('amp', width=20),
                                     Item('vpp', width=20),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=20),
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
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of pi pulse [ns]', label= 'minus pi', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.71e9, desc='frequency plus [Hz]', label='freq1 p [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_2 = Range(low=1, high=20e9, value=2.71e9, desc='frequency plus[Hz]', label='freq2 p[Hz]', mode='text', auto_set=False, enter_set=True)
    freq_3 = Range(low=1, high=20e9, value=2.71e9, desc='frequency plus[Hz]', label='freq3 p[Hz]', mode='text', auto_set=False, enter_set=True)
    freq_4 = Range(low=1, high=20e9, value=2.71e9, desc='frequency minus[Hz]', label='freq1 m[Hz]', mode='text', auto_set=False, enter_set=True)
    freq_5 = Range(low=1, high=20e9, value=2.71e9, desc='frequency minus[Hz]', label='freq2 m[Hz]', mode='text', auto_set=False, enter_set=True)
    freq_6 = Range(low=1, high=20e9, value=2.71e9, desc='frequency minus[Hz]', label='freq3 m[Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    multifid = Bool(False, label='single or double FID',desc = 'false, single FID')
    multifreq = Bool(False, label='single or multi freq for FID',desc = 'false, single freq')
    nfreq = Range(low=2, high=3, value=3, desc='number of freq', label='nfreq', mode='text', auto_set=False, enter_set=True)
    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            pi2_1 = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)   

            # Pulses
            p = {}
            
            if self.multifreq:
                if self.nfreq == 2:
                    p['pi2_1 + 0'] = Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp/2) + Sin( pi2_1, (self.freq_2 - self.freq_center)/sampling, 0 ,self.amp/2)
                    p['pi2_1 + 90']    = Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp/2) + Sin( pi2_1, (self.freq_2 - self.freq_center)/sampling, np.pi/2 ,self.amp/2)
                    
                    p['pi_1 - 0'] = Sin( pi_1, (self.freq_4 - self.freq_center)/sampling, 0 ,self.amp/2) + Sin( pi_1, (self.freq_5 - self.freq_center)/sampling, 0 ,self.amp/2)
                    p['pi_1 - 90']    = Sin( pi_1, (self.freq_4 - self.freq_center)/sampling, np.pi/2 ,self.amp/2) + Sin( pi_1, (self.freq_5 - self.freq_center)/sampling, np.pi/2 ,self.amp/2)
                else:   
                    p['pi2_1 + 0'] = Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp/3) + Sin( pi2_1, (self.freq_2 - self.freq_center)/sampling, 0 ,self.amp/3) + Sin( pi2_1, (self.freq_3 - self.freq_center)/sampling, 0 ,self.amp/3)
                    p['pi2_1 + 90']    = Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp/3) + Sin( pi2_1, (self.freq_2 - self.freq_center)/sampling, np.pi/2 ,self.amp/3) + Sin( pi2_1, (self.freq_3 - self.freq_center)/sampling, 0 ,self.amp/3)
                    
                    p['pi_1 - 0'] = Sin( pi_1, (self.freq_4 - self.freq_center)/sampling, 0 ,self.amp/3) + Sin( pi_1, (self.freq_5 - self.freq_center)/sampling, 0 ,self.amp/3) + Sin( pi_1, (self.freq_6 - self.freq_center)/sampling, 0 ,self.amp/3)
                    p['pi_1 - 90']    = Sin( pi_1, (self.freq_4 - self.freq_center)/sampling, np.pi/2 ,self.amp/3) + Sin( pi_1, (self.freq_5 - self.freq_center)/sampling, np.pi/2 ,self.amp/3) + Sin( pi_1, (self.freq_6 - self.freq_center)/sampling, np.pi/2 ,self.amp/3)
            else:
            # ms= 0 <> ms = +1
                p['pi2_1 + 0']     = Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,1)
                p['pi2_1 + 90']    = Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,1)
            
                p['pi_1 - 0']     = Sin( pi_1, (self.freq_4 - self.freq_center)/sampling, 0 ,1)
                p['pi_1 - 90']    = Sin( pi_1, (self.freq_4 - self.freq_center)/sampling, np.pi/2 ,1)
            
            
            zero = Idle(1)
            mod = Idle(0)
            
            if self.multifid:
                pi2_p_i = [zero, p['pi2_1 + 0'], p['pi_1 - 0'], zero]
                pi2_p_q = [zero, p['pi2_1 + 90'], p['pi_1 - 90'], zero]
                
                pi2_r_i = [zero, p['pi_1 - 0'], p['pi2_1 + 0'],  zero]
                pi2_r_q = [zero, p['pi_1 - 90'], p['pi2_1 + 90'],  zero]
            else:
                pi2_p_i = [zero,p['pi2_1 + 0'],zero]
                pi2_p_q = [zero,p['pi2_1 + 90'],zero]
            
                pi2_r_i = [zero,p['pi2_1 + 0'],zero]
                pi2_r_q = [zero,p['pi2_1 + 90'],zero]
                  
            
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

                map_x = Waveform(name_x, [mod]+pi2_r_i, t_0)
                map_y = Waveform(name_y, [mod]+pi2_r_q , t_0)
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                name = 'DQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                if(repeat_1 == 0):
                    sub_seq.append(sup_x, sup_y)
                    sub_seq.append(map_x, map_y)
                    AWG.upload(sub_seq)
                else:
                    sub_seq.append(sup_x, sup_y)
                    sub_seq.append(evo, evo,repeat=repeat_1)
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
        AWG.set_output( 0b0111 )       
        
    #def _get_sequence_points(self):
        #return 2 * len(self.tau)
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        pi2_1=self.pi2_1
        sequence = []
        for t in tau:
            sub = [ (['awgTrigger'], 100 ),
                    ([], 15000 + t + 500 ),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)
        return sequence
        
    get_set_items = Pulsed.get_set_items + ['freq','freq_2','freq_3','freq_4','freq_5','freq_6','vpp','pi2_1','pi_1','amp','nfreq','multifreq','multifid'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_2',  width=-60),
                                     Item('freq_3',  width=-60),
                                    ),
                              HGroup(
                                     Item('freq_4',  width=-60),
                                     Item('freq_5',  width=-60),
                                     Item('freq_6',  width=-60),
                                    ),       
                              HGroup(      
                                     Item('pi2_1', width=-40),
                                     Item('pi_1', width=-40),
                                     Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     ),             
                              HGroup(      
                                     Item('multifreq',  width=20),
                                     Item('multifid',  width=20),
                                     Item('nfreq', width=-40),
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
class Hahn( Pulsed ):
    #FID 
    
    pi2_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='pi plus', mode='text', auto_set=False, enter_set=True)
    pi_2 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='pi minus', mode='text', auto_set=False, enter_set=True)
    
    freq_center = Range(low=1, high=20e9, value=2.61e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_2 = Range(low=1, high=20e9, value=2.845829e9, desc='frequency 2nd trans [Hz]', label='freq2 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_3 = Range(low=1, high=20e9, value=2.907644e9, desc='frequency 3nd trans [Hz]', label='freq3 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_4 = Range(low=1, high=20e9, value=2.955537e9, desc='frequency 4th trans [Hz]', label='freq4 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_5 = Range(low=1, high=20e9, value=2.907644e9, desc='frequency 5nd trans [Hz]', label='freq5 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_6 = Range(low=1, high=20e9, value=2.955537e9, desc='frequency 6th trans [Hz]', label='freq6 [Hz]', mode='text', auto_set=False, enter_set=True)
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)   
    
    multihahn = Bool(False, label='single or double Hahn',desc = 'false, single coherence Hahn')
    multifreq = Bool(False, label='single or multi freq for Hahn',desc = 'false, single freq')
    nfreq = Range(low=2, high=3, value=3, desc='number of freq', label='nfreq', mode='text', auto_set=False, enter_set=True)
    
    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            p = {}
            
            pi2_1 = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            pi_2 = int(self.pi_2 * sampling/1.0e9)
            
            f1 = (self.freq - self.freq_center)/sampling
            f2 = (self.freq_2 - self.freq_center)/sampling
            f3= (self.freq_3 - self.freq_center)/sampling
            f4 = (self.freq_4 - self.freq_center)/sampling
            f5= (self.freq_5 - self.freq_center)/sampling
            f6 = (self.freq_6 - self.freq_center)/sampling
            
            if self.multifreq:
                if self.nfreq == 2:
                    p['pi2_1 + 0'] = [Sin( pi2_1, f1, 0 ,self.amp/2) + Sin( pi2_1, f2, 0 ,self.amp/2)]
                    p['pi2_1 + 90']    = [Sin( pi2_1, f1, np.pi/2 ,self.amp/2) + Sin( pi2_1, f2, np.pi/2 ,self.amp/2)]
                    
                    p['pi_1 + 0'] = [Sin( pi_1, f1, 0 ,self.amp/2) + Sin( pi_1, f2, 0 ,self.amp/2)]
                    p['pi_1 + 90']    = [Sin( pi_1, f1, np.pi/2 ,self.amp/2) + Sin( pi_1, f2, np.pi/2 ,self.amp/2)]
                    
                    p['pi_2 - 0'] = [Sin( pi_2, f4, 0 ,self.amp/2) + Sin( pi_2, f5, 0 ,self.amp/2)]
                    p['pi_2 - 90']    = [Sin( pi_2, f4, np.pi/2, self.amp/2) + Sin( pi_2, f5, np.pi/2, self.amp/2)]
                else:   
                    p['pi2_1 + 0'] = [Sin( pi2_1, f1, 0 ,self.amp/3) + Sin( pi2_1, f2, 0 ,self.amp/3) + Sin( pi2_1, f3, 0 ,self.amp/3)]
                    p['pi2_1 + 90']    = [Sin( pi2_1, f1, np.pi/2 ,self.amp/3) + Sin( pi2_1, f2, np.pi/2, self.amp/3) + Sin( pi2_1, f3, np.pi/2 ,self.amp/3)]
                    
                    p['pi_1 + 0'] = [Sin( pi_1, f1, 0 ,self.amp/3) + Sin( pi_1, f2 ,0 ,self.amp/3) + Sin( pi_1, f3, 0 ,self.amp/3)]
                    p['pi_1 + 90']    = [Sin( pi_1, f1, np.pi/2 ,self.amp/3) + Sin( pi_1, f2, np.pi/2 ,self.amp/3) + Sin( pi_1, f3, np.pi/2 ,self.amp/3)]
                    
                    p['pi_2 - 0'] = [Sin( pi_2, f4, 0 ,self.amp/3) + Sin( pi_2, f5, 0 ,self.amp/3) + Sin( pi_2, f6, 0 ,self.amp/3)]
                    p['pi_2 - 90']    = [Sin( pi_2, f4, np.pi/2 ,self.amp/3) + Sin( pi_2, f5, np.pi/2 ,self.amp/3) + Sin( pi_2, f6, np.pi/2 ,self.amp/3)]
            else:
            # ms= 0 <> ms = +1
                p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
                p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
                
                p['pi_1 + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp)]
                p['pi_1 + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp)]
            
                p['pi_2 - 0']     = [Sin( pi_2, (self.freq_4 - self.freq_center)/sampling, 0 ,self.amp)]
                p['pi_2 - 90']    = [Sin( pi_2, (self.freq_4 - self.freq_center)/sampling, np.pi/2 ,self.amp)]

            
            zero = Idle(1)
            mod = Idle(0)
            
            if self.multihahn:
            
                p['initial + 0'] =  p['pi2_1 + 0'] + p['pi_2 - 0']
                p['initial + 90'] = p['pi2_1 + 90'] + p['pi_2 - 90']
                
                p['pi + 0'] = p['pi_1 + 0'] + p['pi_2 - 0'] + p['pi_1 + 0']
                p['pi + 90'] = p['pi_1 + 90'] + p['pi_2 - 90'] + p['pi_1 + 90']
                
                p['read + 0'] =  p['pi_2 - 0'] + p['pi2_1 + 0']
                p['read + 90'] = p['pi_2 - 90'] + p['pi2_1 + 90'] 
            else:
                p['initial + 0'] =  p['pi2_1 + 0']
                p['initial + 90'] = p['pi2_1 + 90']
                
                p['pi + 0'] = p['pi_1 + 0']
                p['pi + 90'] = p['pi_1 + 90']
                
                p['read + 0'] = p['pi2_1 + 0']
                p['read + 90'] = p['pi2_1 + 90'] 
                
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
                
                name = 'DQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x, sup_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(map_x, map_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(ref_x, ref_y)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq,wait=True)

            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('Hahn.SEQ')
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
            sub = [ (['awgTrigger'], 100 ),
                    ([], 15000 + 2 * t + 500 ),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)

        return sequence
            
    get_set_items = Pulsed.get_set_items + ['freq','freq_2','freq_3','freq_4','freq_5','freq_6','vpp','amp','multifreq','multihahn','nfreq','pi2_1','pi_1','pi_2'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('amp', width=-40),
                                     Item('vpp', width=-40),                                     
                                     Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     ),       
                              HGroup(Item('freq',  width=-65),
                                     Item('freq_2',  width=-65),
                                     Item('freq_3',  width=-65),
                                     Item('freq_4',  width=-65),
                                     Item('freq_5',  width=-65),
                                     Item('freq_6',  width=-65),
                                     ), 
                             HGroup(Item('multifreq', width=20), 
                                    Item('multihahn', width=20), 
                                    Item('nfreq', width=20), 
                                    ),
                             HGroup(
                                    Item('pi2_1', width=20), 
                                    Item('pi_1', width=20), 
                                    Item('pi_2', width=20), 
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
                       
class DEERpair(Pulsed):                   

    
    freq_center = Range(low=1, high=20e9, value=2.61e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='nv1 frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_2 = Range(low=1, high=20e9, value=2.845829e9, desc='nv1 frequency 2nd trans [Hz]', label='freq2 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_3 = Range(low=1, high=20e9, value=2.907644e9, desc='nv1 frequency 3nd trans [Hz]', label='freq3 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_4 = Range(low=1, high=20e9, value=2.955537e9, desc='nv1 frequency 4th trans [Hz]', label='freq4 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_5 = Range(low=1, high=20e9, value=2.907644e9, desc='nv1 frequency 5nd trans [Hz]', label='freq5 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_6 = Range(low=1, high=20e9, value=2.955537e9, desc='nv1 frequency 6th trans [Hz]', label='freq6 [Hz]', mode='text', auto_set=False, enter_set=True)
    
    ffreq = Range(low=1, high=20e9, value=2.791676e9, desc='nv2 frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    ffreq_2 = Range(low=1, high=20e9, value=2.845829e9, desc='nv2 frequency 2nd trans [Hz]', label='freq2 [Hz]', mode='text', auto_set=False, enter_set=True)
    ffreq_3 = Range(low=1, high=20e9, value=2.907644e9, desc='nv2 frequency 3nd trans [Hz]', label='freq3 [Hz]', mode='text', auto_set=False, enter_set=True)
    ffreq_4 = Range(low=1, high=20e9, value=2.955537e9, desc='nv2 frequency 4th trans [Hz]', label='freq4 [Hz]', mode='text', auto_set=False, enter_set=True)
    ffreq_5 = Range(low=1, high=20e9, value=2.907644e9, desc='nv2 frequency 5nd trans [Hz]', label='freq5 [Hz]', mode='text', auto_set=False, enter_set=True)
    ffreq_6 = Range(low=1, high=20e9, value=2.955537e9, desc='nv2 frequency 6th trans [Hz]', label='freq6 [Hz]', mode='text', auto_set=False, enter_set=True)
    
    amp1 = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp1', mode='text', auto_set=False, enter_set=True)
    amp2 = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp2', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi nv1', mode='text', auto_set=False, enter_set=True)
    pi_1_p   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='plus pi nv1', mode='text', auto_set=False, enter_set=True)
    pi_1_m   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='minus pi nv1', mode='text', auto_set=False, enter_set=True)
    
    pi_2_p   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='plus pi nv2', mode='text', auto_set=False, enter_set=True)
    pi_2_m   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='minus pi nv2', mode='text', auto_set=False, enter_set=True)
    
    tau1 = Range(low=1, high=50e4, value=20e4, desc='first tau in hahn echo [ns]', label='tau1 [ns]', mode='text', auto_set=False, enter_set=True)
    tau_begin = Range(low=0., high=1e8, value=500., desc='tau begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau end [ns]', label='tau end < tau1 [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=1e8, value=50., desc='delta tau [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    
    multico1 = Bool(False, label='multico1',desc = 'false, 0 +1, trur, +1, -1')
    multico2 = Bool(False, label='multico2',desc = 'false, 0 +1, trur, +1, -1')
    nfreq1 = Range(low=2, high=3, value=3, desc='number of freq of nv1', label='nfreq1', mode='text', auto_set=False, enter_set=True)
    nfreq2 = Range(low=2, high=3, value=3, desc='number of freq of nv2', label='nfreq2', mode='text', auto_set=False, enter_set=True)
    multifreq1 = Bool(False, label='multifreq1',desc = 'false, single freq')
    multifreq2 = Bool(False, label='multifreq2',desc = 'false, single freq')
    
    reload = True
    
    def _init_(self):
        super(DEERpair, self).__init__()

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            
            f1 = (self.freq - self.freq_center)/sampling
            f2 = (self.freq_2 - self.freq_center)/sampling
            f3= (self.freq_3 - self.freq_center)/sampling
            f4 = (self.freq_4 - self.freq_center)/sampling
            f5= (self.freq_5 - self.freq_center)/sampling
            f6 = (self.freq_6 - self.freq_center)/sampling
            
            ff1 = (self.ffreq - self.freq_center)/sampling
            ff2 = (self.ffreq_2 - self.freq_center)/sampling
            ff3= (self.ffreq_3 - self.freq_center)/sampling
            ff4 = (self.ffreq_4 - self.freq_center)/sampling
            ff5= (self.ffreq_5 - self.freq_center)/sampling
            ff6 = (self.ffreq_6 - self.freq_center)/sampling
            
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1_p = int(self.pi_1_p * sampling/1.0e9)
            pi_1_m = int(self.pi_1_m * sampling/1.0e9)
            pi_2_p = int(self.pi_2_p * sampling/1.0e9)
            pi_2_m = int(self.pi_2_m * sampling/1.0e9)
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            
            if self.multifreq1:
                if self.nfreq1 == 2:
                    p['pi2_1_nv1 + 0']     = [Sin( pi2_1, f1, 0 , self.amp1/2) + Sin( pi2_1, f2, 0 ,self.amp1/2)]
                    p['pi2_1_nv1 + 90']    = [Sin( pi2_1, f1, np.pi/2 , self.amp1/2) + Sin( pi2_1, f2, np.pi/2 ,self.amp1/2)]
                    
                    p['pi_1_nv1 + 0']     = [Sin( pi_1_p, f1, 0 , self.amp1/2) + Sin( pi_1_p, f2, 0 ,self.amp1/2)]
                    p['pi_1_nv1 + 90']    = [Sin( pi_1_p, f1, np.pi/2 , self.amp1/2) + Sin( pi_1_p, f2, np.pi/2 ,self.amp1/2)]
                    
                    p['pi_1_nv1 - 0']     = [Sin( pi_1_m, f4, 0 , self.amp1/2) + Sin( pi_1_m, f5, 0 ,self.amp1/2) ] 
                    p['pi_1_nv1 - 90']    = [Sin( pi_1_m, f4, np.pi/2 , self.amp1/2) + Sin( pi_1_m, f5, np.pi/2 ,self.amp1/2)]
                else:
                    p['pi2_1_nv1 + 0']     = [Sin( pi2_1, f1, 0 , self.amp1/3) + Sin( pi2_1, f2, 0 ,self.amp1/3) + Sin( pi2_1, f3, 0 ,self.amp1/3)]
                    p['pi2_1_nv1 + 90']    = [Sin( pi2_1, f1, np.pi/2 , self.amp1/3) + Sin( pi2_1, f2, np.pi/2 ,self.amp1/3)  + Sin( pi2_1, f3, np.pi/2 ,self.amp1/3)]
                    
                    p['pi_1_nv1 + 0']     = [Sin( pi_1_p, f1, 0 , self.amp1/3) + Sin( pi_1_p, f2, 0 ,self.amp1/3) + Sin( pi_1_p, f3, 0 ,self.amp1/3)]
                    p['pi_1_nv1 + 90']    = [Sin( pi_1_p, f1, np.pi/2 , self.amp1/3) + Sin( pi_1_p, f2, np.pi/2 ,self.amp1/3) + Sin( pi_1_p, f3, np.pi/2 ,self.amp1/3)]
                    
                    p['pi_1_nv1 - 0']     = [Sin( pi_1_m, f4, 0 , self.amp1/3) + Sin( pi_1_m, f5, 0 ,self.amp1/3) + Sin( pi_1_m, f6, 0 ,self.amp1/3) ] 
                    p['pi_1_nv1 - 90']    = [Sin( pi_1_m, f4, np.pi/2 , self.amp1/3) + Sin( pi_1_m, f5, np.pi/2 ,self.amp1/3) + Sin( pi_1_m, f6, np.pi/2 ,self.amp1/3)]
                
                    p['pi_1_nv1 - 90']    = [Sin( pi_1_m, f4, np.pi/2 , self.amp1/3) + Sin( pi_1_m, f5, np.pi/2 ,self.amp1/3) + Sin( pi_1_m, f6, np.pi/2 ,self.amp1/3)]
            else:
                
                    p['pi2_1_nv1 + 0']     = [Sin( pi2_1, f1, 0 , self.amp1)]
                    p['pi2_1_nv1 + 90']    = [Sin( pi2_1, f1, np.pi/2 , self.amp1)]
                    
                    p['pi_1_nv1 + 0']     = [Sin( pi_1_p, f1, 0 , self.amp1)]
                    p['pi_1_nv1 + 90']    = [Sin( pi_1_p, f1, np.pi/2 , self.amp1)]
                    
                    p['pi_1_nv1 - 0']     = [Sin( pi_1_m, f4, 0 , self.amp1)] 
                    p['pi_1_nv1 - 90']    = [Sin( pi_1_m, f4, np.pi/2 , self.amp1)]    
            
            if self.multifreq2:            
                if self.nfreq2 == 2:
                    p['pi_1_nv2 + 0']     = [Sin( pi_2_p, ff1, 0 , self.amp2/2) + Sin( pi_2_p, ff2, 0 ,self.amp2/2)]
                    p['pi_1_nv2 + 90']    = [Sin( pi_2_p, ff1, np.pi/2 , self.amp2/2) + Sin( pi_2_p, ff2, np.pi/2 ,self.amp2/2)]
                    
                    p['pi_1_nv2 - 0']     = [Sin( pi_2_m, ff4, 0 , self.amp2/2) + Sin( pi_2_m, ff5, 0 ,self.amp2/2) ] 
                    p['pi_1_nv2 - 90']    = [Sin( pi_2_m, ff4, np.pi/2 , self.amp2/2) + Sin( pi_2_m, ff5, np.pi/2 ,self.amp2/2)]
                else:

                    p['pi_1_nv2 + 0']     = [Sin( pi_2_p, ff1, 0 , self.amp2/3) + Sin( pi_2_p, ff2, 0 ,self.amp2/3) + Sin( pi_2_p, ff3, 0 ,self.amp2/3)]
                    p['pi_1_nv2 + 90']    = [Sin( pi_2_p, ff1, np.pi/2 , self.amp2/3) + Sin( pi_2_p, ff2, np.pi/2 ,self.amp2/3) + Sin( pi_2_p, ff3, np.pi/2 ,self.amp2/3)]
                    
                    p['pi_1_nv2 - 0']     = [Sin( pi_2_m, ff4, 0 , self.amp2/3) + Sin( pi_2_m, ff5, 0 ,self.amp2/3) + Sin( pi_2_m, ff6, 0 ,self.amp2/3) ] 
                    p['pi_1_nv2 - 90']    = [Sin( pi_2_m, ff4, np.pi/2 , self.amp2/3) + Sin( pi_2_m, ff5, np.pi/2 ,self.amp2/3) + Sin( pi_2_m, ff6, np.pi/2 ,self.amp2/3)]   
            else:

                    p['pi_1_nv2 + 0']     = [Sin( pi_2_p, ff1, 0 , self.amp2)]
                    p['pi_1_nv2 + 90']    = [Sin( pi_2_p, ff1, np.pi/2 , self.amp2)]
                    
                    p['pi_1_nv2 - 0']     = [Sin( pi_2_m, ff4, 0 , self.amp2)] 
                    p['pi_1_nv2 - 90']    = [Sin( pi_2_m, ff4, np.pi/2 , self.amp2)]        
                
            if self.multico1:    
                if self.multico2:
                
                    p['initial + 0'] = p['pi_1_nv2 + 0']  + [Idle(10)] + p['pi2_1_nv1 + 0']  + [Idle(10)] + p['pi_1_nv1 - 0'] + [Idle(10)]
                    p['initial + 90'] = p['pi_1_nv2 + 90']  + [Idle(10)] + p['pi2_1_nv1 + 90']  + [Idle(10)] + p['pi_1_nv1 - 90'] + [Idle(10)]
                    
                    p['pi_nv1 + 0'] = p['pi_1_nv1 + 0'] + [Idle(10)] + p['pi_1_nv1 - 0'] + [Idle(10)] + p['pi_1_nv1 + 0'] + [Idle(10)]
                    p['pi_nv1 + 90'] = p['pi_1_nv1 + 90'] + [Idle(10)] + p['pi_1_nv1 - 90']  + [Idle(10)] + p['pi_1_nv1 + 90'] + [Idle(10)]
                    
                    p['pi_nv2 + 0'] = p['pi_1_nv2 + 0'] + [Idle(10)] + p['pi_1_nv2 - 0']  + [Idle(10)] + p['pi_1_nv2 + 0'] + [Idle(10)]
                    p['pi_nv2 + 90'] = p['pi_1_nv2 + 90']  + [Idle(10)] + p['pi_1_nv2 - 90']  + [Idle(10)] + p['pi_1_nv2 + 90'] + [Idle(10)]
                    
                    p['read + 0'] = p['pi_1_nv1 - 0'] + [Idle(10)] + p['pi2_1_nv1 + 0'] + [Idle(10)]
                    p['read + 90'] = p['pi_1_nv1 - 90'] + [Idle(10)] + p['pi2_1_nv1 + 90'] + [Idle(10)]
                else:
                
                    p['initial + 0'] =  p['pi2_1_nv1 + 0'] + [Idle(10)] + p['pi_1_nv1 - 0'] + [Idle(10)]
                    p['initial + 90'] = p['pi2_1_nv1 + 90']  + [Idle(10)] + p['pi_1_nv1 - 90'] + [Idle(10)]
                    
                    p['pi_nv1 + 0'] = p['pi_1_nv1 + 0'] + [Idle(10)] + p['pi_1_nv1 - 0'] + [Idle(10)] + p['pi_1_nv1 + 0'] + [Idle(10)]
                    p['pi_nv1 + 90'] = p['pi_1_nv1 + 90'] + [Idle(10)] + p['pi_1_nv1 - 90']  + [Idle(10)]+ p['pi_1_nv1 + 90'] + [Idle(10)]
                    
                    p['pi_nv2 + 0'] = p['pi_1_nv2 + 0'] + [Idle(10)]
                    p['pi_nv2 + 90'] = p['pi_1_nv2 + 90'] + [Idle(10)]
                    
                    p['read + 0'] = p['pi_1_nv1 - 0'] + [Idle(10)] + p['pi2_1_nv1 + 0'] + [Idle(10)]
                    p['read + 90'] = p['pi_1_nv1 - 90'] + [Idle(10)] + p['pi2_1_nv1 + 90'] + [Idle(10)]
            else:
                if self.multico2:
                    p['initial + 0'] = p['pi_1_nv2 + 0']  + [Idle(10)] + p['pi2_1_nv1 + 0'] + [Idle(10)]
                    p['initial + 90'] = p['pi_1_nv2 + 0']  + [Idle(10)] + p['pi2_1_nv1 + 90'] + [Idle(10)]
                    
                    p['pi_nv1 + 0'] =  p['pi_1_nv1 + 0'] + [Idle(10)]
                    p['pi_nv1 + 90'] =  p['pi_1_nv1 + 90'] + [Idle(10)]
                    
                    p['pi_nv2 + 0'] = p['pi_1_nv2 + 0'] + [Idle(10)] + p['pi_1_nv2 - 0'] + [Idle(10)] + p['pi_1_nv2 + 0'] + [Idle(10)]
                    p['pi_nv2 + 90'] = p['pi_1_nv2 + 90']  + [Idle(10)]+ p['pi_1_nv2 - 90']  + [Idle(10)]+ p['pi_1_nv2 + 90'] + [Idle(10)]
                    
                    p['read + 0'] =  p['pi2_1_nv1 + 0'] + [Idle(10)]
                    p['read + 90'] = p['pi2_1_nv1 + 90'] + [Idle(10)]
                
                else:
                    p['initial + 0'] = p['pi2_1_nv1 + 0'] + [Idle(10)]
                    p['initial + 90'] = p['pi2_1_nv1 + 90'] + [Idle(10)]
                    
                    p['pi_nv1 + 0'] =  p['pi_1_nv1 + 0'] + [Idle(10)]
                    p['pi_nv1 + 90'] =  p['pi_1_nv1 + 90'] + [Idle(10)]
                    
                    p['pi_nv2 + 0'] = p['pi_1_nv2 + 0'] + [Idle(10)]
                    p['pi_nv2 + 90'] = p['pi_1_nv2 + 90'] + [Idle(10)]
                    
                    p['read + 0'] =  p['pi2_1_nv1 + 0'] + [Idle(10)]
                    p['read + 90'] = p['pi2_1_nv1 + 90'] + [Idle(10)]

            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('DEER.SEQ')
            
            sup_x = Waveform('Sup1_x', p['initial + 0'])
            sup_y = Waveform('Sup1_y', p['initial + 90'])
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
            
            ref1_x = Waveform(name_x, [mod]+ p['pi_nv1 + 0'], t_0)
            ref1_y = Waveform(name_y, [mod]+ p['pi_nv1 + 90'], t_0)
            self.waves.append(ref1_x)
            self.waves.append(ref1_y)
                                 
            for i, t in enumerate(self.tau):
                t_0 = sup_x.duration + repeat_1 * 256 + ref1_x.duration
                mod.duration = t * 1.2
                t_2 = self.tau1 * 1.2 - t * 1.2
                name_x = 'REF_X%04i.WFM' % i
                name_y = 'REF_Y%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi_nv2 + 0'] + [Idle(t_2)]+ p['read + 0'], t_0)
                ref_y = Waveform(name_y, [mod] + p['pi_nv2 + 90'] + [Idle(t_2)]+ p['read + 90'] , t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                #if(i < len(self.tau) - 5):
                '''
                t_0 = sup_x.duration + repeat_1 * 256
                t_2 = t*1.2 - ref1_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += ref1_x.duration + repeat_2 * 256
                
                name_x = 'REF_X%04i.WFM' % i
                name_y = 'REF_Y%04i.WFM' % i
                
                ref_x = Waveform(name_x, [mod] + p['pi_nv2 + 0'], t_0)
                ref_y = Waveform(name_y, [mod] + p['pi_nv2 + 90'] , t_0)
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                
                t_3 = self.tau1 * 1.2 - t * 1.2 - ref_x.stub
                #t_3 = self.tau1 * 1.2 - repeat_2 * 256 - ref_x.duration - ref1_x.stub
                repeat_3 = int(t_3 / 256)
                mod.duration = int(t_3 % 256)
                #print mod.duration
                t_0 += ref_x.duration + repeat_3 * 256
                
                name_x = 'MAP3_X%04i.WFM' % i
                name_y = 'MAP3_Y%04i.WFM' % i
            
                map_x = Waveform(name_x, [mod]+ p['read + 0'], t_0)
                map_y = Waveform(name_y, [mod]+ p['read + 90'], t_0)
                
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
                 '''
                name = 'DQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x, sup_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(ref_x, ref_y)
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
                    ([], self.pi_2_p + self.pi2_1 + self.pi_1_p + 2*self.pi_1_p+ self.pi_1_m + 2*self.pi_2_p+ self.pi_2_m + self.pi2_1 + self.pi_1_p + self.tau1 *2 + 500 ),
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
        
    get_set_items = Pulsed.get_set_items + ['freq','freq_2','freq_3','freq_4','freq_5','freq_6','ffreq','ffreq_2','ffreq_3','ffreq_4','ffreq_5','ffreq_6','vpp','amp1','amp2','tau1','pi2_1','pi_1_p','pi_1_m','pi_2_p','pi_2_m','nfreq1',
    'nfreq2','multifreq1','multifreq2','multico1','multico2'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp1', width=-40),   
                                     Item('amp2', width=-40),   
                                     Item('tau1', width = 20),
                                     ),                    
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_2', width=-60),   
                                     Item('freq_3', width=-60),   
                                     Item('freq_4', width = -60),
                                     Item('freq_5', width=-60),   
                                     Item('freq_6', width = -60),
                                     ),       
                              HGroup(Item('ffreq',  width=-60),
                                     Item('ffreq_2', width=-60),   
                                     Item('ffreq_3', width=-60),   
                                     Item('ffreq_4', width = -60),
                                     Item('ffreq_5', width=-60),   
                                     Item('ffreq_6', width = -60),
                                     ),      
                              HGroup(Item('multico1',  width=20),
                                     Item('multico2',  width=20),
                                     Item('multifreq1',  width=20),
                                     Item('multifreq2',  width=20),
                                     Item('nfreq1',  width=20),
                                     Item('nfreq2',  width=20),
                                     ),    
                              HGroup(
                                     Item('pi2_1', width = -40),
                                     Item('pi_1_p', width = -40),
                                     Item('pi_1_m', width = -40),
                                     Item('pi_2_p', width = -40),
                                     Item('pi_2_m', width = -40)
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
                       
                       
class HahnPair( Pulsed ):
    #FID 
    
    pi2_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='pi plus', mode='text', auto_set=False, enter_set=True)
    pi_2 = Range(low=1., high=100000., value=25.75, desc='length of pi pulse 1st transition [ns]', label='pi minus', mode='text', auto_set=False, enter_set=True)
    
    freq_center = Range(low=1, high=20e9, value=2.61e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value= 10, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    
    freq = Range(low=1, high=20e9, value=2.791676e9, desc='nv1 frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_2 = Range(low=1, high=20e9, value=2.845829e9, desc='nv1 frequency 2nd trans [Hz]', label='freq2 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_3 = Range(low=1, high=20e9, value=2.907644e9, desc='nv1 frequency 3nd trans [Hz]', label='freq3 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_4 = Range(low=1, high=20e9, value=2.955537e9, desc='nv1 frequency 4th trans [Hz]', label='freq4 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_5 = Range(low=1, high=20e9, value=2.907644e9, desc='nv1 frequency 5nd trans [Hz]', label='freq5 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq_6 = Range(low=1, high=20e9, value=2.955537e9, desc='nv1 frequency 6th trans [Hz]', label='freq6 [Hz]', mode='text', auto_set=False, enter_set=True)
    
    ffreq = Range(low=1, high=20e9, value=2.791676e9, desc='nv2 frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    ffreq_2 = Range(low=1, high=20e9, value=2.845829e9, desc='nv2 frequency 2nd trans [Hz]', label='freq2 [Hz]', mode='text', auto_set=False, enter_set=True)
    ffreq_3 = Range(low=1, high=20e9, value=2.907644e9, desc='nv2 frequency 3nd trans [Hz]', label='freq3 [Hz]', mode='text', auto_set=False, enter_set=True)
    ffreq_4 = Range(low=1, high=20e9, value=2.955537e9, desc='nv2 frequency 4th trans [Hz]', label='freq4 [Hz]', mode='text', auto_set=False, enter_set=True)
    ffreq_5 = Range(low=1, high=20e9, value=2.907644e9, desc='nv2 frequency 5nd trans [Hz]', label='freq5 [Hz]', mode='text', auto_set=False, enter_set=True)
    ffreq_6 = Range(low=1, high=20e9, value=2.955537e9, desc='nv2 frequency 6th trans [Hz]', label='freq6 [Hz]', mode='text', auto_set=False, enter_set=True)
    
    amp1 = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp1', mode='text', auto_set=False, enter_set=True)
    amp2 = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp2', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi nv1', mode='text', auto_set=False, enter_set=True)
    pi_1_p   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='plus pi nv1', mode='text', auto_set=False, enter_set=True)
    pi_1_m   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='minus pi nv1', mode='text', auto_set=False, enter_set=True)
    
    pi2_2   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi nv2', mode='text', auto_set=False, enter_set=True)
    pi_2_p   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='plus pi nv2', mode='text', auto_set=False, enter_set=True)
    pi_2_m   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='minus pi nv2', mode='text', auto_set=False, enter_set=True)
    
    multihahn = Bool(False, label='single or double Hahn',desc = 'false, single coherence Hahn')
    multifreq1 = Bool(False, label='multifreq1',desc = 'false, single freq')
    multifreq2 = Bool(False, label='multifreq2',desc = 'false, single freq')
    nfreq1 = Range(low=2, high=3, value=3, desc='number of freq', label='nfreq1', mode='text', auto_set=False, enter_set=True)
    nfreq2 = Range(low=2, high=3, value=3, desc='number of freq', label='nfreq2', mode='text', auto_set=False, enter_set=True)
    
    
    reload = True

    def prepare_awg(self):
        sampling = 1.2e9
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            p = {}
            
            f1 = (self.freq - self.freq_center)/sampling
            f2 = (self.freq_2 - self.freq_center)/sampling
            f3= (self.freq_3 - self.freq_center)/sampling
            f4 = (self.freq_4 - self.freq_center)/sampling
            f5= (self.freq_5 - self.freq_center)/sampling
            f6 = (self.freq_6 - self.freq_center)/sampling
            
            ff1 = (self.ffreq - self.freq_center)/sampling
            ff2 = (self.ffreq_2 - self.freq_center)/sampling
            ff3= (self.ffreq_3 - self.freq_center)/sampling
            ff4 = (self.ffreq_4 - self.freq_center)/sampling
            ff5= (self.ffreq_5 - self.freq_center)/sampling
            ff6 = (self.ffreq_6 - self.freq_center)/sampling
            
            pi2_1 = int(self.pi2_1 * sampling/1.0e9)
            pi_1_p = int(self.pi_1_p * sampling/1.0e9)
            pi_1_m = int(self.pi_1_m * sampling/1.0e9)
            pi2_2 = int(self.pi2_2 * sampling/1.0e9)
            pi_2_p = int(self.pi_2_p * sampling/1.0e9)
            pi_2_m = int(self.pi_2_m * sampling/1.0e9)
            
            if self.multifreq1:
            
                if self.nfreq1 == 2:
                    p['pi2_1_nv1 + 0']     = [Sin( pi2_1, f1, 0 , self.amp1/2) + Sin( pi2_1, f2, 0 ,self.amp1/2)]
                    p['pi2_1_nv1 + 90']    = [Sin( pi2_1, f1, np.pi/2 , self.amp1/2) + Sin( pi2_1, f2, np.pi/2 ,self.amp1/2)]
                    
                    p['pi_1_nv1 + 0']     = [Sin( pi_1_p, f1, 0 , self.amp1/2) + Sin( pi_1_p, f2, 0 ,self.amp1/2)]
                    p['pi_1_nv1 + 90']    = [Sin( pi_1_p, f1, np.pi/2 , self.amp1/2) + Sin( pi_1_p, f2, np.pi/2 ,self.amp1/2)]
                    
                    p['pi_1_nv1 - 0']     = [Sin( pi_1_m, f4, 0 , self.amp1/2) + Sin( pi_1_m, f5, 0 ,self.amp1/2) ] 
                    p['pi_1_nv1 - 90']    = [Sin( pi_1_m, f4, np.pi/2 , self.amp1/2) + Sin( pi_1_m, f5, np.pi/2 ,self.amp1/2)]
                else:
                    p['pi2_1_nv1 + 0']     = [Sin( pi2_1, f1, 0 , self.amp1/3) + Sin( pi2_1, f2, 0 ,self.amp1/3) + Sin( pi2_1, f3, 0 ,self.amp1/3)]
                    p['pi2_1_nv1 + 90']    = [Sin( pi2_1, f1, np.pi/2 , self.amp1/3) + Sin( pi2_1, f2, np.pi/2 ,self.amp1/3)  + Sin( pi2_1, f3, np.pi/2 ,self.amp1/3)]
                    
                    p['pi_1_nv1 + 0']     = [Sin( pi_1_p, f1, 0 , self.amp1/3) + Sin( pi_1_p, f2, 0 ,self.amp1/3) + Sin( pi_1_p, f3, 0 ,self.amp1/3)]
                    p['pi_1_nv1 + 90']    = [Sin( pi_1_p, f1, np.pi/2 , self.amp1/3) + Sin( pi_1_p, f2, np.pi/2 ,self.amp1/3) + Sin( pi_1_p, f3, np.pi/2 ,self.amp1/3)]
                    
                    p['pi_1_nv1 - 0']     = [Sin( pi_1_m, f4, 0 , self.amp1/3) + Sin( pi_1_m, f5, 0 ,self.amp1/3) + Sin( pi_1_m, f6, 0 ,self.amp1/3) ] 
                    p['pi_1_nv1 - 90']    = [Sin( pi_1_m, f4, np.pi/2 , self.amp1/3) + Sin( pi_1_m, f5, np.pi/2 ,self.amp1/3) + Sin( pi_1_m, f6, np.pi/2 ,self.amp1/3)]
            else:
                
                    p['pi2_1_nv1 + 0']     = [Sin( pi2_1, f1, 0 , self.amp1)]
                    p['pi2_1_nv1 + 90']    = [Sin( pi2_1, f1, np.pi/2 , self.amp1)]
                    
                    p['pi_1_nv1 + 0']     = [Sin( pi_1_p, f1, 0 , self.amp1)]
                    p['pi_1_nv1 + 90']    = [Sin( pi_1_p, f1, np.pi/2 , self.amp1)]
                    
                    p['pi_1_nv1 - 0']     = [Sin( pi_1_m, f4, 0 , self.amp1)] 
                    p['pi_1_nv1 - 90']    = [Sin( pi_1_m, f4, np.pi/2 , self.amp1)]
                    
            if self.multifreq2:        
                if self.nfreq2 == 2:
                
                    p['pi2_1_nv2 + 0']     = [Sin( pi2_2, ff1, 0 , self.amp2/2) + Sin( pi2_2, ff2, 0 ,self.amp2/2)]
                    p['pi2_1_nv2 + 90']    = [Sin( pi2_2, ff1, np.pi/2 , self.amp2/2) + Sin( pi2_2, ff2, np.pi/2 ,self.amp2/2)]
                    
                    p['pi_1_nv2 + 0']     = [Sin( pi_2_p, ff1, 0 , self.amp2/2) + Sin( pi_2_p, ff2, 0 ,self.amp2/2)]
                    p['pi_1_nv2 + 90']    = [Sin( pi_2_p, ff1, np.pi/2 , self.amp2/2) + Sin( pi_2_p, ff2, np.pi/2 ,self.amp2/2)]
                    
                    p['pi_1_nv2 - 0']     = [Sin( pi_2_m, ff4, 0 , self.amp2/2) + Sin( pi_2_m, ff5, 0 ,self.amp2/2) ] 
                    p['pi_1_nv2 - 90']    = [Sin( pi_2_m, ff4, np.pi/2 , self.amp2/2) + Sin( pi_2_m, ff5, np.pi/2 ,self.amp2/2)]
                else:
                    p['pi2_1_nv2 + 0']     = [Sin( pi2_2, ff1, 0 , self.amp2/3) + Sin( pi2_2, ff2, 0 ,self.amp2/3) + Sin( pi2_2, ff3, 0 ,self.amp2/3)]
                    p['pi2_1_nv2 + 90']    = [Sin( pi2_2, ff1, np.pi/2 , self.amp2/3) + Sin( pi2_2, ff2, np.pi/2 ,self.amp2/3)  + Sin( pi2_2, ff3, np.pi/2 ,self.amp2/3)]

                    p['pi_1_nv2 + 0']     = [Sin( pi_2_p, ff1, 0 , self.amp2/3) + Sin( pi_2_p, ff2, 0 ,self.amp2/3) + Sin( pi_2_p, ff3, 0 ,self.amp2/3)]
                    p['pi_1_nv2 + 90']    = [Sin( pi_2_p, ff1, np.pi/2 , self.amp2/3) + Sin( pi_2_p, ff2, np.pi/2 ,self.amp2/3) + Sin( pi_2_p, ff3, np.pi/2 ,self.amp2/3)]
                    
                    p['pi_1_nv2 - 0']     = [Sin( pi_2_m, ff4, 0 , self.amp2/3) + Sin( pi_2_m, ff5, 0 ,self.amp2/3) + Sin( pi_2_m, ff6, 0 ,self.amp2/3) ] 
                    p['pi_1_nv2 - 90']    = [Sin( pi_2_m, ff4, np.pi/2 , self.amp2/3) + Sin( pi_2_m, ff5, np.pi/2 ,self.amp2/3) + Sin( pi_2_m, ff6, np.pi/2 ,self.amp2/3)]   
                    
            else:
                    p['pi2_1_nv2 + 0']     = [Sin( pi2_2, ff1, 0 , self.amp2)]
                    p['pi2_1_nv2 + 90']    = [Sin( pi2_2, ff1, np.pi/2 , self.amp2)]
                    
                    p['pi_1_nv2 + 0']     = [Sin( pi_2_p, ff1, 0 , self.amp2)]
                    p['pi_1_nv2 + 90']    = [Sin( pi_2_p, ff1, np.pi/2 , self.amp2)]
                    
                    p['pi_1_nv2 - 0']     = [Sin( pi_2_m, ff4, 0 , self.amp2)] 
                    p['pi_1_nv2 - 90']    = [Sin( pi_2_m, ff4, np.pi/2 , self.amp2)]
                    
                    
            if self.multihahn:
            
                p['pi2_1 + 0'] = p['pi2_1_nv1 + 0'] + p['pi_1_nv1 - 0']
                p['pi2_1 + 90'] = p['pi2_1_nv1 + 90'] + p['pi_1_nv1 - 90']
                
                p['pi_1 + 0'] = p['pi_1_nv1 + 0'] + p['pi_1_nv1 - 0'] + p['pi_1_nv1 + 0']
                p['pi_1 + 90'] = p['pi_1_nv1 + 90'] + p['pi_1_nv1 - 90'] + p['pi_1_nv1 + 90']
                
                p['pi2_2 + 0'] = p['pi2_1_nv2 + 0'] + p['pi_1_nv2 - 0']
                p['pi2_2 + 90'] = p['pi2_1_nv2 + 90'] + p['pi_1_nv2 - 90']
                
                p['pi_2 + 0'] = p['pi_1_nv2 + 0'] + p['pi_1_nv2 - 0'] + p['pi_1_nv2 + 0']
                p['pi_2 + 90'] = p['pi_1_nv2 + 90'] + p['pi_1_nv2 - 90'] + p['pi_1_nv2 + 90']
               
            else:
            # ms= 0 <> ms = +1
                p['pi2_1 + 0'] = p['pi2_1_nv1 + 0']
                p['pi2_1 + 90'] = p['pi2_1_nv1 + 90']
                
                p['pi_1 + 0'] = p['pi_1_nv1 + 0'] 
                p['pi_1 + 90'] = p['pi_1_nv1 + 90']
                
                p['pi2_2 + 0'] = p['pi2_1_nv2 + 0'] 
                p['pi2_2 + 90'] = p['pi2_1_nv2 + 90'] 
                
                p['pi_2 + 0'] = p['pi_1_nv2 + 0']
                p['pi_2 + 90'] = p['pi_1_nv2 + 90']

            
            zero = Idle(1)
            mod = Idle(0)
            
                
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('Hahn.SEQ')
            
            sup1_x = Waveform('Sup1_x', p['pi2_1 + 0'] + p['pi2_2 + 0'] )
            sup1_y = Waveform('Sup1_y', p['pi2_1 + 90'] + p['pi2_2 + 90'] )
            self.waves.append(sup1_x)
            self.waves.append(sup1_y)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
           
            for i, t in  enumerate(self.tau):
                t_1 = t*1.2 - sup1_x.stub
                repeat_1 = int(t_1 / 256)
                mod.duration = int(t_1 % 256)
                t_0 = sup1_x.duration + repeat_1 * 256
                
                name_x = 'MAP3_X%04i.WFM' % i
                name_y = 'MAP3_Y%04i.WFM' % i
            
                map_x = Waveform(name_x, [mod]+ p['pi_1 + 0'] + p['pi_2 + 0']+[zero], t_0)
                map_y = Waveform(name_y, [mod]+ p['pi_1 + 90'] + p['pi_2 + 90']+[zero], t_0)
                
                self.waves.append(map_x)
                self.waves.append(map_y)
                
                t_2 = t * 1.2 - map_x.stub
                repeat_2 = int(t_2 / 256)
                mod.duration = int(t_2 % 256)
                t_0 += map_x.duration + repeat_2 * 256
                
                name_x = 'ref3_X%04i.WFM' % i
                name_y = 'ref3_Y%04i.WFM' % i
                
                ref_x = Waveform(name_x, [mod]+ p['pi2_1 + 0'] +p['pi2_2 + 0'] +[zero], t_0)
                ref_y = Waveform(name_y, [mod]+ p['pi2_1 + 90']+p['pi2_1 + 90'] +[zero], t_0)
                
                self.waves.append(ref_x)
                self.waves.append(ref_y)
                
               
                
                name = 'DQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup1_x, sup1_y)
                sub_seq.append(evo, evo,repeat=repeat_1)
                sub_seq.append(map_x, map_y)
                sub_seq.append(evo, evo,repeat=repeat_2)
                sub_seq.append(ref_x, ref_y)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq,wait=True)

            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('Hahn.SEQ')
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
            sub = [ (['awgTrigger'], 100 ),
                    ([], 20000 + 2 * t + 500 ),
                    (['laser', 'trigger' ], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)

        return sequence
            
    get_set_items = Pulsed.get_set_items + ['freq','freq_2','freq_3','freq_4','freq_5','freq_6','ffreq','ffreq_2','ffreq_3','ffreq_4','ffreq_5','ffreq_6','vpp','amp1','amp2','pi2_1','pi2_2','pi_1_p','pi_1_m','pi_2_p','pi_2_m','nfreq1','nfreq2','multifreq1','multifreq2','multihahn'] 
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq_center',  width=20),
                                     Item('power', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp1', width=-40),   
                                     Item('amp2', width=-40),   
                                     ),                    
                              HGroup(Item('freq',  width=-60),
                                     Item('freq_2', width=-60),   
                                     Item('freq_3', width=-60),   
                                     Item('freq_4', width = -60),
                                     Item('freq_5', width=-60),   
                                     Item('freq_6', width = -60),
                                     ),       
                              HGroup(Item('ffreq',  width=-60),
                                     Item('ffreq_2', width=-60),   
                                     Item('ffreq_3', width=-60),   
                                     Item('ffreq_4', width = -60),
                                     Item('ffreq_5', width=-60),   
                                     Item('ffreq_6', width = -60),
                                     ),      
                              HGroup(Item('multihahn',  width=20),
                                     Item('multifreq1',  width=20),
                                     Item('multifreq2',  width=20),
                                     Item('nfreq1',  width=20),
                                     Item('nfreq2',  width=20),
                                     ),    
                              HGroup(
                                     Item('pi2_1', width = -40),
                                     Item('pi_1_p', width = -40),
                                     Item('pi_1_m', width = -40),
                                     Item('pi2_2', width = -40),
                                     Item('pi_2_p', width = -40),
                                     Item('pi_2_m', width = -40)
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
                       title='HahnPair',
                       )                          