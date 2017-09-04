import numpy as np

from enthought.traits.api import SingletonHasTraits, Instance, Property, Range, Float, Int, Bool, Array, List, Enum, Trait,\
                                 Button, on_trait_change, cached_property, Code, Str
from enthought.traits.ui.api import View, Item, HGroup, VGroup, Tabbed, EnumEditor, RangeEditor, TextEditor
from enthought.enable.api import Component, ComponentEditor
from enthought.chaco.api import ArrayDataSource, LinePlot, LinearMapper, ArrayPlotData, Plot, CMapImagePlot
from enthought.chaco.tools.api import PanTool, ZoomTool
from enthought.chaco.tools.cursor_tool import CursorTool, CursorTool1D

from traits.api import HasTraits

import threading
import traceback
import sys

import os
os.environ['ETS_TOOLKIT'] = 'qt4'

import traits.etsconfig
traits.etsconfig.enable_toolkit='qt4'
traits.etsconfig.toolkit='qt4'


import time

import logging

from tools.emod import ManagedJob

from tools.utility import GetSetItemsMixin

from hardware.api import PulseGenerator 
from hardware.api import Microwave 
#from hardware.api import Microwave_HMC
from hardware.api import AWG, FastComtec, Counter_SST, Counter
from hardware.awg import *
from hardware.waveform import *

PG = PulseGenerator()
MW = Microwave()
#MW = Microwave_HMC()
FC = FastComtec()
AWG = AWG()
CS = Counter_SST()

class Pulsed(ManagedJob, GetSetItemsMixin):

    """Defines a pulsed measurement."""
    keep_data = Bool(False) # helper variable to decide whether to keep existing data

    resubmit_button = Button(label='resubmit', desc='Submits the measurement to the job manager. Tries to keep previously acquired data. Behaves like a normal submit if sequence or time bins have changed since previous run.')

    sequence = Instance(list, factory=list)
    
    record_length = Float(value=0, desc='length of acquisition record [ms]', label='record length [ms] ', mode='text')
    
    count_data = Array(value=np.zeros(2))
    
    run_time = Float(value=0.0, label='run time [ns]', format_str='%.f')
    stop_time = Range(low=1., value=np.inf, desc='Time after which the experiment stops by itself [s]', label='Stop time [s]', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=0., high=1e5, value=300., desc='tau begin [ns]', label='repetition', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e5, value=4000., desc='tau end [ns]', label='N repetition', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=1e5, value=50., desc='delta tau [ns]', label='delta', mode='text', auto_set=False, enter_set=True)

    tau = Array(value=np.array((0., 1.)))
    sequence_points = Int(value=2, label='number of points', mode='text')
    
    laser_SST = Range(low=1., high=5e6, value=200., desc='laser for SST [ns]', label='laser_SST[ns]', mode='text', auto_set=False, enter_set=True)
    wait_SST = Range(low=1., high=5e6, value=1000., desc='wait for SST[ns]', label='wait_SST [ns]', mode='text', auto_set=False, enter_set=True)
    N_shot = Range(low=1, high=20e5, value=2e3, desc='number of shots in SST', label='N_shot', mode='text', auto_set=False, enter_set=True)
    
    laser = Range(low=1., high=5e4, value=3000, desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
    wait = Range(low=1., high=5e4, value=5000., desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)
    
    freq_center = Range(low=1, high=20e9, value=2.71e9, desc='frequency [Hz]', label='MW freq[Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.4e'))
    power = Range(low=-100., high=25., value=-26, desc='power [dBm]', label='power[dBm]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float))
    freq = Range(low=1, high=20e9, value=2.71e9, desc='frequency [Hz]', label='freq [Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.4e'))
    pi = Range(low=0., high=5e4, value=2e3, desc='pi pulse length', label='pi [ns]', mode='text', auto_set=False, enter_set=True)
    
    amp = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='Amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='Vpp', mode='text', auto_set=False, enter_set=True)

    sweeps = Range(low=1., high=1e4, value=1e2, desc='number of sweeps', label='sweeps', mode='text', auto_set=False, enter_set=True)
    expected_duration = Property(trait=Float, depends_on='sweeps,sequence', desc='expected duration of the measurement [s]', label='expected duration [s]')
    elapsed_sweeps = Float(value=0, desc='Elapsed Sweeps ', label='Elapsed Sweeps ', mode='text')
    elapsed_time = Float(value=0, desc='Elapsed Time [ns]', label='Elapsed Time [ns]', mode='text')
    progress = Int(value=0, desc='Progress [%]', label='Progress [%]', mode='text')

    load_button = Button(desc='compile and upload waveforms to AWG', label='load')
    reload = True
    
    
    readout_interval = Float(1, label='Data readout interval [s]', desc='How often data read is requested from nidaq')
    samples_per_read = Int(200, label='# data points per read', desc='Number of data points requested from nidaq per read. Nidaq will automatically wait for the data points to be aquired.')


    def submit(self):
        """Submit the job to the JobManager."""
        self.keep_data = False
        ManagedJob.submit(self)

    def resubmit(self):
        """Submit the job to the JobManager."""
        self.keep_data = True
        ManagedJob.submit(self)

    def _resubmit_button_fired(self):
        """React to start button. Submit the Job."""
        self.resubmit() 

    def generate_sequence(self):
        return []
        
    def prepare_awg(self):
        """ override this """
        AWG.reset()
        
    def _load_button_changed(self):
        self.load()    
        
    def load(self):
        self.reload = True
        # update record_length, in ms
        self.record_length = self.N_shot * (self.pi + self.laser_SST + self.wait_SST)*1e-6
        #make sure tau is updated
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta) 
        self.prepare_awg()
        self.reload = False    
        
    @cached_property
    def _get_expected_duration(self):
        sequence_length = 0
        for step in self.sequence:
            sequence_length += step[1]
        return self.sweeps * sequence_length * 1e-9  
        
    def _get_sequence_points(self):
        return len(self.tau)    
     
    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""
        """if load button is not used, make sure tau is generated"""
        if(self.tau.shape[0]==2):
            tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta) 
            self.tau = tau         

        self.sequence_points = self._get_sequence_points()
        self.measurement_points = self.sequence_points * int(self.sweeps)
        sequence = self.generate_sequence()
        
        
        if self.keep_data and sequence == self.sequence: # if the sequence and time_bins are the same as previous, keep existing data

            self.previous_sweeps = self.elapsed_sweeps
            self.previous_elapsed_time = self.elapsed_time
            self.keep_data = True # when job manager stops and starts the job, data should be kept. Only new submission should clear data.
        else:
            
            #self.old_count_data = np.zeros((n_laser, n_bins))
            #self.check = True

            self.count_data = np.zeros(self.measurement_points)
            self.old_count_data = np.zeros(self.measurement_points)
            self.previous_sweeps = 0
            self.previous_elapsed_time = 0.0
            self.run_time = 0.0
            self.keep_data = True # when job manager stops and starts the job, data should be kept. Only new submission should clear data.
            
        self.sequence = sequence 
           

        
    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            
            self.prepare_awg()
            MW.setFrequency(self.freq_center)
            MW.setPower(self.power)
         
            AWG.run()
            time.sleep(4.0)
            PG.Sequence(self.sequence, loop=True)
            
            if CS.configure() != 0:   # initialize and start nidaq gated counting task, return 0 if succuessful
                print 'error in nidaq'
                return
            
            start_time = time.time()
            
            aquired_data = np.empty(0)   # new data will be appended to this array

            while True:

               self.thread.stop_request.wait(self.readout_interval)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break

               #threading.current_thread().stop_request.wait(self.readout_interval) # wait for some time before new read command is given. not sure if this is neccessary
                #if threading.current_thread().stop_request.isSet():
                    #break   
                
               points_left = self.measurement_points - len(aquired_data) 
               
               self.elapsed_time = self.previous_elapsed_time + time.time() - start_time
               self.run_time += self.elapsed_time
               

               new_data = CS.read_gated_counts( SampleLength=min(self.samples_per_read, points_left) )   # do not attempt to read more data than neccessary

               aquired_data = np.append( aquired_data, new_data[:min(len(new_data), points_left)] )
               
               self.count_data[:len(aquired_data)] = aquired_data[:]    # length of trace may not change due to plot, so just copy aquired data into trace
               
               sweeps = len(aquired_data) / self.sequence_points
               self.elapsed_sweeps += self.previous_sweeps + sweeps
               self.progress = int( 100 * len(aquired_data) / self.measurement_points ) 

               if self.progress > 99.9:
                  break


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
            
        finally:    
            CS.stop_gated_counting() # stop nidaq task to free counters
                

    get_set_items = ['__doc__', 'record_length','laser','wait','sequence', 'count_data', 'run_time','tau_begin', 'tau_end', 'tau_delta', 'tau','freq_center','power',
                                'laser_SST','wait_SST','amp','vpp','pi','freq','N_shot','readout_interval','samples_per_read']
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                                     
                               HGroup(Item('freq',  width=-70),
                                     Item('freq_center',  width=-70),
                                     Item('amp', width=-30),
                                     Item('vpp', width=-30),                                     
                                     Item('power', width=-40),
                                     Item('pi', width=-70),
                                     ),      

                              HGroup(Item('laser', width=-60),
                                     Item('wait', width=-60),
                                     Item('laser_SST', width=-50),
                                     Item('wait_SST', width=-50),
                                     ),
                                     
                              HGroup(Item('samples_per_read', width=-50),
                                     Item('N_shot', width=-50),
                                     Item('record_length', style='readonly'),
                                     ),        
                                     
                              HGroup(Item('tau_begin', width=30),
                                     Item('tau_end', width=30),
                                     Item('tau_delta', width= 30),
                                     
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=-60),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=-50),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.2f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=-50),
                                     ),
                                   
                              ),
                       title='Pulsed_SST Measurement',
                       )

class SSTCounterTrace(Pulsed):

    tau_begin = Range(low=0., high=1e5, value=1., desc='tau begin [ns]', label='repetition', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e5, value=1000., desc='tau end [ns]', label='N repetition', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=1e5, value=1, desc='delta tau [ns]', label='delta', mode='text', auto_set=False, enter_set=True)
    
    sweeps = Range(low=1., high=1e4, value=1, desc='number of sweeps', label='sweeps', mode='text', auto_set=False, enter_set=True)
    
    def prepare_awg(self):
        sampling = 1.2e9
        N_shot = int(self.N_shot)

        pi = int(self.pi * sampling / 1.0e9)
        laser_SST = int(self.laser_SST * sampling / 1.0e9)
        wait_SST = int(self.wait_SST * sampling / 1.0e9)
        
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            AWG.delete_all()
            
            zero = Idle(1)
            self.waves = []
            sub_seq = []
            p = {}
            
            p['pi + 0'] = Sin(pi, (self.freq - self.freq_center)/sampling, 0, self.amp)
            p['pi + 90'] = Sin(pi, (self.freq - self.freq_center)/sampling, np.pi/2, self.amp)
            
            read_x = Waveform('read_x', [p['pi + 0'],  Idle(laser_SST, marker1 = 1), Idle(wait_SST)])
            read_y = Waveform('read_y', [p['pi + 90'], Idle(laser_SST, marker1 = 1), Idle(wait_SST)])
            self.waves.append(read_x)
            self.waves.append(read_y)
            
            self.main_seq = Sequence('SST.SEQ')
            for i, t in enumerate(self.tau):
                name = 'DQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(read_x, read_y, repeat = N_shot)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('SST.SEQ')
        AWG.set_vpp(self.vpp)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )
        

    def generate_sequence(self):
        points = int(self.sequence_points)
        N_shot = self.N_shot
        laser = self.laser
        wait = self.wait
        laser_SST = self.laser_SST
        wait_SST = self.wait_SST
        pi = self.pi
        record_length = self.record_length*1e+6
        
        sequence = []
        for t in range(points):
            sequence.append( (['laser'], laser) )
            sequence.append( ([ ],  wait) )
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ 'sst' ]       , record_length) )

        return sequence
        
    get_set_items = Pulsed.get_set_items
    
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                                     
                              HGroup(Item('freq',  width=-70),
                                     Item('freq_center',  width=-70),
                                     Item('amp', width=-30),
                                     Item('vpp', width=-30),                                     
                                     Item('power', width=-40),
                                     Item('pi', width=-70),
                                     ),      

                              HGroup(Item('laser', width=-60),
                                     Item('wait', width=-60),
                                     Item('laser_SST', width=-50),
                                     Item('wait_SST', width=-50),
                                     ),
                                     
                              HGroup(Item('samples_per_read', width=-50),
                                     Item('N_shot', width=-50),
                                     Item('record_length', style='readonly'),
                                     ),        
                                     
                              HGroup(Item('tau_begin', width=30),
                                     Item('tau_end', width=30),
                                     Item('tau_delta', width= 30),
                                     
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=-50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.1e'%x), width=-50),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.2f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=-50),
                                     ),
                                   
                              ),
                       title='SST Trace Measurement',
                       )
                       