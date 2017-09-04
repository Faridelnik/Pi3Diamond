# This is the file with AWG sequences for the rreadout via Single Shot

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

from hardware.api import Microwave_HMC, Microwave, Counter_SST, PulseGenerator, AWG, FastComtec

from hardware.awg import *
from hardware.waveform import *
from measurements.odmr import ODMR

PG = PulseGenerator()
#MW= Microwave_HMC()
MW = Microwave()
FC = FastComtec()
AWG = AWG()
CS = Counter_SST()

class Pulsed(ManagedJob, GetSetItemsMixin): #XY8 originally

    resubmit_button = Button(label='resubmit', desc='Submits the measurement to the job manager. Tries to keep previously acquired data. Behaves like a normal submit if sequence or time bins have changed since previous run.')
    sequence = Instance(list, factory=list)

    laser = Range(low=1., high=5e4, value=3000, desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
    wait = Range(low=1., high=5e4, value=5000., desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)

    # DD part
    
    record_length = Range(low=100, high=100000., value=3000, desc='length of acquisition record [ns]', label='record length [ns]', mode='text', auto_set=False, enter_set=True)
    bin_width = Range(low=0.1, high=1000., value=3.2, desc='data bin width [ns]', label='bin width [ns]', mode='text', auto_set=False, enter_set=True)
    
    n_laser = Int(2)
    n_bins = Int(2)
    time_bins = Array(value=np.array((0, 1)))

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power_XY8 = Range(low=-100., high=25., value= 12, desc='power XY8 [dBm]', label='power XY8 [dBm]', mode='text', auto_set=False, enter_set=True)
    freq1 = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq2 = Range(low=1, high=20e9, value=1.8e9, desc='frequency 2nd trans[Hz]', label='freq2 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.71e9, desc='frequency [Hz]', label='mean freq [Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.4e'))
   
    amp_high = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    rabi_contrast = Range(low=1., high=100, value=35.0, desc='Rabi contrast [%]', label='contrast', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=0., high=1e8, value=500., desc='tau/2 begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau/2 end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau/2 [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    
    tau = Array(value=np.array((0., 1.)))
  
    pulse_num = Range(low=1, high=1000, value=1, desc='repetetion of XY8 pulses', label='repetitions', mode='text', auto_set=False, enter_set=True)
    reload = True
    
    # state mapping part
    
    rf_freq = Range(low=1, high=20e9, value=7.2e6, desc='frequency rf [Hz]', label='rf freq [Hz]', mode='text', auto_set=False, enter_set=True)    
    amp_rf = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM rf amp', mode='text', auto_set=False, enter_set=True)
    
    pi_rf=Range(low=1., high=100000., value=99.35, desc='length of nuclear pi pulse [ns]', label='pi RF [ns]', mode='text', auto_set=False, enter_set=True)
   
    # Single shot readout part
    
    keep_data = Bool(False) # helper variable to decide whether to keep existing data
    sequence = Instance(list, factory=list)
    
    record_length_ssr = Float(value=0, desc='length of acquisition record [ms]', label='readout length [ms] ', mode='text')
    
    count_data = Array(value=np.zeros(2))
    
    run_time = Float(value=0.0, label='run time [ns]', format_str='%.f')
    stop_time = Range(low=1., value=np.inf, desc='Time after which the experiment stops by itself [s]', label='Stop time [s]', mode='text', auto_set=False, enter_set=True)
    
    sequence_points = Int(value=2, label='number of points', mode='text')
    
    laser_SST = Range(low=1., high=5e6, value=200., desc='laser for SST [ns]', label='laser_SST[ns]', mode='text', auto_set=False, enter_set=True)
    wait_SST = Range(low=1., high=5e6, value=1000., desc='wait for SST[ns]', label='wait_SST [ns]', mode='text', auto_set=False, enter_set=True)
    N_shot = Range(low=1, high=20e5, value=2e3, desc='number of shots in SST', label='N_shot', mode='text', auto_set=False, enter_set=True)
       
    pi_SSR = Range(low=0., high=5e4, value=2e3, desc='pi pulse length', label='pi [ns]', mode='text', auto_set=False, enter_set=True)
    
    amp_low = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM Amp for SSR', mode='text', auto_set=False, enter_set=True)
    
    sweeps = Range(low=1., high=1e4, value=1e2, desc='number of sweeps', label='sweeps', mode='text', auto_set=False, enter_set=True)
    expected_duration = Property(trait=Float, depends_on='sweeps,sequence', desc='expected duration of the measurement [s]', label='expected duration [s]')
    elapsed_sweeps = Float(value=0, desc='Elapsed Sweeps ', label='Elapsed Sweeps ', mode='text')
    elapsed_time = Float(value=0, desc='Elapsed Time [ns]', label='Elapsed Time [ns]', mode='text')
    progress = Int(value=0, desc='Progress [%]', label='Progress [%]', mode='text')

    load_button = Button(desc='compile and upload waveforms to AWG', label='load')
    
    N_readout_points = Range(low=1., high=1e8, value=50., desc='N readouts', label='N readouts', mode='text', auto_set=False, enter_set=True)
    
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
        #AWG._server_init()
        #print 'come on'
        # update record_length, in ms
        self.record_length_ssr = self.N_shot * (self.pi_SSR + self.laser_SST + self.wait_SST)
        #make sure tau is updated
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta) 
        self.freq=(self.freq1+self.freq2)/2
        self.freq_center=self.freq-0.1e+9
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
               
        n_bins = int(self.record_length / self.bin_width)
        time_bins = self.bin_width * np.arange(n_bins)
        sequence = self.generate_sequence()
        """if load button is not used, make sure tau is generated"""
        if(self.tau.shape[0]==2):
            tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta) 
            self.tau = tau         
        #sequence_points = len(tau)
        n_laser = find_laser_pulses(sequence)
        
        self.sequence_points = self._get_sequence_points()
        self.measurement_points = self.sequence_points * int(self.sweeps)
        self.time_bins = time_bins
        self.n_bins = n_bins
        self.n_laser = n_laser
        
        if self.keep_data and sequence == self.sequence and np.all(time_bins == self.time_bins): # if the sequence and time_bins are the same as previous, keep existing data
            #print(1)
            #self.old_count_data = self.count_data.copy()
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
    
          try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            PG.High([])
            
            self.prepare_awg()
            MW.setFrequency(self.freq_center)
            MW.setPower(self.power_XY8)
         
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
              

    get_set_items = ['__doc__', 'freq1', 'freq2', 'record_length', 'record_length_ssr', 'laser','wait', 'bin_width', 'n_bins', 'time_bins', 'n_laser', 'sequence', 'count_data', 'run_time','tau_begin', 'tau_end', 'tau_delta', 'tau','freq_center','power_XY8',
                                'laser_SST','wait_SST','amp_high','vpp','pi_SSR','freq','N_shot','readout_interval','samples_per_read', 'pi2_1', 'pi_1', 'pulse_num', 'rabi_contrast', 'rf_freq', 'amp_rf', 'pi_rf', 'amp_low', 'N_readout_points' ]
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq1',  editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('freq2',  editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('freq_center',  editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('freq',  editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30)
                                     ),
                              HGroup(Item('power_XY8', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp_high', width=-40),   
                                    ),                         
                              HGroup(
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40),
                                     Item('rabi_contrast', width = -40),
                                     ),  

                              HGroup(Item('rf_freq', width=-60), 
                                     Item('amp_rf', width=-40),
                                     Item('pi_rf', width=-40),
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
                              HGroup(Item('amp_low', width=-30),
                                     Item('N_readout_points', width=-30),                                     
                                     Item('pi_SSR', width=-70),
                                     Item('laser_SST', width=-50),
                                     Item('wait_SST', width=-50),
                                     ),      
                              HGroup(Item('samples_per_read', width=-50),
                                     Item('N_shot', width=-50),
                                     Item('record_length_ssr', style='readonly'),
                                     ),        

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30),
                                     ),
                                                                         
                              ),
                       title='Pulsed AWG with SSR',
                       )

class XY8_with_SSR(Pulsed):    # AWG sequence   

    laser = Range(low=1., high=5e4, value=3000, desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
    wait = Range(low=1., high=5e4, value=5000., desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)

    # DD part
    
    record_length = Range(low=100, high=100000., value=3000, desc='length of acquisition record [ns]', label='record length [ns]', mode='text', auto_set=False, enter_set=True)
    bin_width = Range(low=0.1, high=1000., value=3.2, desc='data bin width [ns]', label='bin width [ns]', mode='text', auto_set=False, enter_set=True)
    
    n_laser = Int(2)
    n_bins = Int(2)
    time_bins = Array(value=np.array((0, 1)))

    freq_center = Range(low=1, high=20e9, value=2.06e9, desc='frequency [Hz]', label='MW freq [Hz]', mode='text', auto_set=False, enter_set=True)
    power_XY8 = Range(low=-100., high=25., value= 12, desc='power XY8 [dBm]', label='power XY8 [dBm]', mode='text', auto_set=False, enter_set=True)
    freq1 = Range(low=1, high=20e9, value=1.8e9, desc='frequency 1st trans[Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq2 = Range(low=1, high=20e9, value=1.8e9, desc='frequency 2nd trans[Hz]', label='freq2 [Hz]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=2.71e9, desc='frequency [Hz]', label='mean freq [Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.4e'))
   
    amp_high = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM amp', mode='text', auto_set=False, enter_set=True)
    vpp = Range(low=0., high=4.5, value=0.6, desc='Amplitude of AWG [Vpp]', label='AWG vpp', mode='text', auto_set=False, enter_set=True)    
    
    pi2_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='half pi', mode='text', auto_set=False, enter_set=True)
    pi_1   = Range(low=1., high=100000., value=99.35, desc='length of half pi pulse [ns]', label='pi_x', mode='text', auto_set=False, enter_set=True)
    rabi_contrast = Range(low=1., high=100, value=35.0, desc='Rabi contrast [%]', label='contrast', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=0., high=1e8, value=500., desc='tau/2 begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau/2 end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=0.0, high=1e8, value=50., desc='delta tau/2 [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
  
    pulse_num = Range(low=1, high=1000, value=1, desc='repetetion of XY8 pulses', label='repetitions', mode='text', auto_set=False, enter_set=True)
    reload = True
    
    # state mapping part
    
    rf_freq = Range(low=1, high=20e9, value=7.2e6, desc='frequency rf [Hz]', label='rf freq [Hz]', mode='text', auto_set=False, enter_set=True)    
    amp_rf = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM rf amp', mode='text', auto_set=False, enter_set=True)
    
    pi_rf=Range(low=1., high=100000., value=99.35, desc='length of nuclear pi pulse [ns]', label='pi RF [ns]', mode='text', auto_set=False, enter_set=True)
    
    # Single shot readout part
    
    keep_data = Bool(False) # helper variable to decide whether to keep existing data
    sequence = Instance(list, factory=list)
    
    record_length_ssr = Float(value=0, desc='length of acquisition record [ms]', label='readout length [ms] ', mode='text')
    
    count_data = Array(value=np.zeros(2))
    
    run_time = Float(value=0.0, label='run time [ns]', format_str='%.f')
    stop_time = Range(low=1., value=np.inf, desc='Time after which the experiment stops by itself [s]', label='Stop time [s]', mode='text', auto_set=False, enter_set=True)
    
    sequence_points = Int(value=2, label='number of points', mode='text')
    
    laser_SST = Range(low=1., high=5e6, value=200., desc='laser for SST [ns]', label='laser_SST[ns]', mode='text', auto_set=False, enter_set=True)
    wait_SST = Range(low=1., high=5e6, value=1000., desc='wait for SST[ns]', label='wait_SST [ns]', mode='text', auto_set=False, enter_set=True)
    N_shot = Range(low=1, high=20e5, value=2e3, desc='number of shots in SST', label='N_shot', mode='text', auto_set=False, enter_set=True)
        
    pi_SSR = Range(low=0., high=5e4, value=2e3, desc='pi pulse length', label='pi [ns]', mode='text', auto_set=False, enter_set=True)
    
    amp_low = Range(low=0., high=1.0, value=1.0, desc='Normalized amplitude of waveform', label='WFM Amp for SSR', mode='text', auto_set=False, enter_set=True)
   
    # sweeps = Range(low=1., high=1e4, value=1e2, desc='number of sweeps', label='sweeps', mode='text', auto_set=False, enter_set=True)
    # expected_duration = Property(trait=Float, depends_on='sweeps,sequence', desc='expected duration of the measurement [s]', label='expected duration [s]')
    # elapsed_sweeps = Float(value=0, desc='Elapsed Sweeps ', label='Elapsed Sweeps ', mode='text')
    # elapsed_time = Float(value=0, desc='Elapsed Time [ns]', label='Elapsed Time [ns]', mode='text')
    # progress = Int(value=0, desc='Progress [%]', label='Progress [%]', mode='text')

    load_button = Button(desc='compile and upload waveforms to AWG', label='load')
    
    N_readout_points = Range(low=1., high=1e8, value=50., desc='N readouts', label='N readouts', mode='text', auto_set=False, enter_set=True)
    
    readout_interval = Float(1, label='Data readout interval [s]', desc='How often data read is requested from nidaq')
    samples_per_read = Int(200, label='# data points per read', desc='Number of data points requested from nidaq per read. Nidaq will automatically wait for the data points to be aquired.')
    
    def prepare_awg(self):
        sampling = 1.2e9
         
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            
            N_shot = int(self.N_shot)
            laser_SST = int(self.laser_SST * sampling / 1.0e9)
            wait_SST = int(self.wait_SST * sampling / 1.0e9)
            
            rf_freq = self.rf_freq
        
            pi2_1  = int(self.pi2_1 * sampling/1.0e9)
            pi_1 = int(self.pi_1 * sampling/1.0e9)
            
            pi_RF = int(self.pi_RF * sampling/1.0e9)
            
            pi_SSR = int(self.pi_SSR * sampling/1.0e9)
          
            zero = Idle(1)
            mod = Idle(0)

            # Pulses
            p = {}
            
            p['pi2_1 + 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp_high)]
            p['pi2_1 + 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp_high)]
            
            p['pi2_1 - 0']     = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp_high)]
            p['pi2_1 - 90']    = [Sin( pi2_1, (self.freq - self.freq_center)/sampling, np.pi*3/2 ,self.amp_high)]
            
            p['pi_1_x + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, 0 ,self.amp_high)]
            p['pi_1_x + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp_high)]
            
            p['pi_1_y + 0']     = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi/2 ,self.amp_high)]
            p['pi_1_y + 90']    = [Sin( pi_1, (self.freq - self.freq_center)/sampling, np.pi ,self.amp_high)]
            
            p['pi_RF_x + 0']     = [Sin( pi_RF, self.rf_freq/sampling, 0 , self.amp_rf)]  # RF pi-pulse for SWAP
            
            p['pi_SSR_l_x + 0']     = [Sin( pi_SSR, (self.freq1 - self.freq_center)/sampling, 0 ,self.amp_low)]   # nuclear state selective pi-pulse, for -1/2
            p['pi_SSR_r_x + 0']     = [Sin( pi_SSR, (self.freq2 - self.freq_center)/sampling, 0 ,self.amp_low)]   # for 1/2          
            
            # Waveforms
            self.waves = []
            sub_seq = []
            self.main_seq = Sequence('XY8_SSR.SEQ')
            
            sup_x_ref = Waveform('Sup_x_ref', p['pi2_1 + 0'])
            sup_y_ref = Waveform('Sup_y_ref', p['pi2_1 + 90'])
            t_sup_ref = sup_x_ref.stub
            
            sup_x = Waveform('Sup_x', [Idle(t_sup_ref)]+p['pi2_1 + 0'])
            sup_y = Waveform('Sup_y', [Idle(t_sup_ref)]+p['pi2_1 + 90'])
            sup_rf= Waveform('Sup_rf', [Idle(sup_x.duration)])
            self.waves.append(sup_x)
            self.waves.append(sup_y)
            self.waves.append(sup_rf)
            
            evo = Waveform('EVO.WFM', Idle(256))
            self.waves.append(evo)
            
            # Reference signal_initial points
            
            for i, t in enumerate(self.tau):
                t_tau = t*1.2*2
                
                name = 'SQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(sup_x, sup_y, sup_rf)
                t_0 = sup_x.duration
                t_1 = t * 1.2
                
                for k in range(int(self.pulse_num)/2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % k
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % k
                    rf_name = 'RF_XY8_%03i' % i + '_%03i.WFM' % k
                    
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    map_rf_1 = Waveform(rf_name, [Idle(map_x_1.duration)], t_0)
                    
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    self.waves.append(map_rf_1)
                    sub_seq.append(map_x_1,map_y_1, map_rf_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub
                    
                for k in range(self.pulse_num%2):
                    x_name = 'X_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    y_name = 'Y_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    rf_name = 'RF_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    map_rf_1 = Waveform(rf_name, [Idle(map_x_1.duration)], t_0)
                    
                    self.waves.append(map_x_1)
                    self.waves.append(map_y_1)
                    self.waves.append(map_rf_1)
                    sub_seq.append(map_x_1,map_y_1, map_rf_1)
                    
                    
                    t_0 += map_x_1.duration
                    t_1 = t_tau - map_x_1.stub        
                    
                   
                    
                mod.duration = t * 1.2 - map_x_1.stub
                name_x = 'Read_x_%04i.WFM' % i
                name_y = 'Read_y_%04i.WFM' % i
                name_rf = 'Read_rf_%04i.WFM' % i
                ref_x = Waveform(name_x, [mod] + p['pi2_1 + 0'],t_0)
                ref_y = Waveform(name_y, [mod] + p['pi2_1 + 90'],t_0)
                ref_rf = Waveform(name_rf, Idle[(ref_x.duration)],t_0)

                self.waves.append(ref_x)
                self.waves.append(ref_y)
                self.waves.append(ref_rf)
                sub_seq.append(ref_x, ref_y, ref_rf) # here the XY8 ends
                                
                # SWAP gate--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
                t_0 += ref_x.duration
                
                name_x = 'Swap_x_%04i.WFM' % i
                name_y = 'Swap_y_%04i.WFM' % i
                name_rf = 'Swap_rf_%04i.WFM' % i
                
                sup_rf_ref = Waveform('Sup_rf_ref',  p['pi_RF_x + 0'])
                sup_ssr_l_ref = Waveform('Sup_ssr_l_ref', p['pi_SSR_l_x + 0'])
                sup_ssr_r_ref = Waveform('Sup_ssr_r_ref', p['pi_SSR_r_x + 0'])
                
                t_1_ref = sup_ssr_l_ref.stub
                t_rf_ref = sup_rf_ref.stub
                t_2_ref = sup_ssr_r_ref.stub
                
                swap_x = Waveform(name_x, p['pi_SSR_l_x + 0']+[Idle(t_rf_ref)]+p['pi_SSR_r_x + 0'],t_0)
                swap_y = Waveform(name_y, [Idle(swap_x.duration)],t_0)
                swap_rf = Waveform(name_rf, [Idle(t_1_ref)]+p['pi_RF_x + 0']+[Idle(t_2_ref)],t_0)
                
                self.waves.append(swap_x)
                self.waves.append(swap_y)
                self.waves.append(swap_rf)
                sub_seq.append(swap_x, swap_y, swap_rf)
                
                # Laser initialize the NV spin for sunsequent readout--------------------------------------------------------------------------------------------------------------------------------------------------------
                                
                t_0 += swap_x.duration
                
                # SSR--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
                name_x='ssr_x' % i
                name_y='ssr_y' % i
                name_rf='ssr_rf' % i
                
                ssr_x = Waveform(name_x, [p['pi_SSR_l_x + 0']+[Idle(laser_SST, marker1 = 1)]+ [Idle(wait_SST)]], t_0)
                ssr_y = Waveform(name_y, [Idle(ssr_x.duration)], t_0)
                ssr_rf = Waveform(name_rf, [Idle(ssr_x.duration)], t_0)
                self.waves.append(ssr_x)
                self.waves.append(ssr_y)
                self.waves.append(ssr_rf)
            
                for i, t in enumerate(self.N_readout_points):
                    sub_seq.append(ssr_x, ssr_y, ssr_rf, repeat = N_shot)
                    
                    
                ssr_x_1 = Waveform('ssr_x_1', [p['pi_SSR_r_x + 0']+[Idle(laser_SST, marker1 = 1)]+ [Idle(wait_SST)]], t_0)
                ssr_y_1 = Waveform('ssr_y_1', [Idle(ssr_x_1.duration)], t_0)
                ssr_rf_1 = Waveform('ssr_rf_1', [Idle(ssr_x_1.duration)], t_0)
                self.waves.append(ssr_x_1)
                self.waves.append(ssr_y_1)
                self.waves.append(ssr_rf_1)
                    
                for i, t in enumerate(self.N_readout_points):
                    sub_seq.append(ssr_x_1, ssr_y_1, ssr_rf_1, repeat = N_shot)
                   
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True) 
                                
                #another projection------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
                # t_tau = t*1.2*2
                # name = 'BSQH_12_%04i.SEQ' % i
                # sub_seq=Sequence(name)
                # sub_seq.append(sup_x,sup_y, sup_rf)
                # t_0 = sup_x.duration
                # t_1 = t * 1.2
                # for k in range(int(self.pulse_num)/2):
                    # x_name = 'BX_XY8_%03i' % i + '_%03i.WFM' % k
                    # y_name = 'BY_XY8_%03i' % i + '_%03i.WFM' % k
                    # rf_name = 'BRF_XY8_%03i' % i + '_%03i.WFM' % k
                    # map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                # +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']\
                                                # +[Idle(t_tau)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                # +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    # map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                # +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']\
                                                # +[Idle(t_tau)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                # +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0)  
                    # map_rf_1 = Waveform(rf_name, [Idle(map_x_1.duration)], t_0)
                    # self.waves.append(map_x_1)
                    # self.waves.append(map_y_1)
                    # self.waves.append(map_rf_1)
                    # sub_seq.append(map_x_1, map_y_1, map_rf_1)
                    
                    
                    # t_0 += map_x_1.duration
                    # t_1 = t_tau - map_x_1.stub
                    
                # for k in range(self.pulse_num%2):
                    # x_name = 'BX_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    # y_name = 'BY_XY8_%03i' % i + '_%03i.WFM' % (self.pulse_num/2)
                    # map_x_1 = Waveform(x_name,   [Idle(t_1)] +p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']\
                                                # +[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0']+[Idle(t_tau)]+p['pi_1_y + 0']+[Idle(t_tau)]+p['pi_1_x + 0'],t_0)
                    # map_y_1 = Waveform(y_name, [Idle(t_1)] +p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']\
                                                # +[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90']+[Idle(t_tau)]+p['pi_1_y + 90']+[Idle(t_tau)]+p['pi_1_x + 90'],t_0) 
                    # map_rf_1 = Waveform(rf_name, [Idle(map_x_1.duration)], t_0)
                    # self.waves.append(map_x_1)
                    # self.waves.append(map_y_1)
                    # self.waves.append(map_rf_1)
                    # sub_seq.append(map_x_1, map_y_1, map_rf_1)
                    
                    
                    # t_0 += map_x_1.duration
                    # t_1 = t_tau - map_x_1.stub   
                    
                # mod.duration = t * 1.2 - map_x_1.stub
                # name_x = 'BRead_x_%04i.WFM' % i
                # name_y = 'BRead_y_%04i.WFM' % i
                # name_rf = 'BRead_rf_%04i.WFM' % i
                # ref_x = Waveform(name_x, [mod] + p['pi2_1 - 0'],t_0)
                # ref_y = Waveform(name_y, [mod] + p['pi2_1 - 90'],t_0)
                # ref_rf = Waveform(name_rf, Idle[(ref_x.duration)],t_0)

                # self.waves.append(ref_x)
                # self.waves.append(ref_y)
                # self.waves.append(ref_rf)
                # sub_seq.append(ref_x, ref_y, ref_rf)
                
                #SWAP gate--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
                # t0 += ref_x.duration
                
                # name_x = 'Swap_x_%04i.WFM' % i
                # name_y = 'Swap_y_%04i.WFM' % i
                # name_rf = 'Swap_rf_%04i.WFM' % i
                
                # sup_rf_ref = Waveform('Sup_rf_ref',  p['pi_RF_x + 0'])
                # sup_ssr_l_ref = Waveform('Sup_ssr_l_ref', p['pi_SSR_l_x + 0'])
                # sup_ssr_r_ref = Waveform('Sup_ssr_r_ref', p['pi_SSR_r_x + 0'])
                
                # t_1_ref = sup_ssr_l_ref.stub
                # t_rf_ref = sup_rf_ref.stub
                # t_2_ref = sup_ssr_r_ref.stub
                
                # swap_x = Waveform(name_x, p['pi_SSR_l_x + 0']+[Idle(t_rf_ref)],t_0)
                # swap_y = Waveform(name_y, [Idle(swap_x.duration)],t_0)
                # swap_rf = Waveform(name_rf, [Idle(t_1_ref)]+p['pi_RF_x + 0'],t_0)
                
                # self.waves.append(swap_x)
                # self.waves.append(swap_y)
                # self.waves.append(swap_rf)
                # sub_seq.append(swap_x, swap_y, swap_rf)
                
                #Laser initialize the NV spin for sunsequent readout--------------------------------------------------------------------------------------------------------------------------------------------------------
                                
                # t0 += swap_x.duration
                
                #SSR--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
                # ssr_x = Waveform('ssr_x', [p['pi_SSR_l_x + 0']+[Idle(laser_SST, marker1 = 1)]+ [Idle(wait_SST)], t0)
                # ssr_y = Waveform('ssr_y', [Idle(ssr_x.duration)], t0)
                # ssr_rf = Waveform('ssr_rf', [Idle(ssr_x.duration)], t0)
                # self.waves.append(ssr_x)
                # self.waves.append(ssr_y)
                # self.waves.append(ssr_rf)
            
                # for i, t in enumerate(self.N_readout_points):
                    # sub_seq.append(ssr_x, ssr_y, ssr_rf, repeat = N_shot)
                    
                    
                # ssr_x_1 = Waveform('ssr_x_1', [p['pi_SSR_r_x + 0']+[Idle(laser_SST, marker1 = 1)]+ [Idle(wait_SST)], t0)
                # ssr_y_1 = Waveform('ssr_y_1', [Idle(ssr_x_1.duration)], t0)
                # ssr_rf_1 = Waveform('ssr_rf_1', [Idle(ssr_x_1.duration)], t0)
                # self.waves.append(ssr_x_1)
                # self.waves.append(ssr_y_1)
                # self.waves.append(ssr_rf_1)
                    
                # for i, t in enumerate(self.N_readout_points):
                    # sub_seq.append(ssr_x, ssr_y, ssr_rf, repeat = N_shot)
                
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)

            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            #time.sleep(70)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('XY8_SSR.SEQ')
            
        AWG.set_vpp(self.vpp)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0111 )  
        
    def generate_sequence(self):
    
    
        points = int(self.N_readout_points)
        N_shot = self.N_shot
        laser = self.laser
        wait = self.wait
        laser_SST = self.laser_SST
        wait_SST = self.wait_SST
        pi = self.pi
        pi_SSR = self.pi_SSR
        pi_RF = self.pi_RF
        record_length_ssr = self.record_length_ssr*1e+6        
        tau = self.tau
        pi2_1  = self.pi2_1
        pi_1 = self.pi_1
        sequence = []
        
        for i, t in enumerate(tau):
            sequence.append( (['laser', 'trigger'], laser) )
            sequence.append( ([ ]                 , wait) )
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ ]                 , t*16 * self.pulse_num + 2*pi2_1 + 8*self.pulse_num*pi_1 + 2000+pi_SSR+pi.RF) )
                        
            for i in 2*range(points):
                #sequence.append( (['laser'], laser) )
                #sequence.append( ([ ],  wait) )
                #sequence.append( (['awgTrigger']      , 100) )
                sequence.append( ([ 'sst' ]       , record_length_ssr) )
                            
            # sequence.append( (['awgTrigger']      , 100) )
            # sequence.append( ([ ]                 , t*16 * self.pulse_num + 2*pi2_1 + 8*self.pulse_num*pi_1 + 2000) )
            # sequence.append( (['laser', 'trigger'], laser) )
            # sequence.append( ([ ]                 , wait) )
        return sequence    
                    
    get_set_items = Pulsed.get_set_items + ['pi2_1','pi_1','pulse_num', 'rabi_contrast']   
        
        
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq1',  width=-60),
                                     Item('freq2',  width=-60),
                                     Item('freq_center',  width=20),
                                     Item('freq',  width=-60)
                                     ),                         
                              HGroup(
                                     Item('pi2_1', width = -40),
                                     Item('pi_1', width = -40),
                                     Item('pulse_num', width = -40),
                                     Item('rabi_contrast', width = -40),
                                     Item('power_XY8', width=-40),
                                     Item('vpp', width=-40),   
                                     Item('amp_high', width=-40), 
                                     ),  

                              HGroup(Item('rf_freq', width=-60), 
                                     Item('amp_rf', width=-40),
                                     Item('pi_rf', width=-40),
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
                              HGroup(Item('amp_low', width=-30),
                                     Item('N_readout_points', width=-30),                                     
                                     Item('pi_SSR', width=-70),
                                     ),      

                              HGroup(
                                     Item('laser_SST', width=-50),
                                     Item('wait_SST', width=-50),
                                     ),
                                     
                              HGroup(Item('samples_per_read', width=-50),
                                     Item('N_shot', width=-50),
                                     Item('record_length_ssr', style='readonly'),
                                     ),        

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f', width=50),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30),
                                     ),
                                                                         
                              ),
                       title='XY8_with_SSR',
                       )     