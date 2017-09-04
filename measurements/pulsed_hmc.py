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

from hardware.waveform import *

from hardware.api import PulseGenerator 
from hardware.api import Microwave 
from hardware.api import Microwave_HMC
from hardware.api import AWG, FastComtec
from hardware import awg

from tools.emod import ManagedJob

from tools.utility import GetSetItemsMixin

PG = PulseGenerator()
MW = Microwave()
MW_HMC = Microwave_HMC() # only one MW source HMC
FC = FastComtec()
AWG = awg.AWG()

"""
Several options to decide when to start  and when to restart a job, i.e. when to clear data, etc.

1. set a 'new' flag on every submit button

pro: simple, need not to think about anything in subclass

con: continue of measurement only possible by hack (manual submit to JobManager without submit button)
     submit button does not do what it says
     
2. check at start time whether this is a new measurement.

pro: 

con: complicated checking needed
     checking has to be reimplemented on sub classes
     no explicit way to restart the same measurement

3. provide user settable clear / keep flag

pro: explicit

con: user can forget

4. provide two different submit buttons: submit, resubmit

pro: explicit

con: two buttons that user may not understand
     user may use wrong button
     wrong button can result in errors

"""

# utility functions
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

def sequence_length(sequence):
    t = 0
    for c, ti in sequence:
        t += ti
    return t

def sequence_union(s1, s2):
    """
    Return the union of two pulse sequences s1 and s2.
    """
    # make sure that s1 is the longer sequence and s2 is merged into it
    if sequence_length(s1) < sequence_length(s2):
        sp = s2
        s2 = s1
        s1 = sp
    s = []
    c1, dt1 = s1.pop(0)
    c2, dt2 = s2.pop(0)
    while True:
        if dt1 < dt2:
            s.append((set(c1) | set(c2), dt1))
            dt2 -= dt1
            try:
                c1, dt1 = s1.pop(0)
            except:
                break
        elif dt2 < dt1:
            s.append((set(c1) | set(c2), dt2))
            dt1 -= dt2
            try:
                c2, dt2 = s2.pop(0)
            except:
                c2 = []
                dt2 = np.inf
        else:
            s.append((set(c1) | set(c2), dt1))
            try:
                c1, dt1 = s1.pop(0)
            except:
                break
            try:
                c2, dt2 = s2.pop(0)
            except:
                c2 = []
                dt2 = np.inf            
    return s

def sequence_remove_zeros(sequence):
    return filter(lambda x: x[1] != 0.0, sequence)

class Pulsed(ManagedJob, GetSetItemsMixin): # Rabi measurements without AWG
    
    """Defines a pulsed measurement."""
    
    keep_data = Bool(False) # helper variable to decide whether to keep existing data

    resubmit_button = Button(label='resubmit', desc='Submits the measurement to the job manager. Tries to keep previously acquired data. Behaves like a normal submit if sequence or time bins have changed since previous run.')

    sequence = Instance(list, factory=list)
    
    record_length = Range(low=100, high=100000., value=3000, desc='length of acquisition record [ns]', label='record length [ns]', mode='text', auto_set=False, enter_set=True)
    bin_width = Range(low=0.1, high=1000., value=3.2, desc='data bin width [ns]', label='bin width [ns]', mode='text', auto_set=False, enter_set=True)
    
    n_laser = Int(2)
    n_bins = Int(2)
    time_bins = Array(value=np.array((0, 1)))
    
    count_data = Array(value=np.zeros((2, 2)))
    
    run_time = Float(value=0.0, label='run time [ns]', format_str='%.f')
    stop_time = Range(low=1., value=np.inf, desc='Time after which the experiment stops by itself [s]', label='Stop time [s]', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=0., high=1e8, value=300., desc='tau begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=1e8, value=50., desc='delta tau [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)

    tau = Array(value=np.array((0., 1.)))

    laser = Range(low=1., high=5e6, value=3000., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
    wait = Range(low=1., high=5e6, value=5000., desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)
    
    freq_LO = Range(low=0.8e9, high=20e9, value=2.0e9, desc='frequency [Hz]', label='frequency_LO [Hz]', mode='text', auto_set=False, enter_set=True)
    power_LO = Range(low=-10., high=25., value=6, desc='power [dBm]', label='power_LO [dBm]', mode='text', auto_set=False, enter_set=True)
    
    freq = Range(low=1, high=20e9, value=2.71e9, desc='frequency [Hz]', label='frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value=-26, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)

    sweeps = Range(low=1., high=1e10, value=1e6, desc='number of sweeps', label='sweeps', mode='text', auto_set=False, enter_set=True)
    expected_duration = Property(trait=Float, depends_on='sweeps,sequence', desc='expected duration of the measurement [s]', label='expected duration [s]')
    elapsed_sweeps = Float(value=0, desc='Elapsed Sweeps ', label='Elapsed Sweeps ', mode='text')
    elapsed_time = Float(value=0, desc='Elapsed Time [ns]', label='Elapsed Time [ns]', mode='text')
    progress = Int(value=0, desc='Progress [%]', label='Progress [%]', mode='text')

    import_code = Code()

    import_button = Button(desc='set parameters such as pulse length, frequency, power, etc. by executing import code specified in settings', label='import')

    def __init__(self):
        super(Pulsed, self).__init__()
        
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
        tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta)   
        self.tau = tau

        sequence = self.generate_sequence()
        n_laser = find_laser_pulses(sequence)
        
        self.sequence = sequence 
        self.sequence_points = self._get_sequence_points()
        self.time_bins = time_bins
        self.n_bins = n_bins
        self.n_laser = n_laser

        if self.keep_data and sequence == self.sequence and np.all(time_bins == self.time_bins): # if the sequence and time_bins are the same as previous, keep existing data
            self.old_count_data = self.count_data.copy()
            self.previous_sweeps = self.elapsed_sweeps
            self.previous_elapsed_time = self.elapsed_time
        else:
             #self.old_count_data = np.zeros((n_laser, n_bins))
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.check = True
            self.old_count_data = np.zeros(FC.GetData().shape)
            self.previous_sweeps = 0
            self.previous_elapsed_time = 0.0
            self.run_time = 0.0
        
        
        self.keep_data = True # when job manager stops and starts the job, data should be kept. Only new submission should clear data.
           

        
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
            #self.previous_sweeps = 0
            #self.previous_count_data = FC.GetData()
            MW_HMC.setFrequency(self.freq_LO)
            MW_HMC.setPower(self.power_LO)
            MW.setFrequency(self.freq-self.freq_LO)
            MW.setPower(self.power)
            time.sleep(2.0)
            FC.Start()
            time.sleep(0.1)
            PG.Sequence(self.sequence, loop=True)

            start_time = time.time()

            while self.run_time < self.stop_time:
               self.thread.stop_request.wait(1.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               self.elapsed_time = time.time() - start_time
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
            MW_HMC.Off()
            PG.High(['laser'])
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  

        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'
                

    get_set_items = ['__doc__', 'record_length','laser','wait', 'bin_width', 'n_bins', 'time_bins', 'n_laser', 'sequence', 'count_data', 'run_time','tau_begin', 'tau_end', 'tau_delta', 'tau','freq_LO','power_LO','freq','power']
    
    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq_LO', width=40),
                                     Item('power_LO', width=40),
                                     Item('freq', width=40),
                                     Item('power', width=40),
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
                                     Item('run_time', style='readonly', format_str='%.f'),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=40),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=40),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=40),
                                     ),
                                   
                              ),
                       title='Pulsed_HMC Measurement',
                       )

             
class Rabi( Pulsed ):
    """Rabi measurement.
    """
    def __init__(self):
        super(Rabi, self).__init__()
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        sequence = []
        for t in tau:
            sequence.append(  (['mw'               ] , t       )  )
            sequence.append(  (['laser', 'trigger' ] , laser   )  )
            sequence.append(  ([                   ] , wait    )  )
        return sequence
        
    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq_LO', width=40),
                                     Item('power_LO', width=40),
                                     Item('freq', width=40),
                                     Item('power', width=40),
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
                       title='Rabi_HMC Measurement',
                       )
                       
class FID( Pulsed ):
    """Hahn echo measurement using standard pi/2-pi-pi/2 sequence.
    """

    t_pi2 = Range(low=1., high=100000., value=1000., desc='length of pi/2 pulse [ns]', label='pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
   
    def __init__(self):
        super(FID, self).__init__()
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait  = self.wait
        t_pi2  = self.t_pi2
        sequence = []
        for t in tau:
            sequence.append(  (['mw'   ],            t_pi2  )  )
            sequence.append(  ([       ],            t      )  )
            sequence.append(  (['mw'   ],            t_pi2   )  )
            sequence.append(  (['laser', 'trigger'], laser  )  )
            sequence.append(  ([       ],            wait   )  )
        return sequence
        
    get_set_items = Pulsed.get_set_items + ['t_pi2']

    traits_view = View(  VGroup( HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                                HGroup( Item('freq_LO', width=40),
                                        Item('power_LO', width=40), 
                                        Item('freq', width=40),
                                        Item('power', width=20),
                                        Item('t_pi2', width=20),
                                      ),
                                         
                                 HGroup( Item('tau_begin', width=20),
                                         Item('tau_end', width=20),
                                         Item('tau_delta', width=20),),
                                         
                                 HGroup( Item('laser', width=40),
                                         Item('wait', width=40),
                                         Item('bin_width', width=40),),
                                         
                                 HGroup( Item('state', style='readonly'),
                                         Item('run_time', style='readonly', format_str='%.f', width=50),
                                         Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=40),
                                         Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.3e'%x), width=40),
                                         Item('progress', style='readonly'),
                                         Item('elapsed_time', style='readonly'),
                                       ),
                                ),
                                         
                 title='FID_HMC',
                 )                       

class Hahn( Pulsed ):
    """Hahn echo measurement using standard pi/2-pi-pi/2 sequence.
    """

    t_pi2 = Range(low=1., high=100000., value=1000., desc='length of pi/2 pulse [ns]', label='pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
    t_pi   = Range(low=1., high=100000., value=1000., desc='length of pi pulse [ns]', label='pi [ns]', mode='text', auto_set=False, enter_set=True)
    t_3pi2 = Range(low=1., high=100000., value=1000., desc='length of 3pi/2 pulse [ns]', label='3pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
    
    def __init__(self):
        super(Hahn, self).__init__()
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait  = self.wait
        t_pi2  = self.t_pi2
        t_pi   = self.t_pi
        t_3pi2 = self.t_3pi2
        sequence = []
        for t in tau:
            sequence.append(  (['mw'   ],            t_pi2  )  )
            sequence.append(  ([       ],            t      )  )
            sequence.append(  (['mw'   ],            t_pi   )  )
            sequence.append(  ([       ],            t      )  )
            sequence.append(  (['mw'   ],            t_pi2  )  )
            sequence.append(  (['laser', 'trigger'], laser  )  )
            sequence.append(  ([       ],            wait   )  )
        return sequence
        
    get_set_items = Pulsed.get_set_items + ['t_pi2','t_pi']

    traits_view = View(  VGroup( HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                                  HGroup(
                                     Item('freq_LO', width=40),
                                     Item('power_LO', width=40),
                                     ),
                                HGroup(  Item('freq', width=40),
                                         Item('power', width=20),
                                         Item('t_pi2', width=20),
                                         Item('t_pi', width=20),
                                         Item('t_3pi2', width=20),),
                                         
                                 HGroup( Item('tau_begin', width=20),
                                         Item('tau_end', width=20),
                                         Item('tau_delta', width=20),),
                                         
                                 HGroup( Item('laser', width=40),
                                         Item('wait', width=40),
                                         Item('bin_width', width=40),),
                                         
                                 HGroup( Item('state', style='readonly'),
                                         Item('run_time', style='readonly', format_str='%.f', width=50),
                                         Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=40),
                                         Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.3e'%x), width=40),
                                         Item('progress', style='readonly'),
                                         Item('elapsed_time', style='readonly'),
                                       ),
                                ),
                                         
                 title='Hahn_HMC',
                 )
                 
                 
class PulsedAWG(ManagedJob, GetSetItemsMixin):

    """Defines a pulsed measurement."""
    keep_data = Bool(False) # helper variable to decide whether to keep existing data

    resubmit_button = Button(label='resubmit', desc='Submits the measurement to the job manager. Tries to keep previously acquired data. Behaves like a normal submit if sequence or time bins have changed since previous run.')

    sequence = Instance(list, factory=list)
    
    record_length = Range(low=100, high=100000., value=3000, desc='length of acquisition record [ns]', label='record length [ns]', mode='text', auto_set=False, enter_set=True)
    bin_width = Range(low=0.1, high=1000., value=3.2, desc='data bin width [ns]', label='bin width [ns]', mode='text', auto_set=False, enter_set=True)
    
    n_laser = Int(2)
    n_bins = Int(2)
    time_bins = Array(value=np.array((0, 1)))
    
    count_data = Array(value=np.zeros((2, 2)))
    
    run_time = Float(value=0.0, label='run time [ns]', format_str='%.f')
    stop_time = Range(low=1., value=np.inf, desc='Time after which the experiment stops by itself [s]', label='Stop time [s]', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=0., high=1e8, value=300., desc='tau begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=4000., desc='tau end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=1e8, value=50., desc='delta tau [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)

    tau = Array(value=np.array((0., 1.)))

    laser = Range(low=1., high=5e6, value=3000., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
    wait = Range(low=1., high=5e6, value=5000., desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)
    
    freq_center = Range(low=1, high=20e9, value=2.71e9, desc='frequency [Hz]', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power = Range(low=-100., high=25., value=-26, desc='power [dBm]', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    freq = Range(low=1, high=20e9, value=1.8e9, desc='frequency [Hz]', label='freq1 [Hz]', mode='text', auto_set=False, enter_set=True)
    
    sweeps = Range(low=1., high=1e10, value=1e6, desc='number of sweeps', label='sweeps', mode='text', auto_set=False, enter_set=True)
    expected_duration = Property(trait=Float, depends_on='sweeps,sequence', desc='expected duration of the measurement [s]', label='expected duration [s]')
    elapsed_sweeps = Float(value=0, desc='Elapsed Sweeps ', label='Elapsed Sweeps ', mode='text')
    elapsed_time = Float(value=0, desc='Elapsed Time [ns]', label='Elapsed Time [ns]', mode='text')
    progress = Int(value=0, desc='Progress [%]', label='Progress [%]', mode='text')

    load_button = Button(desc='compile and upload waveforms to AWG', label='load')
    reload = True


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
        #make sure tau is updated
        
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta) 
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
        self.time_bins = time_bins
        self.n_bins = n_bins
        self.n_laser = n_laser
        
        if self.keep_data and sequence == self.sequence and np.all(time_bins == self.time_bins): # if the sequence and time_bins are the same as previous, keep existing data
            #print(1)
            self.old_count_data = self.count_data.copy()
            self.previous_sweeps = self.elapsed_sweeps
            self.previous_elapsed_time = self.elapsed_time
            self.keep_data = True # when job manager stops and starts the job, data should be kept. Only new submission should clear data.
        else:
            
            #self.old_count_data = np.zeros((n_laser, n_bins))
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.check = True
            self.old_count_data = np.zeros(FC.GetData().shape)
            self.count_data = np.zeros(FC.GetData().shape)
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
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            FC.Configure(self.laser, self.bin_width, self.sequence_points)
            #self.previous_time = 0
            #self.previous_count_data = FC.GetData()
            self.prepare_awg()
            MW_HMC.setFrequency(self.freq_center)
            MW_HMC.setPower(self.power)
            AWG.run()
            time.sleep(4.0)
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
               #print(FC.GetData()[0:20])
               if self.elapsed_sweeps > self.sweeps:
                  break   

            FC.Halt()
            time.sleep(0.1)
            MW_HMC.Off()
            time.sleep(0.1)
            PG.High(['laser'])
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
            #AWG._server_close()
                

    get_set_items = ['__doc__', 'record_length','laser','wait', 'bin_width', 'n_bins', 'time_bins', 'n_laser', 'sequence', 'count_data', 'run_time','tau_begin', 'tau_end', 'tau_delta', 'tau','freq_center','power', 'freq']
    
    traits_view = View(VGroup(HGroup(Item('load_button', show_label=False),
                                     Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq', width=30),
                                     Item('power', width=20),
                                     ),       
                              HGroup(Item('laser', width=20),
                                     Item('wait', width=20),
                                     Item('bin_width', width= 20, enabled_when='state != "run"'),
                                     Item('record_length', width= 20, enabled_when='state != "run"'),
                                     ),
                                     
                              HGroup(Item('tau_begin', width=40),
                                     Item('tau_end', width=40),
                                     Item('tau_delta', width= 40),
                                     ),       

                              HGroup(Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f'),
                                     Item('sweeps', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.3e'%x), width=30),
                                     Item('expected_duration', style='readonly', editor=TextEditor(evaluate=float, format_func=lambda x:'%.f'%x), width=30),
                                     Item('progress', style='readonly'),
                                     Item('elapsed_time', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.f'%x), width=30),
                                     ),
                                   
                              ),
                       title='Pulsed AWG Measurement',
                       )
                       
class RabiAWG( PulsedAWG ):
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

class Hahn( PulsedAWG ):
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
                    (['laser', 'trigger'], laser ),
                    ([], wait ),
                    
                    (['awgTrigger'], 100 ),
                    ([], 2 * t + pi_1 + pi2_1*2 + 100),
                    (['laser', 'trigger'], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)

        return sequence
            
    get_set_items = PulsedAWG.get_set_items + ['freq','vpp','amp','pi2_1','pi_1', 'rabi_contrast'] 
    
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

class T1( PulsedAWG ):
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
                    (['laser', 'trigger'], laser ),
                    ([], wait ),
                    
                    (['awgTrigger'], 100 ),
                    ([], t + pi_1 * 2 + 100),
                    (['laser', 'trigger'], laser ),
                    ([], wait )
                  ]
            sequence.extend(sub)

        return sequence
            
    get_set_items = PulsedAWG.get_set_items + ['freq','vpp','amp','pi_1'] 
    
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
                       
                       
       