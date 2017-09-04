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

from hardware.api import PulseGenerator 
from hardware.api import Microwave 
from hardware.api import AWG, FastComtec
import hardware.api as ha

from tools.emod import ManagedJob

from tools.utility import GetSetItemsMixin

PG = PulseGenerator()
MW = Microwave()
FC = FastComtec()
#AWG = AWG()

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

class Pulsed(ManagedJob, GetSetItemsMixin):
    
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
            MW.setFrequency(self.freq)
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
            PG.High(['laser'])
            if self.elapsed_sweeps < self.sweeps:
                self.state = 'idle'
            else:
                self.state='done'  

        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'
                

    get_set_items = ['__doc__', 'record_length','laser','wait', 'bin_width', 'n_bins', 'time_bins', 'n_laser', 'sequence', 'count_data', 'run_time','tau_begin', 'tau_end', 'tau_delta', 'tau','power']
    
    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                              HGroup(Item('freq', width=40),
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
                       title='Pulsed Measurement',
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
                              HGroup(Item('freq', width=40),
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
                       title='Rabi Measurement',
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
                                HGroup(  Item('freq', width=40),
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
                                         
                 title='FID',
                 )                       

class Hahn( Pulsed ):
    """Hahn echo measurement using standard pi/2-pi-pi/2 sequence.
    """

    t_pi2 = Range(low=1., high=100000., value=1000., desc='length of pi/2 pulse [ns]', label='pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
    t_pi   = Range(low=1., high=100000., value=1000., desc='length of pi pulse [ns]', label='pi [ns]', mode='text', auto_set=False, enter_set=True)
    t_3pi2 = Range(low=1., high=100000., value=1000., desc='length of 3pi/2 pulse [ns]', label='3pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
    rabi_contrast=Range(low=1., high=100, value=30.0, desc='Rabi contrast [%]', label='contrast', mode='text', auto_set=False, enter_set=True)
    
    def __init__(self):
        super(Hahn, self).__init__()
     
    def _get_sequence_points(self):
        return 2 * len(self.tau) 
        
    def generate_sequence(self):
    
        tau = self.tau
        laser = self.laser
        wait  = self.wait
        t_pi2  = self.t_pi2
        t_pi   = self.t_pi
        t_3pi2 = self.t_3pi2
        sequence = []
        for t in tau:
        
            sub = [ (['mw'   ],            t_pi2  ),
                    ([       ],            t      ),
                    (['mw'   ],            t_pi   ),
                    ([       ],            t      ),
                    (['mw'   ],            t_pi2  ),
                    (['laser', 'trigger'], laser  ),
                    ([       ],            wait   ),
                    
                    
                    (['mw'   ],            t_pi2  ),
                    ([       ],            t      ),
                    (['mw'   ],            t_pi   ),
                    ([       ],            t      ),
                    (['mw'   ],            t_3pi2  ),
                    (['laser', 'trigger'], laser  ),
                    ([       ],            wait   )
                  ]
            sequence.extend(sub)
    
        return sequence
        
    get_set_items = Pulsed.get_set_items + ['t_pi2','t_pi', 't_3pi2']

    traits_view = View(  VGroup( HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                                HGroup(  Item('freq', width=40),
                                         Item('power', width=20),
                                         Item('t_pi2', width=20),
                                         Item('t_pi', width=20),
                                         Item('t_3pi2', width=20),),
                                         
                                 HGroup( Item('tau_begin', width=20),
                                         Item('tau_end', width=20),
                                         Item('tau_delta', width=20),
                                         Item('rabi_contrast', width=20)),
                                         
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
                                         
                 title='Hahn',
                 )
                 
class T1(Hahn):
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait  = self.wait
        t_pi2  = self.t_pi2
        t_pi   = self.t_pi
        t_3pi2 = self.t_3pi2
        sequence = []
        for t in tau:       
            #sequence.append(  (['mw'   ],            t_3pi2  )  )
            #sequence.append(  (['mw'   ],            t_pi  )  )
            sequence.append(  ([       ],            t      )  )
            sequence.append(  (['laser', 'trigger'], laser  )  )
            sequence.append(  ([       ],            wait   )  )
        #for t in tau:
            sequence.append(  (['mw'       ],            t_pi      )  )
            sequence.append(  ([       ],            t      )  )
            sequence.append(  (['laser', 'trigger'], laser  )  )
            sequence.append(  ([       ],            wait   )  )
        return sequence
        
    def _get_sequence_points(self):
        return 2 * len(self.tau) 

    get_set_items = Pulsed.get_set_items + ['t_pi2','t_pi', 't_3pi2']

    traits_view = View(  VGroup( HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     ),
                                HGroup(  Item('freq', width=40),
                                         Item('power', width=20),
                                         Item('t_pi2', width=20),
                                         Item('t_pi', width=20),
                                         Item('t_3pi2', width=20),),
                                         
                                 HGroup( Item('tau_begin', width=20),
                                         Item('tau_end', width=20),
                                         Item('tau_delta', width=20),
                                         Item('rabi_contrast', width=20)),
                                         
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
                                         
                 title='T1',
                 )
                 

