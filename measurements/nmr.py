import numpy as np

from traits.api import Trait, Instance, Property, String, Range, Float, Int, Bool, Array, Enum
from traitsui.api import View, Item, HGroup, VGroup, VSplit, Tabbed, EnumEditor, TextEditor, Group, Label

import logging
import time

import hardware.api as ha

from pulsed import Pulsed

class NMR(Pulsed):
    
    mwA_frequency = Range(low=1, high=20e9, value=2.87405e+09, desc='microwave A frequency', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    mwA_power = Range(low= -100., high=25., value= -19, desc='microwave A power', label='MW power [dBm]', mode='text', auto_set=False, enter_set=True)
    t_pi_A = Range(low=1., high=100000., value=1040., desc='length of pi pulse of MW A[ns]', label='pi [ns]', mode='text', auto_set=False, enter_set=True)

    mwB_frequency = Range(low=1, high=20e9, value=2.880965e9, desc='microwave B frequency', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    mwB_power = Range(low= -100., high=25., value= -19, desc='microwave B power', label='MW power [dBm]', mode='text', auto_set=False, enter_set=True)
    t_pi_B = Range(low=1., high=100000., value=1020., desc='length of pi pulse of MW B[ns]', label='pi [ns]', mode='text', auto_set=False, enter_set=True)


    rf_power = Range(low= -130., high=25., value= -20, desc='RF power', label='RF power [dBm]', mode='text', auto_set=False, enter_set=True)
    rf_begin = Range(low=1, high=200e6, value=2.4e6, desc='Start Frequency [Hz]', label='RF Begin [Hz]', mode='text', auto_set=False, enter_set=True)
    rf_end = Range(low=1, high=200e6, value=2.7e6, desc='Stop Frequency [Hz]', label='RF End [Hz]', mode='text', auto_set=False, enter_set=True)
    rf_delta = Range(low=1e-3, high=200e6, value=2.0e3, desc='frequency step [Hz]', label='Delta [Hz]', mode='text', auto_set=False, enter_set=True)
    rf_t_pi = Range(low=1., high=1.e7, value=1.e6, desc='length of pi pulse of RF[ns]', label='RF pi [ns]', mode='text', auto_set=False, enter_set=True)
    # ESR:
    """
    rf_begin        = Range(low=1,      high=10e9, value=2.8e9,    desc='Start Frequency [Hz]',    label='RF Begin [Hz]')
    rf_end          = Range(low=1,      high=10e9, value=2.9e9,    desc='Stop Frequency [Hz]',     label='RF End [Hz]')
    rf_delta        = Range(low=1e-3,   high=10e9, value=1e6,       desc='frequency step [Hz]',     label='Delta [Hz]')
    """
    laser = Range(low=1., high=1.0e7, value=3000., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
    wait = Range(low=1., high=1.0e7, value=2.0e6, desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)

    seconds_per_point = Range(low=0.1, high=10., value=0.5, desc='Seconds per point', label='Seconds per point', mode='text', auto_set=False, enter_set=True)
    sweeps_per_point = Int()
    run_time = Float(value=0.0)
        
    frequencies = Array(value=np.array((0., 1.)))   
    
      
    def generate_sequence(self):
        # ESR:
        #return 100*[ (['laser','aom'], self.laser), ([],self.wait), (['mw'], self.mw_t_pi), (['sequence'], 10) ]
        return 100 * [ (['laser', 'aom'], self.laser), ([], self.wait), (['mw_b'], self.t_pi_B), (['rf'], self.rf_t_pi), ([], 500), (['aom'], self.laser), ([], self.wait), (['mw'], self.t_pi_A), (['rf'], self.rf_t_pi), ([], 500), (['mw'], self.t_pi_A), (['sequence'], 10) ]
    
    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""

        frequencies = np.arange(self.rf_begin, self.rf_end + self.rf_delta, self.rf_delta)
        n_bins = int(self.record_length / self.bin_width)
        time_bins = self.bin_width * np.arange(n_bins)
        sequence = self.generate_sequence()

        if not (self.keep_data and sequence == self.sequence and np.all(time_bins == self.time_bins) and np.all(frequencies == self.frequencies)): # if the sequence and time_bins are the same as previous, keep existing data
            self.count_data = np.zeros((len(frequencies), n_bins))
            self.run_time = 0.0
        
        self.frequencies = frequencies
        self.sequence = sequence 
        self.time_bins = time_bins
        self.n_bins = n_bins
        # ESR:
        #self.sweeps_per_point = int(self.seconds_per_point * 1e9 / (self.laser+self.wait+self.mw_t_pi))
        self.sweeps_per_point = int(np.max((1, int(self.seconds_per_point * 1e9 / (self.laser*2 + self.wait*2 + 2 * self.t_pi_A + self.t_pi_B+ 2*self.rf_t_pi+1000.0)))))
        self.keep_data = True # when job manager stops and starts the job, data should be kept. Only new submission should clear data.
    

    def _run(self):
        """Acquire data."""
        
        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()
            ha.PulseGenerator().Night()
            ha.Microwave().setOutput(self.mwA_power, self.mwA_frequency)
            ha.MicrowaveD().setOutput(self.mwB_power, self.mwB_frequency)
            tagger = ha.TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), 1, 0, 2, 3)
            tagger.setMaxCounts(self.sweeps_per_point)
            ha.PulseGenerator().Sequence(self.sequence)
            ha.RFSource().setMode()
            
            while True:

                if self.thread.stop_request.isSet():
                    break

                t_start = time.time()
                
                for i, fi in enumerate(self.frequencies):
                    
                    # ESR:
                    #ha.Microwave().setOutput(self.mw_power,fi)
                    ha.RFSource().setOutput(self.rf_power, fi)                    
                    tagger.clear()
                    while not tagger.ready():
                        time.sleep(1.1 * self.seconds_per_point)
                    self.count_data[i, :] += tagger.getData()[0]
                                                            
                self.trait_property_changed('count_data', self.count_data)
                self.run_time += time.time() - t_start
                
            del tagger
            ha.PulseGenerator().Light()
            ha.Microwave().setOutput(None, self.mwA_frequency)
            ha.MicrowaveD().setOutput(None, self.mwB_frequency)
            ha.RFSource().setOutput(None, self.rf_begin)
        finally: # if anything fails, recover
            self.state = 'idle'        
            ha.PulseGenerator().Light()
        
    get_set_items = Pulsed.get_set_items + ['mwA_frequency', 'mwA_power', 't_pi_A', 'mwB_frequency', 'mwB_power', 't_pi_B',
                                                       'rf_power', 'rf_begin', 'rf_end', 'rf_delta', 'rf_t_pi',
                                                       'laser', 'wait', 'seconds_per_point', 'frequencies', 'count_data', 'sequence']

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority', width= -40),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('mwA_power', width= -40),
                                                   Item('mwA_frequency', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x)),
                                                   Item('t_pi_A', width= -80),
                                                   ),
                                            HGroup(Item('mwB_power', width= -40),
                                                   Item('mwB_frequency', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x)),
                                                   Item('t_pi_B', width= -80),
                                                   ),
                                            HGroup(Item('rf_power', width= -40),
                                                   Item('rf_begin', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x)),
                                                   Item('rf_end', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x)),
                                                   Item('rf_delta', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x)),
                                                   Item('rf_t_pi', width= -80),
                                                   ),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser', width= -80, enabled_when='state == "idle"'),
                                                   Item('wait', width= -80, enabled_when='state == "idle"'),
                                                   Item('record_length', width= -80, enabled_when='state == "idle"'),
                                                   Item('bin_width', width= -80, enabled_when='state == "idle"'),
                                                   ),
                                            HGroup(Item('seconds_per_point', width= -120, enabled_when='state == "idle"'),
                                                   Item('sweeps_per_point', width= -120, style='readonly'),
                                                   ),
                                            label='settings'
                                            ),
                                     ),
                              ),
                        title='NMR, use frequencies to fit',
                        )
