import numpy as np

from traits.api import Range, Array
from traitsui.api import View, Item, Tabbed, HGroup, VGroup, VSplit, EnumEditor, TextEditor

import logging

import hardware.api as ha

from pulsed import Pulsed

class NuclearT1(Pulsed):

    """Defines a Nuclear t1 measurement. use t1p to fit"""

    mwA_frequency = Range(low=1, high=20e9, value=2.830495e9, desc='microwave A frequency', label='MW A frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    mwA_power = Range(low= -100., high=25., value= -8, desc='microwave A power', label='MW A power [dBm]', mode='text', auto_set=False, enter_set=True)
    t_pi_A = Range(low=1., high=100000., value=1150., desc='length of pi pulse A [ns]', label='pi A[ns]', mode='text', auto_set=False, enter_set=True)
    mwB_frequency = Range(low=1, high=20e9, value=2.832651e9, desc='microwave B frequency', label='MW B frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    mwB_power = Range(low= -100., high=25., value= -8, desc='microwave B power', label='MW B power [dBm]', mode='text', auto_set=False, enter_set=True)
    t_pi_B = Range(low=1., high=100000., value=1150., desc='length of pi pulse B[ns]', label='pi B[ns]', mode='text', auto_set=False, enter_set=True)

    rf_frequency = Range(low=1, high=20e6, value=2.788e6, desc='RF frequency', label='RF frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    rf_power = Range(low= -130., high=25., value=23, desc='RF power', label='RF power [dBm]', mode='text', auto_set=False, enter_set=True)
    t_pi_rf = Range(low=1., high=1.0e9, value=3.5e5, desc='length of pi pulse RF[ns]', label='pi RF[ns]', mode='text', auto_set=False, enter_set=True)
    
    tau_begin = Range(low=0., high=1e9, value=300.0, desc='tau begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e9, value=2.0e6, desc='tau end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=5.0e8, value=10000.0, desc='delta tau [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    laser = Range(low=1., high=1.0e7, value=3000.0, desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
    wait = Range(low=1., high=1.0e8, value=4.0e5, desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)

    tau = Array(value=np.array((0., 1.)))

    get_set_items = Pulsed.get_set_items + ['mwA_frequency', 'mwA_power', 't_pi_A', 'mwB_frequency', 'mwB_power', 't_pi_B', 'rf_frequency', 't_pi_rf', 'rf_power', 'tau_begin', 'tau_end', 'tau_delta', 'laser', 'wait', 'tau']

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('mwA_frequency', width= -80, enabled_when='state == "idle"'),
                                                   Item('mwA_power', width= -80, enabled_when='state == "idle"'),
                                                   Item('t_pi_A', width= -80, enabled_when='state == "idle"'),
                                                   ),
                                            HGroup(Item('mwB_frequency', width= -80, enabled_when='state == "idle"'),
                                                   Item('mwB_power', width= -80, enabled_when='state == "idle"'),
                                                   Item('t_pi_B', width= -80, enabled_when='state == "idle"'),
                                                   ),
                                            HGroup(Item('rf_frequency', width= -80, enabled_when='state == "idle"'),
                                                   Item('rf_power', width= -80, enabled_when='state == "idle"'),
                                                   Item('t_pi_rf', width= -80, enabled_when='state == "idle"'),
                                                   ),
                                            HGroup(Item('tau_begin', width= -80, enabled_when='state == "idle"'),
                                                   Item('tau_end', width= -80, enabled_when='state == "idle"'),
                                                   Item('tau_delta', width= -80, enabled_when='state == "idle"'),),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser', width= -80, enabled_when='state == "idle"'),
                                                   Item('wait', width= -80, enabled_when='state == "idle"'),
                                                   Item('record_length', width= -80, enabled_when='state == "idle"'),
                                                   Item('bin_width', width= -80, enabled_when='state == "idle"'),),
                                                   label='settings'),
                              ),
                        ),
                       title='Nuclear T1 Measurement',
                  )

    def generate_sequence(self):
        t_pi_A = self.t_pi_A
        t_pi_B = self.t_pi_B
        t_pi_rf = self.t_pi_rf
        laser = self.laser
        tau = self.tau
        wait = self.wait
        sequence = []
        for t in tau:
            sequence.append((['mw_a'], t_pi_A))
            sequence.append((['rf'], t_pi_rf))
            sequence.append((['laser', 'aom'], t))
            sequence.append((['mw_b'], t_pi_B))
            sequence.append((['laser', 'aom'], laser))
            sequence.append(([], wait))
        sequence.append((['sequence'], 100))
        return sequence

    def apply_parameters(self):
        """Overwrites apply_parameters() from pulsed. Prior to generating sequence, etc., generate the tau mesh."""
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta)
        Pulsed.apply_parameters(self)
        
    def start_up(self):
        ha.PulseGenerator().Night()
        ha.MicrowaveA().setOutput(self.mwA_power, self.mwA_frequency)
        ha.MicrowaveB().setOutput(self.mwB_power, self.mwB_frequency)
        ha.RFSource().setOutput(self.rf_power, self.rf_frequency)

    def shut_down(self):
        ha.PulseGenerator().Light()
        ha.MicrowaveA().setOutput(None, self.mwA_frequency)
        ha.MicrowaveB().setOutput(None, self.mwB_frequency)
        ha.RFSource().setOutput(None, self.rf_frequency)

    
