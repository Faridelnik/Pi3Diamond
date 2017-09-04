import numpy as np

from traits.api import Range, Array
from traitsui.api import View, Item, Tabbed, HGroup, VGroup, VSplit, EnumEditor, TextEditor

import logging

import hardware.api as ha

from pulsed import Pulsed

class NHahn( Pulsed ):

    """Defines a Nuclear Rabi measurement."""

    mw_frequency   = Range(low=1,      high=20e9,  value=2.874356e+09, desc='microwave frequency', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    mw_power       = Range(low=-100.,  high=25.,   value=-18,      desc='microwave power',     label='MW power [dBm]',    mode='text', auto_set=False, enter_set=True)
    t_pi           = Range(low=1.,     high=100000., value=1190., desc='length of pi pulse [ns]', label='pi [ns]', mode='text', auto_set=False, enter_set=True)

    rf_frequency   = Range(low=1,      high=20e6,  value=2.762e6, desc='RF frequency', label='RF frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    rf_power       = Range(low=-130.,  high=25.,   value=-14,      desc='RF power',     label='RF power [dBm]',    mode='text', auto_set=False, enter_set=True)
    t_pi2_rf = Range(low=1., high=1.0e7, value=21.0e3, desc='length of pi/2 pulse of RF[ns]', label='pi2 [ns]', mode='text', auto_set=False, enter_set=True)
    t_pi_rf = Range(low=1., high=1.0e7, value=42.0e3, desc='length of pi pulse of RF[ns]', label='pi [ns]', mode='text', auto_set=False, enter_set=True)
 
    tau_begin   = Range(low=0., high=1e8,       value=1.5,     desc='tau begin [ns]',  label='tau begin [ns]',   mode='text', auto_set=False, enter_set=True)
    tau_end     = Range(low=1., high=1e9,       value=2.4e7,     desc='tau end [ns]',    label='tau end [ns]',     mode='text', auto_set=False, enter_set=True)
    tau_delta   = Range(low=1., high=1e9,       value=0.3e6,      desc='delta tau [ns]',  label='delta tau [ns]',   mode='text', auto_set=False, enter_set=True)
    laser       = Range(low=1., high=1.0e7,     value=3000.0,    desc='laser [ns]',      label='laser [ns]',       mode='text', auto_set=False, enter_set=True)
    wait        = Range(low=1., high=1.0e8,   value=1.0e6,    desc='wait [ns]',       label='wait [ns]',        mode='text', auto_set=False, enter_set=True)

    tau = Array( value=np.array((0.,1.)) )

    get_set_items = Pulsed.get_set_items + ['mw_frequency','mw_power','t_pi','rf_frequency','rf_power','tau_begin','tau_end','tau_delta','laser','wait','tau']

    traits_view = View(VGroup(HGroup(Item('submit_button',   show_label=False),
                                     Item('remove_button',   show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('mw_frequency',     width=-80, enabled_when='state == "idle"'),
                                                   Item('mw_power',         width=-80, enabled_when='state == "idle"'),
                                                   Item('t_pi',             width=-80, enabled_when='state == "idle"'),
                                                   ),
                                            HGroup(Item('rf_frequency',     width=-80, enabled_when='state == "idle"'),
                                                   Item('rf_power',         width=-80, enabled_when='state == "idle"'),
                                                   Item('t_pi2_rf',             width=-80, enabled_when='state == "idle"'),
                                                   Item('t_pi_rf',             width=-80, enabled_when='state == "idle"'),
                                                   ),
                                            HGroup(Item('tau_begin',     width=-80, enabled_when='state == "idle"'),
                                                   Item('tau_end',       width=-80, enabled_when='state == "idle"'),
                                                   Item('tau_delta',     width=-80, enabled_when='state == "idle"'),),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser',         width=-80, enabled_when='state == "idle"'),
                                                   Item('wait',          width=-80, enabled_when='state == "idle"'),
                                                   Item('record_length', width=-80, enabled_when='state == "idle"'),
                                                   Item('bin_width',     width=-80, enabled_when='state == "idle"'),),
                                                   label='settings'),
                              ),
                        ),
                       title='Nuclear Rabi Measurement',
                  )

    def generate_sequence(self):
        t_pi=self.t_pi
        laser = self.laser
        tau= self.tau
        wait = self.wait
        sequence = []
        for t in tau:
            sequence.append(  (['mw'], t_pi       )  )
            sequence.append(  (['rf'], self.t_pi2_rf       )  )
            sequence.append(  ([], 0.5*t       )  )
            sequence.append(  (['rf'], self.t_pi_rf       )  )
            sequence.append(  ([], 0.5*t       )  )
            sequence.append(  (['rf'], self.t_pi2_rf       )  )
            sequence.append(  ([], 500))
            sequence.append(  (['mw'], t_pi       )  )
            sequence.append(  (['laser','aom'], laser   )  )
            sequence.append(  ([], wait    )  )
        sequence.append(  (['sequence'], 100  )  )
        return sequence

    def apply_parameters(self):
        """Overwrites apply_parameters() from pulsed. Prior to generating sequence, etc., generate the tau mesh."""
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta)
        Pulsed.apply_parameters(self)
        
    def start_up(self):
        ha.PulseGenerator().Night()
        ha.Microwave().setOutput(self.mw_power, self.mw_frequency)
        ha.RFSource().setOutput(self.rf_power, self.rf_frequency)
        ha.RFSource().setMode()

    def shut_down(self):
        ha.PulseGenerator().Light()
        ha.Microwave().setOutput(None, self.mw_frequency)
        ha.RFSource().setOutput(None, self.rf_frequency)

    