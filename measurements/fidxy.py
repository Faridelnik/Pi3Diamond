import numpy as np

from traits.api import Trait, Instance, Property, String, Range, Float, Int, Bool, Array, Enum
from traitsui.api import View, Item, HGroup, VGroup, VSplit, Tabbed, EnumEditor, TextEditor, Group, Label

import logging
import time

import hardware.api as ha

from pulsed import Rabi
    
class FIDXY(Rabi):
    
    """Defines a FID measurement with both pi/2 and 3pi/2 readout pulse."""

    t_pi2 = Range(low=1., high=100000., value=1000., desc='pi/2 pulse length', label='pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
    t_3pi2 = Range(low=1., high=100000., value=1000., desc='3pi/2 pulse length', label='3pi/2 [ns]', mode='text', auto_set=False, enter_set=True)

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        t_pi2 = self.t_pi2
        t_3pi2 = self.t_3pi2
        sequence = []
        for t in tau:
            sequence.append((['mw'   ], t_pi2))
            sequence.append(([       ], t))
            sequence.append((['mw_y'   ], t_pi2))
            sequence.append((['laser', 'aom'], laser))
            sequence.append(([       ], wait))
        for t in tau:
            sequence.append((['mw'   ], t_pi2))
            sequence.append(([       ], t))
            sequence.append((['mw_y'   ], t_3pi2))
            sequence.append((['laser', 'aom'], laser))
            sequence.append(([       ], wait))
        sequence.append((['sequence'], 100))
        return sequence
    
    #items to be saved
    get_set_items = Rabi.get_set_items + ['t_pi2', 't_3pi2']

    # gui elements
    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f'),
                                     Item('stop_time'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('frequency', width= -80, enabled_when='state != "run"'),
                                                   Item('power', width= -80, enabled_when='state != "run"'),
                                                   Item('t_pi2', width= -80, enabled_when='state != "run"'),
                                                   Item('t_3pi2', width= -80, enabled_when='state != "run"'),),
                                            HGroup(Item('tau_begin', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_end', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_delta', width= -80, enabled_when='state != "run"'),
                                                   ),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser', width= -80, enabled_when='state != "run"'),
                                                   Item('wait', width= -80, enabled_when='state != "run"'),
                                                   Item('record_length', width= -80, enabled_when='state != "run"'),
                                                   Item('bin_width', width= -80, enabled_when='state != "run"'),),
                                            label='settings'),
                                     ),
                              ),
                       title='FIDXY',
                       )
  