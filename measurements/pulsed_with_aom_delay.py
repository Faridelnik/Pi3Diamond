from pulsed import PulsedTau, sequence_remove_zeros, sequence_union

from traits.api import Range, Int, Float, Bool, Array, Instance, Enum, on_trait_change, Button
from traitsui.api import View, Item, Tabbed, HGroup, VGroup, VSplit, EnumEditor, TextEditor

from hardware.api import PulseGenerator, Microwave

class Rabi( PulsedTau ):
    
    """Defines a Rabi measurement."""
    
    frequency  = Range(low=1,      high=20e9,  value=2.8705e9, desc='microwave frequency', label='frequency [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    power      = Range(low=-100.,  high=25.,   value=-20,      desc='microwave power',     label='power [dBm]',    mode='text', auto_set=False, enter_set=True)
    switch     = Enum( 'mw_x', 'mw_y',   desc='switch to use for microwave pulses',     label='switch', editor=EnumEditor(cols=3, values={'mw_x':'1:X', 'mw_y':'2:Y'}) )
    
    laser      = Range(low=1., high=100000.,  value=3000., desc='laser [ns]',      label='laser [ns]',       mode='text', auto_set=False, enter_set=True)
    decay_init = Range(low=0., high=10000.,   value=1000., desc='time to let the system decay after laser pulse [ns]',       label='decay init [ns]',        mode='text', auto_set=False, enter_set=True)
    decay_read = Range(low=0., high=10000.,   value=0.,    desc='time to let the system decay before laser pulse [ns]',       label='decay read [ns]',        mode='text', auto_set=False, enter_set=True)
    
    aom_delay  = Range(low=0., high=1000.,   value=0.,    desc='If set to a value other than 0.0, the aom triggers are applied\nearlier by the specified value. Use with care!', label='aom delay [ns]', mode='text', auto_set=False, enter_set=True)
    
    def start_up(self):
        PulseGenerator().Night()
        Microwave().setOutput(self.power, self.frequency)

    def shut_down(self):
        PulseGenerator().Light()
        Microwave().setOutput(None, self.frequency)

    def generate_sequence(self):
        MW = self.switch
        tau = self.tau
        laser = self.laser
        decay_init = self.decay_init
        decay_read = self.decay_read
        aom_delay = self.aom_delay
        if aom_delay == 0.0:
            sequence = [ (['aom'],laser) ]
            for t in tau:
                sequence += [ ([],decay_init), ([MW],t), ([],decay_read), (['laser','aom'],laser) ]
            sequence += [ (['sequence'], 100) ]
            sequence = sequence_remove_zeros(sequence)
        else:
            s1 = [ (['aom'],laser) ]
            s2 = [ ([],aom_delay+laser) ]
            for t in tau:
                s1 += [ ([], decay_init+t+decay_read), (['aom'], laser) ]
                s2 += [ ([], decay_init), ([MW],t), ([],decay_read), (['laser'],laser) ]
            s2 += [ (['sequence'],100) ]
            s1 = sequence_remove_zeros(s1)            
            s2 = sequence_remove_zeros(s2)            
            sequence = sequence_union(s1,s2)
        return sequence

    get_set_items = PulsedTau.get_set_items + ['frequency','power','switch','laser','decay_init','decay_read','aom_delay']

    traits_view = View(VGroup(HGroup(Item('submit_button',   show_label=False),
                                     Item('remove_button',   show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly'),
                                     Item('stop_time'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('frequency',     width=-120, enabled_when='state != "run"'),
                                                   Item('power',         width=-60, enabled_when='state != "run"'),
                                                   Item('switch',        style='custom', enabled_when='state != "run"'),
                                                   Item('aom_delay',     width=-80, enabled_when='state != "run"'),
                                                   ),
                                            HGroup(Item('laser',         width=-80, enabled_when='state != "run"'),
                                                   Item('decay_init',    width=-80, enabled_when='state != "run"'),
                                                   Item('decay_read',    width=-80, enabled_when='state != "run"'),
                                                   ),
                                            HGroup(Item('tau_begin',     width=-80, enabled_when='state != "run"'),
                                                   Item('tau_end',       width=-80, enabled_when='state != "run"'),
                                                   Item('tau_delta',     width=-80, enabled_when='state != "run"'),
                                                   ),
                                            label='manipulation'),
                                     VGroup(HGroup(Item('record_length', width=-80, enabled_when='state != "run"'),
                                                   Item('bin_width',     width=-80, enabled_when='state != "run"'),
                                                   ),
                                     label='acquisition'),
                              ),
                        ),
                       title='Rabi Measurement AOM Delay',
                  )

class T1pi( Rabi ):
    
    """T1 measurement with pi-pulse."""
    
    t_pi        = Range(low=0., high=100000.,   value=1000.,    desc='pi pulse length',     label='pi [ns]',      mode='text', auto_set=False, enter_set=True)
    
    def generate_sequence(self):
        MW = self.switch
        tau = self.tau
        laser = self.laser
        decay_init = self.decay_init
        decay_read = self.decay_read
        aom_delay = self.aom_delay
        t_pi = self.t_pi
        if aom_delay == 0.0:
            sequence = [ (['aom'],laser) ]
            for t in tau:
                sequence += [ ([],decay_init), ([MW],t_pi), ([],t+decay_read), (['laser','aom'],laser) ]
            sequence += [ (['sequence'], 100) ]
            sequence = sequence_remove_zeros(sequence)
        else:
            s1 = [ (['aom'],laser) ]
            s2 = [ ([],aom_delay+laser) ]
            for t in tau:
                s1 += [ ([],decay_init+t_pi+t+decay_read), (['aom','laser'],laser) ]
                s2 += [ ([],decay_init), ([MW],t_pi), ([],t+decay_read+laser) ]
            s2 += [ (['sequence'],100) ]            
            s1 = sequence_remove_zeros(s1)            
            s2 = sequence_remove_zeros(s2)            
            sequence = sequence_union(s1,s2)
        return sequence

    get_set_items = Rabi.get_set_items + ['t_pi']

    traits_view = View(VGroup(HGroup(Item('submit_button',   show_label=False),
                                     Item('remove_button',   show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority', width=-60),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly'),
                                     Item('stop_time'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('frequency',     width=-120, enabled_when='state != "run"'),
                                                   Item('power',         width=-60, enabled_when='state != "run"'),
                                                   Item('switch',        style='custom', enabled_when='state != "run"'),
                                                   Item('aom_delay',     width=-80, enabled_when='state != "run"'),
                                                   ),
                                            HGroup(Item('laser',         width=-80, enabled_when='state != "run"'),
                                                   Item('decay_init',    width=-80, enabled_when='state != "run"'),
                                                   Item('t_pi',          width=-80, enabled_when='state != "run"'),
                                                   Item('decay_read',    width=-80, enabled_when='state != "run"'),
                                                   ),
                                            HGroup(Item('tau_begin',     width=-80, enabled_when='state != "run"'),
                                                   Item('tau_end',       width=-80, enabled_when='state != "run"'),
                                                   Item('tau_delta',     width=-80, enabled_when='state != "run"'),
                                                   ),
                                            label='manipulation'),
                                     VGroup(HGroup(Item('record_length', width=-80, enabled_when='state != "run"'),
                                                   Item('bin_width',     width=-80, enabled_when='state != "run"'),
                                                   ),
                                     label='acquisition'),
                              ),
                        ),
                       title='T1pi AOM Delay',
                       )
