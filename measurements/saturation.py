import numpy as np

from traits.api import Range, Array, Instance
from traitsui.api import View, Item, HGroup, VGroup
from enable.api import Component, ComponentEditor
from chaco.api import ArrayPlotData, Plot

from traitsui.file_dialog import save_file
from traitsui.menu import Action, Menu, MenuBar

import logging

from tools.emod import ManagedJob
from tools.utility import GetSetItemsMixin

from hardware.api import TimeTagger, Laser, PowerMeter 
import time

from tools.utility import GetSetItemsHandler, GetSetItemsMixin

class CustomHandler( GetSetItemsHandler ):

    def savePlot(self, info):
        filename = save_file(title='Save Plot')
        if filename is '':
            return
        else:
            if filename.find('.png')==-1:
                filename=filename+'.png'
            info.object.save_plot(filename)

class Saturation( ManagedJob, GetSetItemsMixin ):
    """
    Measures saturation curves.
    
    written by: helmut.fedder@gmail.com
    last modified: 2012-08-17
    """
    
    v_begin   = Range(low=0., high=5.,       value=0.,     desc='begin [V]',  label='begin [V]',   mode='text', auto_set=False, enter_set=True)
    v_end     = Range(low=0., high=5.,       value=5.,     desc='end [V]',    label='end [V]',     mode='text', auto_set=False, enter_set=True)
    v_delta   = Range(low=0., high=5.,       value=.1,     desc='delta [V]',  label='delta [V]',   mode='text', auto_set=False, enter_set=True)
    
    seconds_per_point = Range(low=1e-3, high=1000., value=1., desc='Seconds per point', label='Seconds per point', mode='text', auto_set=False, enter_set=True)

    voltage = Array()
    power = Array()
    rate = Array()
    
    plot_data = Instance( ArrayPlotData )
    plot = Instance( Plot )

    get_set_items=['__doc__', 'v_begin', 'v_end', 'v_delta', 'seconds_per_point', 'voltage', 'power', 'rate' ]

    traits_view = View(VGroup(HGroup(Item('submit_button',   show_label=False),
                                     Item('remove_button',   show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     ),
                              HGroup(Item('v_begin'),
                                     Item('v_end'),
                                     Item('v_delta'),
                                     ),
                              HGroup(Item('seconds_per_point'),
                                     ),
                              Item('plot', editor=ComponentEditor(), show_label=False, resizable=True),
                              ),
                       menubar = MenuBar(Menu(Action(action='savePlot', name='Save Plot (.png)'),
                                              Action(action='save', name='Save (.pyd or .pys)'),
                                              Action(action='load', name='Load'),
                                              Action(action='_on_close', name='Quit'),
                                              name='File')),
                       title='Saturation', buttons=[], resizable=True, handler=CustomHandler
                       )

    
    def __init__(self):
        super(Saturation, self).__init__()
        self._create_plot()
        self.on_trait_change(self._update_index,    'power',    dispatch='ui')
        self.on_trait_change(self._update_value,    'rate',     dispatch='ui')
        
    def _run(self):

        try:
            self.state='run'
            voltage = np.arange(self.v_begin, self.v_end, self.v_delta)
    
            power = np.zeros_like(voltage)
            rate = np.zeros_like(voltage)
    
            counter_0 = TimeTagger.Countrate(0)
            counter_1 = TimeTagger.Countrate(1)
    
            for i,v in enumerate(voltage):
                Laser().voltage = v
                power[i] = PowerMeter().getPower()
                counter_0.clear()
                counter_1.clear()
                self.thread.stop_request.wait(self.seconds_per_point)
                if self.thread.stop_request.isSet():
                    logging.getLogger().debug('Caught stop signal. Exiting.')
                    self.state = 'idle'
                    break            
                rate[i] = counter_0.getData() + counter_1.getData()
                power[i] = PowerMeter().getPower()
            else:
                self.state = 'done'
                
            del counter_0
            del counter_1
            
            self.voltage = voltage
            self.power = power
            self.rate = rate

        finally:
            self.state = 'idle'

    def _create_plot(self):
        plot_data = ArrayPlotData(power=np.array(()), rate=np.array(()),)
        plot = Plot(plot_data, padding=8, padding_left=64, padding_bottom=64)
        plot.plot(('power','rate'), color='blue')
        plot.index_axis.title = 'Power [mW]'
        plot.value_axis.title = 'rate [kcounts/s]'
        self.plot_data = plot_data
        self.plot = plot

    def _update_index(self, new):
        self.plot_data.set_data('power', new*1e3)
        
    def _update_value(self, new):
        self.plot_data.set_data('rate', new*1e-3)

    def save_plot(self, filename):
        self.save_figure(self.plot, filename)

    
if __name__=='__main__':

    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().info('Starting logger.')
    
    from tools.emod import JobManager
    
    JobManager().start()

    saturation = Saturation()
    saturation.edit_traits()
    