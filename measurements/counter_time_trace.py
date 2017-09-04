import numpy as np
import cPickle

# enthought library imports
from traits.api import SingletonHasTraits, HasTraits, Trait, Instance, Property, Int, Float, Range,\
                                 Bool, Array, Str, Enum, Button, on_trait_change
from traitsui.api import View, Item, Group, HGroup, VGroup, Tabbed, EnumEditor, Action, Menu, MenuBar
from enable.api import ComponentEditor, Component
from chaco.api import Plot, ArrayPlotData

from tools.utility import GetSetItemsHandler, GetSetItemsMixin

import threading
import time

import logging

import hardware.api as ha

from tools.emod import FreeJob

class CounterTimeTrace( FreeJob, GetSetItemsMixin ):

    trace_length        = Range(low=10, high=10000, value=100, desc='Length of Count Trace', label='Trace Length')
    seconds_per_point   = Range(low=0.001, high=1, value=0.1, desc='Seconds per point [s]', label='Seconds per point [s]')
    refresh_interval    = Range(low=0.01, high=1, value=0.1, desc='Refresh interval [s]', label='Refresh interval [s]')

    # trace data
    C = Array()
    T = Array()
    
    trace_plot = Instance( Plot )
    trace_data = Instance( ArrayPlotData )

    def __init__(self):
        super(CounterTimeTrace, self).__init__()
        self.on_trait_change(self.update_T, 'T', dispatch='ui')
        self.on_trait_change(self.update_C, 'C', dispatch='ui')

    def _C_default(self):
        return np.zeros((self.trace_length,))   

    def _T_default(self):
        return self.seconds_per_point*np.arange(self.trace_length)

    def update_T(self):
        self.trace_data.set_data('t', self.T)

    def update_C(self):
        self.trace_data.set_data('y', self.C)

    def _trace_length_changed(self):
        self.C = self._C_default()
        self.T = self._T_default()
        
    def _seconds_per_point_changed(self):
        self.T = self._T_default()

    def _trace_data_default(self):
        return ArrayPlotData(t=self.T, y=self.C)
    def _trace_plot_default(self):
        plot = Plot(self.trace_data, width=500, height=500, resizable='hv')
        plot.plot(('t','y'), type='line', color='blue')
        return plot

    def _run(self):
        """Acquire Count Trace"""
        try:
            self.state='run'
            counter = ha.CountTask(self.seconds_per_point, self.trace_length)
        except Exception as e:
            logging.getLogger().exception(e)
            raise
        else:
            while True:
                threading.current_thread().stop_request.wait(self.refresh_interval)
                if threading.current_thread().stop_request.isSet():
                    break
                try:
                    self.C = counter.getData() / self.seconds_per_point
                except Exception as e:
                    logging.getLogger().exception(e)
                    raise
        finally:
            self.state='idle'

    traits_view = View(VGroup(HGroup(Item('start_button', show_label=False),
                                     Item('stop_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),),
                              Item('trace_plot', editor=ComponentEditor(), show_label=False),
                              HGroup(Item('trace_length'),
                                     Item ('seconds_per_point'),
                                     Item ('refresh_interval'),
                                     ),
                              ),
                       title='Counter Time Trace', width=800, height=600, buttons=[], resizable=True,
                       handler=GetSetItemsHandler()
                       )

if __name__ == '__main__':

    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().info('Starting logger.')

    c = CounterTimeTrace()
    c.edit_traits()
    c.trace_length = 10
    c.bin_width = 1.0
    c.refresh_interval = 1.0
    