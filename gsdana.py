from traits.api import HasTraits, Instance, Property, Range, Float, Int, Bool, Array, List, Str, Tuple, Enum, on_trait_change, cached_property, DelegatesTo, Any
from traitsui.api import View, Item, Tabbed, Group, HGroup, VGroup, VSplit, EnumEditor, TextEditor, InstanceEditor
from enable.api import ComponentEditor
from chaco.api import ArrayDataSource, LinePlot, LinearMapper, ArrayPlotData, Plot, Spectral, PlotLabel, Legend

from traitsui.file_dialog import save_file
from traitsui.menu import Action, Menu, MenuBar

import threading
import time
import logging

import fitting

from tools.emod import ManagedJob
from tools.utility import GetSetItemsHandler, GetSetItemsMixin

class PulsedAnaHandler(GetSetItemsHandler):

    """Provides handling of menu."""

    def save_matrix_plot(self, info):
        filename = save_file(title='Save Matrix Plot')
        if filename is '':
            return
        else:
            if filename.find('.png') == -1:
                filename = filename + '.png'
            info.object.save_matrix_plot(filename)

    def save_line_plot(self, info):
        filename = save_file(title='Save Line Plot')
        if filename is '':
            return
        else:
            if filename.find('.png') == -1:
                filename = filename + '.png'
            info.object.save_line_plot(filename)

menubar = MenuBar(Menu(Action(action='save', name='Save (.pyd or .pys)'),
                   Action(action='load', name='Load (.pyd or .pys)'),
                   Action(action='save_matrix_plot', name='Save Matrix Plot (.png)'),
                   Action(action='save_line_plot', name='Save Line Plot (.png)'),
                   Action(action='_on_close', name='Quit'),
                   name='File'
                   ),
              )

class GSDAna(ManagedJob):
    plot = Instance(Plot)
    line_data = Instance(ArrayPlotData)
    
    # overwrite this to change the window title
    traits_view = View(VGroup(Item(name='measurement', style='custom', show_label=False),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     Item('run_sum'),
                                     ),
                              VSplit(Item('plot', show_label=False, width=500, height=300, resizable=True),
                                    ),
                              ),
                       title='GSDAna',
                       menubar=menubar,
                       buttons=[], resizable=True, handler=PulsedAnaHandler)
    

    def __init__(self):
        
        super(GSDAna, self).__init__()
        self.measurement = confocal
        
    # overwrite the line_plot such that the x-axis label is time 
    def _create_plot(self):
        line_data = ArrayPlotData(index=np.array((0, 1)), spin_state=np.array((0, 0)),)
        plot = Plot(line_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('index', 'spin_state'), color='blue', name='spin_state')
        plot.index_axis.title = 'time [micro s]'
        plot.value_axis.title = 'spin state'
        self.line_data = line_data
        self.line_plot = plot

    # overwrite this one to throw out setting of index data according to length of spin_state
    def _update_line_plot_value(self):
        self.line_data.set_data('spin_state', self.spin_state)

    # provide method for update of tau
    def _on_tau_change(self, new):
        self.line_data.set_data('index', new * 1e-3)

