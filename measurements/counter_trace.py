import numpy
import cPickle

# enthought library imports
from traits.api import SingletonHasTraits, HasTraits, Trait, Instance, Property, Int, Float, Range,\
                                 Bool, Array, String, Str, Enum, Button, Tuple, List, on_trait_change,\
                                 cached_property, DelegatesTo
from traitsui.api import View, Item, Group, HGroup, VGroup, VSplit,Tabbed, EnumEditor

from traitsui.file_dialog import save_file
from traitsui.menu import Action, Menu, MenuBar

from enable.api import ComponentEditor, Component
from chaco.api import Plot, ScatterPlot, CMapImagePlot, ArrayPlotData,\
                                Spectral, ColorBar, LinearMapper, DataView,\
                                LinePlot, ArrayDataSource, HPlotContainer,hot
#from chaco.tools.api import ZoomTool
from chaco.tools.cursor_tool import CursorTool, BaseCursorTool

from tools.utility import GetSetItemsHandler, GetSetItemsMixin

import threading
import logging

import time

from hardware.api import Scanner   

from tools.emod import FreeJob

counter_tr= Scanner()

class StartThreadHandler( GetSetItemsHandler ):

    def init(self, info):
        info.object.start()
        
class CounterTrace( FreeJob, GetSetItemsMixin ):

    TraceLength = Range(low=10, high=10000, value=100, desc='Length of Count Trace', label='Trace Length')
    SecondsPerPoint = Range(low=0.001, high=1, value=0.1, desc='Seconds per point [s]', label='Seconds per point [s]')
    RefreshRate = Range(low=0.01, high=1, value=0.1, desc='Refresh rate [s]', label='Refresh rate [s]')

    # trace data
    C = Array()
    T = Array()
    
    counts = Float(0.0)
    
    TracePlot = Instance( Plot )
    TraceData = Instance( ArrayPlotData )
    
    digits_data = Instance( ArrayPlotData )
    digits_plot = Instance( Plot )

    def __init__(self):
        super(CounterTrace, self).__init__()
        self.on_trait_change(self._update_T, 'T', dispatch='ui')
        self.on_trait_change(self._update_C, 'C', dispatch='ui')
        #counter.startCounter()
        
        self._create_digits_plot()
        self.update_digits_plot()
        
    def _C_default(self):
        return numpy.zeros((self.TraceLength,))   

    def _T_default(self):
        return self.SecondsPerPoint*numpy.arange(self.TraceLength)
        
    def _TraceLength_changed(self):
        self.C = self._C_default()
        self.T = self._T_default()
    
    def _counts_default(self):
        return 0    

    def _SecondsPerPoint_changed(self):
        self.T = self._T_default() 

    def _update_T(self):
        self.TraceData.set_data('t', self.T)

    def _update_C(self):
        self.TraceData.set_data('y0', self.C/1e3)
        #self.TracePlot.request_redraw()
 
    def _TraceData_default(self):
        return ArrayPlotData(t=self.T, y0=self.C/1e3)
    
    def _TracePlot_default(self):
        plot = Plot(self.TraceData, width=500, height=500, resizable='hv')
        plot.plot(('t','y0'), type='line', color='black',fontsize=20)
        plot.value_axis.title = 'kcounts / s'
        return plot
        
    def _create_digits_plot(self):
        data = ArrayPlotData(image=numpy.zeros((2,2)))
        plot = Plot(data, width=500, height=500, resizable='hv', aspect_ratio=37.0/9, padding=8, padding_left=48, padding_bottom=36)
        plot.img_plot('image',
                      xbounds=(0, 1),
                      ybounds=(0, 1),
                      colormap=hot)
        plot.plots['plot0'][0].value_range.high_setting = 1
        plot.plots['plot0'][0].value_range.low_setting = 0
        plot.x_axis = None
        plot.y_axis = None
        self.digits_data = data
        self.digits_plot = plot
    
    @on_trait_change('counts')
    def update_digits_plot(self):
        string = ('%5.1f' % self.counts)[:5] + 'k'
            
        data = numpy.zeros((37,9))
        for i, char in enumerate(string):
            data[6*i+1:6*i+6,1:-1] = DIGIT[char].transpose()
        if self.counts >= 2e3:
            data *= 0.4
        self.digits_data.set_data('image', data.transpose()[::-1])    

    def _run(self):
        """Acquire Count Trace"""
        try:
            self.state='run'
            counter_tr.startCounter()
        except Exception as e:
            logging.getLogger().exception(e)
            raise
        else:
            while True:
                threading.current_thread().stop_request.wait(self.RefreshRate)
                if threading.current_thread().stop_request.isSet():
                    break
                try:
                    self.C = numpy.append(self.C[1:], counter_tr.Count())
                    self.counts = self.C[-1]/1000
                except Exception as e:
                    logging.getLogger().exception(e)
                    raise
        finally:
            self.state='idle'
            counter_tr.stopCounter()    

    traits_view = View(VGroup(HGroup(Item('start_button', show_label=False),
                                     Item('stop_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),),
                              VSplit(Item('TracePlot', editor=ComponentEditor(), show_label=False),
                                     Item('digits_plot', editor=ComponentEditor(), show_label=False)),
                              Item('TraceLength'),
                              HGroup( Item ('SecondsPerPoint'),
                                      Item ('RefreshRate'),
                                    ),
                              ),
                       title='Counter Time Trace', width=600, height=600, buttons=[], resizable=True,
                       handler=GetSetItemsHandler()
                       )
    
DIGIT = {}
DIGIT['0'] = numpy.array([0,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,1,1,
                       1,0,1,0,1,
                       1,1,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['1'] = numpy.array([0,0,1,0,0,
                       0,1,1,0,0,
                       1,0,1,0,0,
                       0,0,1,0,0,
                       0,0,1,0,0,
                       0,0,1,0,0,
                       1,1,1,1,1]).reshape(7,5)
DIGIT['2'] = numpy.array([0,1,1,1,0,
                       1,0,0,0,1,
                       0,0,0,0,1,
                       0,0,0,1,0,
                       0,0,1,0,0,
                       0,1,0,0,0,
                       1,1,1,1,1]).reshape(7,5)
DIGIT['3'] = numpy.array([0,1,1,1,0,
                       1,0,0,0,1,
                       0,0,0,0,1,
                       0,0,0,1,0,
                       0,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['4'] = numpy.array([0,0,1,0,0,
                       0,1,0,0,0,
                       0,1,0,0,0,
                       1,0,0,1,0,
                       1,1,1,1,1,
                       0,0,0,1,0,
                       0,0,0,1,0]).reshape(7,5)
DIGIT['5'] = numpy.array([1,1,1,1,1,
                       1,0,0,0,0,
                       1,0,0,0,0,
                       1,1,1,1,0,
                       0,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['6'] = numpy.array([0,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,0,0,
                       1,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['7'] = numpy.array([1,1,1,1,1,
                       0,0,0,0,1,
                       0,0,0,0,1,
                       0,0,0,1,0,
                       0,0,1,0,0,
                       0,1,0,0,0,
                       1,0,0,0,0]).reshape(7,5)
DIGIT['8'] = numpy.array([0,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['9'] = numpy.array([0,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,1,
                       0,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['.'] = numpy.array([0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,1,0,0]).reshape(7,5)
DIGIT['k'] = numpy.array([1,0,0,0,0,
                       1,0,0,0,0,
                       1,0,0,0,1,
                       1,0,0,1,0,
                       1,0,1,0,0,
                       1,1,0,1,0,
                       1,0,0,0,1]).reshape(7,5)
DIGIT[' '] = numpy.array([0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0]).reshape(7,5)    
                       
if __name__=='__main__':

    import logging
    
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().info('Starting logger.')

    from tools.emod import JobManager
    JobManager().start()

    p = CounterTrace()
    p.edit_traits()