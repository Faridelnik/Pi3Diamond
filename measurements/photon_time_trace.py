import numpy as np
import cPickle

# enthought library imports
from traits.api import SingletonHasTraits, HasTraits, Trait, Instance, Property, Int, Float, Range,\
                                 Bool, Array, String, Str, Enum, Button, Tuple, List, on_trait_change,\
                                 cached_property, DelegatesTo, Font
from traitsui.api import View, Item, Group, HGroup, VGroup, VSplit, Tabbed, EnumEditor

from traitsui.file_dialog import save_file
from traitsui.menu import Action, Menu, MenuBar

from enable.api import ComponentEditor, Component
from chaco.api import Plot, ScatterPlot, CMapImagePlot, ArrayPlotData,\
                                Spectral, ColorBar, LinearMapper, DataView,\
                                LinePlot, ArrayDataSource, HPlotContainer, hot
#from chaco.tools.api import ZoomTool
from chaco.tools.cursor_tool import CursorTool, BaseCursorTool

from tools.utility import GetSetItemsHandler, GetSetItemsMixin

import threading
import time

from tools.emod import Job
from tools.color import scheme

class StartThreadHandler( GetSetItemsHandler ):

    def init(self, info):
        info.object.start()
        
class PhotonTimeTrace( Job, GetSetItemsMixin ):

    TraceLength = Range(low=10, high=10000, value=1000, desc='Length of Count Trace', label='Trace Length')
    SecondsPerPoint = Range(low=0.001, high=1, value=0.05, desc='Seconds per point [s]', label='Seconds per point [s]')
    RefreshRate = Range(low=0.01, high=1, value=0.1, desc='Refresh rate [s]', label='Refresh rate [s]')

    # trace data
    C0 = Array()
    C1 = Array()
    C0C1 = Array()
    T = Array()
    
    counts = Float(0.0)
    throttle = 0
    throttle_level = 2
    
    c_enable0 = Bool(False, label='channel 0', desc='enable channel 0')
    c_enable1 = Bool(False, label='channel 1', desc='enable channel 1')
    sum_enable = Bool(True, label='c0 + c1', desc='enable sum c0 + c1')
    
    baseline = Bool(True, label='baseline', desc='show baseline')
    
    TracePlot = Instance( Plot )
    TraceData = Instance( ArrayPlotData )
    
    digits_data = Instance( ArrayPlotData )
    digits_plot = Instance( Plot )
    
    
    def __init__(self, time_tagger, **kwargs):
        super(PhotonTimeTrace, self).__init__(**kwargs)
        self.time_tagger = time_tagger
        self.on_trait_change(self._update_T, 'T', dispatch='ui')
        self.on_trait_change(self._update_C0, 'C0', dispatch='ui')
        self.on_trait_change(self._update_C1, 'C1', dispatch='ui')
        self.on_trait_change(self._update_C0C1, 'C0C1', dispatch='ui')
        self._create_counter()
        
        self._create_digits_plot()
        self.update_digits_plot()

    def _create_counter(self):
        self._counter0 = self.time_tagger.Counter(0, int(self.SecondsPerPoint*1e12), self.TraceLength) 
        self._counter1 = self.time_tagger.Counter(1, int(self.SecondsPerPoint*1e12), self.TraceLength) 
        
    def _C0_default(self):
        return np.zeros((self.TraceLength,))   
         
    def _C1_default(self):
        return np.zeros((self.TraceLength,))
         
    def _C0C1_default(self):
        return np.zeros((self.TraceLength,))
    
    def _counts_default(self):
        return 0
    
    def _T_default(self):
        return self.SecondsPerPoint*np.arange(self.TraceLength)

    def _update_T(self):
        self.TraceData.set_data('t', self.T)

    def _update_C0(self):
        self.TraceData.set_data('y0', self.C0)
        #self.TracePlot.request_redraw()

    def _update_C1(self):
        self.TraceData.set_data('y1', self.C1)
        #self.TracePlot.request_redraw()
    
    def _update_C0C1(self):
        self.TraceData.set_data('y8', self.C0C1)
        #self.TracePlot.request_redraw()

    def _TraceLength_changed(self):
        self.C0 = self._C0_default()
        self.C1 = self._C1_default()
        self.C0C1 = self._C0C1_default()
        self.T = self._T_default()
        self._create_counter()
        
    def _SecondsPerPoint_changed(self):
        self.T = self._T_default()
        self._create_counter()

    def _TraceData_default(self):
        return ArrayPlotData(t=self.T, y0=self.C0, y1=self.C1, y8=self.C0C1)
    
    def _TracePlot_default(self):
        plot = Plot(self.TraceData, width=500, height=500, resizable='hv')
        plot.plot(('t','y8'), type='line', line_style='solid', color=0xFFFFFF, line_width=2, render_style='connectedpoints', name='ch0 & ch1')
        plot.bgcolor = scheme['background']
        plot.value_range.low = 0.0
        plot.x_grid = None
        plot.y_grid = None
        return plot
    
    @on_trait_change('baseline,c_enable0,c_enable1,sum_enable')
    def _replot(self):
        
        self.TracePlot = Plot(self.TraceData, width=500, height=500, resizable='hv')
        if self.baseline:
            self.TracePlot.value_range.low = 0.0
        if not self.baseline:
            self.TracePlot.value_range.low = 'auto'
        self.TracePlot.legend.align = 'll'
        self.TracePlot.bgcolor = scheme['background']
        self.TracePlot.x_grid = None
        self.TracePlot.y_grid = None
        
        n=0
        if self.c_enable0:
            self.TracePlot.plot(('t','y0'), type='line', line_style='solid', color=scheme['data 1'], line_width=2, render_style='connectedpoints',  name='channel 0')
            n+=1
        if self.c_enable1:
            self.TracePlot.plot(('t','y1'), type='line', line_style='solid', color=scheme['data 2'], line_width=2, render_style='connectedpoints', name='channel 1')
            n+=1
        if self.sum_enable:
            self.TracePlot.plot(('t','y8'), type='line', line_style='solid', color=0xFFFFFF, line_width=2, render_style='connectedpoints', name='ch0 & ch1')
            n+=1
        if n > 3:
            self.TracePlot.legend.visible = True
        else:
            self.TracePlot.legend.visible = False
    
    def _baseline_changed(self):
        if self.baseline:
            self.TracePlot.value_range.low = 0.0
        if not self.baseline:
            self.TracePlot.value_range.low = 'auto'
    
    def _run(self):
        """Acquire Count Trace"""
        while True:
            threading.current_thread().stop_request.wait(self.RefreshRate)
            if threading.current_thread().stop_request.isSet():
                break
            self.C0 = self._counter0.getData() / self.SecondsPerPoint / 1000.0 # kcounts / s
            self.C1 = self._counter1.getData() / self.SecondsPerPoint / 1000.0 # kcounts / s
            self.C0C1 = self.C0 + self.C1
            self.throttle = (self.throttle + 1) % self.throttle_level
            if self.throttle == 0:
                self.counts = self.C0C1[-1]
                
    
    # DIGITS PLOT
    def _create_digits_plot(self):
        data = ArrayPlotData(image=np.zeros((2,2)))
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
            
        data = np.zeros((37,9))
        for i, char in enumerate(string):
            data[6*i+1:6*i+6,1:-1] = DIGIT[char].transpose()
        if self.counts >= 2e3:
            data *= 0.4
        self.digits_data.set_data('image', data.transpose()[::-1])
        
        
    traits_view = View( VGroup(VSplit(Item('TracePlot', editor=ComponentEditor(), show_label=False),
                                      Item('digits_plot', editor=ComponentEditor(), show_label=False)),
                               #VGroup(Item('c_enable0'),Item('c_enable1'),Item('c_enable2'),Item('c_enable3'),Item('c_enable4'),Item('c_enable5'),Item('c_enable6'),Item('c_enable7'),Item('sum_enable'))
                               HGroup(Item('c_enable0'),Item('c_enable1'),Item('sum_enable'),Item('baseline'))
                        ),
                        Item('TraceLength'),
                        Item ('SecondsPerPoint'),
                        Item ('RefreshRate'),
                        title='Counter',
                        width=895,
                        height=1200,
                        buttons=[],
                        resizable=True,
                        x=1025,
                        y=0,
                        handler=StartThreadHandler
                  )
    
DIGIT = {}
DIGIT['0'] = np.array([0,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,1,1,
                       1,0,1,0,1,
                       1,1,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['1'] = np.array([0,0,1,0,0,
                       0,1,1,0,0,
                       1,0,1,0,0,
                       0,0,1,0,0,
                       0,0,1,0,0,
                       0,0,1,0,0,
                       1,1,1,1,1]).reshape(7,5)
DIGIT['2'] = np.array([0,1,1,1,0,
                       1,0,0,0,1,
                       0,0,0,0,1,
                       0,0,0,1,0,
                       0,0,1,0,0,
                       0,1,0,0,0,
                       1,1,1,1,1]).reshape(7,5)
DIGIT['3'] = np.array([0,1,1,1,0,
                       1,0,0,0,1,
                       0,0,0,0,1,
                       0,0,0,1,0,
                       0,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['4'] = np.array([0,0,1,0,0,
                       0,1,0,0,0,
                       0,1,0,0,0,
                       1,0,0,1,0,
                       1,1,1,1,1,
                       0,0,0,1,0,
                       0,0,0,1,0]).reshape(7,5)
DIGIT['5'] = np.array([1,1,1,1,1,
                       1,0,0,0,0,
                       1,0,0,0,0,
                       1,1,1,1,0,
                       0,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['6'] = np.array([0,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,0,0,
                       1,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['7'] = np.array([1,1,1,1,1,
                       0,0,0,0,1,
                       0,0,0,0,1,
                       0,0,0,1,0,
                       0,0,1,0,0,
                       0,1,0,0,0,
                       1,0,0,0,0]).reshape(7,5)
DIGIT['8'] = np.array([0,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['9'] = np.array([0,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,1,
                       0,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['.'] = np.array([0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,1,0,0]).reshape(7,5)
DIGIT['k'] = np.array([1,0,0,0,0,
                       1,0,0,0,0,
                       1,0,0,0,1,
                       1,0,0,1,0,
                       1,0,1,0,0,
                       1,1,0,1,0,
                       1,0,0,0,1]).reshape(7,5)
DIGIT[' '] = np.array([0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0]).reshape(7,5)

if __name__=='__main__':
    pass