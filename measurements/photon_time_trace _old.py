import numpy
import cPickle

# enthought library imports
from traits.api import SingletonHasTraits, HasTraits, Trait, Instance, Property, Int, Float, Range,\
                                 Bool, Array, String, Str, Enum, Button, Tuple, List, on_trait_change,\
                                 cached_property, DelegatesTo
from traitsui.api import View, Item, Group, HGroup, VGroup, Tabbed, EnumEditor

from traitsui.file_dialog import save_file
from traitsui.menu import Action, Menu, MenuBar

from enable.api import ComponentEditor, Component
from chaco.api import Plot, ScatterPlot, CMapImagePlot, ArrayPlotData,\
                                Spectral, ColorBar, LinearMapper, DataView,\
                                LinePlot, ArrayDataSource, HPlotContainer
#from chaco.tools.api import ZoomTool
from chaco.tools.cursor_tool import CursorTool, BaseCursorTool

from tools.utility import GetSetItemsHandler, GetSetItemsMixin

import threading
import time

from hardware.api import TimeTagger

from tools.emod import Job

class StartThreadHandler( GetSetItemsHandler ):

    def init(self, info):
        info.object.start()
        
class PhotonTimeTrace( Job, GetSetItemsMixin ):

    TraceLength = Range(low=10, high=10000, value=100, desc='Length of Count Trace', label='Trace Length')
    SecondsPerPoint = Range(low=0.001, high=1, value=0.1, desc='Seconds per point [s]', label='Seconds per point [s]')
    RefreshRate = Range(low=0.01, high=1, value=0.1, desc='Refresh rate [s]', label='Refresh rate [s]')

    # trace data
    C0 = Array()
    C1 = Array()
    C2 = Array()
    C3 = Array()
    C4 = Array()
    C5 = Array()
    C6 = Array()
    C7 = Array()
    C0C1 = Array()
    T = Array()
    
    c_enable0 = Bool(False, label='channel 0', desc='enable channel 0')
    c_enable1 = Bool(False, label='channel 1', desc='enable channel 1')
    c_enable2 = Bool(False, label='channel 2', desc='enable channel 2')
    c_enable3 = Bool(False, label='channel 3', desc='enable channel 3')
    c_enable4 = Bool(False, label='channel 4', desc='enable channel 4')
    c_enable5 = Bool(False, label='channel 5', desc='enable channel 5')
    c_enable6 = Bool(False, label='channel 6', desc='enable channel 6')
    c_enable7 = Bool(False, label='channel 7', desc='enable channel 7')
    sum_enable = Bool(True, label='c0 + c1', desc='enable sum c0 + c1')
    
    TracePlot = Instance( Plot )
    TraceData = Instance( ArrayPlotData )

    def __init__(self):
        super(PhotonTimeTrace, self).__init__()
        self.on_trait_change(self._update_T, 'T', dispatch='ui')
        self.on_trait_change(self._update_C0, 'C0', dispatch='ui')
        self.on_trait_change(self._update_C1, 'C1', dispatch='ui')
        self.on_trait_change(self._update_C2, 'C2', dispatch='ui')
        self.on_trait_change(self._update_C3, 'C3', dispatch='ui')
        self.on_trait_change(self._update_C4, 'C4', dispatch='ui')
        self.on_trait_change(self._update_C5, 'C5', dispatch='ui')
        self.on_trait_change(self._update_C6, 'C6', dispatch='ui')
        #self.on_trait_change(self._update_C7, 'C7', dispatch='ui')
        self.on_trait_change(self._update_C0C1, 'C0C1', dispatch='ui')
        self._create_counter()

    def _create_counter(self):
        self._counter0 = TimeTagger.Counter(0, int(self.SecondsPerPoint*1e12), self.TraceLength) 
        self._counter1 = TimeTagger.Counter(1, int(self.SecondsPerPoint*1e12), self.TraceLength) 
        self._counter2 = TimeTagger.Counter(2, int(self.SecondsPerPoint*1e12), self.TraceLength) 
        self._counter3 = TimeTagger.Counter(3, int(self.SecondsPerPoint*1e12), self.TraceLength) 
        self._counter4 = TimeTagger.Counter(4, int(self.SecondsPerPoint*1e12), self.TraceLength) 
        self._counter5 = TimeTagger.Counter(5, int(self.SecondsPerPoint*1e12), self.TraceLength) 
        self._counter6 = TimeTagger.Counter(6, int(self.SecondsPerPoint*1e12), self.TraceLength) 
        #self._counter7 = TimeTagger.Counter(7, int(self.SecondsPerPoint*1e12), self.TraceLength) # ToDo: does not work when using channel 7
        
    def _C0_default(self):
        return numpy.zeros((self.TraceLength,))   
         
    def _C1_default(self):
        return numpy.zeros((self.TraceLength,))
         
    def _C2_default(self):
        return numpy.zeros((self.TraceLength,))
         
    def _C3_default(self):
        return numpy.zeros((self.TraceLength,))

    def _C4_default(self):
        return numpy.zeros((self.TraceLength,))

    def _C5_default(self):
        return numpy.zeros((self.TraceLength,))

    def _C6_default(self):
        return numpy.zeros((self.TraceLength,))

    def _C7_default(self):
        return numpy.zeros((self.TraceLength,))
    
    def _C0C1_default(self):
        return numpy.zeros((self.TraceLength,))

    def _T_default(self):
        return self.SecondsPerPoint*numpy.arange(self.TraceLength)

    def _update_T(self):
        self.TraceData.set_data('t', self.T)

    def _update_C0(self):
        self.TraceData.set_data('y0', self.C0)
        #self.TracePlot.request_redraw()

    def _update_C1(self):
        self.TraceData.set_data('y1', self.C1)
        #self.TracePlot.request_redraw()

    def _update_C2(self):
        self.TraceData.set_data('y2', self.C2)
        #self.TracePlot.request_redraw()

    def _update_C3(self):
        self.TraceData.set_data('y3', self.C3)
        #self.TracePlot.request_redraw()

    def _update_C4(self):
        self.TraceData.set_data('y4', self.C4)
        #self.TracePlot.request_redraw()

    def _update_C5(self):
        self.TraceData.set_data('y5', self.C5)
        #self.TracePlot.request_redraw()

    def _update_C6(self):
        self.TraceData.set_data('y6', self.C6)
        #self.TracePlot.request_redraw()

    def _update_C7(self):
        self.TraceData.set_data('y7', self.C7)
        #self.TracePlot.request_redraw()
        
    def _update_C0C1(self):
        self.TraceData.set_data('y8', self.C0C1)
        #self.TracePlot.request_redraw()

    def _TraceLength_changed(self):
        self.C0 = self._C0_default()
        self.C1 = self._C1_default()
        self.C2 = self._C2_default()
        self.C3 = self._C3_default()
        self.C4 = self._C4_default()
        self.C5 = self._C5_default()
        self.C6 = self._C6_default()
        self.C7 = self._C7_default()
        self.C0C1 = self._C0C1_default()
        self.T = self._T_default()
        self._create_counter()
        
    def _SecondsPerPoint_changed(self):
        self.T = self._T_default()
        self._create_counter()

    def _TraceData_default(self):
        return ArrayPlotData(t=self.T, y0=self.C0, y1=self.C1, y2=self.C2, y3=self.C3, y4=self.C4, y5=self.C5, y6=self.C6, y7=self.C7, y8=self.C0C1)
    
    def _TracePlot_default(self):
        plot = Plot(self.TraceData, width=500, height=500, resizable='hv')
        plot.plot(('t','y0'), type='line', color='black')
        return plot
    
    #@on_trait_change('c_enable0,c_enable1,c_enable2,c_enable3,c_enable4,c_enable5,c_enable6,c_enable7,sum_enable') # ToDo: fix channel 7
    @on_trait_change('c_enable0,c_enable1,c_enable2,c_enable3,c_enable4,c_enable5,c_enable6,sum_enable')
    def _replot(self):
        
        self.TracePlot = Plot(self.TraceData, width=500, height=500, resizable='hv')
        self.TracePlot.legend.align = 'll'
        
        n=0
        if self.c_enable0:
            self.TracePlot.plot(('t','y0'), type='line', color='blue',  name='channel 0')
            n+=1
        if self.c_enable1:
            self.TracePlot.plot(('t','y1'), type='line', color='red',   name='channel 1')
            n+=1
        if self.c_enable2:
            self.TracePlot.plot(('t','y2'), type='line', color='green', name='channel 2')
            n+=1
        if self.c_enable3:
            self.TracePlot.plot(('t','y3'), type='line', color='black', name='channel 3')
            n+=1
        if self.c_enable4:
            self.TracePlot.plot(('t','y4'), type='line', color='blue',  name='channel 4')
            n+=1
        if self.c_enable5:
            self.TracePlot.plot(('t','y5'), type='line', color='red',   name='channel 5')
            n+=1
        if self.c_enable6:
            self.TracePlot.plot(('t','y6'), type='line', color='green', name='channel 6')
            n+=1
        #if self.c_enable7:
        #    self.TracePlot.plot(('t','y7'), type='line', color='black', name='channel 7')
        if self.sum_enable:
            self.TracePlot.plot(('t','y8'), type='line', color='black', name='sum c0 + c1')
            n+=1

        if n > 1:
            self.TracePlot.legend.visible = True
        else:
            self.TracePlot.legend.visible = False

    def _run(self):
        """Acquire Count Trace"""
        while True:
            threading.current_thread().stop_request.wait(self.RefreshRate)
            if threading.current_thread().stop_request.isSet():
                break
            self.C0 = self._counter0.getData() / self.SecondsPerPoint
            self.C1 = self._counter1.getData() / self.SecondsPerPoint
            self.C2 = self._counter2.getData() / self.SecondsPerPoint
            self.C3 = self._counter3.getData() / self.SecondsPerPoint
            self.C4 = self._counter4.getData() / self.SecondsPerPoint
            self.C5 = self._counter5.getData() / self.SecondsPerPoint
            self.C6 = self._counter6.getData() / self.SecondsPerPoint
            #self.C7 = self._counter7.getData() / self.SecondsPerPoint
            self.C0C1 = self.C0 + self.C1

    traits_view = View( HGroup(Item('TracePlot', editor=ComponentEditor(), show_label=False),
                               #VGroup(Item('c_enable0'),Item('c_enable1'),Item('c_enable2'),Item('c_enable3'),Item('c_enable4'),Item('c_enable5'),Item('c_enable6'),Item('c_enable7'),Item('sum_enable'))
                               VGroup(Item('c_enable0'),Item('c_enable1'),Item('c_enable2'),Item('c_enable3'),Item('c_enable4'),Item('c_enable5'),Item('c_enable6'),Item('sum_enable'))
                        ),
                        Item('TraceLength'),
                        Item ('SecondsPerPoint'),
                        Item ('RefreshRate'),
                        title='Counter', width=800, height=600, buttons=[], resizable=True,
                        handler=StartThreadHandler
                  )


if __name__=='__main__':

    import logging
    
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().info('Starting logger.')

    from tools.emod import JobManager
    JobManager().start()

    p = PhotonTimeTrace()
    p.edit_traits()
    
    