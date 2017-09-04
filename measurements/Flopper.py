import os
os.environ['ETS_TOOLKIT'] = 'qt4'

import enthought.etsconfig
enthought.etsconfig.enable_toolkit='qt4'
enthought.etsconfig.toolkit='qt4'

from enthought.traits.api import SingletonHasTraits, Instance, Range, Int, Bool, Property, Array, on_trait_change
from enthought.traits.ui.api import View, Item, VGroup, HGroup

from enthought.enable.api import Component, ComponentEditor
from enthought.chaco.api import DataView, ArrayDataSource, LinePlot, LinearMapper, BarPlot
from enthought.chaco.tools.api import PanTool, ZoomTool

from enthought.traits.api import *
from enthought.traits.ui.api import *

from enthought.enable.api import Component, ComponentEditor
from enthought.chaco.api import *
from enthought.chaco.tools.api import PanTool, ZoomTool

import threading, time

import ok
import struct

import numpy
import analysis.fitting as fitting

from tools.utility import GetSetItemsMixin
#from pi3diamond import pi3d, CloseHandler

xem = ok.FrontPanel()
if (xem.OpenBySerial('') != 0):
    raise RuntimeError, 'Failed to open USB connection.'

PLL = ok.PLL22150()
xem.GetPLL22150Configuration(PLL)
PLL.SetVCOParameters(100,48)
PLL.SetOutputSource(0,2)
PLL.SetOutputEnable(0,2)
#PLL.SetDiv1(1,10)
xem.SetPLL22150Configuration(PLL)
import datetime

if (xem.ConfigureFPGA(r'D:\pi3diamondnew\toplevel.bit') != 0):
    raise RuntimeError, 'Failed to upload bit file to fpga.'    
    
class Flopper( SingletonHasTraits, GetSetItemsMixin ):

    BufferLength = Range(low=1, high=512, value=512)
    Threshold = Range(low=0, high=10000, value=100)
    Pulse = Bool(True)
    Stream = Bool(True)
    Chunks = Range(low=1, high=10000, value=10)
    PulseLength = Int(10)

    Trace = Array()
    trace_binned = Array()
    
    file_path_timetrace = Str(r'D:\data\Timetrace')
    file_name_timetrace = Str('enter filename')
    save_timetrace = Button()
    
    binning = Range(low = 1, high = 100, value = 1, mode='text')
    refresh_hist = Button()    
    
    file_path_hist = Str(r'D:\data\Histograms')
    file_name_hist = Str('enter filename')
    save_hist = Button()
    
    #for readout protocol
    bits_readout = Int(6, label='# of substeps-bits for readout protocol (steps = 2**x)')
    
    # FileNameTrace = Str(r'D:\data\Trace' + str(datetime.date.today()) + '_01.dat')
    # FileNameHist = Str(r'D:\data\Histogram' + str(datetime.date.today()) + '_01.dat')
    view = View( VGroup( HGroup(Item('Pulse'),
                                Item('Loop'),
                                Item('Run'),
                                Item('Threshold'),
                                Item('Chunks')),
                         HGroup( VGroup(Item('TracePlot', editor=ComponentEditor(), show_label=False, width=500, height=300, resizable=True),
                                        HGroup(Item('file_name_timetrace', label='Filename of timetrace:'),
                                               Item('save_timetrace', label = 'Save Timetrace', show_label=False))),
                                 VGroup(Item('HistPlot', editor=ComponentEditor(), show_label=False, width=500, height=300, resizable=True),
                                        HGroup(Item('binning',        label='             # of bins'),
                                               Item('refresh_hist', label = 'Refresh histogram', show_label=False)),
                                        HGroup(Item('file_name_hist', label='Filename of histogram:'),
                                               Item('save_hist', label = 'Save Histogram', show_label=False)))),
                         ),
                 title='Flopper', width=700, height=500, buttons=['OK'], resizable=True)

    def _Trace_default(self):
        return numpy.zeros((self.BufferLength*self.Chunks,))

    def _BufferLength_default(self):
        xem.SetWireInValue(0x00, 512)
        xem.UpdateWireIns()
        self._Buf = '\x00'*2*512
        return 512
            
    def _BufferLength_changed(self):
        xem.SetWireInValue(0x00, self.BufferLength)
        xem.UpdateWireIns()
        self._Buf = '\x00'*2*self.BufferLength

    def _Threshold_changed(self):
        xem.SetWireInValue(0x02, self.Threshold)
        xem.UpdateWireIns()

    def _PulseLength_changed(self):
        xem.SetWireInValue(0x04, self.PulseLength)
        xem.UpdateWireIns()

    @on_trait_change('Pulse,Stream')
    def UpdateFlags(self):
        xem.SetWireInValue(0x03, (self.Pulse << 1) | self.Stream )
        xem.UpdateWireIns()

    def Reset(self):
        self.Stream=False
        xem.ActivateTriggerIn(0x40, 0)

    def ReadPipe(self):
        M = self.BufferLength
        buf = '\x00'*2*M
        xem.ReadFromBlockPipeOut(0xA0, 2*M, buf)
        return numpy.array(struct.unpack('%iH'%M, buf))

    def GetTrace_old(self):
        M = self.BufferLength
        self.Trace = numpy.zeros((self.BufferLength*self.Chunks,))
        temp = self.Trace.copy()
        self.Reset()
        self.Stream=True
        for i in range(self.Chunks):
            if self.abort.isSet():
                break
            temp[i*M:(i+1)*M] = self.ReadPipe()
            self.Trace = temp    
            self.HistogramN, self.HistogramBins = numpy.histogram(self.Trace[:(i+1)*M], bins=numpy.arange(self.Trace.min(),self.Trace.max(),1))            
            self._Trace_changed()
    HistogramN = Array(value=numpy.array((0,1)))
    HistogramBins = Array(value=numpy.array((0,0)))
    
    def GetTrace(self):
        M = self.BufferLength
        self.Trace = numpy.zeros((self.BufferLength*self.Chunks,))
        #temp = self.Trace.copy()
        self.Reset()
        self.Stream=True
        for i in range(self.Chunks):
            if self.abort.isSet():
                break
            self.Trace[i*M:(i+1)*M] = self.ReadPipe()
            
            xem.UpdateWireOuts()
            ep21 = xem.GetWireOutValue(0x21)
            ep20 = xem.GetWireOutValue(0x20)
            print 'Output values', ep20, ep21
            print 'Trace[0:9] values', self.Trace[0:9]
            
            #self.Trace = temp
            #if i % 10 == 0:
            #    self.HistogramN, self.HistogramBins = numpy.histogram(self.Trace[:(i+1)*M], bins=numpy.arange(self.Trace.min(),self.Trace.max(),1))            
            #    self._Trace_changed()
        self.HistogramN, self.HistogramBins = numpy.histogram(self.Trace[:(i+1)*M], bins=numpy.arange(self.Trace.min(),self.Trace.max(),1))            
        self._Trace_changed()
        
    def GetTrace2(self, weightfnct):
        '''Get trace with weighting function'''
        M = self.BufferLength
        bits = self.bits_readout
        if len(weightfnct) != 64:
            raise RuntimeError('Length of weightfunction != 64')
        N = len(weightfnct)
        weightfnct_init = weightfnct[::-1]
        points = M / N
        self.Trace = numpy.zeros((self.BufferLength*self.Chunks))
        self.Reset()
        self.Stream = True
        Trace_init = []
        Trace_read = []           
        for i in range(self.Chunks):
            if self.abort.isSet():
                break
            for k in range(N):
                temp = self.ReadPipe()
                for j in range(points):
                    value_init = 0
                    value_read = 0
                    for k in range(N):
                        value_init += temp[j*N+k] * weightfnct_init[k]
                        value_read += temp[j*N+k] * weightfnct[k]
                    Trace_init.append(value_init)
                    Trace_read.append(value_read)
        self.Trace = numpy.array(Trace_read)  
        self.Trace_init = numpy.array(Trace_init)                  
        #self.HistogramN, self.HistogramBins = numpy.histogram(self.Trace[:(i+1)*M], bins=numpy.arange(self.Trace.min(),self.Trace.max(),1))            
        self.HistogramN, self.HistogramBins = numpy.histogram(self.Trace, bins=numpy.arange(self.Trace.min(),self.Trace.max(),1))
        self._Trace_changed()
        
    HistogramN = Array(value=numpy.array((0,1)))
    HistogramBins = Array(value=numpy.array((0,0)))
    
# continuous data acquisition in thread

    Loop = Bool(False)

    Run = Property( trait=Bool )

    abort = threading.Event()
    abort.clear()

    _StopTimeout = 10.

    def _get_Run(self):
        if hasattr(self, 'Thread'):
            return self.Thread.isAlive()
        else:
            return False

    def _set_Run(self, value):
        if value == True:
            self.Start()
        else:
            self.Stop()

    def Start(self):
        """Start Measurement in a thread."""
        self.Stop()
        self.Thread = threading.Thread(target=self.run)
        self.Thread.start()

    def Stop(self):
        if hasattr(self, 'Thread'):
            self.abort.set()
            self.Thread.join(self._StopTimeout)
            self.abort.clear()
            if self.Thread.isAlive():
                self.ready.set()
            del self.Thread

    def run(self):
        self.Traces = []
        self.GetTrace()
        while self.Loop:
            if self.abort.isSet():
                break
            self.GetTrace()
        self.Traces.append( self.Export() )
        self.trace=self.Export()
    
    # trace and histogram plots

    TracePlot = Instance( Component )
    HistPlot = Instance( Component )
    
    def _TracePlot_default(self):
         return self._create_TracePlot_component()

    def _HistPlot_default(self):
         return self._create_HistPlot_component()

    def _create_TracePlot_component(self):
        plot = DataView(border_visible = True)
        line = LinePlot(value = ArrayDataSource(self.Trace),
                        index = ArrayDataSource(numpy.arange(len(self.Trace))),
                        color = 'blue',
                        index_mapper = LinearMapper(range=plot.index_range),
                        value_mapper = LinearMapper(range=plot.value_range))
        plot.index_range.sources.append(line.index)
        plot.value_range.sources.append(line.value)
        plot.add(line)
        plot.index_axis.title = 'index'
        plot.value_axis.title = 'Fluorescence [ counts / s ]'
        plot.tools.append(PanTool(plot))
        plot.overlays.append(ZoomTool(plot))
        self.TraceLine=line
        return plot
    
    def _create_HistPlot_component(self):
        plot = DataView(border_visible = True)
        line = LinePlot(index = ArrayDataSource(self.HistogramBins),
                        value = ArrayDataSource(self.HistogramN),
                        color = 'blue',
                        #fill_color='blue',
                        index_mapper = LinearMapper(range=plot.index_range),
                        value_mapper = LinearMapper(range=plot.value_range))
        plot.index_range.sources.append(line.index)
        plot.value_range.sources.append(line.value)
        plot.add(line)
        plot.index_axis.title = 'Fluorescence counts'
        plot.value_axis.title = 'number of occurences'
        plot.tools.append(PanTool(plot))
        plot.overlays.append(ZoomTool(plot))
        self.HistLine=line
        return plot
    
    def _save_timetrace_changed(self):
        fil = open(self.file_path_timetrace+'\\'+self.file_name_timetrace+'_Trace'+r'.asc','w')
        fil.write('[Data]')
        fil.write('\n')
        for x in self.trace.Normal():
            fil.write('%i '%x)
            fil.write('\n')
        fil.close()
    
    def _save_hist_changed(self):
        fil = open(self.file_path_hist+'\\'+self.file_name_hist+'_Hist'+r'.asc','w')
        fil.write('binning = %i'%self.binning)
        fil.write('\n')
        fil.write('[Data]')
        fil.write('\n')
        for x in range(len(self.HistogramBins)-1):
            fil.write('%i'%self.HistogramBins[x]+'   '+'%i'%self.HistogramN[x]+'\n')
        fil.close()           
    
    def _refresh_hist_changed(self):
        self.trace_binned = numpy.zeros((self.BufferLength*self.Chunks/self.binning,))
        a=0
        for i in range(len(self.Trace[:-(self.binning)])):
            if i % self.binning == 0:
                self.trace_binned[a] = self.Trace[i:(i+self.binning)].sum()
                a = a+1
        self.HistogramN, self.HistogramBins = numpy.histogram(self.trace_binned, bins=numpy.arange(self.trace_binned.min(),self.trace_binned.max(),1))
        self._Trace_changed()
        self._HistogramN_changed()
        self._HistogramBins_changed()
        
    def __init__(self):
        super(Flopper, self).__init__()

        # TracePlotData = ArrayPlotData(Trace=self.Trace)
        # HistPlotData = ArrayPlotData(Bins=self.HistogramBins, N=self.HistogramN)

        # TracePlot = Plot( TracePlotData )
        # HistPlot = Plot( HistPlotData )

        # TracePlot.index_axis.title = 'index'
        # TracePlot.value_axis.title = 'Fluorescence counts'
        # HistPlot.index_axis.title = 'Fluorescence counts'
        # HistPlot.value_axis.title = 'number'

        # TraceRenderer = TracePlot.plot('Trace', type='line', color='blue')[0]
        # HistRenderer = HistPlot.plot(('Bins','N'), type='line', color='blue')[0]

        # TracePlot.tools.append(PanTool(TracePlot, drag_button='right'))
        # TracePlot.tools.append(ZoomTool(TracePlot, tool_mode='range'))
        #TracePlot.overlays.append()

        # self.TracePlot = TracePlot
        # self.HistPlot = HistPlot
        # self.TraceRenderer=TraceRenderer
        # self.HistRenderer=HistRenderer

        self._PulseLength_changed()
        self._Threshold_changed()
        self.BufferLength=512
        self.Pulse=False
        self.Stream=True
        self._Buf = '\x00'*2*self.BufferLength
        self.Reset()

    def _Trace_changed(self):
        if len(self.Trace) > 40000:     #Program can't handle very long traces
            Trace = self.Trace[0:40000]
        else:
            Trace = self.Trace
        self.TraceLine.value.set_data(Trace)
        self.TraceLine.index.set_data(numpy.arange(len(Trace)))

    def _HistogramN_changed(self):
        self.HistLine.value.set_data(self.HistogramN)
        
    def _HistogramBins_changed(self):
        self.HistLine.index.set_data(self.HistogramBins)
        
    def Export(self):
        return fitting.Trace(self.Trace, self.Threshold, self.Pulse)
        
    # def SaveAsc(self):
        # Ytrace=self.Trace
        # fil = open( self.FileNameTrace,'w')
        # for i in range(len(Ytrace)):
            # if i==0:
                # fil.write('Trace\n')
                # fil.write('%f\n'%(Ytrace[i]) )
            # else: 
                # fil.write('%f\n'%(Ytrace[i]) )    
        # fil.close()
        
        # Yhist= self.HistogramN
        Xhist= self.HistogramBins
        # fil = open( self.FileNameHist,'w')
        # for i in range(len(Yhist)):
            # if i==0:
                # fil.write('Histogram\n')
                # fil.write('%f\n'%(Yhist[i]) )
            # else: 
                # fil.write('%f\n'%(Yhist[i]) )    
        # fil.close()         
#    def __del__(self):
#        del self.xem

    def __getstate__(self):
        """Returns current state of a selection of traits.
        Overwritten HasTraits.
        """
        state = SingletonHasTraits.__getstate__(self)
        for key in ['Thread','abort']:
            if state.has_key( key ):
                del state[ key ]
        return state

#flopper = Flopper()
#flopper.edit_traits()


