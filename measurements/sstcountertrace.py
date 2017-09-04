import numpy
import threading
import time
import traceback
import sys

import os
os.environ['ETS_TOOLKIT'] = 'qt4'

import traits.etsconfig
traits.etsconfig.enable_toolkit='qt4'
traits.etsconfig.toolkit='qt4'

from traits.api import *
from traitsui.api import *

from enable.api import Component, ComponentEditor
from chaco.api import ArrayDataSource, LinePlot, LinearMapper, ArrayPlotData, jet, Plot, CMapImagePlot
from chaco.tools.api import PanTool, ZoomTool
import datetime

#from pi3diamond import pi3d, CloseHandler
from hardware.api import SSTtrace, PulseGenerator, Microwave_HMC, AWG, FastComtec
from tools.utility import GetSetItemsMixin

from tools.emod import ManagedJob

from tools.utility import GetSetItemsMixin

#from hardware.api import Microwave 

from hardware.awg import *
from hardware.waveform import *

PG = PulseGenerator()
#MW = Microwave()
MW = Microwave_HMC()
FC = FastComtec()
AWG = AWG()

ssttrace = SSTtrace()

class SSTCounterTrace(SingletonHasTraits, GetSetItemsMixin):

    readout_interval = Float(0.01, label='Data readout interval [s]', desc='How often data read is requested from nidaq')
    samples_per_read = Int(200, label='# data points per read', desc='Number of data points requested from nidaq per read. Nidaq will automatically wait for the data points to be aquired.')
    max_sampling_rate = Property(trait=Int, depends_on='readout_interval,samples_per_read', label='Max sampling rate [kS/s]', 
                                 desc='Maximum sampling rate (i.e. data points per second) of aquisition. If gate signal is faster, buffer overflow will occur')
    points = Int(1000, label='# points')
    progress = Int(0, label='progress [points]')
    state = Enum('idle', 'count')
    thread = Trait()
    
    hist_binning = Int(1)
    refresh_hist = Button()
    trace = Array()
    histogram = Array()
    histogram_bins = Array()

    trace_plot_data = Instance( ArrayPlotData, transient=True )
    trace_plot = Instance( Plot, transient=True )
    hist_plot_data = Instance( ArrayPlotData, transient=True )
    hist_plot = Instance( Plot, transient=True )
    
    file_path_timetrace = Str(r'D:\data\Timetrace')
    file_name_timetrace = Str('enter filename')
    save_timetrace = Button()
    file_path_hist = Str(r'D:\data\Histograms')
    file_name_hist = Str('enter filename')
    save_hist = Button()
    
    
    def prepare_awg(self):
        sampling = 1.2e9
        N_shot = int(self.N_shot)
        points = self.points
        pi = int(self.pi * sampling / 1.0e9)
        laser_SST = int(self.laser_SST * sampling / 1.0e9)
        wait_SST = int(self.wait_SST * sampling / 1.0e9)
        
        if self.reload:
            AWG.stop()
            AWG.set_output( 0b0000 )
            AWG.delete_all()
            
            zero = Idle(1)
            self.waves = []
            sub_seq = []
            p = {}
            
            p['pi + 0'] = Sin(pi, (self.freq - self.freq_center)/sampling, 0, self.amp)
            p['pi + 90'] = Sin(pi, (self.freq - self.freq_center)/sampling, np.pi/2, self.amp)
            
            read_x = Waveform('read_x', [p['pi + 0'],  Idle(laser_SST, marker1 = 1), Idle(wait_SST)])
            read_y = Waveform('read_y', [p['pi + 90'], Idle(laser_SST, marker1 = 1), Idle(wait_SST)])
            self.waves.append(read_x)
            self.waves.append(read_y)
            
            self.main_seq = Sequence('SST.SEQ')
            for i in range(points):
                name = 'DQH_12_%04i.SEQ' % i
                sub_seq=Sequence(name)
                sub_seq.append(read_x, read_y, repeat = N_shot)
                AWG.upload(sub_seq)
                
                self.main_seq.append(sub_seq, wait=True)
            for w in self.waves:
                w.join()
            AWG.upload(self.waves)
            AWG.upload(self.main_seq)
            AWG.tell('*WAI')
            AWG.load('SST.SEQ')
        AWG.set_vpp(self.vpp)
        AWG.set_sample( sampling/1.0e9 )
        AWG.set_mode('S')
        AWG.set_output( 0b0011 )
        

    def generate_sequence(self):
        points = self.points
        N_shot = self.N_shot
        laser = self.laser
        wait = self.wait
        laser_SST = self.laser_SST
        wait_SST = self.wait_SST
        pi = self.pi
        record_length = self.record_length
        
        sequence = []
        for t in range(points):
            sequence.append( (['laser'], laser) )
            sequence.append( ([ ],  wait) )
            sequence.append( (['awgTrigger']      , 100) )
            sequence.append( ([ 'sst' ]       , record_length) )
            '''
            for n in range(int(N_shot)):
                sequence.append( ([ 'trigger' ]       , pi) )
                sequence.append( (['laser', 'trigger'], laser_SST) )
                sequence.append( ([ 'trigger']        , wait_SST) )
                '''
        #sequence.append(  ([                   ] , 12.5  )  )
        return sequence
    
    def count(self):
        """"""
        try:
            if ssttrace.start_gated_counting() != 0:   # initialize and start nidaq gated counting task, return 0 if successful
                print 'error in nidaq'
                return
            # make sure to start data acquisition shortly after, as nidaq is waiting for data to read, and may time out eventually (check timeout of nidaq.read_gated_counts function)
            aquired_data = numpy.empty(0)   # new data will be appended to this array
            self.trace = numpy.zeros(self.points)   # reset trace
            while self.points - len(aquired_data) > 0:
                points_left = self.points - len(aquired_data)
                threading.current_thread().stop_request.wait(self.readout_interval) # wait for some time before new read command is given. not sure if this is necessary
                if threading.current_thread().stop_request.isSet():
                    break
                new_data = ssttrace.read_gated_counts( samples=min(self.samples_per_read, points_left) )   # do not attempt to read more data than necessary
                aquired_data = numpy.append( aquired_data, new_data[:min(len(new_data), points_left)] )
                self.trace[:len(aquired_data)] = aquired_data[:]    # length of trace may not change due to plot, so just copy acquired data into trace
                self._trace_changed()   # will not be triggered by command above
                self.progress = len(aquired_data)
            self.histogram, self.histogram_bins = numpy.histogram(self.trace, bins=numpy.arange(self.trace.min() - 1,self.trace.max() + 1, 1))
        except Exception as e:
            self.thread.stop_request.set()
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)
        finally:
            ssttrace.stop_gated_counting() # stop nidaq task to free counters
            self.state = 'idle'
        
    def get_trace(self, turn_dtg_on=False):
        self.state = 'count'
        time.sleep(0.1)
        while self.state == 'count':
            time.sleep(0.1)
        time.sleep(1)
        return numpy.array(self.trace.copy(), dtype=numpy.uint16)
        
    def _trace_changed(self):
        if len(self.trace) > 10000:     # if trace to plot is too long memory errors / performance problems may occur
            trace = self.trace[0:10000]
        else:
            trace = self.trace
        self.trace_plot_data.set_data('y', trace)
        self.trace_plot_data.set_data('x', numpy.arange(len(trace)))
        self.trace_plot.request_redraw()
        
    def _histogram_bins_changed(self):
        self.hist_plot_data.set_data('x', self.histogram_bins)
        
    def _histogram_changed(self):
        self.hist_plot_data.set_data('y', self.histogram)
        self.hist_plot.request_redraw()
        
    def _state_changed(self):
        self.stop_thread()
        if self.state == 'count':
            self.thread = threading.Thread(target=self.count)
            self.thread.stop_request = threading.Event()
            self.thread.start()

    def start(self, abort):
        self.state = 'count'
        while self.thread is None:
            time.sleep(0.1)
        self.thread.stop_request = abort


    def stop_thread(self):
        if isinstance(self.thread, threading.Thread):
            if self.thread is None or self.thread is threading.current_thread() or not self.thread.isAlive():
                return
            self.thread.stop_request.set()
            self.thread.join()
            self.thread = None

    def _trace_default(self):
        return numpy.zeros(1000)
    
    def _trace_plot_data_default(self):
        return ArrayPlotData(x=numpy.arange(len(self.trace)), y=self.trace)
    
    def _trace_plot_default(self):
        plot = Plot(self.trace_plot_data, padding_left=50, padding_top=10, padding_right=10, padding_bottom=30)
        plot.plot(('x','y'), style='line', color='blue', name='trace')
        plot.index_axis.title = '# bin'
        plot.value_axis.title = '# counts'
        plot.tools.append(PanTool(plot))
        plot.overlays.append(ZoomTool(plot))
        return plot
    
    def _histogram_bins_default(self):
        return numpy.arange(10)
    def _histogram_default(self):
        return numpy.zeros(self.histogram_bins.shape)
    
    def _hist_plot_data_default(self):
        return ArrayPlotData(x=self.histogram_bins, y=self.histogram)
    
    def _hist_plot_default(self):
        plot = Plot(self.hist_plot_data, padding_left=50, padding_top=10, padding_right=10, padding_bottom=30)
        plot.plot(('x','y'), style='line', color='blue', name='hist')
        plot.index_axis.title = '# counts'
        plot.value_axis.title = '# occurences'
        plot.tools.append(PanTool(plot))
        plot.overlays.append(ZoomTool(plot))
        return plot
    
    def _refresh_hist_changed(self):
        self.trace_binned = numpy.zeros(int(len(self.trace)/self.hist_binning))
        a=0
        for i in range(len(self.trace[:-(self.hist_binning)])):
            if i % self.hist_binning == 0:
                self.trace_binned[a] = self.trace[i:(i+self.hist_binning)].sum()
                a = a+1
        self.histogram, self.histogram_bins = numpy.histogram(self.trace_binned, bins=numpy.arange(self.trace_binned.min(),self.trace_binned.max(),1))
        self._histogram_changed()
        self._histogram_bins_changed()
    
    def _save_timetrace_changed(self):
        path = self.file_path_timetrace+'/'+self.file_name_timetrace+'_Trace'
        filename = path + '.dat'
        if filename in os.listdir(self.file_path_timetrace):
            print 'File already exists! Data NOT saved!'
            print 'Choose other filename!'
            return
        fil = open(filename,'w')
        fil.write('[Data]')
        fil.write('\n')
        for x in self.trace:
            fil.write('%i '%x)
            fil.write('\n')
        fil.close()
        self.save_figure(self.trace_plot, path+'plot.png')
        print 'saved gated counter trace' + self.file_name_timetrace
    
    def _save_hist_changed(self):
        path = self.file_path_hist+'/'+self.file_name_hist+'_Hist'
        filename = path+'.dat'
        if filename in os.listdir(self.file_path_hist):
            print 'File already exists! Data NOT saved!'
            print 'Choose other filename!'
            return
        fil = open(filename,'w')
        fil.write('binning = %i'%self.hist_binning)
        fil.write('\n')
        fil.write('[Data]')
        fil.write('\n')
        for x in range(len(self.histogram_bins)-1):
            fil.write('%i'%self.histogram_bins[x]+'\t'+'%i'%self.histogram[x]+'\n')
        fil.close() 
        self.save_figure(self.hist_plot, path+'plot.png')
        print 'saved gated counter histogram ' + self.file_name_hist

    @cached_property
    def _get_max_sampling_rate(self):
        return round(self.samples_per_read * 1./self.readout_interval * 1e-3, 2)    

    def reset_settings(self):
        self.points = 2000
        self.samples_per_read = 400

    view = View( VGroup( HGroup(Item('points', enabled_when='self.state=="idle"'),
                                Item('state', style='custom', show_label=False),
                                Item('samples_per_read'),
                                Item('progress', style='readonly', width=35),
                                ),
                         HGroup( VGroup(Item('trace_plot', editor=ComponentEditor(), show_label=False, width=500, height=300, resizable=True),
                                        HGroup(Item('file_name_timetrace', label='Filename of timetrace:'),
                                               Item('save_timetrace', label = 'Save Timetrace', show_label=False))),
                                 VGroup(Item('hist_plot', editor=ComponentEditor(), show_label=False, width=500, height=300, resizable=True),
                                        HGroup(Item('hist_binning', label='# of bins'),
                                               Item('refresh_hist', label = 'Refresh histogram', show_label=False)),
                                        HGroup(Item('file_name_hist', label='Filename of histogram:'),
                                               Item('save_hist', label = 'Save Histogram', show_label=False)))),
                         ),
                 title='Gated Counter', width=700, height=500,x=0, buttons=['OK'], resizable=True)#, handler=CloseHandler )

    def __getstate__(self):
        """Returns current state of a selection of traits.
        Overwritten HasTraits.
        """
        state = SingletonHasTraits.__getstate__(self)
        for key in ['thread']:
            if state.has_key( key ):
                del state[ key ]
        return state