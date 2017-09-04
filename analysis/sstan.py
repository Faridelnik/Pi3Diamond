import numpy as np
from fitting import find_edge, run_sum   
    
from traits.api import HasTraits, Trait, Instance, Property, Range, Float, Int, String, Bool, Array, List, Str, Tuple, Enum, on_trait_change, cached_property, DelegatesTo, Any, Button
from traitsui.api import View, Item, Tabbed, Group, HGroup, VGroup, VSplit, EnumEditor, TextEditor, InstanceEditor
from enable.api import ComponentEditor
from chaco.api import ArrayDataSource, LinePlot, LinearMapper, ArrayPlotData, Plot, Spectral, PlotLabel, Legend
from enthought.chaco.tools.api import PanTool, ZoomTool

from traitsui.file_dialog import save_file
from traitsui.menu import Action, Menu, MenuBar

import matplotlib.pyplot as plt

import threading
import time
import logging

import fitting

from tools.utility import GetSetItemsHandler, GetSetItemsMixin

#import measurements.pulsed_awg as mp
#import measurements.prog_gate_awg as pga
#import measurements.pair_search as ps
#import measurements.shallow_NV as sensing
#import measurements.nmr as nmr
import measurements.singleshot as ss
#import measurements.opticalrabi as orabi
#import measurements.nuclear_rabi as nr
#import measurements.rabi as newrabi
#from measurements.odmr import ODMR


def spin_state(c, n_points):
    
    """
    Compute the spin state from a full array of count data.
    
    Parameters:
    
        c    = count data
  n_points   = number of sequence points
        
    """
    if len(c)>=n_points:
        n_sweep = int(len(c)/n_points)
       
        #c_new = np.zeros(n_sweep * n_points)
        c_new = c[:(n_sweep * n_points)]
        #print(c_new)
        #c_new = c[0:(n_sweep * n_points)]
        c_array = c_new.reshape(n_sweep, n_points)
        y = c_array.sum(0)
        #print(1,y)    
        if len(y) != n_points:
            print('data size is wrong!')
    else:
        y = np.zeros(n_points)
        #print(2,y)    

    return y

class PulsedAnaHandler(GetSetItemsHandler):

    """Provides handling of menu."""

    def save_hist_plot(self, info):
        filename = save_file(title='Save Hist Plot')
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
    
    def save_all(self, info):
        filename = save_file(title='Save All')
        if filename is '':
            return
        else:
            info.object.save_all(filename)

menubar = MenuBar(Menu(Action(action='save', name='Save (.pyd or .pys)'),
                   Action(action='load', name='Load (.pyd or .pys)'),
                   Action(action='save_hist_plot', name='Save Hist Plot (.png)'),
                   Action(action='save_line_plot', name='Save Line Plot (.png)'),
                   Action(action='save_all', name='Save All'),
                   Action(action='_on_close', name='Quit'),
                   name='File'
                   ),
              )

class PulsedFit(HasTraits, GetSetItemsMixin):

    """
    Base class for a pulsed fit. Provides calculation of normalized intensity.
    Derive from this to create fits for pulsed measurements.
    """

    measurement = Instance(ss.Pulsed, factory=ss.Pulsed)
    
    x_tau = Array(value=np.array((0., 1.)))
    spin_state = Array(value=np.array((0., 0.)))
    spin_state_error = Array(value=np.array((0., 0.)))
    fit_parameters = Array(value=np.array((np.nan, np.nan, np.nan, np.nan)))

    def __init__(self):
        super(PulsedFit, self).__init__()
        #self.on_trait_change(self.update_spin_state, 'spin_state, x_tau, measurement.count_data', dispatch='ui')
        self.on_trait_change(self.update_plot_spin_state, 'spin_state, measurement.count_data', dispatch='ui')
        
    
    @on_trait_change('measurement.count_data, measurement.progress')
    def update_spin_state(self):
        if self.measurement is None:
            return
            
        self.x_tau = self.measurement.tau
        sequence_points = len(self.x_tau)
        y = spin_state(c=self.measurement.count_data,
                       n_points = sequence_points)
        self.spin_state = y
        self.spin_state_error = y ** 0.5
        

    # The following is a bit tricky. Data for plots is passed to the PulsedAnalyzer through these two attributes.
    # The first one is a standard chaco ArrayPlotData instance. It is very important to specify a proper initial instance,
    # otherwise the PulsedAnalyzer will not start. The line below specifies an initial instance through the factory and kw argument.
    line_data = Instance(ArrayPlotData,
                         factory=ArrayPlotData,
                         kw={'tau':np.array((0, 1)),
                             'spin_state':np.array((0, 0)),
                             }
                         )
    # The second line is a list that is interpreted to create appropriate plots through the chaco Plot.plot() method.
    # The list elements are dictionaries that are directly passed to the Plot.plot() command as keyword arguments through the **kwagrs expansion 
    plots = [ {'data':('tau', 'spin_state'), 'color':'blue', 'name':'pulsed'} ]
        
    def update_plot_spin_state(self):
              
        old_mesh = self.x_tau
        old_mesh = self.line_data.get_data('pulse_number')
        if old_mesh is not None and len(old_mesh) != len(self.spin_state):
            self.line_data.set_data('tau', np.arange(len(self.spin_state)))
        self.line_data.set_data('spin_state', self.spin_state)
        self.line_data.set_data('tau', old_mesh)
        

    traits_view = View(title='Pulsed Fit')

    get_set_items = ['__doc__', 'spin_state', 'spin_state_error']
    
    
class TracePulsedAnalyzerHandler(GetSetItemsHandler):

    """Provides handling of menu."""

    # save plots
    
    def save_hist_plot(self, info):
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
            
    def save_all(self, info):
        filename = save_file(title='Save All')
        if filename is '':
            return
        else:
            info.object.save_all(filename)
            
    def new_trace_measurement(self, info):
        info.object.measurement = ss.SSTCounterTrace()


    def new_pulsed_fit(self, info):
        info.object.fit = PulsedFit()
        
    def new_unknown_fit(self, info):
        info.object.fit = SSTtraceFit()
        
    def new_odmr_fit_XY8(self, info):
        info.object.fit = OdmrFitXY8()
           
class TracePulsedAnalyzer(HasTraits, GetSetItemsMixin):
    
    """Fits a pulsed measurement with a pulsed fit.
    
    """
    
    measurement = Instance(ss.Pulsed, factory=ss.Pulsed)
    fit = Instance(PulsedFit, factory=PulsedFit)

        
    hist_binning = Int(1)
    refresh_hist = Button()
    histogram = Array()
    histogram_bins = Array()
    
    hist_plot_data = Instance( ArrayPlotData, transient=True )
    hist_plot = Instance( Plot, transient=True )
    
    line_plot = Instance(Plot)
    # line_data is provided by fit class

    def __init__(self):
        super(TracePulsedAnalyzer, self).__init__()
        self.on_trait_change(self.refresh_line_data, 'measurement.count_data, fit.spin_state', dispatch='ui')
        self.on_trait_change(self.refresh_hist_plot_data, 'fit.spin_state', dispatch='ui')
        self.on_trait_change(self._update_processed_plot_data_fit, 'fit.fit_parameters', dispatch='ui')
        

        self.histogram, self.histogram_bins = np.histogram(self.fit.spin_state, bins=np.arange(self.fit.spin_state.min(),self.fit.spin_state.max() + 1, 1))   
        
    def _histogram_bins_changed(self):
        self.hist_plot_data.set_data('x', self.histogram_bins[1:])
        
    def _histogram_changed(self):
        self.hist_plot_data.set_data('y', self.histogram)
        self.hist_plot.request_redraw()

    def _measurement_changed(self, new):
        self.fit = PulsedFit()
        
    def _histogram_bins_default(self):
        return np.arange(10)
        
    def _histogram_default(self):
        return np.zeros(self.histogram_bins[1:].shape)
    
    def _hist_plot_data_default(self):
        return ArrayPlotData(x=self.histogram_bins[1:], y=self.histogram)
        
    def refresh_hist_plot_data(self):
        self.histogram, self.histogram_bins = np.histogram(self.fit.spin_state, bins=np.arange(self.fit.spin_state.min(), self.fit.spin_state.max() + 1, 1))
        self.hist_plot_data.set_data('x', self.histogram_bins[1:])
        self.hist_plot_data.set_data('y', self.histogram)    
        
    def _update_processed_plot_data_fit(self):
       
        if not np.isnan(self.fit.fit_parameters[0]):     
            
            self.hist_plot_data.set_data('fit', fitting.NLorentzians(*self.fit.fit_parameters)(self.histogram_bins[1:]))
            self.hist_plot.plot(('x', 'fit'), color='red', style='line', line_width = 1)
            p = self.fit.fit_parameters
            f = p[1::3]
            w = p[2::3]
            N = len(p) / 3
            contrast = np.empty(N)
            c = p[0]
            pp = p[1:].reshape((N, 3))
            for i, pi in enumerate(pp):
                a = pi[2]
                g = pi[1]
                A = np.abs(a / (np.pi * g))
                if a > 0:
                    contrast[i] = 100 * A / (A + c)
                else:
                    contrast[i] = 100 * A / c
            s = ''
            
            for i, fi in enumerate(self.fit.fit_times):
                s += 'f %i: %.1f, HWHM %.1f, contrast %.1f%%\n,' % (i + 1, fi, self.fit.fit_line_width[i], contrast[i])
            self.text = s
        
    def refresh_line_data(self):
        self.fit.line_data.set_data('tau', self.fit.x_tau)
        self.fit.line_data.set_data('spin_state', self.fit.spin_state)   
    
    def _hist_plot_default(self):
        plot = Plot(self.hist_plot_data, padding_left=50, padding_top=10, padding_right=10, padding_bottom=30)
        plot.plot(('x','y'), style='line', color='blue', name='hist')
        plot.index_axis.title = '# counts'
        plot.value_axis.title = '# of events'
        plot.tools.append(PanTool(plot))
        plot.overlays.append(ZoomTool(plot))
        return plot
    
    def _refresh_hist_changed(self):
        self.trace_binned = np.zeros(int(len(self.fit.spin_state)/self.hist_binning))
        a=0
        for i in range(len(self.fit.spin_state[:-(self.hist_binning)])):
            if i % self.hist_binning == 0:
                self.trace_binned[a] = self.fit.spin_state[i:(i+self.hist_binning)].sum()
                a = a+1
        self.histogram, self.histogram_bins = np.histogram(self.trace_binned, bins=np.arange(self.trace_binned.min(),self.trace_binned.max()+1,1))
        self._histogram_changed()
        self._histogram_bins_changed()    
    

    def _line_plot_default(self):
        plot = Plot(self.fit.line_data, padding=8, padding_left=64, padding_bottom=36)
        for item in self.fit.plots:
            plot.plot(**item)
        plot.index_axis.title = '# bin'
        plot.value_axis.title = 'spin state'
        return plot
        

    def _fit_changed(self, name, old, new):
        old.measurement = None
        new.measurement = self.measurement
        plot = self.line_plot
        # delete all old plots
        for key in plot.plots.keys():
            plot.delplot(key)
        # set new data source
        plot.data = self.fit.line_data
        # make new plots
        for item in self.fit.plots:
            plot.plot(**item)
        # if the fit has an attr 'text' attached to it, print it in the lower left corner of the plot
        if hasattr(old, 'text'):
            label = plot.overlays[0]
            old.sync_trait('text', label, 'text', mutual=False, remove=True)
            plot.overlays = []
        if hasattr(new, 'text'):
            label = PlotLabel(text=self.fit.text, hjustify='left', vjustify='bottom', position=[64, 32])
            new.sync_trait('text', label, 'text', mutual=False)
            plot.overlays = [label]
    
    def save_hist_plot(self, filename):
        self.save_figure(self.hist_plot, filename)
    
    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename)
        
    def save_all(self, filename):
        self.save_line_plot(filename + 'line.png')
        self.save_hist_plot(filename + 'hist.png')
        self.save(filename + '.pyd')
        
    # Menu of the PulsedAnalyzer, only include tested measurements and Fits please!
    
    traits_view = View(HGroup(VGroup(Item(name='measurement', style='custom', show_label=False),
                                     Item('line_plot', name='normalized intensity', editor=ComponentEditor(), show_label=False, width=300, height=300, resizable=True), 
                                     ),
                              VGroup(Item(name='fit', style='custom', show_label=False),
                                        HGroup(Item('hist_binning', label='# of bins'),
                                               Item('refresh_hist', label = 'Refresh histogram', show_label=False)),
                                     Item('hist_plot', editor=ComponentEditor(), show_label=False, width=300, height=300, resizable=True),
                                     ),
                              ),
                       menubar=MenuBar(Menu(Action(action='save', name='Save (.pyd or .pys)'),
                                              Action(action='load', name='Load (.pyd or .pys)'),
                                              Action(action='save_hist_plot', name='Save Hist Plot (.png)'),
                                              Action(action='save_line_plot', name='Save Line Plot (.png)'),
                                              Action(action='save_all', name='Save All'),
                                              Action(action='_on_close', name='Quit'),
                                              name='File'),
                                         Menu(Action(action='new_trace_measurement', name='SSTTrace'),
                                             
                                              name='Measurement'),
                                         Menu(Action(action='new_pulsed_fit', name='Pulsed'),
                                              Action(action='new_unknown_fit', name='SSTtraceFit'),
                                              Action(action = 'new_odmr_fit_XY8', name = 'Fit Lorentzians'),
                                              name='Fit'),
                                           
                                              
                                 ),
                       title='TracePulsedAnalyzer',width=1400, height=600, buttons=[], resizable=True, handler=TracePulsedAnalyzerHandler)

    get_set_items = ['measurement', 'fit']
    get_set_order = ['measurement', 'fit']    
    
    
    
class OdmrFitXY8(PulsedFit):

    fit = Instance(PulsedFit, factory=PulsedFit)

    text = Str('')
     
    number_of_resonances = Trait('auto', String('auto', auto_set=False, enter_set=True), Int(10000., desc='Number of Lorentzians used in fit', label='N', auto_set=False, enter_set=True))
    threshold = Range(low= -99, high=99., value= 50., desc='Threshold for detection of resonances [%]. The sign of the threshold specifies whether the resonances are negative or positive.', 
    label='threshold [%]', mode='text', auto_set=False, enter_set=True)
    
    perform_fit = Bool(False, label='perform fit')
    
    fit_times = Array(value=np.array((np.nan,)), label='time [ns]') 
    fit_line_width = Array(value=np.array((np.nan,)), label='line_width [ns]') 
    fit_contrast = Array(value=np.array((np.nan,)), label='contrast [%]')

    def __init__(self):
        super(OdmrFitXY8, self).__init__()
        #self.on_trait_change(self._update_processed_plot_data_fit, 'fit_parameters', dispatch='ui')
        self.on_trait_change(self._update_fit, 'spin_state, number_of_resonances, threshold, perform_fit', dispatch='ui')
    
    def _update_fit(self):
  
        if self.perform_fit:
            
            histogram, histogram_bins = np.histogram(self.spin_state, bins=np.arange(self.spin_state.min(),self.spin_state.max() + 1, 1))
              
            N = self.number_of_resonances 
            
            if N != 'auto':
               N = int(N)
               
            try:
                self.fit_parameters = fitting.fit_multiple_lorentzians(histogram_bins[1:], histogram, N, threshold=self.threshold * 0.01)
            except Exception:
                logging.getLogger().debug('XY8 fit failed.', exc_info=True)
                self.fit_parameters = np.nan * np.empty(4)
        else:
            self.fit_parameters = np.nan * np.empty(4)
        
        p=self.fit_parameters
        self.fit_times = p[1::3]
        self.fit_line_width = p[2::3]
        N = len(p) / 3
        contrast = np.empty(N)
        c = p[0]
        pp = p[1:].reshape((N, 3))
        for i, pn in enumerate(pp):
            a = pn[2]
            g = pn[1]
            A = np.abs(a/(np.pi * g))
            if a > 0:
                contrast[i] = 100 * A / (A + c)
            else:
                contrast[i] = 100 * A / c
        self.fit_contrast = contrast
                
    plots = [{'data':('tau', 'spin_state'), 'color':'blue', 'name':'pulsed'},
             {'data':('x','y'), 'color':'blue', 'name':'hist'},
             {'data':('x', 'fit'), 'color':'purple', 'name':'fitting','label':'Fit'}]
  

    # def _update_processed_plot_data_fit(self):
       
        # if not np.isnan(self.fit_parameters[0]):     
            # histogram, histogram_bins = np.histogram(self.spin_state, bins=np.arange(self.spin_state.min(),self.spin_state.max() + 1, 1))
            
            # self.hist_plot_data.set_data('fit', fitting.NLorentzians(*self.fit_parameters)(histogram_bins[1:]))
            
            # p = self.fit_parameters
            # f = p[1::3]
            # w = p[2::3]
            # N = len(p) / 3
            # contrast = np.empty(N)
            # c = p[0]
            # pp = p[1:].reshape((N, 3))
            # for i, pi in enumerate(pp):
                # a = pi[2]
                # g = pi[1]
                # A = np.abs(a / (np.pi * g))
                # if a > 0:
                    # contrast[i] = 100 * A / (A + c)
                # else:
                    # contrast[i] = 100 * A / c
            # s = ''
            
            # for i, fi in enumerate(self.fit_times):
                # s += 'f %i: %.1f, HWHM %.1f, contrast %.1f%%\n,' % (i + 1, fi, self.fit_line_width[i], contrast[i])
            # self.text = s
                 
    traits_view = View(Tabbed(VGroup(HGroup(Item('number_of_resonances', width= -60),
                                            Item('threshold', width= -60),
                                            Item('perform_fit'),
                                     ),
                                      HGroup(Item('fit_contrast', width= -90,style='readonly'),
                                             Item('fit_line_width', width= -90,style='readonly'),
                                             Item('fit_times', width= -90,style='readonly'),
                                            ),
                                     label='fit_parameter'
                              ),
                              ),
                       title='Fit shape',
                       )

    get_set_items = PulsedFit.get_set_items + ['fit_parameters', 'fit_line_width', 'fit_contrast',  'text', 'fit_times']  
    
    
    
              
class SSTtraceFit(PulsedFit):

    #state = Enum('idle', 'count')
    #thread = Trait()
    
    hist_binning = Int(1)
    refresh_hist = Button()
    spin_state = Array()
    x_tau = Array()
    histogram = Array()
    histogram_bins = Array()

    trace_plot_data = Instance( ArrayPlotData, transient=True )
    trace_plot = Instance( Plot, transient=True )
    hist_plot_data = Instance( ArrayPlotData, transient=True )
    hist_plot = Instance( Plot, transient=True )
    
    measurement = Instance(ss.Pulsed, factory=ss.Pulsed)
    
    def _init_():
        super(SSTtraceFit, self).__init__()
        self.on_trait_change(self._update_plot_trace, 'spin_state', dispatch='ui')
        self.on_trait_change(self.update_plot_spin_state, 'spin_state', dispatch='ui')
        
        
    def _trace_plot_data_default(self):
        return ArrayPlotData(x=np.arange(len(self.trace)), y=self.trace)    
        
    def _trace_plot_default(self):
        plot = Plot(self.trace_plot_data, padding_left=50, padding_top=10, padding_right=10, padding_bottom=30)
        plot.plot(('x','y'), style='line', color='blue', name='trace')
        plot.index_axis.title = '# bin'
        plot.value_axis.title = '# counts'
        plot.tools.append(PanTool(plot))
        plot.overlays.append(ZoomTool(plot))
        return plot    
        
    def _update_plot_trace(self):
        if len(self.trace) > 10000:     # if trace to plot is too long memory errors / performance problems may occur
            trace = self.trace[0:10000]
        else:
            trace = self.trace
        self.trace_plot_data.set_data('counts', trace)
        self.trace_plot_data.set_data('time', numpy.arange(len(trace)))
        self.trace_plot.request_redraw()    

    # The following is a bit tricky. Data for plots is passed to the PulsedAnalyzer through these two attributes.
    # The first one is a standard chaco ArrayPlotData instance. It is very important to specify a proper initial instance,
    # otherwise the PulsedAnalyzer will not start. The line below specifies an initial instance through the factory and kw argument.
    line_data = Instance(ArrayPlotData,
                         factory=ArrayPlotData,
                         kw={'pulse_number':np.array((0, 1)),
                             'spin_state':np.array((0, 0)),
                             }
                         )
    # The second line is a list that is interpreted to create appropriate plots through the chaco Plot.plot() method.
    # The list elements are dictionaries that are directly passed to the Plot.plot() command as keyword arguments through the **kwagrs expansion 
    plots = [ {'data':('pulse_number', 'spin_state'), 'color':'blue', 'name':'pulsed'} ]    
        
   

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
                 title='Gated Counter', width=700, height=500,x=0, buttons=['OK'], resizable=True)    