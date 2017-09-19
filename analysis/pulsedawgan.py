"""
This module provides analysis of pulsed measurements.

The first part provides simple numeric functions.

The second part provides Trait GUIs
"""

import numpy as np
from fitting import find_edge, run_sum

def spin_state(c, dt, T, t0=0.0, t1= -1.):
    
    """
    Compute the spin state from a 2D array of count data.
    
    Parameters:
    
        c    = count data
        dt   = time step
        t0   = beginning of integration window relative to the edge
        t1   = None or beginning of integration window for normalization relative to edge
        T    = width of integration window
        
    Returns:
    
        y       = 1D array that contains the spin state
        profile = 1D array that contains the pulse profile
        edge    = position of the edge that was found from the pulse profile
        
    If t1<0, no normalization is performed. If t1>=0, each data point is divided by
    the value from the second integration window and multiplied with the mean of
    all normalization windows.
    """

    profile = c.sum(0)   # pulse plot
    edge = find_edge(profile)   # flank
    
    I = int(round(T / float(dt)))  # number of data points in the integration window
    i0 = edge + int(round(t0 / float(dt))) # position of  the integr window
    y = np.empty((c.shape[0],))
    for i, slot in enumerate(c):
        y[i] = slot[i0:i0 + I].sum()
    if t1 >= 0:
        i1 = edge + int(round(t1 / float(dt)))    
        y1 = np.empty((c.shape[0],))
        for i, slot in enumerate(c):
            y1[i] = slot[i1:i1 + I].sum()
        y = y / y1 * y1.mean()
    return y, profile, edge


#########################################
# Trait GUIs for pulsed fits
#########################################

from traits.api import HasTraits, Trait, Instance, Property, Range, Float, Int, String, Bool, Array, List, Str, Tuple, Enum, on_trait_change, cached_property, DelegatesTo, Any
from traitsui.api import View, Item, Tabbed, Group, HGroup, VGroup, VSplit, EnumEditor, TextEditor, InstanceEditor
from enable.api import ComponentEditor
from chaco.api import ArrayDataSource, LinePlot, LinearMapper, ArrayPlotData, Plot, Spectral, PlotLabel, Legend

from traitsui.file_dialog import save_file
from traitsui.menu import Action, Menu, MenuBar

import matplotlib.pyplot as plt

import threading
import time
import logging

import fitting

from tools.utility import GetSetItemsHandler, GetSetItemsMixin

import measurements.pulsed_awg as mp
import measurements.prog_gate_awg as pga
import measurements.pair_search as ps
import measurements.shallow_NV as sensing
import measurements.nmr as nmr
import measurements.singleshot as ss
#import measurements.opticalrabi as orabi
import measurements.nuclear_rabi as nr
import measurements.rabi as newrabi
from measurements.odmr import ODMR

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
    
    def save_all(self, info):
        filename = save_file(title='Save All')
        if filename is '':
            return
        else:
            info.object.save_all(filename)

menubar = MenuBar(Menu(Action(action='save', name='Save (.pyd or .pys)'),
                   Action(action='load', name='Load (.pyd or .pys)'),
                   Action(action='save_matrix_plot', name='Save Matrix Plot (.png)'),
                   Action(action='save_line_plot', name='Save Line Plot (.png)'),
                   Action(action='save_all', name='Save All'),
                   Action(action='_on_close', name='Quit'),
                   name='File'
                   ),
              )
              
class SSTAnaHandler(GetSetItemsHandler):

    """Provides handling of menu."""


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
                   Action(action='save_line_plot', name='Save Line Plot (.png)'),
                   Action(action='save_all', name='Save All'),
                   Action(action='_on_close', name='Quit'),
                   name='File'
                   ),
              )              

              
class PulsedAna(HasTraits, GetSetItemsMixin):

    """
    Base class for a pulsed analysis. Provides calculation of spin state
    and plotting.
    Derive from this to create analysis tools for pulsed measurements.
    """

    # the measurement to analyze
    measurement = Any(editor=InstanceEditor)
    
    # parameters for calculating spin state
    integration_width = Range(low=10., high=1000., value=200., desc='time window for pulse analysis [ns]', label='integr. width [ns]', mode='text', auto_set=False, enter_set=True)
    position_signal = Range(low= -100., high=1000., value=0., desc='position of signal window relative to edge [ns]', label='pos. signal [ns]', mode='text', auto_set=False, enter_set=True)
    position_normalize = Range(low= -1., high=10000., value= -1., desc='position of normalization window relative to edge [ns]. If negative, no normalization is performed', label='pos. norm. [ns]', mode='text', auto_set=False, enter_set=True)
    run_sum = Range(low=1, high=1000, value=1, desc='running sum over n samples', label='running sum', mode='text', auto_set=False, enter_set=True)

    # analysis data
    pulse = Array(value=np.array((0., 0.)))
    flank = Float(value=0.0)
    spin_state = Array(value=np.array((0., 0.)))

    # plotting
    matrix_data = Instance(ArrayPlotData)
    line_data = Instance(ArrayPlotData)
    pulse_data = Instance(ArrayPlotData)
    matrix_plot = Instance(Plot, editor=ComponentEditor())
    pulse_plot = Instance(Plot, editor=ComponentEditor())
    line_plot = Instance(Plot, editor=ComponentEditor())

    get_set_items = ['__doc__', 'measurement', 'integration_width', 'position_signal',
                     'position_normalize', 'run_sum', 'pulse', 'flank', 'spin_state']

    traits_view = View(VGroup(Item(name='measurement', style='custom', show_label=False),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     Item('run_sum'),
                                     ),
                              VSplit(Item('matrix_plot', show_label=False, width=500, height=300, resizable=True),
                                     Item('line_plot', show_label=False, width=500, height=300, resizable=True),
                                     Item('pulse_plot', show_label=False, width=500, height=300, resizable=True),
                                     ),
                              ),
                       title='Pulsed Analysis',
                       menubar=menubar,
                       buttons=[], resizable=True, handler=PulsedAnaHandler)

    def __init__(self, **kwargs):
        super(PulsedAna, self).__init__(**kwargs)
        self._create_matrix_plot()
        self._create_pulse_plot()
        self._create_line_plot()
        self.on_trait_change(self._update_matrix_index, 'measurement.time_bins,measurement.n_laser', dispatch='ui')
        self.on_trait_change(self._update_matrix_value, 'measurement.count_data', dispatch='ui')
        self.on_trait_change(self._update_pulse_index, 'measurement.time_bins', dispatch='ui')
        self.on_trait_change(self._update_pulse_value, 'pulse', dispatch='ui')
        self.on_trait_change(self._update_line_plot_value, 'spin_state', dispatch='ui')
        self.on_trait_change(self._on_flank_change, 'flank', dispatch='ui')

    @on_trait_change('measurement.count_data,integration_width,position_signal,position_normalize')
    def _analyze_count_data(self):
        m = self.measurement
        if m is None:
            return
        y, profile, flank = spin_state(c=m.count_data,
                                       dt=m.bin_width,
                                       T=self.integration_width,
                                       t0=self.position_signal,
                                       t1=self.position_normalize,
                                       )
        self.spin_state = y
        self.pulse = profile
        self.flank = m.time_bins[flank]
        self.x_axis_data = m.tau

    # plotting
    def _create_matrix_plot(self):
        matrix_data = ArrayPlotData(image=np.zeros((2, 2)))
        plot = Plot(matrix_data, width=500, height=500, resizable='hv', padding=8, padding_left=48, padding_bottom=36)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'laser pulse #'
        plot.img_plot('image',
                      xbounds=(self.measurement.time_bins[0], self.measurement.time_bins[-1]),
                      ybounds=(0, self.measurement.n_laser),
                      colormap=Spectral)[0]
        self.matrix_data = matrix_data
        self.matrix_plot = plot
    
    def _create_pulse_plot(self):
        pulse_data = ArrayPlotData(x=np.array((0., 0.1, 0.2)), y=np.array((0, 1, 2)))
        plot = Plot(pulse_data, padding=8, padding_left=64, padding_bottom=36)    
        line = plot.plot(('x', 'y'), style='line', color='blue', name='data')[0]
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'intensity'
        edge_marker = LinePlot(index=ArrayDataSource(np.array((0, 0))),
                               value=ArrayDataSource(np.array((0, 1e9))),
                               color='red',
                               index_mapper=LinearMapper(range=plot.index_range),
                               value_mapper=LinearMapper(range=plot.value_range),
                               name='marker')
        plot.add(edge_marker)
        self.pulse_data = pulse_data
        self.pulse_plot = plot
        
    def _create_line_plot(self):
        line_data = ArrayPlotData(index=np.array((0, 1)), spin_state=np.array((0, 0)),)
        plot = Plot(line_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('index', 'spin_state'), color='blue', name='spin_state')
        plot.index_axis.title = 'pulse #'
        plot.value_axis.title = 'spin state'
        self.line_data = line_data
        self.line_plot = plot

    def _update_matrix_index(self):
        self.matrix_plot.components[0].index.set_data((self.measurement.time_bins[0], self.measurement.time_bins[-1]), (0.0, float(self.measurement.n_laser)))
    def _update_matrix_value(self):
        s = self.measurement.count_data.shape
        if not s[0] * s[1] > 1000000:
            self.matrix_data.set_data('image', self.measurement.count_data)
            
    def _update_pulse_index(self):
        self.pulse_data.set_data('x', self.measurement.time_bins)        
    def _update_pulse_value(self):
        self.pulse_data.set_data('y', self.pulse)
    def _on_flank_change(self, new):
        self.pulse_plot.components[1].index.set_data(np.array((new, new)))

    def _update_line_plot_value(self):
        y = self.spin_state
        n = len(y)
        old_index = self.line_data.get_data('index')
        if old_index is not None and len(old_index) != n:
            #self.line_data.set_data('index', np.arange(n))
            self.line_data.set_data('index', self.x_axis_data)
        self.line_data.set_data('spin_state', y)
        self.line_data.set_data('index', self.x_axis_data)

    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)
    
    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename)
        
    def save_all(self, filename):
        self.save_line_plot(filename + '.png')
        self.save_matrix_plot(filename + '.png')
        self.save(filename + '.pyd')
        
   


class PulsedAnaTau(PulsedAna):

    """
    Analysis of a pulsed measurement with a 'tau' as index-data.
    """

    # overwrite measurement such that the default measurement is one that actually has a 'tau'
    measurement = Instance(mp.Pulsed, factory=mp.Rabi)

    # overwrite __init__ such that change of 'tau' causes plot update 
    def __init__(self):
        super(PulsedAnaTau, self).__init__()
        # self.on_trait_change(self._on_tau_change, 'measurement.tau', dispatch='ui')
        self.on_trait_change(self._on_tau_change, 'measurement.tau') # ToDo: fix this

    # overwrite the line_plot such that the x-axis label is time 
    def _create_line_plot(self):
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

    # overwrite this to change the window title
    traits_view = View(VGroup(Item(name='measurement', style='custom', show_label=False),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     Item('run_sum'),
                                     ),
                              VSplit(Item('matrix_plot', show_label=False, width=500, height=300, resizable=True),
                                     Item('line_plot', show_label=False, width=500, height=300, resizable=True),
                                     Item('pulse_plot', show_label=False, width=500, height=300, resizable=True),
                                     ),
                              ),
                       title='Pulsed Analysis Tau',
                       menubar=menubar,
                       buttons=[], resizable=True, handler=PulsedAnaHandler)


class PulsedAnaTauRef(PulsedAnaTau):

    """
    Analysis of a pulsed measurement with a 'tau' as index-data.
    and bright / dark reference points at the end of the sequence.
    """
    
    # overwrite the line_plot such that the bright and dark state reference lines are plotted 
    def _create_line_plot(self):
        line_data = ArrayPlotData(index=np.array((0, 1)),
                                    spin_state=np.array((0, 0)),
                                    bright=np.array((1, 1)),
                                    dark=np.array((0, 0)))
        plot = Plot(line_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('index', 'spin_state'), color='blue', name='pulsed')
        plot.plot(('index', 'bright'), color='red', name='bright')
        plot.plot(('index', 'dark'), color='black', name='dark')
        plot.index_axis.title = 'time [micro s]'
        plot.value_axis.title = 'spin state'
        self.line_data = line_data
        self.line_plot = plot
    
    # overwrite this one to provide splitting up of spin_state array and setting of bright and dark state data 
    def _update_line_plot_value(self):
        y = self.spin_state
        n_ref = self.measurement.n_ref
        n = len(y) - 2 * n_ref
        self.line_data.set_data('spin_state', y[:n])
        self.line_data.set_data('bright', np.mean(y[n:n + n_ref]) * np.ones(n))
        self.line_data.set_data('dark', np.mean(y[n + n_ref:n + 2 * n_ref]) * np.ones(n))




class FitAnaTau(PulsedAnaTau):

    """
    Base class for PulsedAna with a tau and fit.
    """

    # fit results
    fit_result = Tuple()
    label_text = Str('')

    # add fit results to the get_set_items
    get_set_items = PulsedAnaTau.get_set_items + ['fit_result', 'label_text']

    # overwrite __init__ to trigger update events
    def __init__(self):
        super(FitAnaTau, self).__init__()
        self.on_trait_change(self._update_fit, 'spin_state', dispatch='ui')
        self.on_trait_change(self._on_fit_result_change, 'fit_result', dispatch='ui')
        self.on_trait_change(self._on_label_text_change, 'label_text', dispatch='ui')
    
    def _update_fit(self):
        pass
        
    # overwrite the line_plot to include fit and text label 
    def _create_line_plot(self):
        line_data = ArrayPlotData(index=np.array((0, 1)),
                                    spin_state=np.array((0, 0)),
                                    fit=np.array((0, 0)))
        plot = Plot(line_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('index', 'spin_state'), color='blue', name='spin_state')
        plot.plot(('index', 'fit'), color='red', name='fit')
        plot.index_axis.title = 'time [micro s]'
        plot.value_axis.title = 'spin state'
        plot.overlays.insert(0, PlotLabel(text=self.label_text, hjustify='left', vjustify='bottom', position=[64, 32]))
        self.line_data = line_data
        self.line_plot = plot

    def _on_fit_result_change(self, new):
        pass
    
    def _on_label_text_change(self, new):
        self.line_plot.overlays[0].text = new    
    
    
class RabiAna(FitAnaTau):
    
    """
    Analysis of a Rabi measurement.
    """
    
    # fit results
    contrast = Tuple((np.nan, np.nan), editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str=' %.1f+-%.1f %%'))
    period = Tuple((np.nan, np.nan), editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str=' %.2f+-%.2f'))
    q = Float(np.nan, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x))
    t_pi2 = Tuple((np.nan, np.nan), editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str=' %.2f+-%.2f'))
    t_pi = Tuple((np.nan, np.nan), editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str=' %.2f+-%.2f'))
    t_3pi2 = Tuple((np.nan, np.nan), editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str=' %.2f+-%.2f'))
    
    # add fit results to the get_set_items
    get_set_items = FitAnaTau.get_set_items + ['contrast', 'period', 't_pi2', 't_pi', 't_3pi2']
        
    def _update_fit(self, y): #ToDo: test and test whether 'y' can be changed to e.g. 'y'
        # try to compute new fit from spin_state and tau. If it fails, set result to NaNs
        try:
            fit_result = fitting.fit_rabi(self.measurement.tau, y, y ** 0.5)
        except:
            fit_result = (np.NaN * np.zeros(3), np.NaN * np.zeros((3, 3)), np.NaN, np.NaN)

        p, v, q, chisqr = fit_result
        a, T, c = p
        a_var, T_var, c_var = v.diagonal()

        # compute some relevant parameters from fit result
        contrast = 200 * a / (c + a)
        contrast_delta = 200. / c ** 2 * (a_var * c ** 2 + c_var * a ** 2) ** 0.5
        T_delta = abs(T_var) ** 0.5
        pi2 = 0.25 * T
        pi = 0.5 * T
        threepi2 = 0.75 * T
        pi2_delta = 0.25 * T_delta
        pi_delta = 0.5 * T_delta
        threepi2_delta = 0.75 * T_delta
        
        # set respective attributes
        self.q = q
        self.period = T, T_delta
        self.contrast = contrast, contrast_delta
        self.t_pi2 = pi2, pi2_delta
        self.t_pi = pi, pi_delta
        self.t_3pi2 = threepi2, threepi2_delta

        # create a summary of fit result as a text string
        s = 'q: %.2e\n' % q
        s += 'contrast: %.1f+-%.1f%%\n' % (contrast, contrast_delta)
        s += 'period: %.2f+-%.2f ns\n' % (T, T_delta)
        #s += 'pi/2: %.2f+-%.2f ns\n'%(pi2, pi2_delta)
        #s += 'pi: %.2f+-%.2f ns\n'%(pi, pi_delta)
        #s += '3pi/2: %.2f+-%.2f ns\n'%(threepi2, threepi2_delta)
        
        self.fit_result = fit_result
        self.text = s
                
    def _on_fit_result_change(self, new):
        if len(new) > 0 and new[0][0] is not np.NaN:
            self.line_data.set_data('fit', fitting.Cosinus(*new[0])(self.measurement.tau))             

    traits_view = View(VGroup(VGroup(Item(name='measurement', style='custom', show_label=False),
                                     Group(VGroup(HGroup(Item('contrast', style='readonly', width= -100),
                                                          Item('period', style='readonly', width= -100),
                                                          Item('q', style='readonly', width= -100),
                                                          ),
                                                   HGroup(Item('t_pi2', style='readonly', width= -100),
                                                          Item('t_pi', style='readonly', width= -100),
                                                          Item('t_3pi2', style='readonly', width= -100),
                                                          ),
                                                   label='fit_result',
                                                   ),
                                            VGroup(HGroup(Item('integration_width'),
                                                          Item('position_signal'),
                                                          Item('position_normalize'),
                                                          Item('run_sum'),
                                                          ),
                                                   label='fit_parameter',
                                                   ),
                                            orientation='horizontal', layout='tabbed', springy=False,
                                            ),
                                     ),
                              VSplit(Item('matrix_plot', show_label=False, width=500, height=300, resizable=True),
                                     Item('line_plot', show_label=False, width=500, height=300, resizable=True),
                                     Item('pulse_plot', show_label=False, width=500, height=300, resizable=True),
                                     ),
                              ),
                       title='Rabi Analysis',
                       menubar=menubar,
                       buttons=[], resizable=True, handler=PulsedAnaHandler)
 
class SSTFit(HasTraits, GetSetItemsMixin):

    """
    Base class for a pulsed fit. Provides calculation of normalized intensity.
    Derive from this to create fits for pulsed measurements.
    """

    measurement = Instance(ss.Pulsed, factory=ss.Pulsed)
    
    pulse = Array(value=np.array((0., 0.)))
    x_tau = Array(value=np.array((0., 1.)))
    spin_state = Array(value=np.array((0., 0.)))
    spin_state_error = Array(value=np.array((0., 0.)))
    
    
    def __init__(self):
        super(SSTFit, self).__init__()
        self.on_trait_change(self.update_plot_spin_state, 'spin_state', dispatch='ui')
    
    @on_trait_change('measurement.count_data')
    def update_spin_state(self):
        if self.measurement is None:
            return
        
        n_row = self.measurement.tau.shape[0]
        dat = self.measurement.data_matrix.sum(1)
        self.spin_state = dat[0:n_row]
        self.spin_state_error = self.spin_state ** 0.5
        self.x_tau = self.measurement.tau

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
        
    def update_plot_spin_state(self):
        old_mesh = self.x_tau
        #old_mesh = self.line_data.get_data('pulse_number')
        if old_mesh is not None and len(old_mesh) != len(self.spin_state):
            self.line_data.set_data('pulse_number', np.arange(len(self.spin_state)))
        self.line_data.set_data('spin_state', self.spin_state)
        self.line_data.set_data('pulse_number', old_mesh)

    traits_view = View(title='Pulsed Fit',
                       )

    get_set_items = ['__doc__', 'spin_state', 'spin_state_error']
    
class PulsedFit(HasTraits, GetSetItemsMixin):

    """
    Base class for a pulsed fit. Provides calculation of normalized intensity.
    Derive from this to create fits for pulsed measurements.
    """

    measurement = Instance(mp.Pulsed, factory=mp.Pulsed)
    
    pulse = Array(value=np.array((0., 0.)))
    flank = Float(value=0.0)
    x_tau = Array(value=np.array((0., 1.)))
    spin_state = Array(value=np.array((0., 0.)))
    spin_state_error = Array(value=np.array((0., 0.)))
    
    integration_width = Range(low=10., high=1000., value=200., desc='time window for pulse analysis [ns]', label='integr. width [ns]', mode='text', auto_set=False, enter_set=True)
    position_signal = Range(low= -100., high=1000., value=0., desc='position of signal window relative to edge [ns]', label='pos. signal [ns]', mode='text', auto_set=False, enter_set=True)
    position_normalize = Range(low=0., high=10000., value=2200., desc='position of normalization window relative to edge [ns]', label='pos. norm. [ns]', mode='text', auto_set=False, enter_set=True)
    
    def __init__(self):
        super(PulsedFit, self).__init__()
        self.on_trait_change(self.update_plot_spin_state, 'spin_state', dispatch='ui')
    
    @on_trait_change('measurement.count_data,integration_width,position_signal,position_normalize')
    def update_spin_state(self):
        if self.measurement is None:
            return
  
        y, profile, flank = spin_state(c=self.measurement.count_data,
                                       dt=self.measurement.bin_width,
                                       T=self.integration_width,
                                       t0=self.position_signal,
                                       t1=self.position_normalize,)
        self.spin_state = y
        self.spin_state_error = y ** 0.5
        self.pulse = profile
        self.flank = self.measurement.time_bins[flank]
        self.x_tau = self.measurement.tau


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
        
    def update_plot_spin_state(self):
        old_mesh = self.x_tau
        #old_mesh = self.line_data.get_data('pulse_number')
        if old_mesh is not None and len(old_mesh) != len(self.spin_state):
            self.line_data.set_data('pulse_number', np.arange(len(self.spin_state)))
        self.line_data.set_data('spin_state', self.spin_state)
        self.line_data.set_data('pulse_number', old_mesh)

    traits_view = View(HGroup(Item('integration_width'),
                              Item('position_signal'),
                              Item('position_normalize'),
                              ),
                       title='Pulsed Fit',
                       )

    get_set_items = ['__doc__', 'integration_width', 'position_signal', 'position_normalize', 'pulse', 'flank', 'spin_state', 'spin_state_error']
    
class DoublePulsedFit(HasTraits, GetSetItemsMixin):

    """
    Base class for a pulsed fit. Provides calculation of normalized intensity.
    Derive from this to create fits for pulsed measurements.
    """

    measurement = Instance(mp.Pulsed, factory=mp.Pulsed)
    
    pulse = Array(value=np.array((0., 0.)))
    flank = Float(value=0.0)
    x_tau = Array(value=np.array((0., 1.)))
    spin_state1 = Array(value=np.array((0., 0.)))
    spin_state_error1 = Array(value=np.array((0., 0.)))
    spin_state2 = Array(value=np.array((0., 0.)))
    spin_state_error2 = Array(value=np.array((0., 0.)))
    
    normalized_counts=Array(value=np.array((0., 0.)))
    normalized_counts_error=Array(value=np.array((0., 0.)))
    free_evolution_time=Array(value=np.array((0., 1.)))
    fit_parameters = Array(value=np.array((np.nan, np.nan, np.nan, np.nan)))
    #processed_plot_data=Instance(ArrayPlotData)
    T1 = Bool(False, label='T1')
    locking = Bool(True, label='locking')
    
    integration_width = Range(low=10., high=1000., value=200., desc='time window for pulse analysis [ns]', label='integr. width [ns]', mode='text', auto_set=False, enter_set=True)
    position_signal = Range(low= -100., high=1000., value=0., desc='position of signal window relative to edge [ns]', label='pos. signal [ns]', mode='text', auto_set=False, enter_set=True)
    position_normalize = Range(low=0., high=10000., value=2200., desc='position of normalization window relative to edge [ns]', label='pos. norm. [ns]', mode='text', auto_set=False, enter_set=True)
    
    def __init__(self):
        super(DoublePulsedFit, self).__init__()
        self.on_trait_change(self.update_plot_spin_state, 'spin_state1, spin_state2, normalized_counts', dispatch='ui')
    
    @on_trait_change('measurement.count_data, integration_width, position_signal, position_normalize')
    def update_spin_state(self):
        if self.measurement is None:
            return
        y1, profile, flank = spin_state(c=self.measurement.count_data[0::2], # 0, 2, 4, 6...
                                        dt=self.measurement.bin_width,
                                        T=self.integration_width,
                                        t0=self.position_signal,
                                        t1=self.position_normalize,)
        y2, profile, flank = spin_state(c=self.measurement.count_data[1::2], # 1, 3, 5, 7...
                                        dt=self.measurement.bin_width,
                                        T=self.integration_width,
                                        t0=self.position_signal,
                                        t1=self.position_normalize,)                               
        self.spin_state1 = y1
        self.spin_state_error1 = y1 ** 0.5
        self.spin_state2 = y2
        self.spin_state_error2 = y2 ** 0.5
        self.pulse = profile
        self.flank = self.measurement.time_bins[flank]
        self.x_tau = self.measurement.tau
        
        contrast=self.measurement.rabi_contrast
        c1=self.spin_state1
        c2=self.spin_state2
        l=(c1+c2)/2.
        baseline=sum(l)/len(l)
        C0_up=baseline/(1-0.01*contrast/2)
        C0_down=C0_up*(1-0.01*contrast)
        counts=-np.abs(c2-c1)
        
        self.normalized_counts=(counts)/(C0_up-C0_down)
        self.normalized_counts_error=self.normalized_counts*(self.spin_state_error1/self.spin_state1)
        
        
        if self.T1:
            self.free_evolution_time=(self.x_tau+self.measurement.pi_1)
        elif self.locking:
            self.free_evolution_time=self.x_tau
        else:
            self.free_evolution_time=(2*self.x_tau+self.measurement.pi_1)
        
    # The following is a bit tricky. Data for plots is passed to the PulsedAnalyzer through these two attributes.
    # The first one is a standard chaco ArrayPlotData instance. It is very important to specify a proper initial instance,
    # otherwise the PulsedAnalyzer will not start. The line below specifies an initial instance through the factory and kw argument.
    line_data = Instance(ArrayPlotData,
                         factory=ArrayPlotData,
                         kw={'tau':np.array((0, 1)),
                             'spin_state1':np.array((0, 0)),
                             'spin_state2':np.array((0, 0)),
                            }
                         )
                         
    processed_plot_data = Instance(ArrayPlotData,
                                   factory=ArrayPlotData,
                                   kw={'x':np.array((0, 1)),
                                       'y':np.array((0, 0)),
                                      }
                                   )
                 
    # The second line is a list that is interpreted to create appropriate plots through the chaco Plot.plot() method.
    # The list elements are dictionaries that are directly passed to the Plot.plot() command as keyword arguments through the **kwagrs expansion 
    plots = [ {'data':('tau', 'spin_state1'), 'color':'blue','name':'pulsed'},
              {'data':('tau', 'spin_state2'), 'color':'green', 'name':'pulsed2'},
              {'data':('x', 'y'), 'color':'black','name':'processed'},
            ]
        
    def update_plot_spin_state(self):
        old_mesh = self.x_tau
        #old_mesh = self.line_data.get_data('pulse_number')
        if old_mesh is not None and len(old_mesh) != len(self.spin_state1):
            self.line_data.set_data('tau', np.arange(len(self.spin_state1)))
            self.processed_plot_data.set_data('y', np.arange(len(self.spin_state1)))
            #self.line_data2.set_data('pulse_number', np.arange(len(self.spin_state1)))
        self.line_data.set_data('spin_state1', self.spin_state1)
        self.line_data.set_data('spin_state2', self.spin_state2)
        
        self.line_data.set_data('tau', old_mesh)
        #self.line_data2.set_data('spin_state', self.spin_state2)
        #self.line_data2.set_data('pulse_number', old_mesh)
        
        self.processed_plot_data.set_data('x', self.free_evolution_time)
        self.processed_plot_data.set_data('y', self.normalized_counts)

    traits_view = View(HGroup(Item('integration_width'),
                              Item('position_signal'),
                              Item('position_normalize'),
                              Item('T1'),
                              Item('locking')
                              ),
                       title='DoublePulsed Fit',
                       )

    get_set_items = ['__doc__', 'integration_width', 'position_signal', 'position_normalize', 'pulse', 'flank', 'spin_state1', 'spin_state_error1','spin_state2', 'spin_state_error2', 'normalized_counts', 'free_evolution_time']

class OdmrFit(PulsedFit):

    """Provides fits and plots for a Odmr measurement."""

    measurement = Instance(mp.Pulsed, factory=mp.ODMR)
    
    text = Str('')
    
    number_of_resonances = Trait('auto', String('auto', auto_set=False, enter_set=True), Int(10000., desc='Number of Lorentzians used in fit', label='N', auto_set=False, enter_set=True))
    threshold = Range(low= -99, high=99., value= -50., desc='Threshold for detection of resonances [%]. The sign of the threshold specifies whether the resonances are negative or positive.', 
    label='threshold [%]', mode='text', auto_set=False, enter_set=True)
    
    fit_parameters = Array(value=np.array((np.nan, np.nan, np.nan, np.nan)))
    fit_frequencies = Array(value=np.array((np.nan,)), label='frequency [Hz]') 
    fit_line_width = Array(value=np.array((np.nan,)), label='line_width [Hz]') 
    fit_contrast = Array(value=np.array((np.nan,)), label='contrast [%]')

    def __init__(self):
        super(OdmrFit, self).__init__()
        self.on_trait_change(self.update_fit, 'spin_state', dispatch='ui')
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')
        self.on_trait_change(self.update_plot_fit, 'fit_parameters', dispatch='ui')
        
    @on_trait_change('measurement.count_data,integration_width,position_signal,position_normalize')    
    def update_spin_state(self):
        if self.measurement is None:
            return
        y, profile, flank = spin_state(c=self.measurement.count_data,
                                       dt=self.measurement.bin_width,
                                       T=self.integration_width,
                                       t0=self.position_signal,
                                       t1=self.position_normalize,)
        self.spin_state = y
        self.spin_state_error = y ** 0.5
        self.pulse = profile
        self.flank = self.measurement.time_bins[flank]
        self.x_tau = self.measurement.tau/1e6
        
    def update_fit(self):
        # try to compute new fit from spin_state and tau. If it fails, set result to NaNs
        N = self.number_of_resonances 
        if N != 'auto':
            N = int(N)
        try:
            p = fitting.fit_multiple_lorentzians(self.x_tau, self.spin_state, N, threshold=self.threshold * 0.01)
        except Exception:
            logging.getLogger().debug('ODMR fit failed.', exc_info=True)
            p = np.nan * np.empty(4)

        self.fit_parameters = p
        self.fit_frequencies = p[1::3]
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
        
        s = ''
        for i, fi in enumerate(self.fit_frequencies):
            s += 'f %i: %.6e MHz, HWHM %.3e KHz, contrast %.1f%%\n' % (i + 1, fi, self.fit_line_width[i]*1e3, contrast[i])
        self.text = s
        
    plots = [{'data':('tau', 'spin_state'), 'color':'blue', 'name':'rabi'},
             {'data':('tau', 'fit'), 'color':'red', 'name':'cos fit'} ]
    
    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'spin_state':np.array((0, 0)),
                                                                    'fit':np.array((0, 0))})
    def _line_plot_default(self):
        plot = Plot(self.fit.line_data, padding=8, padding_left=64, padding_bottom=36)
        #plot.plot(('tau','spin_state1'), style='line', color='blue')
        #plot.plot(('tau','spin_state2'), style='line', color='green')
        for item in self.fit.plots:
            plot.plot(**item)
  
        plot.index_axis.title = 'Frequency [MHz]'
        plot.value_axis.title = 'spin state'
        return plot
        
    def update_plot_spin_state(self):
        self.line_data.set_data('spin_state', self.spin_state)

    def update_plot_tau(self):
        self.line_data.set_data('tau', self.x_tau)

    def update_plot_fit(self):
        if not np.isnan(self.fit_parameters[0]):            
            self.line_data.set_data('fit', fitting.NLorentzians(*self.fit_parameters)(self.x_tau))
    
    traits_view = View(Tabbed(VGroup(HGroup(Item('number_of_resonances', width= -60),
                                            Item('threshold', width= -60),
                                     ),
                                      HGroup(Item('fit_contrast', width= -90,style='readonly'),
                                            Item('fit_line_width', width= -90,style='readonly'),
                                            Item('fit_frequencies', width= -90,style='readonly'),
                                            ),
                                     label='fit_parameter'
                              ),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     label='settings'),
                              ),
                       title='Odmr Fit',
                       )

    get_set_items = PulsedFit.get_set_items + ['fit_parameters', 'fit_frequencies', 'fit_line_width', 'fit_contrast',  'text']   
        

class RabiFit(PulsedFit):

    """Provides fits and plots for a Rabi measurement."""

    measurement = Instance(mp.Pulsed, factory=mp.Rabi)
    
    fit_result = Tuple()
    text = Str('')
    
    period = Tuple((0., 0.)) #Property( depends_on='fit_result', label='period' )
    contrast = Tuple((0., 0.)) #Property( depends_on='fit_result', label='contrast' )
    q = Float(0.) #Property( depends_on='fit_result', label='q' )
    t_pi2 = Tuple((0., 0.)) #Property( depends_on='fit_result', label='pi/2' )
    t_pi = Tuple((0., 0.)) #Property( depends_on='fit_result', label='pi' )
    t_3pi2 = Tuple((0., 0.)) #Property( depends_on='fit_result', label='3pi/2' )

    def __init__(self):
        super(RabiFit, self).__init__()
        self.on_trait_change(self.update_fit, 'spin_state', dispatch='ui')
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')
        self.on_trait_change(self.update_plot_fit, 'fit_result', dispatch='ui')

    def update_fit(self):
        # try to compute new fit from spin_state and tau. If it fails, set result to NaNs
        try:
            fit_result = fitting.fit_rabi(self.measurement.tau, self.spin_state, self.spin_state_error)
        except:
            fit_result = (np.NaN * np.zeros(3), np.NaN * np.zeros((3, 3)), np.NaN, np.NaN)

        p, v, q, chisqr = fit_result
        a, T, c = p
        a_var, T_var, c_var = v.diagonal()

        # compute some relevant parameters from fit result
        contrast = 200 * a / (c + a)
        contrast_delta = 200. / c ** 2 * (a_var * c ** 2 + c_var * a ** 2) ** 0.5
        T_delta = abs(T_var) ** 0.5
        pi2 = 0.25 * T
        pi = 0.5 * T
        threepi2 = 0.75 * T
        pi2_delta = 0.25 * T_delta
        pi_delta = 0.5 * T_delta
        threepi2_delta = 0.75 * T_delta
        
        # set respective attributes
        self.q = q
        self.period = T, T_delta
        self.contrast = contrast, contrast_delta
        self.t_pi2 = pi2, pi2_delta
        self.t_pi = pi, pi_delta
        self.t_3pi2 = threepi2, threepi2_delta

        # create a summary of fit result as a text string
        s = 'q: %.2e\n' % q
        s += 'contrast: %.1f+-%.1f%%\n' % (contrast, contrast_delta)
        s += 'period: %.2f+-%.2f ns\n' % (T, T_delta)
        #s += 'pi/2: %.2f+-%.2f ns\n'%(pi2, pi2_delta)
        #s += 'pi: %.2f+-%.2f ns\n'%(pi, pi_delta)
        #s += '3pi/2: %.2f+-%.2f ns\n'%(threepi2, threepi2_delta)
        
        # 
        self.text = s

        self.fit_result = fit_result
        
    plots = [{'data':('tau', 'spin_state'), 'color':'blue', 'name':'rabi'},
             {'data':('tau', 'fit'), 'color':'red', 'name':'cos fit'} ]
    
    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'spin_state':np.array((0, 0)),
                                                                    'fit':np.array((0, 0))})

    def update_plot_spin_state(self):
        self.line_data.set_data('spin_state', self.spin_state)

    def update_plot_tau(self):
        self.line_data.set_data('tau', self.x_tau)

    def update_plot_fit(self):
        if self.fit_result[0][0] is not np.NaN:
            self.line_data.set_data('fit', fitting.Cosinus(*self.fit_result[0])(self.measurement.tau))            
    
    traits_view = View(Tabbed(VGroup(HGroup(Item('contrast', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.1f+-%.1f%%' % x)),
                                            Item('period', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('q', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x)),
                                     ),
                                     HGroup(Item('t_pi2', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('t_pi', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('t_3pi2', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                     ),
                                     label='fit_parameter'
                              ),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     label='settings'),
                              ),
                       title='Rabi Fit',
                       )

    get_set_items = PulsedFit.get_set_items + ['fit_result', 'contrast', 'period', 't_pi2', 't_pi', 't_3pi2', 'text']

class RabiFit_phase(PulsedFit):

    """Provides fits and plots for a Rabi measurement."""

    measurement = Instance(mp.Pulsed, factory=mp.Rabi)
    
    fit_result = Tuple()
    text = Str('')
    
    period = Tuple((0., 0.)) #Property( depends_on='fit_result', label='period' )
    contrast = Tuple((0., 0.)) #Property( depends_on='fit_result', label='contrast' )
    phase = Tuple((0., 0.)) #Property( depends_on='fit_result', label='phase' )
    q = Float(0.) #Property( depends_on='fit_result', label='q' )
    t_pi2 = Tuple((0., 0.)) #Property( depends_on='fit_result', label='pi/2' )
    t_pi = Tuple((0., 0.)) #Property( depends_on='fit_result', label='pi' )
    t_3pi2 = Tuple((0., 0.)) #Property( depends_on='fit_result', label='3pi/2' )

    def __init__(self):
        super(RabiFit_phase, self).__init__()
        self.on_trait_change(self.update_fit, 'spin_state', dispatch='ui')
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')
        self.on_trait_change(self.update_plot_fit, 'fit_result', dispatch='ui')

    def update_fit(self):
        # try to compute new fit from spin_state and tau. If it fails, set result to NaNs
        try:
            fit_result = fitting.fit_rabi_phase(self.measurement.tau, self.spin_state, self.spin_state_error)
        except:
            fit_result = (np.NaN * np.zeros(4), np.NaN * np.zeros((4, 4)), np.NaN, np.NaN)

        p, v, q, chisqr = fit_result
        a, x0, T, c = p
        a_var, x0_var, T_var, c_var = v.diagonal()

        # compute some relevant parameters from fit result
        contrast = 200 * a / (c + a)
        contrast_delta = 200. / c ** 2 * (a_var * c ** 2 + c_var * a ** 2) ** 0.5
        T_delta = abs(T_var) ** 0.5
        phase = x0
        phase_delta = abs(x0_var) ** 0.5
        pi2 = 0.25 * T
        pi = 0.5 * T
        threepi2 = 0.75 * T
        pi2_delta = 0.25 * T_delta
        pi_delta = 0.5 * T_delta
        threepi2_delta = 0.75 * T_delta
        
        # set respective attributes
        self.q = q
        self.period = T, T_delta
        self.contrast = contrast, contrast_delta
        self.phase = phase, phase_delta
        self.t_pi2 = pi2, pi2_delta
        self.t_pi = pi, pi_delta
        self.t_3pi2 = threepi2, threepi2_delta

        # create a summary of fit result as a text string
        s = 'q: %.2e\n' % q
        s += 'contrast: %.1f+-%.1f%%\n' % (contrast, contrast_delta)
        s += 'period: %.2f+-%.2f ns\n' % (T, T_delta)
        s += 'phase: %.2f+-%.2f ns\n' % (phase, phase_delta)
        #s += 'pi/2: %.2f+-%.2f ns\n'%(pi2, pi2_delta)
        #s += 'pi: %.2f+-%.2f ns\n'%(pi, pi_delta)
        #s += '3pi/2: %.2f+-%.2f ns\n'%(threepi2, threepi2_delta)
        
        # 
        self.text = s

        self.fit_result = fit_result
        
    plots = [{'data':('tau', 'spin_state'), 'color':'blue', 'name':'rabi'},
             {'data':('tau', 'fit'), 'color':'red', 'name':'cos fit'} ]
    
    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'spin_state':np.array((0, 0)),
                                                                    'fit':np.array((0, 0))})

    def update_plot_spin_state(self):
        self.line_data.set_data('spin_state', self.spin_state)

    def update_plot_tau(self):
        self.line_data.set_data('tau', self.x_tau)

    def update_plot_fit(self):
        if self.fit_result[0][0] is not np.NaN:
            self.line_data.set_data('fit', fitting.Cosinus_phase(*self.fit_result[0])(self.measurement.tau))            
    
    traits_view = View(Tabbed(VGroup(HGroup(Item('contrast', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.1f+-%.1f%%' % x)),
                                            Item('period', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('phase', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('q', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x)),
                                     ),
                                     HGroup(Item('t_pi2', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('t_pi', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('t_3pi2', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                     ),
                                     label='fit_parameter'
                              ),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     label='settings'),
                              ),
                       title='Rabi Fit_phase',
                       )

    get_set_items = PulsedFit.get_set_items + ['fit_result', 'contrast', 'period', 'phase','t_pi2', 't_pi', 't_3pi2', 'text']
 
class Damped_Cosine(PulsedFit):

    """Provides fits and plots for a Rabi measurement."""

    measurement = Instance(mp.Pulsed, factory=mp.Rabi)
    
    fit_result = Tuple()
    text = Str('')
    
    period = Tuple((0., 0.)) #Property( depends_on='fit_result', label='period' )
    contrast = Tuple((0., 0.)) #Property( depends_on='fit_result', label='contrast' )
    phase = Tuple((0., 0.)) #Property( depends_on='fit_result', label='phase' )
    tau = Tuple((0., 0.)) #Property( depends_on='fit_result', label='phase' )
    q = Float(0.) #Property( depends_on='fit_result', label='q' )
    t_pi2 = Tuple((0., 0.)) #Property( depends_on='fit_result', label='pi/2' )
    t_pi = Tuple((0., 0.)) #Property( depends_on='fit_result', label='pi' )
    t_3pi2 = Tuple((0., 0.)) #Property( depends_on='fit_result', label='3pi/2' )

    def __init__(self):
        super(Damped_Cosine, self).__init__()
        self.on_trait_change(self.update_fit, 'spin_state', dispatch='ui')
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')
        self.on_trait_change(self.update_plot_fit, 'fit_result', dispatch='ui')

    def update_fit(self):
        # try to compute new fit from spin_state and tau. If it fails, set result to NaNs
        try:
            fit_result = fitting.fit_decaying_cosine(self.measurement.tau, self.spin_state, self.spin_state_error)
            #print fit_result
        except:
            fit_result = (np.NaN * np.zeros(5), np.NaN * np.zeros((5, 5)), np.NaN, np.NaN)

        p, v, q, chisqr = fit_result
        a, x0, T, c, tau = p
        a_var, x0_var, T_var, c_var, tau_var = v.diagonal()

        # compute some relevant parameters from fit result
        contrast = 200 * a / (c + a)
        contrast_delta = 200. / c ** 2 * (a_var * c ** 2 + c_var * a ** 2) ** 0.5
        T_delta = abs(T_var) ** 0.5
        phase = x0
        phase_delta = abs(x0_var) ** 0.5
        tau_delta=abs(tau_var) ** 0.5
        pi2 = 0.25 * T
        pi = 0.5 * T
        threepi2 = 0.75 * T
        pi2_delta = 0.25 * T_delta
        pi_delta = 0.5 * T_delta
        threepi2_delta = 0.75 * T_delta
        
        # set respective attributes
        self.q = q
        self.period = T, T_delta
        self.contrast = contrast, contrast_delta
        self.phase = phase, phase_delta
        self.t_pi2 = pi2, pi2_delta
        self.t_pi = pi, pi_delta
        self.t_3pi2 = threepi2, threepi2_delta
        self.tau = tau, tau_delta 

        # create a summary of fit result as a text string
        s = 'q: %.2e\n' % q
        s += 'contrast: %.1f+-%.1f%%\n' % (contrast, contrast_delta)
        s += 'period: %.2f+-%.2f ns\n' % (T, T_delta)
        s += 'phase: %.2f+-%.2f ns\n' % (phase, phase_delta)
        #s += 'pi/2: %.2f+-%.2f ns\n'%(pi2, pi2_delta)
        #s += 'pi: %.2f+-%.2f ns\n'%(pi, pi_delta)
        #s += '3pi/2: %.2f+-%.2f ns\n'%(threepi2, threepi2_delta)
        
        # 
        self.text = s

        self.fit_result = fit_result
        
    plots = [{'data':('tau', 'spin_state'), 'color':'blue', 'name':'rabi'},
             {'data':('tau', 'fit'), 'color':'red', 'name':'cos fit'} ]
    
    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'spin_state':np.array((0, 0)),
                                                                    'fit':np.array((0, 0))})

    def update_plot_spin_state(self):
        self.line_data.set_data('spin_state', self.spin_state)

    def update_plot_tau(self):
        self.line_data.set_data('tau', self.x_tau)

    def update_plot_fit(self):
        if self.fit_result[0][0] is not np.NaN:
            self.line_data.set_data('fit', fitting.Damping_cosinus_phase(*self.fit_result[0])(self.measurement.tau))            
    
    traits_view = View(Tabbed(VGroup(HGroup(Item('contrast', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.1f+-%.1f%%' % x)),
                                            Item('period', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('phase', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('q', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x)),
                                     ),
                                     HGroup(Item('t_pi2', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('t_pi', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('t_3pi2', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('tau', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                     ),
                                     label='fit_parameter'
                              ),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     label='settings'),
                              ),
                       title='Damped Cosine',
                       )

    get_set_items = PulsedFit.get_set_items + ['fit_result', 'contrast', 'period', 'phase','t_pi2', 't_pi', 't_3pi2', 'text', 'tau'] 
    
class DoubleExponentialDecayFit(DoublePulsedFit):
    """Provides fits and plots for Hahn echo and T1 measurements."""

    #fit = Instance(DoublePulsedFit, factory=DoublePulsedFit)
    
    fit_result = Tuple()
    text = Str('')
     
    decay_time = Float(value=0.0, desc='Decay time [us]', label='Decay time [us]', editor=TextEditor(auto_set=False, enter_set=False, evaluate=float, format_str='%.1f'))
    standart_deviation = Float(value=0.0, desc='Standart deviation [us]', label='StDev [us]', editor=TextEditor(auto_set=False, enter_set=False, evaluate=float, format_str='%.1f'))
    percent = Float(value=0.0, desc='percent', label='%', editor=TextEditor(auto_set=False, enter_set=False, evaluate=float, format_str='%.1f'))

    def __init__(self):
        super(DoubleExponentialDecayFit, self).__init__()
        #self.on_trait_change(self.update_plot_tau, 'free_evolution_time', dispatch='ui')
        self.on_trait_change(self.update_fit, 'normalized_counts', dispatch='ui')
        self.on_trait_change(self.update_plot_fit, 'fit_result', dispatch='ui')
     
    #def update_plot_tau(self):
        #self.line_data.set_data('tau', self.x_tau) 
        
    def update_fit(self):
        # try to compute new fit from spin_state and tau. If it fails, set result to NaNs
        try:
            fit_result = fitting.fit_exp_decay(self.free_evolution_time, self.normalized_counts, self.normalized_counts_error)
        except:
            fit_result = (np.NaN * np.zeros(4))

        self.fit_result = fit_result
        
        
        # create a summary of fit result as a text string
        
        #s = 'tau: %.2f+-%.2f us\n' % (self.fit_result[0][0]*1e-3, T_delta)
        s = 'tau: %.2f us\n' % (self.fit_result[0][0]*1e-3)
                   
        # 
        self.text = s

    processed_plot_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'x':np.array((0, 1)),
                                                                             'y':np.array((0, 0)),
                                                                             'fit':np.array((0,0)),
                                                                             })    
        
    plots = [{'data':('tau', 'spin_state1'), 'color':'blue', 'name':'signal1','label':'Reference'},
             {'data':('tau', 'spin_state2'), 'color':'green', 'name':'signal2','label':'Signal'},
             {'data':('x', 'y'), 'color':'black','name':'processed'},
             {'data':('x', 'fit'), 'color':'purple', 'name':'fitting','label':'Fit'}]
           
    #legends = {'loc':'upper right'}       
            
    # def update_plot_spin_state(self):
        # old_mesh = self.x_tau
        # self.line_data.set_data('tau', old_mesh)
        # self.line_data.set_data('spin_state1', self.spin_state1)
        # self.line_data.set_data('spin_state2', self.spin_state2)
     
    def calc_st_dev(self):
        #y=ax, a=1/decay_time 
         
        x=self.free_evolution_time
        u=self.normalized_counts
        
        y=np.log(-u)
        
        n=len(x)
        
        Sa=np.sqrt((np.sum(x**2)*np.sum(y**2)-np.sum(x*y)**2)/((n-1)*np.sum(x**2)**2))
        
        a=np.sum(x*y)/np.sum(x**2)
        
        decay_time=-1.0/a
        error=Sa/a**2
        #self.fit_result[0][0]=decay_time
        return decay_time, error

    def update_plot_fit(self):
       
        if self.fit_result[0][0] is not np.NaN:
            #self.processed_plot_data.set_data('fit', fitting.ExponentialZero(*self.fit_result[0])(self.free_evolution_time)) 
            #self.decay_time=self.fit_result[0][0]*1e-3
            
            self.standart_deviation=self.calc_st_dev()[1]*1e-3
            self.decay_time=self.calc_st_dev()[0]*1e-3
            self.processed_plot_data.set_data('fit', fitting.ExponentialZero(*self.fit_result[0])(self.free_evolution_time))   
            self.percent=(self.standart_deviation/self.decay_time)*100
    
    traits_view = View(Tabbed(VGroup(HGroup(Item('decay_time', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.2f')),
                                            Item('standart_deviation', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.2f')),
                                            Item('percent', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.1f'))
                                                                               ),
                                     label='fit_parameter'
                              ),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     Item('T1'),
                                     label='settings'),
                              ),
                       title='Exponential decay',
                       )
    #get_set_items = DoublePulsedFitPulsedFit.get_set_items + ['fit_result', 'contrast', 'period', 'phase','t_pi2', 't_pi', 't_3pi2', 'text']
    get_set_items = DoublePulsedFit.get_set_items + ['fit_result']   
    
class DoubleGaussianDecayFit(DoublePulsedFit):
    """Provides fits and plots for Hahn echo and T1 measurements."""
    
    fit_result = Tuple()
    text = Str('')
     
    decay_time = Float(value=0.0, desc='Decay time [us]', label='Decay time [us]', editor=TextEditor(auto_set=False, enter_set=False, evaluate=float, format_str='%.1f'))
    standart_deviation = Float(value=0.0, desc='Standart deviation [us]', label='StDev [us]', editor=TextEditor(auto_set=False, enter_set=False, evaluate=float, format_str='%.1f'))
    percent = Float(value=0.0, desc='percent', label='%', editor=TextEditor(auto_set=False, enter_set=False, evaluate=float, format_str='%.1f'))

    def __init__(self):
        super(DoubleGaussianDecayFit, self).__init__()
        self.on_trait_change(self.update_fit, 'normalized_counts', dispatch='ui')
        self.on_trait_change(self.update_plot_fit, 'fit_result', dispatch='ui')
     
        
    def update_fit(self):
        # try to compute new fit from spin_state and tau. If it fails, set result to NaNs
        try:
            fit_result = fitting.fit_gaussian_decay(self.free_evolution_time, self.normalized_counts, self.normalized_counts_error)
        except:
            fit_result = (np.NaN * np.zeros(4))

        self.fit_result = fit_result
        
        # create a summary of fit result as a text string
        #s = 'tau: %.2f+-%.2f us\n' % (self.fit_result[0][0]*1e-3, T_delta)
        s = 'tau: %.2f us\n' % (self.fit_result[0][1]*1e-3)         
        # 
        self.text = s

    processed_plot_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'x':np.array((0, 1)),
                                                                             'y':np.array((0, 0)),
                                                                             'fit':np.array((0,0)),
                                                                             })    
        
    plots = [{'data':('tau', 'spin_state1'), 'color':'blue', 'name':'signal1','label':'Reference'},
             {'data':('tau', 'spin_state2'), 'color':'green', 'name':'signal2','label':'Signal'},
             {'data':('x', 'y'), 'color':'black','name':'processed'},
             {'data':('x', 'fit'), 'color':'purple', 'name':'fitting','label':'Fit'}]
             
    def calculate_errors(self):
        #y-c=ln(a)-p*x/decay_time
         
        x=self.free_evolution_time
        u=self.normalized_counts
        
        y=np.log(-u+self.fit_result[0][3])
        
        n=len(x)
        
        x_mean = np.sum(x)/n
        y_mean = np.sum(y)/n
        
        Sx=np.sqrt(np.sum((x-x_mean)**2)/(n-1))
        Sy=np.sqrt(np.sum((y-y_mean)**2)/(n-1))
        
        r=np.sum((x-x_mean)*(y-y_mean))/((n-1)*Sx*Sy)
        
        a=r*Sy/Sx       
        Sa=np.sqrt(((1-r**2)*Sy**2)/((n-2)*Sx**2))
        
        decay_time=-self.fit_result[0][2]/a
        error=Sa*self.fit_result[0][2]/a**2
        
        #self.fit_result[0][0]=decay_time
        
        return decay_time, error
               
    def update_plot_fit(self):
       
        if self.fit_result[0][0] is not np.NaN:
            #self.processed_plot_data.set_data('fit', fitting.ExponentialPowerZero(*self.fit_result[0])(self.free_evolution_time)) 
            #self.decay_time=self.fit_result[0][1]*1e-3
            self.standart_deviation=self.calculate_errors()[1]*1e-3
            self.decay_time=self.calculate_errors()[0]*1e-3
            self.processed_plot_data.set_data('fit', fitting.ExponentialPowerZero(*self.fit_result[0])(self.free_evolution_time))   
            self.percent=(self.standart_deviation/self.decay_time)*100
    
    traits_view = View(Tabbed(VGroup(HGroup(Item('decay_time', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.2f')),
                                    Item('standart_deviation', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.2f')),
                                    Item('percent', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.1f')),
                                                                               ),
                                     label='fit_parameter'
                              ),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     label='settings'),
                              ),
                       title='Gaussian decay',
                       )
    get_set_items = DoublePulsedFit.get_set_items + ['fit_result'] 
    
class DoubleRabiFit_phase(DoublePulsedFit):

    """Provides fits and plots for a Rabi measurement."""

    measurement = Instance(mp.Pulsed, factory=mp.Double_RF_sweep)
    
    fit_result = Tuple()
    fit_result_2 = Tuple()
    text = Str('')
    
    period = Tuple((0., 0.)) #Property( depends_on='fit_result', label='period' )
    contrast = Tuple((0., 0.)) #Property( depends_on='fit_result', label='contrast' )
    phase = Tuple((0., 0.)) #Property( depends_on='fit_result', label='phase' )
    q = Float(0.) #Property( depends_on='fit_result', label='q' )
    t_pi2 = Tuple((0., 0.)) #Property( depends_on='fit_result', label='pi/2' )
    t_pi = Tuple((0., 0.)) #Property( depends_on='fit_result', label='pi' )
    t_3pi2 = Tuple((0., 0.)) #Property( depends_on='fit_result', label='3pi/2' )
    
    period_2 = Tuple((0., 0.)) #Property( depends_on='fit_result', label='period' )
    contrast_2 = Tuple((0., 0.)) #Property( depends_on='fit_result', label='contrast' )
    phase_2 = Tuple((0., 0.)) #Property( depends_on='fit_result', label='phase' )
    q_2 = Float(0.) #Property( depends_on='fit_result', label='q' )
    t_pi2_2 = Tuple((0., 0.)) #Property( depends_on='fit_result', label='pi/2' )
    t_pi_2 = Tuple((0., 0.)) #Property( depends_on='fit_result', label='pi' )
    t_3pi2_2 = Tuple((0., 0.)) #Property( depends_on='fit_result', label='3pi/2' )

    def __init__(self):
        super(DoubleRabiFit_phase, self).__init__()
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')
        #self.on_trait_change(self.update_plot_spin_state, 'spin_state1,spin_state2', dispatch='ui')
        self.on_trait_change(self.update_fit, 'spin_state1,spin_state2', dispatch='ui')
        self.on_trait_change(self.update_plot_fit, 'fit_result, fit_result_2', dispatch='ui')
        #self.on_trait_change(self.update_fit, 'spin_state2', dispatch='ui')
        #self.on_trait_change(self.update_plot_fit2, 'fit_result_2', dispatch='ui')
     
    def update_plot_tau(self):
        self.line_data.set_data('tau', self.x_tau) 
        
    def update_fit(self):
        # try to compute new fit from spin_state and tau. If it fails, set result to NaNs
        try:
            fit_result = fitting.fit_rabi_phase(self.measurement.tau, self.spin_state1, self.spin_state_error1)
            fit_result_2 = fitting.fit_rabi_phase(self.measurement.tau, self.spin_state2, self.spin_state_error2)
        except:
            fit_result = (np.NaN * np.zeros(4), np.NaN * np.zeros((4, 4)), np.NaN, np.NaN)
            fit_result_2 = (np.NaN * np.zeros(4), np.NaN * np.zeros((4, 4)), np.NaN, np.NaN)

        p, v, q, chisqr = fit_result
        a, x0, T, c = p
        a_var, x0_var, T_var, c_var = v.diagonal()
        
        p_2, v_2, q_2, chisqr_2 = fit_result_2
        a_2, x0_2, T_2, c_2 = p_2
        a_var_2, x0_var_2, T_var_2, c_var_2 = v_2.diagonal()

        # compute some relevant parameters from fit result
        contrast = 200 * a / (c + a)
        contrast_delta = 200. / c ** 2 * (a_var * c ** 2 + c_var * a ** 2) ** 0.5
        T_delta = abs(T_var) ** 0.5
        phase = x0
        phase_delta = abs(x0_var) ** 0.5
        pi2 = 0.25 * T
        pi = 0.5 * T
        threepi2 = 0.75 * T
        pi2_delta = 0.25 * T_delta
        pi_delta = 0.5 * T_delta
        threepi2_delta = 0.75 * T_delta
        
        # set respective attributes
        self.q = q
        self.period = T, T_delta
        self.contrast = contrast, contrast_delta
        self.phase = phase, phase_delta
        self.t_pi2 = pi2, pi2_delta
        self.t_pi = pi, pi_delta
        self.t_3pi2 = threepi2, threepi2_delta
        
        contrast_2 = 200 * a_2 / (c_2 + a_2)
        contrast_delta_2 = 200. / c_2 ** 2 * (a_var_2 * c_2 ** 2 + c_var_2 * a_2 ** 2) ** 0.5
        T_delta_2 = abs(T_var_2) ** 0.5
        phase_2 = x0_2
        phase_delta_2 = abs(x0_var_2) ** 0.5
        pi2_2 = 0.25 * T_2
        pi_2 = 0.5 * T_2
        threepi2_2 = 0.75 * T_2
        pi2_delta_2 = 0.25 * T_delta_2
        pi_delta_2 = 0.5 * T_delta_2
        threepi2_delta_2 = 0.75 * T_delta_2
        
        # set respective attributes
        self.q_2 = q_2
        self.period_2 = T_2, T_delta_2
        self.contrast_2 = contrast_2, contrast_delta_2
        self.phase_2 = phase_2, phase_delta_2
        self.t_pi2_2 = pi2_2, pi2_delta_2
        self.t_pi_2 = pi_2, pi_delta_2
        self.t_3pi2_2 = threepi2_2, threepi2_delta_2

        # create a summary of fit result as a text string
        s = 'q: %.2e\n' % q
        s += 'contrast: %.1f+-%.1f%%\n' % (contrast, contrast_delta)
        s += 'period: %.2f+-%.2f ns\n' % (T, T_delta)
        s += 'phase: %.2f+-%.2f ns\n' % (phase, phase_delta)
        s += 'q_2: %.2e\n' % q_2
        s += 'contrast_2: %.1f+-%.1f%%\n' % (contrast_2, contrast_delta_2)
        s += 'period_2: %.2f+-%.2f ns\n' % (T_2, T_delta_2)
        s += 'phase_2: %.2f+-%.2f ns\n' % (phase_2, phase_delta_2)
        #s += 'pi/2: %.2f+-%.2f ns\n'%(pi2, pi2_delta)
        #s += 'pi: %.2f+-%.2f ns\n'%(pi, pi_delta)
        #s += '3pi/2: %.2f+-%.2f ns\n'%(threepi2, threepi2_delta)
        
        # 
        self.text = s

        self.fit_result = fit_result
        self.fit_result_2 = fit_result_2

    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                        'spin_state1':np.array((0, 0)),
                                                                        'spin_state2':np.array((0,0)),
                                                                        'fit1':np.array((0, 0)),
                                                                        'fit2':np.array((0, 0)),
                                                                        })    
        
    plots = [{'data':('tau', 'spin_state1'), 'color':'blue', 'name':'signal1','label':'Reference'},
             {'data':('tau', 'spin_state2'), 'color':'green', 'name':'signal2','label':'Signal'},
             {'data':('tau', 'fit1'), 'color':'red', 'name':'fit1','label':'Reference_Fit'},
             {'data':('tau', 'fit2'), 'color':'purple', 'name':'fit2','label':'Signal_Fit'}
            ]
    #legends = {'loc':'upper right'}       
            
    def update_plot_spin_state(self):
        old_mesh = self.x_tau
        self.line_data.set_data('tau', old_mesh)
        #self.line_data.set_data('tau', old_mesh)
        #self.line_data.set_data('spin_state1', self.spin_state1)
        #self.line_data.set_data('spin_state2', self.spin_state2)
        self.line_data.set_data('spin_state1', self.spin_state1)
        self.line_data.set_data('spin_state2', self.spin_state2)
        

    def update_plot_fit(self):
        #self.line_data_fit.set_data('tau', self.x_tau)
        if self.fit_result[0][0] is not np.NaN:
            self.line_data.set_data('fit1', fitting.Cosinus_phase(*self.fit_result[0])(self.measurement.tau))
        if self.fit_result_2[0][0] is not np.NaN:
            self.line_data.set_data('fit2', fitting.Cosinus_phase(*self.fit_result_2[0])(self.measurement.tau))     
    #def update_plot_fit2(self):
       # if self.fit_result_2[0][0] is not np.NaN:
            #self.line_data.set_data('fit2', fitting.Cosinus_phase(*self.fit_result_2[0])(self.measurement.tau))  
    
    traits_view = View(Tabbed(VGroup(HGroup(Item('contrast', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.1f+-%.1f%%' % x)),
                                            Item('period', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('phase', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('q', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x)),
                                     ),
                                     HGroup(Item('contrast_2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.1f+-%.1f%%' % x)),
                                            Item('period_2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('phase_2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('q_2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x)),
                                            ),
                                     HGroup(Item('t_pi2', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('t_pi', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x))),
                                        
                                     HGroup(Item('t_pi2_2', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('t_pi_2', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),       
                                            
                                     ),
                                     label='fit_parameter'
                              ),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     label='settings'),
                              ),
                       title='Double Rabi Fit_phase',
                       )
    #get_set_items = DoublePulsedFitPulsedFit.get_set_items + ['fit_result', 'contrast', 'period', 'phase','t_pi2', 't_pi', 't_3pi2', 'text']
    get_set_items = DoublePulsedFit.get_set_items + ['fit_result','fit_result_2', 'contrast','period', 'phase','t_pi2', 't_pi', 't_3pi2', 'contrast_2', 'period_2', 'phase_2', 't_pi_2', 't_3pi2_2','text', 't_pi2_2']
   
class OdmrFitXY8(DoublePulsedFit):

    """Provides fits and plots for xy8 measurement."""

    #fit = Instance(DoublePulsedFit, factory=DoublePulsedFit)
    text = Str('')
     
    number_of_resonances = Trait('auto', String('auto', auto_set=False, enter_set=True), Int(10000., desc='Number of Lorentzians used in fit', label='N', auto_set=False, enter_set=True))
    threshold = Range(low= -99, high=99., value= -50., desc='Threshold for detection of resonances [%]. The sign of the threshold specifies whether the resonances are negative or positive.', 
    label='threshold [%]', mode='text', auto_set=False, enter_set=True)
    
    perform_fit = Bool(False, label='perform fit')
    
    fit_frequencies = Array(value=np.array((np.nan,)), label='frequency [MHz]') 
    fit_times = Array(value=np.array((np.nan,)), label='time [ns]') 
    fit_line_width = Array(value=np.array((np.nan,)), label='line_width [ns]') 
    fit_contrast = Array(value=np.array((np.nan,)), label='contrast [%]')

    def __init__(self):
        super(OdmrFitXY8, self).__init__()
        self.on_trait_change(self._update_processed_plot_data_fit, 'fit_parameters', dispatch='ui')
        self.on_trait_change(self._update_fit, 'normalized_counts, number_of_resonances, threshold, perform_fit', dispatch='ui')
        #self.on_trait_change(self._update_plot_tau, 'fit.free_evolution_time', dispatch='ui')
    
    def _update_fit(self):
  
        if self.perform_fit:
              
            N = self.number_of_resonances 
            
            if N != 'auto':
               N = int(N)
               
            try:
                
                self.fit_parameters = fitting.fit_multiple_lorentzians(self.free_evolution_time, self.normalized_counts, N, threshold=self.threshold * 0.01)
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
        self.fit_frequencies=1e+3/(2*self.fit_times)
        
       
    processed_plot_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'x':np.array((0, 1)),
                                                                             'y':np.array((0, 0)),
                                                                             'fit':np.array((0,0)),
                                                                             })    
        
    plots = [{'data':('tau', 'spin_state1'), 'color':'blue', 'name':'signal1','label':'Reference'},
             {'data':('tau', 'spin_state2'), 'color':'green', 'name':'signal2','label':'Signal'},
             {'data':('x', 'y'), 'color':'black','name':'processed'},
             {'data':('x', 'fit'), 'color':'purple', 'name':'fitting','label':'Fit'}]
             
    # def _update_plot_tau(self):
        # self.processed_plot_data.set_data('y', self.fit.free_evolution_time)

    def _update_processed_plot_data_fit(self):
       
        if not np.isnan(self.fit_parameters[0]):              
            self.processed_plot_data.set_data('fit', fitting.NLorentzians(*self.fit_parameters)(self.free_evolution_time))
            p = self.fit_parameters
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
            
            for i, fi in enumerate(self.fit_times):
                s += 'f %i: %.6e ns, HWHM %.3e ns, contrast %.1f%%\n, freq %.3e MHz' % (i + 1, fi, self.fit_line_width[i], contrast[i], self.fit_frequencies[i])
            self.text = s
                 
    traits_view = View(Tabbed(VGroup(HGroup(Item('number_of_resonances', width= -60),
                                            Item('threshold', width= -60),
                                            Item('perform_fit'),
                                     ),
                                      HGroup(Item('fit_contrast', width= -90,style='readonly'),
                                             Item('fit_line_width', width= -90,style='readonly'),
                                             Item('fit_frequencies', width= -90,style='readonly'),
                                             Item('fit_times', width= -90,style='readonly'),
                                            ),
                                     label='fit_parameter'
                              ),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     label='settings'),
                              ),
                       title='Noise spectrum Fit',
                       )

    get_set_items = DoublePulsedFit.get_set_items + ['fit_parameters', 'fit_frequencies', 'fit_line_width', 'fit_contrast',  'text', 'fit_times']      
#########################################
# Pulsed Analyzer Tool
#########################################
class Cdecay_ebath_fit(PulsedFit):

    """Provides fits and plots for a XY8 measurement."""

    measurement = Instance(mp.Pulsed, factory=mp.Rabi)
    
    fit_result = Tuple()
    text = Str('')
    
    period = Tuple((0., 0.)) #Property( depends_on='fit_result', label='period' )
    contrast = Tuple((0., 0.)) #Property( depends_on='fit_result', label='contrast' )
    phase = Tuple((0., 0.)) #Property( depends_on='fit_result', label='phase' )
    q = Float(0.) #Property( depends_on='fit_result', label='q' )
    t_pi2 = Tuple((0., 0.)) #Property( depends_on='fit_result', label='pi/2' )
    t_pi = Tuple((0., 0.)) #Property( depends_on='fit_result', label='pi' )
    t_3pi2 = Tuple((0., 0.)) #Property( depends_on='fit_result', label='3pi/2' )

    def __init__(self):
        super(Cdecay_ebath_fit, self).__init__()
        self.on_trait_change(self.update_fit, 'spin_state', dispatch='ui')
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')
        self.on_trait_change(self.update_plot_fit, 'fit_result', dispatch='ui')

    def update_fit(self):
        # try to compute new fit from spin_state and tau. If it fails, set result to NaNs
        try:
            fit_result = fitting.fit_coherence_decay_ebath(self.measurement.tau, self.spin_state, self.spin_state_error)
        except:
            fit_result = (np.NaN * np.zeros(3), np.NaN * np.zeros((3, 3)), np.NaN, np.NaN)

        p, v, q, chisqr = fit_result
        y0, yp, Td = p
        y0_var, yp_var, Td_var = v.diagonal()

        # compute some relevant parameters from fit result
        contrast = 100 * abs(yp) / (y0)
        contrast_delta = 100 * abs(yp_var) ** 0.5/ (y0)
        Td_delta = abs(Td_var) ** 0.5
        
        # set respective attributes
        self.q = q
        self.period = Td, Td_delta
        self.contrast = contrast, contrast_delta

        # create a summary of fit result as a text string
        s = 'q: %.2e\n' % q
        s += 'contrast: %.1f+-%.1f%%\n' % (contrast, contrast_delta)
        s += 'period: %.2f+-%.2f ns\n' % (Td, T_delta)
        #s += 'pi/2: %.2f+-%.2f ns\n'%(pi2, pi2_delta)
        #s += 'pi: %.2f+-%.2f ns\n'%(pi, pi_delta)
        #s += '3pi/2: %.2f+-%.2f ns\n'%(threepi2, threepi2_delta)
        
        # 
        self.text = s

        self.fit_result = fit_result
        
    plots = [{'data':('tau', 'spin_state'), 'color':'blue', 'name':'rabi'},
             {'data':('tau', 'fit'), 'color':'red', 'name':'cos fit'} ]
    
    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'spin_state':np.array((0, 0)),
                                                                    'fit':np.array((0, 0))})

    def update_plot_spin_state(self):
        self.line_data.set_data('spin_state', self.spin_state)

    def update_plot_tau(self):
        self.line_data.set_data('tau', self.x_tau)

    def update_plot_fit(self):
        if self.fit_result[0][0] is not np.NaN:
            self.line_data.set_data('fit', fitting.Cosinus_phase(*self.fit_result[0])(self.measurement.tau))            
    
    traits_view = View(Tabbed(VGroup(HGroup(Item('contrast', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.1f+-%.1f%%' % x)),
                                            Item('period', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            #Item('phase', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('q', style='readonly', width= -200, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x)),
                                     ),
                                     label='fit_parameter'
                              ),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     label='settings'),
                              ),
                       title='Cdecay_ebath_fit',
                       )

    get_set_items = PulsedFit.get_set_items + ['fit_result', 'contrast', 'period' 'text']
    
class PulsedAnalyzerHandler(GetSetItemsHandler):

    """Provides handling of menu."""

    # save plots
    
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
            
    def save_all(self, info):
        filename = save_file(title='Save All')
        if filename is '':
            return
        else:
            info.object.save_all(filename)
            
    #def save_all(self, info):
    #    filename = save_file(title='Save All')
    #    if filename is '':
    #        return
    #    else:
    #        info.object.save_all(filename)
    #        info.object.save_line_plot(filename)
            
    # new measurements

    def new_t1_measurement(self, info):
        info.object.measurement = mp.T1()
        #if pulsed.measurement.state=='run':
        #    logging.getLogger().exception(str(RuntimeError('Measurement running. Stop it and try again!')))
        #    raise RuntimeError('Measurement running. Stop it and try again!')
        
    def new_odmr_measurement(self, info):
        """info.object.measurement = mp.Rabi()"""
        info.object.measurement = mp.ODMR()    
    def new_rabi_measurement(self, info):
        """info.object.measurement = mp.Rabi()"""
        info.object.measurement = mp.Rabi()
    def new_fid_measurement(self, info):
        info.object.measurement = mp.Sing_FID()    
    def new_dqfid_measurement(self, info):
        info.object.measurement = mp.DQFID()      
    def new_dqhahn_measurement(self, info):
        """info.object.measurement = mp.Rabi()"""
        info.object.measurement = mp.DQHahn()         
    def new_nmr_measurement(self, info):
        info.object.measurement = mp.Double_RF_sweep()
    def new_nmr_measurement_1(self, info):
        info.object.measurement = mp.N14_Ms0_RF_sweep()
    def new_nmr_srabi_measurement(self, info):    
        info.object.measurement = mp.Single_RF_Rabi()
    def new_nmr_rabi_measurement(self, info):
        info.object.measurement = mp.Double_RF_Rabi()
    def new_pump_test_measurement(self, info):
        info.object.measurement = mp.DNP_pump_test()        
    def new_nmr_t1_measurement(self, info):
        info.object.measurement = mp.DNP_N_T1()    
    def new_dnp_rabi_measurement(self, info):
        info.object.measurement = mp.DNP_Rabi()   
    def new_dnp_nucle_rabi_measurement(self, info):
        info.object.measurement = mp.DNP_Nucle_Rabi() 
    def new_nucle_rabi_dnp_measurement(self, info):
        info.object.measurement = mp.Nucle_Rabi_with_DNP()     
    def new_opt_test_measurement(self, info):
        info.object.measurement = mp.OptPulseTest()    
    def new_opt_test_measurement2(self, info):
        info.object.measurement = mp.OptPulseCom()        
    def new_qsim_test_measurement(self, info):
        info.object.measurement = mp.QS_test() 
    def new_qsim_measurement(self, info):
        info.object.measurement = mp.QSim()  
    def new_qsim_ref_measurement(self, info):
        info.object.measurement = mp.QSim_Ref() 
    def new_qsim_fft_ground_measurement(self, info):
        info.object.measurement = mp.QSim_FFT_ground()  
    def new_qsim_fft_excited_measurement(self, info):
        info.object.measurement = mp.QSim_FFT_excited()
    def new_qsim_fft_phase_ground_measurement(self, info):
        info.object.measurement = mp.QSim_FFT_phase_ground()      
        
      
    '''     
    def new_hahn_measurement(self, info):
        info.object.measurement = mp.Hahn()
    #def new_opticalrabi_measurement(self, info):
    #    info.object.measurement = orabi.OpticalRabi()
    def new_fid3pi2_measurement(self, info):
        info.object.measurement = mp.FID3pi2()
    def new_hahn3pi2_measurement(self, info):
        info.object.measurement = mp.Hahn3pi2()
    def new_hartmannhahn_measurement(self, info):
        info.object.measurement = mp.HH()
    def new_hartmannhahn3pi2_measurement(self, info):
        info.object.measurement = mp.HH3pi2()
    def new_deer_measurement(self, info):
        info.object.measurement = mp.DEER()
    def new_deer3pi2_measurement(self, info):
        info.object.measurement = mp.DEER3pi2()
    def new_dqtfid3pi2_measurement(self, info):
        info.object.measurement = mp.DQTFID3pi2()
    def new_dqtrabi_measurement(self, info):
        info.object.measurement = mp.DQTRabi()
    def new_t1pi_measurement(self, info):
        info.object.measurement = mp.T1pi()
    def new_HardDQTFID_measurement(self, info):        
        info.object.measurement = mp.HardDQTFID()
    def new_HardDQTFIDTauMod_measurement(self, info):
        info.object.measurement = mp.HardDQTFIDTauMod()    
    def new_CPMG3pi2_measurement(self, info):
        info.object.measurement = mp.CPMG3pi2()    
    def new_CPMGxy_measurement(self, info):
        info.object.measurement = mp.CPMGxy()    
    def new_CPMG_measurement(self, info):
        info.object.measurement = mp.CPMG()    
    def new_PulseCal_measurement(self, info):
        info.object.measurement = mp.PulseCal()
    def new_Pi2Pi_measurement(self, info):
        info.object.measurement = mp.Pi2PiCal()
    '''
    
    # new fits 
    def new_pulsed_fit(self, info):
        info.object.fit = PulsedFit()
    def new_odmr_fit(self, info):
        info.object.fit = OdmrFit()
    def new_rabi_fit(self, info):
        info.object.fit = RabiFit()
    def new_rabi_fit_phase(self, info):
        info.object.fit = RabiFit_phase()
    def new_Cdecay_ebath_fit(self, info):
        info.object.fit = Cdecay_ebath_fit()        
    def new_doublepulsed_fit(self, info):
        info.object.fit = DoublePulsedFit()
    def new_doublerabi_fit_phase(self, info):
        info.object.fit = DoubleRabiFit_phase()    
    def new_odmr_fit_XY8(self, info):
        info.object.fit = OdmrFitXY8()
    def new_exponential_fit(self, info):
        info.object.fit = DoubleExponentialDecayFit()
    def new_gaussian_fit(self, info):
        info.object.fit = DoubleGaussianDecayFit()
    def new_decaying_cosine_fit(self, info):
        info.object.fit = Damped_Cosine()
        
    '''    
    def new_frequency_fit(self, info):
        info.object.fit = FrequencyFit()
    def new_frequency3pi2_fit(self, info):
        info.object.fit = Frequency3pi2Fit()
    def new_pulsed_fit_tau(self, info):
        info.object.fit = PulsedFitTau()
    def new_fit_tau_ref(self, info):
        info.object.fit = PulsedFitTauRef()

    def new_hahn_fit(self, info):
        info.object.fit = HahnFit()
    def new_double_fit(self, info):
        info.object.fit = DoubleFit()
    def new_double_fit_tau(self, info):
        info.object.fit = DoubleFitTau()
    def new_double_fit_tau_ref(self, info):
        info.object.fit = DoubleFitTauRef()
    def new_hahn3pi2_fit(self, info):
        info.object.fit = Hahn3pi2Fit()
    def new_t1pi_fit(self, info):
        info.object.fit = T1piFit()
    def new_pulse_extraction_fit(self, info):
        info.object.fit = PulseExtraction()
    def new_pulse_cal_fit(self, info):
        info.object.fit = PulseCalFit()
    '''    
   
class PulsedAnalyzer(HasTraits, GetSetItemsMixin):
    
    """Fits a pulsed measurement with a pulsed fit.
    
    Example of usage:

        a = PulsedAnalyzer()
        a.measurement = Rabi
        a.fit = RabiFit()
        a.edit_traits()
        
    """
    
    measurement = Instance(mp.Pulsed, factory=mp.Pulsed)
    fit = Instance(PulsedFit, factory=PulsedFit)

    matrix_plot_data = Instance(ArrayPlotData)
    pulse_plot_data = Instance(ArrayPlotData)
    matrix_plot = Instance(Plot)
    pulse_plot = Instance(Plot)
    line_plot = Instance(Plot)
    # line_data is provided by fit class

    def __init__(self):
        super(PulsedAnalyzer, self).__init__()
        self.on_trait_change(self.refresh_matrix_axis, 'measurement.time_bins,measurement.n_laser', dispatch='ui')
        self.on_trait_change(self.refresh_matrix, 'measurement.count_data', dispatch='ui')
        self.on_trait_change(self.refresh_pulse, 'fit.pulse', dispatch='ui')
        self.on_trait_change(self.refresh_time_bins, 'measurement.time_bins', dispatch='ui')
        self.on_trait_change(self.refresh_flank, 'fit.flank', dispatch='ui')

    def _measurement_changed(self, new):
        self.fit = PulsedFit()
    
    def _matrix_plot_data_default(self):
        return ArrayPlotData(image=np.zeros((self.measurement.n_laser, self.measurement.n_bins)))
    def _pulse_plot_data_default(self):
        return ArrayPlotData(x=self.measurement.time_bins, y=self.fit.pulse)
    
    def _matrix_plot_default(self):
        plot = Plot(self.matrix_plot_data, width=500, height=500, resizable='hv', padding=8, padding_left=48, padding_bottom=36)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'laser pulse'
        plot.img_plot('image',
                      xbounds=(self.measurement.time_bins[0], self.measurement.time_bins[-1]),
                      ybounds=(0, self.measurement.n_laser),
                      colormap=Spectral)[0]
        return plot
    def _pulse_plot_default(self):
        plot = Plot(self.pulse_plot_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('x', 'y'), style='line', color='blue', name='data')
        edge_marker = LinePlot(index=ArrayDataSource(np.array((0, 0))),
                               value=ArrayDataSource(np.array((0, 1e9))),
                               color='red',
                               index_mapper=LinearMapper(range=plot.index_range),
                               value_mapper=LinearMapper(range=plot.value_range),
                               name='marker')
        plot.add(edge_marker)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'intensity'
        return plot
    def _line_plot_default(self):
        plot = Plot(self.fit.line_data, padding=8, padding_left=64, padding_bottom=36)
        for item in self.fit.plots:
            plot.plot(**item)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'spin state'
        return plot

    def refresh_matrix_axis(self):
        self.matrix_plot.components[0].index.set_data((self.measurement.time_bins[0], self.measurement.time_bins[-1]), (0.0, float(self.measurement.n_laser)))
        
    def refresh_matrix(self):
        s = self.measurement.count_data.shape
        if not s[0] * s[1] > 1000000:
            self.matrix_plot_data.set_data('image', self.measurement.count_data)

    def refresh_pulse(self):
        self.pulse_plot_data.set_data('y', self.fit.pulse)
    
    def refresh_time_bins(self):
        self.pulse_plot_data.set_data('x', self.measurement.time_bins)
    
    def refresh_flank(self):
        self.pulse_plot.components[1].index.set_data(np.array((self.fit.flank, self.fit.flank)))

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
    
    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)
    
    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename)
        
    def save_all(self, filename):
        self.save_line_plot(filename + '_LinePlot.png')
        self.save_matrix_plot(filename + '_MatrixPlot.png')
        self.save(filename + '.pyd')
        
    # Menu of the PulsedAnalyzer, only include tested measurements and Fits please!
    
    traits_view = View(HGroup(VGroup(Item(name='measurement', style='custom', show_label=False),
                                     Item('line_plot', name='normalized intensity', editor=ComponentEditor(), show_label=False, width=-400, height=-400, resizable=True), 
                                     ),
                              VGroup(Item(name='fit', style='custom', show_label=False),
                                     Tabbed(Item('matrix_plot', editor=ComponentEditor(), show_label=False, width=-300, height=-400, resizable=True),
                                            Item('pulse_plot', name='pulse profile', editor=ComponentEditor(), show_label=False, width=-300, height=-400, resizable=True),
                                            ),
                                     ),
                              ),
                       menubar=MenuBar(Menu(Action(action='save', name='Save (.pyd or .pys)'),
                                              Action(action='load', name='Load (.pyd or .pys)'),
                                              Action(action='save_matrix_plot', name='Save Matrix Plot (.png)'),
                                              Action(action='save_line_plot', name='Save Line Plot (.png)'),
                                              Action(action='save_all', name='Save All'),
                                              Action(action='_on_close', name='Quit'),
                                              name='File'),
                                         Menu(Action(action='new_odmr_measurement', name='ODMR'),
                                              Action(action='new_rabi_measurement', name='Rabi'),
                                              Action(action='new_fid_measurement', name='FID'),
                                              Action(action='new_dqfid_measurement', name='DQFID'),
                                              Action(action='new_dqhahn_measurement', name='DQHahn'),
                                              Action(action='new_nmr_measurement_1', name='Ms0_NMR'),
                                              Action(action='new_nmr_t1_measurement', name='N_t1'),
                                              Action(action='new_nmr_srabi_measurement', name='RF_Rabi'),
                                              Action(action='new_dnp_rabi_measurement', name='DNP_Rabi'),
                                              Action(action='new_dnp_nucle_rabi_measurement', name='DNP_Nucle_Rabi'),
                                              Action(action='new_nucle_rabi_dnp_measurement', name='Nucle_Rabi_afte_DNP'),
                                              Action(action='new_opt_test_measurement', name='OptPulseTest'),
                                              Action(action='new_opt_test_measurement2', name='OptPulseCom'),
                                              Action(action='new_qsim_test_measurement', name='QS_test'),
                                              Action(action='new_qsim_measurement', name='QSim'),
                                              Action(action='new_qsim_fft_ground_measurement', name='QSim_fft_ground'),
                                              Action(action='new_qsim_fft_phase_ground_measurement', name='QuantumSim FFT_phase'),
                                              Action(action='new_qsim_fft_excited_measurement', name='QSim_fft_excited'),
                                              name='Measurement'),
                                         Menu(Action(action='new_pulsed_fit', name='Pulsed'),
                                              Action(action='new_odmr_fit', name='Odmr Fit'),
                                              Action(action='new_rabi_fit', name='Rabi'),
                                              Action(action='new_rabi_fit_phase', name='Rabi_phase'),
                                              Action(action='new_decaying_cosine_fit', name = 'Damped Cosine'),
                                              name='Fit'),
                                              """
                                              Action(action='new_doublepulsed_fit', name='DoublePulsed'),
                                              Action(action='new_frequency_fit', name='Frequency'),
                                              Action(action='new_frequency3pi2_fit', name='Frequency3pi2'),
                                              Action(action='new_pulsed_fit_tau', name='Tau'),
                                              Action(action='new_fit_tau_ref', name='Tau Ref'),
                                              Action(action='new_rabi_fit', name='Rabi'),
                                              Action(action='new_hahn_fit', name='Hahn'),
                                              Action(action='new_double_fit', name='Double'),
                                              Action(action='new_double_fit_tau', name='Double Tau'),
                                              Action(action='new_double_fit_tau_ref', name='Double Tau Ref'),
                                              Action(action='new_t1pi_fit', name='T1pi'),
                                              Action(action='new_hahn3pi2_fit', name='Hahn3pi2'),
                                              Action(action='new_pulse_extraction_fit', name='PulseExtraction'),
                                              Action(action='new_pulse_cal_fit', name='PulseCal'),
                                              
                                              Action(action='new_t1_measurement', name='T1'),
                                              Action(action='new_t1pi_measurement', name='T1pi'),
                                              Action(action='new_nmr_measurement', name='NMR'),
                                              #Action(action='new_opticalrabi_measurement', name='OpticalRabi'),
                                              Action(action='new_nr_measurement', name='NuclearRabi'),
                                              Action(action='new_PulseCal_measurement', name='PulseCal'),
                                              Action(action='new_hahn_measurement', name='Hahn'),
                                              Action(action='new_fid3pi2_measurement', name='FID3pi2'),
                                              Action(action='new_hahn3pi2_measurement', name='Hahn3pi2'),
                                              Action(action='new_deer_measurement', name='DEER'),
                                              Action(action='new_deer3pi2_measurement', name='DEER3pi2'),
                                              Action(action='new_hartmannhahn_measurement', name='HartmannHahn'),
                                              Action(action='new_hartmannhahn3pi2_measurement', name='HartmannHahn3pi2'),
                                              Action(action='new_dqtfid3pi2_measurement', name='DQTFID3pi2'),
                                              Action(action='new_dqtrabi_measurement', name='DQTRabi'),
                                              Action(action='new_HardDQTFID_measurement', name='HardDQTFID'),
                                              Action(action='new_HardDQTFIDTauMod_measurement', name='HardDQTFIDTauMod'),
                                              Action(action='new_CPMG_measurement', name='CPMG'),
                                              Action(action='new_CPMG3pi2_measurement', name='CPMG3pi2'),
                                              Action(action='new_CPMGxy_measurement', name='CPMGxy'),
                                              Action(action='new_Pi2Pi_measurement', name='Pi2Pi'),
                                             
                                              """
                                              
                                 ),
                       title='PulsedAWGAnalyzer', buttons=[], resizable=True, handler=PulsedAnalyzerHandler)

    get_set_items = ['measurement', 'fit']
    get_set_order = ['measurement', 'fit']
    
class DoublePulsedAnalyzer(HasTraits, GetSetItemsMixin):
    
    """Fits a pulsed measurement with a pulsed fit.
    
    Example of usage:

        a = PulsedAnalyzer()
        a.measurement = Rabi
        a.fit = RabiFit()
        a.edit_traits()
        
    """
    
    measurement = Instance(mp.Pulsed, factory=mp.Pulsed)
    fit = Instance(DoublePulsedFit, factory=DoublePulsedFit)

    matrix_plot_data = Instance(ArrayPlotData)
    pulse_plot_data = Instance(ArrayPlotData)
    matrix_plot = Instance(Plot)
    pulse_plot = Instance(Plot)
    line_plot = Instance(Plot)
    # line_data is provided by fit class

    def __init__(self):
        super(DoublePulsedAnalyzer, self).__init__()
        self.on_trait_change(self.refresh_matrix_axis, 'measurement.time_bins,measurement.n_laser', dispatch='ui')
        self.on_trait_change(self.refresh_matrix, 'measurement.count_data', dispatch='ui')
        self.on_trait_change(self.refresh_pulse, 'fit.pulse', dispatch='ui')
        self.on_trait_change(self.refresh_time_bins, 'measurement.time_bins', dispatch='ui')
        self.on_trait_change(self.refresh_flank, 'fit.flank', dispatch='ui')

    def _measurement_changed(self, new):
        self.fit = DoublePulsedFit()
    
    def _matrix_plot_data_default(self):
        return ArrayPlotData(image=np.zeros((self.measurement.n_laser, self.measurement.n_bins)))
    def _pulse_plot_data_default(self):
        return ArrayPlotData(x=self.measurement.time_bins, y=self.fit.pulse)
    
    def _matrix_plot_default(self):
        plot = Plot(self.matrix_plot_data, width=500, height=500, resizable='hv', padding=8, padding_left=48, padding_bottom=36)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'laser pulse'
        plot.img_plot('image',
                      xbounds=(self.measurement.time_bins[0], self.measurement.time_bins[-1]),
                      ybounds=(0, self.measurement.n_laser),
                      colormap=Spectral)[0]
        return plot
    def _pulse_plot_default(self):
        plot = Plot(self.pulse_plot_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('x', 'y'), style='line', color='blue', name='data')
        edge_marker = LinePlot(index=ArrayDataSource(np.array((0, 0))),
                               value=ArrayDataSource(np.array((0, 1e9))),
                               color='red',
                               index_mapper=LinearMapper(range=plot.index_range),
                               value_mapper=LinearMapper(range=plot.value_range),
                               name='marker')
        plot.add(edge_marker)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'intensity'
        return plot
    def _line_plot_default(self):
        plot = Plot(self.fit.line_data, padding=8, padding_left=64, padding_bottom=36)
        #plot.plot(('tau','spin_state1'), style='line', color='blue')
        #plot.plot(('tau','spin_state2'), style='line', color='green')
        for item in self.fit.plots:
            plot.plot(**item)
  
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'spin state'
        return plot

    def refresh_matrix_axis(self):
        self.matrix_plot.components[0].index.set_data((self.measurement.time_bins[0], self.measurement.time_bins[-1]), (0.0, float(self.measurement.n_laser)))
        
    def refresh_matrix(self):
        s = self.measurement.count_data.shape
        if not s[0] * s[1] > 1000000:
            self.matrix_plot_data.set_data('image', self.measurement.count_data)

    def refresh_pulse(self):
        self.pulse_plot_data.set_data('y', self.fit.pulse)
    
    def refresh_time_bins(self):
        self.pulse_plot_data.set_data('x', self.measurement.time_bins)
    
    def refresh_flank(self):
        self.pulse_plot.components[1].index.set_data(np.array((self.fit.flank, self.fit.flank)))

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
        #if hasattr(self.fit,'legends'):
            #plot.legend(**self.fit.legends)
        # if the fit has an attr 'text' attached to it, print it in the lower left corner of the plot
        if hasattr(old, 'text'):
            label = plot.overlays[0]
            old.sync_trait('text', label, 'text', mutual=False, remove=True)
            plot.overlays = []
        if hasattr(new, 'text'):
            label = PlotLabel(text=self.fit.text, hjustify='left', vjustify='bottom', position=[64, 32])
            new.sync_trait('text', label, 'text', mutual=False)
            plot.overlays = [label]
    
    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)
    
    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename)
        
    def save_all(self, filename):
        self.save_line_plot(filename + '.png')
        self.save_matrix_plot(filename + '.png')
        self.save(filename + '.pyd')
        
    # Menu of the PulsedAnalyzer, only include tested measurements and Fits please!
    
    traits_view = View(HGroup(VGroup(Item(name='measurement', style='custom', show_label=False),
                                     Item('line_plot', name='normalized intensity', editor=ComponentEditor(), show_label=False, width=300, height=300, resizable=True), 
                                     ),
                              VGroup(Item(name='fit', style='custom', show_label=False),
                                     Tabbed(Item('matrix_plot', editor=ComponentEditor(), show_label=False, width=300, height=300, resizable=True),
                                            Item('pulse_plot', name='pulse profile', editor=ComponentEditor(), show_label=False, width=300, height=300, resizable=True),
                                            ),
                                     ),
                              ),
                       menubar=MenuBar(Menu(Action(action='save', name='Save (.pyd or .pys)'),
                                              Action(action='load', name='Load (.pyd or .pys)'),
                                              Action(action='save_matrix_plot', name='Save Matrix Plot (.png)'),
                                              Action(action='save_line_plot', name='Save Line Plot (.png)'),
                                              Action(action='save_all', name='Save All'),
                                              Action(action='_on_close', name='Quit'),
                                              name='File'),
                                       Menu(Action(action='new_rabi_measurement', name='Rabi'),
                                              Action(action='new_fid_measurement', name='FID'),
                                              Action(action='new_dqfid_measurement', name='DQFID'),
                                              Action(action='new_dqhahn_measurement', name='DQHahn'),
                                              Action(action='new_pump_test_measurement', name='DNP_pump_test'),
                                              Action(action='new_nmr_measurement', name='Double_RF_sweep'),
                                              #Action(action='new_opticalrabi_measurement', name='OpticalRabi'),
                                              Action(action='new_nmr_rabi_measurement', name='Double_RF_Rabi'),
                                              Action(action='new_qsim_ref_measurement', name='Quantum Sim with Ref'),
                                             
                                              name='Measurement'),
                                            
                                            #  Action(action='new_t1_measurement', name='T1'),
                                             # Action(action='new_t1pi_measurement', name='T1pi'),
                                            #  Action(action='new_PulseCal_measurement', name='PulseCal'),
                                            #  Action(action='new_hahn_measurement', name='Hahn'),
                                             # Action(action='new_fid3pi2_measurement', name='FID3pi2'),
                                             # Action(action='new_hahn3pi2_measurement', name='Hahn3pi2'),
                                             # Action(action='new_deer_measurement', name='DEER'),
                                            #  Action(action='new_deer3pi2_measurement', name='DEER3pi2'),
                                             # Action(action='new_hartmannhahn_measurement', name='HartmannHahn'),
                                             # Action(action='new_hartmannhahn3pi2_measurement', name='HartmannHahn3pi2'),
                                             # Action(action='new_dqtfid3pi2_measurement', name='DQTFID3pi2'),
                                             # Action(action='new_dqtrabi_measurement', name='DQTRabi'),
                                              #Action(action='new_HardDQTFID_measurement', name='HardDQTFID'),
                                              #Action(action='new_HardDQTFIDTauMod_measurement', name='HardDQTFIDTauMod'),
                                             # Action(action='new_CPMG_measurement', name='CPMG'),
                                              #Action(action='new_CPMG3pi2_measurement', name='CPMG3pi2'),
                                              #Action(action='new_CPMGxy_measurement', name='CPMGxy'),
                                              #Action(action='new_Pi2Pi_measurement', name='Pi2Pi'),
                                             
                                         Menu(
                                              Action(action='new_doublepulsed_fit', name='DoublePulsed'),
                                              Action(action='new_doublerabi_fit_phase', name='Doublerabi_fit_phase'),
                                              Action(action='new_odmr_fit_XY8', name='fit lorentzians'),
                                              Action(action='new_exponential_fit', name = 'Exponential fit'),
                                              Action(action='new_gaussian_fit', name = 'Gausian fit'),
                                              Action(action='new_decaying_cosine_fit', name='Damped Cosine'),
                                              name='Fit'
                                              ),
                                              '''
                                              Action(action='new_pulsed_fit',name='Pulsed'),
                                              Action(action='new_frequency_fit', name='Frequency'),
                                              Action(action='new_frequency3pi2_fit', name='Frequency3pi2'),
                                              Action(action='new_pulsed_fit_tau', name='Tau'),
                                              Action(action='new_fit_tau_ref', name='Tau Ref'),
                                              Action(action='new_rabi_fit', name='Rabi'),
                                              Action(action='new_hahn_fit', name='Hahn'),
                                              Action(action='new_double_fit', name='Double'),
                                              Action(action='new_double_fit_tau', name='Double Tau'),
                                              Action(action='new_double_fit_tau_ref', name='Double Tau Ref'),
                                              Action(action='new_t1pi_fit', name='T1pi'),
                                              Action(action='new_hahn3pi2_fit', name='Hahn3pi2'),
                                              Action(action='new_pulse_extraction_fit', name='PulseExtraction'),
                                              Action(action='new_pulse_cal_fit', name='PulseCal'),
                                              '''
                                              
                                 ),
                       title='DoublePulsedAnalyzer', buttons=[], resizable=True, handler=PulsedAnalyzerHandler)

    get_set_items = ['measurement', 'fit']
    get_set_order = ['measurement', 'fit']    

class ProgPulsedAnalyzerHandler(GetSetItemsHandler):

    """Provides handling of menu."""

    # save plots
    
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
            
    def save_all(self, info):
        filename = save_file(title='Save All')
        if filename is '':
            return
        else:
            info.object.save_all(filename)
            
    #def save_all(self, info):
    #    filename = save_file(title='Save All')
    #    if filename is '':
    #        return
    #    else:
    #        info.object.save_all(filename)
    #        info.object.save_line_plot(filename)
            
    # new measurements
    
    def new_Tomo_measurement(self, info):        
        info.object.measurement = pga.EspinTomo_diag() 
        
    def new_Tomo2_measurement(self, info):        
        info.object.measurement = pga.EspinTomo_NLoC()     
        
    def new_Tomo1_measurement(self, info):        
        info.object.measurement = pga.EspinTomo_LoC2() 
    def new_Tomo3_measurement(self, info):        
        info.object.measurement = pga.EspinTomo_LoC1()     

    def new_DEER_measurement(self, info):
        """info.object.measurement = mp.Rabi()"""
        info.object.measurement = pga.DEERpair()     

        
                
    # new fits 

    def new_pulsed_fit(self, info):
        info.object.fit = PulsedFit()
    def new_rabi_fit(self, info):
        info.object.fit = RabiFit()
    def new_rabi_fit_phase(self, info):
        info.object.fit = RabiFit_phase()    
    def new_doublepulsed_fit(self, info):
        info.object.fit = DoublePulsedFit()
    def new_doublerabi_fit_phase(self, info):
        info.object.fit = DoubleRabiFit_phase()    
    def new_odmr_fit_XY8(self, info):
        info.object.fit = OdmrFitXY8()
    def new_exponential_fit(self, info):
        info.object.fit = DoubleExponentialDecayFit()
    def new_gaussian_fit(self, info):
        info.object.fit = DoubleGaussianDecayFit()
        
class ProgPulsedAnalyzer(HasTraits, GetSetItemsMixin):
    
    """Fits a pulsed measurement with a pulsed fit.
    
    Example of usage:

        a = PulsedAnalyzer()
        a.measurement = Rabi
        a.fit = RabiFit()
        a.edit_traits()
        
    """
    
    measurement = Instance(pga.Pulsed, factory=pga.Pulsed)
    fit = Instance(PulsedFit, factory=PulsedFit)

    matrix_plot_data = Instance(ArrayPlotData)
    pulse_plot_data = Instance(ArrayPlotData)
    matrix_plot = Instance(Plot)
    pulse_plot = Instance(Plot)
    line_plot = Instance(Plot)
    # line_data is provided by fit class

    def __init__(self):
        super(ProgPulsedAnalyzer, self).__init__()
        self.on_trait_change(self.refresh_matrix_axis, 'measurement.time_bins,measurement.n_laser', dispatch='ui')
        self.on_trait_change(self.refresh_matrix, 'measurement.count_data', dispatch='ui')
        self.on_trait_change(self.refresh_pulse, 'fit.pulse', dispatch='ui')
        self.on_trait_change(self.refresh_time_bins, 'measurement.time_bins', dispatch='ui')
        self.on_trait_change(self.refresh_flank, 'fit.flank', dispatch='ui')

    def _measurement_changed(self, new):
        self.fit = PulsedFit()
    
    def _matrix_plot_data_default(self):
        return ArrayPlotData(image=np.zeros((self.measurement.n_laser, self.measurement.n_bins)))
    def _pulse_plot_data_default(self):
        return ArrayPlotData(x=self.measurement.time_bins, y=self.fit.pulse)
    
    def _matrix_plot_default(self):
        plot = Plot(self.matrix_plot_data, width=500, height=500, resizable='hv', padding=8, padding_left=48, padding_bottom=36)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'laser pulse'
        plot.img_plot('image',
                      xbounds=(self.measurement.time_bins[0], self.measurement.time_bins[-1]),
                      ybounds=(0, self.measurement.n_laser),
                      colormap=Spectral)[0]
        return plot
    def _pulse_plot_default(self):
        plot = Plot(self.pulse_plot_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('x', 'y'), style='line', color='blue', name='data')
        edge_marker = LinePlot(index=ArrayDataSource(np.array((0, 0))),
                               value=ArrayDataSource(np.array((0, 1e9))),
                               color='red',
                               index_mapper=LinearMapper(range=plot.index_range),
                               value_mapper=LinearMapper(range=plot.value_range),
                               name='marker')
        plot.add(edge_marker)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'intensity'
        return plot
    def _line_plot_default(self):
        plot = Plot(self.fit.line_data, padding=8, padding_left=64, padding_bottom=36)
        for item in self.fit.plots:
            plot.plot(**item)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'spin state'
        return plot

    def refresh_matrix_axis(self):
        self.matrix_plot.components[0].index.set_data((self.measurement.time_bins[0], self.measurement.time_bins[-1]), (0.0, float(self.measurement.n_laser)))
        
    def refresh_matrix(self):
        s = self.measurement.count_data.shape
        if not s[0] * s[1] > 1000000:
            self.matrix_plot_data.set_data('image', self.measurement.count_data)

    def refresh_pulse(self):
        self.pulse_plot_data.set_data('y', self.fit.pulse)
    
    def refresh_time_bins(self):
        self.pulse_plot_data.set_data('x', self.measurement.time_bins)
    
    def refresh_flank(self):
        self.pulse_plot.components[1].index.set_data(np.array((self.fit.flank, self.fit.flank)))

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
    
    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)
    
    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename)
    # Menu of the PulsedAnalyzer, only include tested measurements and Fits please!
    
    traits_view = View(HGroup(VGroup(Item(name='measurement', style='custom', show_label=False),
                                     Item('line_plot', name='normalized intensity', editor=ComponentEditor(), show_label=False, width=-400, height=-400, resizable=True), 
                                     ),
                              VGroup(Item(name='fit', style='custom', show_label=False),
                                     Tabbed(Item('matrix_plot', editor=ComponentEditor(), show_label=False, width=-300, height=-400, resizable=True),
                                            Item('pulse_plot', name='pulse profile', editor=ComponentEditor(), show_label=False, width=-300, height=-400, resizable=True),
                                            ),
                                     ),
                              ),
                              menubar=MenuBar(Menu(Action(action='save', name='Save (.pyd or .pys)'),
                                              Action(action='load', name='Load (.pyd or .pys)'),
                                              Action(action='save_matrix_plot', name='Save Matrix Plot (.png)'),
                                              Action(action='save_line_plot', name='Save Line Plot (.png)'),
                                              Action(action='save_all', name='Save All'),
                                              Action(action='_on_close', name='Quit'),
                                              name='File'),
                                              Menu(Action(action='new_Tomo_measurement', name='EspinTomo_diag'),
                                                    Action(action='new_Tomo1_measurement', name='EspinTomo_LoC2'),
                                                    Action(action='new_Tomo3_measurement', name='EspinTomo_LoC1'),
                                                    Action(action='new_Tomo2_measurement', name='EspinTomo_NLoC'),
                                                   Action(action='new_DEER_measurement', name='DEER_pair'), 
                                              name='Measurement'),
                                              Menu(Action(action='new_pulsed_fit', name='Pulsed'),
                                              Action(action='new_rabi_fit', name='Rabi'),
                                              Action(action='new_rabi_fit_phase', name='Rabi_phase'),
                                              name='Fit'),
                                                                              
                                 ),
                       title='ProgPulsedAWGAnalyzer', buttons=[], resizable=True, handler=ProgPulsedAnalyzerHandler)

    get_set_items = ['measurement', 'fit']
    get_set_order = ['measurement', 'fit']    
    
class ProgPulsedAnalyzerHandler(GetSetItemsHandler):

    """Provides handling of menu."""

    # save plots
    
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
            
    def save_all(self, info):
        filename = save_file(title='Save All')
        if filename is '':
            return
        else:
            info.object.save_all(filename)
            
    #def save_all(self, info):
    #    filename = save_file(title='Save All')
    #    if filename is '':
    #        return
    #    else:
    #        info.object.save_all(filename)
    #        info.object.save_line_plot(filename)
            
    # new measurements
    
    def new_Tomo_measurement(self, info):        
        info.object.measurement = pga.EspinTomo_diag() 
        
    def new_Tomo2_measurement(self, info):        
        info.object.measurement = pga.EspinTomo_NLoC()     
        
    def new_Tomo1_measurement(self, info):        
        info.object.measurement = pga.EspinTomo_LoC2() 
    def new_Tomo3_measurement(self, info):        
        info.object.measurement = pga.EspinTomo_LoC1()     

    def new_DEER_measurement(self, info):
        """info.object.measurement = mp.Rabi()"""
        info.object.measurement = pga.DEERpair()     

        
                
    # new fits 

    def new_pulsed_fit(self, info):
        info.object.fit = PulsedFit()
    def new_rabi_fit(self, info):
        info.object.fit = RabiFit()
    def new_rabi_fit_phase(self, info):
        info.object.fit = RabiFit_phase()    
    def new_doublepulsed_fit(self, info):
        info.object.fit = DoublePulsedFit()
    def new_doublerabi_fit_phase(self, info):
        info.object.fit = DoubleRabiFit_phase()   
    def new_odmr_fit_XY8(self, info):
        info.object.fit = OdmrFitXY8()
    def new_exponential_fit(self, info):
        info.object.fit = DoubleExponentialDecayFit()
    def new_gaussian_fit(self, info):
        info.object.fit = DoubleGaussianDecayFit()
        
class PairPulsedAnalyzerHandler(GetSetItemsHandler):

    """Provides handling of menu."""

    # save plots
    
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
            
            
    def save_all(self, info):
        filename = save_file(title='Save All')
        if filename is '':
            return
        else:
            info.object.save_all(filename)
            
    #def save_all(self, info):
    #    filename = save_file(title='Save All')
    #    if filename is '':
    #        return
    #    else:
    #        info.object.save_all(filename)
    #        info.object.save_line_plot(filename)
            
    # new measurements
    
    def new_rabi_measurement(self, info):        
        info.object.measurement = ps.Rabi() 
        
    def new_FID_measurement(self, info):        
        info.object.measurement = ps.FID()    
        
    def new_Hahn_measurement(self, info):        
        info.object.measurement = ps.Hahn()     

    def new_Hahndeer_measurement(self, info):
        """info.object.measurement = mp.Rabi()"""        
        info.object.measurement = ps.HahnPair()     
    def new_DEER_measurement(self, info):
        """info.object.measurement = mp.Rabi()"""
        info.object.measurement = ps.DEERpair()     

        
                
    # new fits 

    def new_pulsed_fit(self, info):
        info.object.fit = PulsedFit()
    def new_rabi_fit(self, info):
        info.object.fit = RabiFit()
    def new_rabi_fit_phase(self, info):
        info.object.fit = RabiFit_phase()    
    def new_doublepulsed_fit(self, info):
        info.object.fit = DoublePulsedFit()
    def new_doublerabi_fit_phase(self, info):
        info.object.fit = DoubleRabiFit_phase()          
    def new_odmr_fit_XY8(self, info):
        info.object.fit = OdmrFitXY8()
    def new_exponential_fit(self, info):
        info.object.fit = DoubleExponentialDecayFit()
    def new_gaussian_fit(self, info):
        info.object.fit = DoubleGaussianDecayFit()
        
class PairPulsedAnalyzer(HasTraits, GetSetItemsMixin):
    
    """Fits a pulsed measurement with a pulsed fit.
    
    Example of usage:

        a = PulsedAnalyzer()
        a.measurement = Rabi
        a.fit = RabiFit()
        a.edit_traits()
        
    """
    
    measurement = Instance(ps.Pulsed, factory=ps.Pulsed)
    fit = Instance(PulsedFit, factory=PulsedFit)

    matrix_plot_data = Instance(ArrayPlotData)
    pulse_plot_data = Instance(ArrayPlotData)
    matrix_plot = Instance(Plot)
    pulse_plot = Instance(Plot)
    line_plot = Instance(Plot)
    # line_data is provided by fit class

    def __init__(self):
        super(PairPulsedAnalyzer, self).__init__()
        self.on_trait_change(self.refresh_matrix_axis, 'measurement.time_bins,measurement.n_laser', dispatch='ui')
        self.on_trait_change(self.refresh_matrix, 'measurement.count_data', dispatch='ui')
        self.on_trait_change(self.refresh_pulse, 'fit.pulse', dispatch='ui')
        self.on_trait_change(self.refresh_time_bins, 'measurement.time_bins', dispatch='ui')
        self.on_trait_change(self.refresh_flank, 'fit.flank', dispatch='ui')

    def _measurement_changed(self, new):
        self.fit = PulsedFit()
    
    def _matrix_plot_data_default(self):
        return ArrayPlotData(image=np.zeros((self.measurement.n_laser, self.measurement.n_bins)))
    def _pulse_plot_data_default(self):
        return ArrayPlotData(x=self.measurement.time_bins, y=self.fit.pulse)
    
    def _matrix_plot_default(self):
        plot = Plot(self.matrix_plot_data, width=500, height=500, resizable='hv', padding=8, padding_left=48, padding_bottom=36)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'laser pulse'
        plot.img_plot('image',
                      xbounds=(self.measurement.time_bins[0], self.measurement.time_bins[-1]),
                      ybounds=(0, self.measurement.n_laser),
                      colormap=Spectral)[0]
        return plot
    def _pulse_plot_default(self):
        plot = Plot(self.pulse_plot_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('x', 'y'), style='line', color='blue', name='data')
        edge_marker = LinePlot(index=ArrayDataSource(np.array((0, 0))),
                               value=ArrayDataSource(np.array((0, 1e9))),
                               color='red',
                               index_mapper=LinearMapper(range=plot.index_range),
                               value_mapper=LinearMapper(range=plot.value_range),
                               name='marker')
        plot.add(edge_marker)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'intensity'
        return plot
    def _line_plot_default(self):
        plot = Plot(self.fit.line_data, padding=8, padding_left=64, padding_bottom=36)
        for item in self.fit.plots:
            plot.plot(**item)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'spin state'
        return plot

    def refresh_matrix_axis(self):
        self.matrix_plot.components[0].index.set_data((self.measurement.time_bins[0], self.measurement.time_bins[-1]), (0.0, float(self.measurement.n_laser)))
        
    def refresh_matrix(self):
        s = self.measurement.count_data.shape
        if not s[0] * s[1] > 1000000:
            self.matrix_plot_data.set_data('image', self.measurement.count_data)

    def refresh_pulse(self):
        self.pulse_plot_data.set_data('y', self.fit.pulse)
    
    def refresh_time_bins(self):
        self.pulse_plot_data.set_data('x', self.measurement.time_bins)
    
    def refresh_flank(self):
        self.pulse_plot.components[1].index.set_data(np.array((self.fit.flank, self.fit.flank)))

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
    
    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)
    
    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename)
        
    def save_all(self, filename):
        self.save_line_plot(filename + '.png')
        self.save_matrix_plot(filename + '.png')
        self.save(filename + '.pyd')
    # Menu of the PulsedAnalyzer, only include tested measurements and Fits please!
    
    traits_view = View(HGroup(VGroup(Item(name='measurement', style='custom', show_label=False),
                                     Item('line_plot', name='normalized intensity', editor=ComponentEditor(), show_label=False, width=-400, height=-400, resizable=True), 
                                     ),
                              VGroup(Item(name='fit', style='custom', show_label=False),
                                     Tabbed(Item('matrix_plot', editor=ComponentEditor(), show_label=False, width=-300, height=-400, resizable=True),
                                            Item('pulse_plot', name='pulse profile', editor=ComponentEditor(), show_label=False, width=-300, height=-400, resizable=True),
                                            ),
                                     ),
                              ),
                              menubar=MenuBar(Menu(Action(action='save', name='Save (.pyd or .pys)'),
                                              Action(action='load', name='Load (.pyd or .pys)'),
                                              Action(action='save_matrix_plot', name='Save Matrix Plot (.png)'),
                                              Action(action='save_line_plot', name='Save Line Plot (.png)'),
                                              Action(action='save_all', name='Save All'),
                                              Action(action='_on_close', name='Quit'),
                                              name='File'),
                                              Menu(Action(action='new_rabi_measurement', name='Rabi'),
                                                    Action(action='new_FID_measurement', name='FID'),
                                                    Action(action='new_Hahn_measurement', name='Hahn'),
                                                    Action(action='new_Hahndeer_measurement', name='HahnPair'), 
                                                    Action(action='new_DEER_measurement', name='DEER_pair'), 
                                              name='Measurement'),
                                              Menu(Action(action='new_pulsed_fit', name='Pulsed'),
                                              Action(action='new_rabi_fit', name='Rabi'),
                                              Action(action='new_rabi_fit_phase', name='Rabi_phase'),
                                              name='Fit'),
                                                                              
                                 ),
                       title='PairPulsedAWGAnalyzer', buttons=[], resizable=True, handler=PairPulsedAnalyzerHandler)

    get_set_items = ['measurement', 'fit']
    get_set_order = ['measurement', 'fit']             
   
class SensingPulsedAnalyzerHandler(GetSetItemsHandler):

    """Provides handling of menu."""

    # save plots
    odmr = Instance( ODMR )
    
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
            
    def save_all(self, info):
        filename = save_file(title='Save All')
        if filename is '':
            return
        else:
            info.object.save_all(filename)
            
    #def save_all(self, info):
    #    filename = save_file(title='Save All')
    #    if filename is '':
    #        return
    #    else:
    #        info.object.save_all(filename)
    #        info.object.save_line_plot(filename)
            
    # new measurements
    
    def new_rabi_measurement(self, info):        
        info.object.measurement = sensing.Rabi() 
        
        
    def new_FID_measurement(self, info):        
        info.object.measurement = sensing.FID()    
        
    def new_Hahn_measurement(self, info):        
        info.object.measurement = sensing.Hahn()    
        
    def new_T1_measurement(self, info):        
        info.object.measurement = sensing.T1()     
        
    def new_T1_Ms0_measurement(self, info):        
        info.object.measurement = sensing.T1_Ms0()         
        
    def new_XY8_measurement(self, info):        
        info.object.measurement = sensing.XY8()     
        
    def new_XY4_measurement(self, info):        
        info.object.measurement = sensing.XY4()     

    def new_CPMG4_measurement(self, info):        
        info.object.measurement = sensing.CPMG4()             
        
    def new_SLD_measurement(self, info):        
        info.object.measurement = sensing.SpinLocking_decay()   
        
    def new_SLD_no_polarization_measurement(self, info):        
        info.object.measurement = sensing.Spin_Locking_without_polarization()
        
    def new_pulsepol_measurement(self, info):
        info.object.measurement = sensing.PulsePol()
        
    def new_kddpol_measurement(self, info):
        info.object.measurement = sensing.KDDxy_polarization()
        
    def new_PPC_measurement(self, info):        
        info.object.measurement = sensing.pulse_phase_check()         
        
        
    def new_HartmannHahn_measurement(self, info):        
        info.object.measurement = sensing.HartmannHahn()   
        
    def new_HartmannHahnOnwway_measurement(self, info):        
        info.object.measurement = sensing.HartmannHahnOneway()     
        
    def new_HartmannHahn_ampscan_measurement(self, info):        
        info.object.measurement = sensing.HartmannHahnamp()  
    
    def new_HartmannHahn_ampscanOneway_measurement(self, info):        
        info.object.measurement = sensing.HartmannHahnampOneway()     
        
    def new_HartmannHahn_scanOneway_fsweep_measurement(self, info):        
        info.object.measurement = sensing.HartmannHahnOnewayfsweep()   
        
    def new_HartmannHahn_ampscanOneway_fsweep_measurement(self, info):        
        info.object.measurement = sensing.HartmannHahnampOnewayfsweep()       

    def new_XY4_Ref_measurement(self, info):        
        info.object.measurement = sensing.XY4_Ref()     
        
    def new_XY8_Ref_measurement(self, info):        
        info.object.measurement = sensing.XY8_Ref()     
        
    def new_XY16_Ref_measurement(self, info):        
        info.object.measurement = sensing.XY16_Ref()         
        
    def new_Corr_XY8_phase_measurement(self, info):     
        info.object.measurement = sensing.Spec_XY8_phase_check()     
        
    def new_Correlation_Spec_XY8_measurement(self, info):        
        info.object.measurement = sensing.Correlation_Spec_XY8()   
        
    def new_Correlation_Spec_XY8_phase_measurement(self, info):        
        info.object.measurement = sensing.Correlation_Spec_XY8_phase()   

    def new_Correlation_Spec_XY16_phase_measurement(self, info):        
        info.object.measurement = sensing.Correlation_Spec_XY16_phase()   
        
    def new_Correlation_Nuclear_Rabi(self, info):
        info.object.measurement = sensing.Correlation_Nuclear_Rabi()       
        
        
    def new_Correlation_Spec_XY4_measurement(self, info):        
        info.object.measurement = sensing.Correlation_Spec_XY4()      

    def new_Correlation_Spec_CPMG4_measurement(self, info):        
        info.object.measurement = sensing.Correlation_Spec_CPMG4()             
        
    def new_RF_Endor_measurement(self, info):        
        info.object.measurement = sensing.RF_sweep()  
        
    def new_RF_Rabi_measurement(self, info):        
        info.object.measurement = sensing.Single_RF_Rabi()      
        
    def new_Nuclea_Hahn_measurement(self, info):        
        info.object.measurement = sensing.Nuclea_Hahn()               
    
    def new_proton_lmzdet_freq_measurement(self, info):        
        info.object.measurement = sensing.Proton_longmagzationdet_freq()     
        
    def new_proton_lmzdet_time_measurement(self, info):        
        info.object.measurement = sensing.Proton_longmagzationdet_time()     

    def new_Hahn_AC_measurement(self, info):        
        info.object.measurement = sensing.Hahn_AC()  
        
    def new_FID_AC_measurement(self, info):        
        info.object.measurement = sensing.FID_AC() 
        
    def new_FID_Db_measurement(self, info):        
        info.object.measurement = sensing.FID_Db()    
        
    def new_Hahn_AC_phase_measurement(self, info):        
        info.object.measurement = sensing.Hahn_AC_phase()                 

    #def new_Hahndeer_measurement(self, info):
        """info.object.measurement = mp.Rabi()"""        
        #info.object.measurement = ps.HahnPair()     
    #def new_DEER_measurement(self, info):
        """info.object.measurement = mp.Rabi()"""
        #info.object.measurement = ps.DEERpair()     

        
                
    # new fits 

    def new_pulsed_fit(self, info):
        info.object.fit = PulsedFit()
    def new_rabi_fit(self, info):
        info.object.fit = RabiFit()
        
    def new_odmr_fit(self, info):
        info.object.fit = OdmrFit()    
    def new_rabi_fit_phase(self, info):
        info.object.fit = RabiFit_phase()          
        
    def new_doublepulsed_fit(self, info):
        info.object.fit = DoublePulsedFit()

    def new_doublerabi_fit_phase(self, info):
        info.object.fit = DoubleRabiFit_phase()              
    
    def new_odmr_fit_XY8(self, info):
        info.object.fit = OdmrFitXY8()
       
    def new_exponential_fit(self, info):
        info.object.fit = DoubleExponentialDecayFit()
        
    def new_gaussian_fit(self, info):
        info.object.fit = DoubleGaussianDecayFit()
    def new_decaying_cosine_fit(self, info):
        info.object.fit = Damped_Cosine()
        
class SensingPulsedAnalyzer(HasTraits, GetSetItemsMixin):
    
    """Fits a pulsed measurement with a pulsed fit.
    
    Example of usage:

        a = PulsedAnalyzer()
        a.measurement = Rabi
        a.fit = RabiFit()
        a.edit_traits()
        
    """
    
    measurement = Instance(ps.Pulsed, factory=ps.Pulsed)
    fit = Instance(PulsedFit, factory=PulsedFit)

    matrix_plot_data = Instance(ArrayPlotData)
    pulse_plot_data = Instance(ArrayPlotData)
    matrix_plot = Instance(Plot)
    pulse_plot = Instance(Plot)
    line_plot = Instance(Plot)
    # line_data is provided by fit class

    def __init__(self):
        super(SensingPulsedAnalyzer, self).__init__()
        self.on_trait_change(self.refresh_matrix_axis, 'measurement.time_bins,measurement.n_laser', dispatch='ui')
        self.on_trait_change(self.refresh_matrix, 'measurement.count_data', dispatch='ui')
        self.on_trait_change(self.refresh_pulse, 'fit.pulse', dispatch='ui')
        self.on_trait_change(self.refresh_time_bins, 'measurement.time_bins', dispatch='ui')
        self.on_trait_change(self.refresh_flank, 'fit.flank', dispatch='ui')

    def _measurement_changed(self, new):
        self.fit = PulsedFit()
    
    def _matrix_plot_data_default(self):
        return ArrayPlotData(image=np.zeros((self.measurement.n_laser, self.measurement.n_bins)))
    def _pulse_plot_data_default(self):
        return ArrayPlotData(x=self.measurement.time_bins, y=self.fit.pulse)
    
    def _matrix_plot_default(self):
        plot = Plot(self.matrix_plot_data, width=500, height=500, resizable='hv', padding=8, padding_left=48, padding_bottom=36)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'laser pulse'
        plot.img_plot('image',
                      xbounds=(self.measurement.time_bins[0], self.measurement.time_bins[-1]),
                      ybounds=(0, self.measurement.n_laser),
                      colormap=Spectral)[0]
        return plot
    def _pulse_plot_default(self):
        plot = Plot(self.pulse_plot_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('x', 'y'), style='line', color='blue', name='data')
        edge_marker = LinePlot(index=ArrayDataSource(np.array((0, 0))),
                               value=ArrayDataSource(np.array((0, 1e9))),
                               color='red',
                               index_mapper=LinearMapper(range=plot.index_range),
                               value_mapper=LinearMapper(range=plot.value_range),
                               name='marker')
        plot.add(edge_marker)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'intensity'
        return plot
    def _line_plot_default(self):
        plot = Plot(self.fit.line_data, padding=8, padding_left=64, padding_bottom=36)
        for item in self.fit.plots:
            plot.plot(**item)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'spin state'
        return plot

    def refresh_matrix_axis(self):
        self.matrix_plot.components[0].index.set_data((self.measurement.time_bins[0], self.measurement.time_bins[-1]), (0.0, float(self.measurement.n_laser)))
        
    def refresh_matrix(self):
        s = self.measurement.count_data.shape
        if not s[0] * s[1] > 1000000:
            self.matrix_plot_data.set_data('image', self.measurement.count_data)

    def refresh_pulse(self):
        self.pulse_plot_data.set_data('y', self.fit.pulse)
    
    def refresh_time_bins(self):
        self.pulse_plot_data.set_data('x', self.measurement.time_bins)
    
    def refresh_flank(self):
        self.pulse_plot.components[1].index.set_data(np.array((self.fit.flank, self.fit.flank)))

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
    
    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename + '_Matrix_Plot.png')
    
    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename + '_Line_Plot.png')
        
    def save_all(self, filename):
        self.save_line_plot(filename)
        self.save_matrix_plot(filename)
        self.save(filename + '.pyd')
    # Menu of the PulsedAnalyzer, only include tested measurements and Fits please!
    
    traits_view = View(HGroup(VGroup(Item(name='measurement', style='custom', show_label=False),
                                     Item('line_plot', name='normalized intensity', editor=ComponentEditor(), show_label=False, width=-400, height=-400, resizable=True), 
                                     ),
                              VGroup(Item(name='fit', style='custom', show_label=False),
                                     Tabbed(Item('matrix_plot', editor=ComponentEditor(), show_label=False, width=-300, height=-400, resizable=True),
                                            Item('pulse_plot', name='pulse profile', editor=ComponentEditor(), show_label=False, width=-300, height=-400, resizable=True),
                                            ),
                                     ),
                              ),
                              menubar=MenuBar(Menu(Action(action='save', name='Save (.pyd or .pys)'),
                                              Action(action='load', name='Load (.pyd or .pys)'),
                                              Action(action='save_matrix_plot', name='Save Matrix Plot (.png)'),
                                              Action(action='save_line_plot', name='Save Line Plot (.png)'),
                                              Action(action='save_all', name='Save All'),
                                              Action(action='_on_close', name='Quit'),
                                              name='File'),
                                              Menu(Action(action='new_rabi_measurement', name='Rabi'),
                                                    Action(action='new_FID_measurement', name='FID'),
                                                    Action(action='new_XY8_measurement', name='XY8'),
                                                    Action(action='new_XY4_measurement', name='XY4'),
                                                    Action(action='new_CPMG4_measurement', name='CPMG4'),
                                                    Action(action='new_PPC_measurement', name='pulse_phase_check'), 
                                                    Action(action='new_SLD_no_polarization_measurement', name='Alternating locking'),
                                                    Action(action='new_pulsepol_measurement', name='PulsePol'),
                                                    Action(action='new_HartmannHahn_measurement', name='HartmannHahn'),  
                                                    Action(action='new_HartmannHahnOnwway_measurement', name='HartmannHahnOneway'),  
                                                    Action(action='new_HartmannHahn_ampscan_measurement', name='HartmannHahnscan'),    
                                                    Action(action='new_HartmannHahn_ampscanOneway_measurement', name='HartmannHahnscanOneway'),  
                                                    Action(action='new_HartmannHahn_scanOneway_fsweep_measurement', name='HartmannHahnscanOnewayfsweep'),  
                                                    Action(action='new_HartmannHahn_ampscanOneway_fsweep_measurement', name='HartmannHahnampscanOnewayfsweep'),                                                 
                                                    Action(action='new_Correlation_Spec_XY4_measurement', name='Correlation_Spec_XY4'),      
                                                    Action(action='new_Correlation_Spec_CPMG4_measurement', name='Correlation_Spec_CPMG4'),       
                                                    Action(action='new_Correlation_Spec_XY8_phase_measurement', name='Correlation_Spec_XY8_phase'),  
                                                    Action(action='new_Correlation_Spec_XY16_phase_measurement', name='Correlation_Spec_XY16_phase'),  
                                                    Action(action='new_Correlation_Nuclear_Rabi', name = 'Correlation_Nuclear_Rabi'),
                                                                                                 
                                                    Action(action='new_RF_Endor_measurement', name='RF_Endor'),       
                                                    Action(action='new_RF_Rabi_measurement', name='RF_Rabi'), 

                                                    #Action(action='new_Hahndeer_measurement', name='HahnPair'), 
                                                    #Action(action='new_DEER_measurement', name='DEER_pair'), 
                                              name='Measurement'),
                                              Menu(Action(action='new_pulsed_fit', name='Pulsed'),
                                              Action(action='new_odmr_fit', name='Odmr_Fit'),
                                              Action(action='new_rabi_fit', name='Rabi'),
                                              Action(action='new_rabi_fit_phase', name='Rabi_phase'),
                                              Action(action='new_Cdecay_ebath_fit', name='Cdecay_ebath'),
                                              Action(action='new_exponential_fit', name = 'Exponential fit'),
                                              Action(action='new_gaussian_fit', name = 'Gaussian fit'),
                                              Action(action='new_decaying_cosine_fit', name= 'Damped cosine'),
                                              name='Fit'),
                                                                              
                                 ),
                       title='SensingPulsedAWGAnalyzer', buttons=[], resizable=True, handler=SensingPulsedAnalyzerHandler)

    get_set_items = ['measurement', 'fit']
    get_set_order = ['measurement', 'fit']       


class DoubleSensingPulsedAnalyzer(HasTraits, GetSetItemsMixin):
    
    """Fits a pulsed measurement with a pulsed fit.
    
    Example of usage:

        a = PulsedAnalyzer()
        a.measurement = Rabi
        a.fit = RabiFit()
        a.edit_traits()
        
    """
    
    measurement = Instance(mp.Pulsed, factory=mp.Pulsed)
    fit = Instance(DoublePulsedFit, factory=DoublePulsedFit)
    
    matrix_plot_data = Instance(ArrayPlotData)
    pulse_plot_data = Instance(ArrayPlotData)
    
    matrix_plot = Instance(Plot)
    pulse_plot = Instance(Plot)
    line_plot = Instance(Plot)
    processed_plot = Instance(Plot)
    
    # line_data and processed_plot_data are provided by fit class

    def __init__(self):
        super(DoubleSensingPulsedAnalyzer, self).__init__()
        self.on_trait_change(self.refresh_matrix_axis, 'measurement.time_bins,measurement.n_laser', dispatch='ui')
        self.on_trait_change(self.refresh_matrix, 'measurement.count_data', dispatch='ui')
        self.on_trait_change(self.refresh_pulse, 'fit.pulse', dispatch='ui')
        self.on_trait_change(self.refresh_time_bins, 'measurement.time_bins', dispatch='ui')
        
        #self.on_trait_change(self.refresh_evol_time, 'fit.free_evolution_time', dispatch='ui')
        #self.on_trait_change(self.refresh_norm_counts, 'measurement.count_data, fit.normalized_counts', dispatch='ui')
        #self.on_trait_change(self.refresh_plot_fit, 'fit.fit_parameters', dispatch='ui')
    
        self.on_trait_change(self.refresh_flank, 'fit.flank', dispatch='ui')
        
    def _measurement_changed(self, new):
        self.fit = DoublePulsedFit()
    
    def _matrix_plot_data_default(self):
        return ArrayPlotData(image=np.zeros((self.measurement.n_laser, self.measurement.n_bins)))
        
    def _pulse_plot_data_default(self):
        return ArrayPlotData(x=self.measurement.time_bins, y=self.fit.pulse)
        
    # def _processed_plot_data_default(self): 
        # return ArrayPlotData(x=self.fit.free_evolution_time, y=self.fit.normalized_counts, fit=np.array((0., 0.)))
    
    def _matrix_plot_default(self):
        plot = Plot(self.matrix_plot_data, width=500, height=500, resizable='hv', padding=8, padding_left=48, padding_bottom=36)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'laser pulse'
        plot.img_plot('image',
                      xbounds=(self.measurement.time_bins[0], self.measurement.time_bins[-1]),
                      ybounds=(0, self.measurement.n_laser),
                      colormap=Spectral)[0]
        return plot
        
    def _pulse_plot_default(self):
        plot = Plot(self.pulse_plot_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('x', 'y'), style='line', color='blue', name='data')
        edge_marker = LinePlot(index=ArrayDataSource(np.array((0, 0))),
                               value=ArrayDataSource(np.array((0, 1e9))),
                               color='red',
                               index_mapper=LinearMapper(range=plot.index_range),
                               value_mapper=LinearMapper(range=plot.value_range),
                               name='marker')
        plot.add(edge_marker)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'intensity'
        return plot
        
    def _line_plot_default(self):
        plot = Plot(self.fit.line_data, padding=8, padding_left=64, padding_bottom=36)
        for item in self.fit.plots[0:2]:
            plot.plot(**item)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'spin state'
        return plot
        
    def _processed_plot_default(self):
        plot = Plot(self.fit.processed_plot_data, padding=8, padding_left=64, padding_bottom=36)
        for item in self.fit.plots[2:]:
            plot.plot(**item)
        plot.index_axis.title = 'free evolution time [ns]'
        plot.value_axis.title = 'spin state'
        return plot

    def refresh_matrix_axis(self):
        self.matrix_plot.components[0].index.set_data((self.measurement.time_bins[0], self.measurement.time_bins[-1]), (0.0, float(self.measurement.n_laser)))
        
    def refresh_matrix(self):
        s = self.measurement.count_data.shape
        if not s[0] * s[1] > 1000000:
            self.matrix_plot_data.set_data('image', self.measurement.count_data)

    def refresh_pulse(self):
        self.pulse_plot_data.set_data('y', self.fit.pulse)
    
    def refresh_time_bins(self):
        self.pulse_plot_data.set_data('x', self.measurement.time_bins)
    
    def refresh_flank(self):
        self.pulse_plot.components[1].index.set_data(np.array((self.fit.flank, self.fit.flank)))
        
    # def refresh_evol_time(self):
        # self.fit.processed_plot_data.set_data('x', self.fit.free_evolution_time)
        
    # def refresh_norm_counts(self):
        # self.fit.processed_plot_data.set_data('y', self.fit.normalized_counts)
        
    # def refresh_plot_fit(self):
        # if not np.isnan(self.fit.fit_parameters[0]):  
            # self.fit.processed_plot_data.set_data('fit', fitting.NLorentzians(*self.fit.fit_parameters)(self.fit.free_evolution_time))
            # self.processed_plot.plot(('x', 'fit'), color='red', style='line', line_width = 1) 

    def _fit_changed(self, name, old, new):
        old.measurement = None
        new.measurement = self.measurement
        plot = self.line_plot
        plot2=self.processed_plot
        # delete all old plots
        for key in plot.plots.keys()[0:2]:
            plot.delplot(key)
        for key in plot2.plots.keys()[2:]:
            plot2.delplot(key)   
        
        # set new data source
        plot.data = self.fit.line_data
        plot2.data = self.fit.processed_plot_data
                
        # make new plots
        for item in self.fit.plots[0:2]:
            plot.plot(**item)
            
        for item in self.fit.plots[2:]:
            plot2.plot(**item)
        #if hasattr(self.fit,'legends'):
            #plot.legend(**self.fit.legends)
        # if the fit has an attr 'text' attached to it, print it in the lower left corner of the plot
        if hasattr(old, 'text'):
            label = plot.overlays[0]
            old.sync_trait('text', label, 'text', mutual=False, remove=True)
            plot.overlays = []
        if hasattr(new, 'text'):
            label = PlotLabel(text=self.fit.text, hjustify='left', vjustify='bottom', position=[64, 32])
            new.sync_trait('text', label, 'text', mutual=False)
            plot.overlays = [label]
                    
        #self.processed_plot.plot(('x', 'y'), color='red', line_width = 2)
    
    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)
    
    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename)
        
    def save_processed_plot(self, filename):
        self.save_figure(self.processed_plot, filename)
        
    def save_all(self, filename):
        self.save_line_plot(filename + '_LinePlot.png')
        self.save_matrix_plot(filename + '_MatrixPlot.png')
        self.save_processed_plot(filename + '_ProcPlot.png')
        self.save(filename + '.pyd')
        
    # Menu of the PulsedAnalyzer, only include tested measurements and Fits please!
    
    traits_view = View(HGroup(VGroup(Item(name='measurement', style='custom', show_label=False),
                                     Item('line_plot', name='normalized intensity', editor=ComponentEditor(), show_label=False, width=300, height=300, resizable=True), 
                                     ),
                              VGroup(Item(name='fit', style='custom', show_label=False),
                                     Tabbed(Item('matrix_plot', editor=ComponentEditor(), show_label=False, width=300, height=300, resizable=True),
                                            Item('pulse_plot', name='pulse profile', editor=ComponentEditor(), show_label=False, width=300, height=300, resizable=True),
                                            Item('processed_plot', name='normalized intensity2', editor=ComponentEditor(), show_label=False, width=300, height=300, resizable=True), 
                                            ),
                                     ),
                              ),
                       menubar=MenuBar(Menu(Action(action='save', name='Save (.pyd or .pys)'),
                                              Action(action='load', name='Load (.pyd or .pys)'),
                                              Action(action='save_matrix_plot', name='Save Matrix Plot (.png)'),
                                              Action(action='save_line_plot', name='Save Line Plot (.png)'),
                                              Action(action='save_all', name='Save All'),
                                              Action(action='_on_close', name='Quit'),
                                              name='File'),
                                       Menu(Action(action='new_XY4_Ref_measurement', name='XY4_Ref'),
                                            Action(action='new_XY8_Ref_measurement', name='XY8_Ref'),
                                            Action(action='new_XY16_Ref_measurement', name='XY16_Ref'),
                                            Action(action='new_Hahn_measurement', name='Hahn'),
                                            Action(action='new_T1_measurement', name='T1'),
                                            Action(action='new_kddpol_measurement', name = 'KDDxy polarisation'),
                                            Action(action='new_T1_Ms0_measurement', name='T1_Ms0'),
                                            Action(action='new_SLD_measurement', name='Spinlocking_decay'),
                                            Action(action='new_SLD_no_polarization_measurement', name='Alternating locking'),
                                            Action(action='new_Corr_XY8_phase_measurement', name='Spec_XY8_phase_check'),
                                            Action(action='new_proton_lmzdet_freq_measurement', name='Proton_longmagzationdet_freq'),
                                            Action(action='new_proton_lmzdet_time_measurement', name='Proton_longmagzationdet_time'),
                                            Action(action='new_Hahn_AC_measurement', name='Hahn_AC'),
                                            Action(action='new_FID_AC_measurement', name='FID_AC'),
                                            Action(action='new_FID_Db_measurement', name='FID_Db'),
                                            Action(action='new_Hahn_AC_phase_measurement', name='Hahn_AC_phase'),
                                            Action(action='new_Nuclea_Hahn_measurement', name='Nuclea_Hahn'),   
                                            Action(action='new_Correlation_Spec_XY8_measurement', name='Correlation_Spec_XY8'),  
                                            
                                            
                                              name='Measurement'),
                                             
                                         Menu(
                                              Action(action='new_doublepulsed_fit', name='DoublePulsed'),
                                              Action(action='new_doublerabi_fit_phase', name='Doublerabi_fit_phase'),
                                              Action(action='new_odmr_fit_XY8', name='fit lorentzians'),
                                              Action(action='new_exponential_fit', name = 'Exponential fit'),
                                              Action(action='new_gaussian_fit', name = 'Gaussian fit'),
                                              Action(action='new_decaying_cosine_fit', name='Damped Cosine'),
                                              name='Fit'
                                              ),
                                              '''
                                              Action(action='new_pulsed_fit',name='Pulsed'),
                                              Action(action='new_frequency_fit', name='Frequency'),
                                              Action(action='new_frequency3pi2_fit', name='Frequency3pi2'),
                                              Action(action='new_pulsed_fit_tau', name='Tau'),
                                              Action(action='new_fit_tau_ref', name='Tau Ref'),
                                              Action(action='new_rabi_fit', name='Rabi'),
                                              Action(action='new_hahn_fit', name='Hahn'),
                                              Action(action='new_double_fit', name='Double'),
                                              Action(action='new_double_fit_tau', name='Double Tau'),
                                              Action(action='new_double_fit_tau_ref', name='Double Tau Ref'),
                                              Action(action='new_t1pi_fit', name='T1pi'),
                                              Action(action='new_hahn3pi2_fit', name='Hahn3pi2'),
                                              Action(action='new_pulse_extraction_fit', name='PulseExtraction'),
                                              Action(action='new_pulse_cal_fit', name='PulseCal'),
                                              '''
                                              
                                 ),
                       title='DoubleSensingPulsedAnalyzer', buttons=[], resizable=True, handler=SensingPulsedAnalyzerHandler)

    get_set_items = ['measurement', 'fit']
    get_set_order = ['measurement', 'fit']        
    
    
class SSTAnalyzerHandler(GetSetItemsHandler):

    def new_sst_measurement(self, info):
        info.object.measurement = ss.SST_Threshold()
    
    # new fits 
    def new_pulsed_fit(self, info):
        info.object.fit = SSTFit()
    
   
class SSTAnalyzer(HasTraits, GetSetItemsMixin):
    
    """Fits a pulsed measurement with a pulsed fit.
    
    Example of usage:

        a = PulsedAnalyzer()
        a.measurement = Rabi
        a.fit = RabiFit()
        a.edit_traits()
        
    """
    
    measurement = Instance(ss.Pulsed, factory=ss.Pulsed)
    fit = Instance(SSTFit, factory=SSTFit)

    line_plot = Instance(Plot)
    # line_data is provided by fit class

    def __init__(self):
        super(SSTAnalyzer, self).__init__()
        #self.on_trait_change(self.refresh_matrix, 'measurement.count_data', dispatch='ui')
        #self.on_trait_change(self.refresh_pulse, 'fit.pulse', dispatch='ui')
        

    def _measurement_changed(self, new):
        self.fit = SSTFit()
    
    def _line_plot_default(self):
        plot = Plot(self.fit.line_data, padding=8, padding_left=64, padding_bottom=36)
        for item in self.fit.plots:
            plot.plot(**item)
        plot.index_axis.title = 'time [ns]'
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
    
  
    
    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename)
        
    def save_all(self, filename):
        self.save_line_plot(filename + '.png')
        #self.save_matrix_plot(filename + '.png')
        self.save(filename + '.pyd')
        
    # Menu of the PulsedAnalyzer, only include tested measurements and Fits please!
    
    traits_view = View(HGroup(VGroup(Item(name='measurement', style='custom', show_label=False),
                                     Item('line_plot', name='normalized intensity', editor=ComponentEditor(), show_label=False, width=-400, height=-400, resizable=True), 
                                     ),
                              VGroup(Item(name='fit', style='custom', show_label=False),
                                     #Item('line_plot', name='normalized intensity', editor=ComponentEditor(), show_label=False, width=-400, height=-400, resizable=True), 
                                     #Tabbed(Item('matrix_plot', editor=ComponentEditor(), show_label=False, width=-300, height=-400, resizable=True),
                                            #Item('pulse_plot', name='pulse profile', editor=ComponentEditor(), show_label=False, width=-300, height=-400, resizable=True),
                                            #),
                                     ),
                              ),
                       menubar=MenuBar(Menu(Action(action='save', name='Save (.pyd or .pys)'),
                                              Action(action='load', name='Load (.pyd or .pys)'),
                                              Action(action='save_matrix_plot', name='Save Matrix Plot (.png)'),
                                              Action(action='save_line_plot', name='Save Line Plot (.png)'),
                                              Action(action='save_all', name='Save All'),
                                              Action(action='_on_close', name='Quit'),
                                              name='File'),
                                         Menu(Action(action='new_sst_measurement', name='SST'),
                                              name='Measurement'),
                                         Menu(Action(action='new_pulsed_fit', name='SSTFit'),
                                            
                                              name='Fit'),
                                              """
                                             
                                              """
                                              
                                 ),
                       title='SSTAnalyzerHandler', buttons=[], resizable=True, handler=SSTAnalyzerHandler)

    get_set_items = ['measurement', 'fit']
    get_set_order = ['measurement', 'fit']    
#########################################
# testing
#########################################

if __name__ == '__main__':

    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().info('Starting logger.')
    
    from tools.emod import JobManager
    
    JobManager().start()
        
    import measurements.pulsed
    
    # basic example to analyze a Rabi measurement
    #m = measurements.pulsed.Rabi()
    #a = PulsedAna()
    #a = PulsedAnaTau()
    #a = PulsedAnaDoubleTauRef()
    a = RabiAna()
    #a = PulseCalAna()
    #a.measurement = measurements.pulsed.Rabi()
    #a.measurement = mp.PulseCal()
    a.edit_traits()
    #a.load('')
    #a.measurement = m
