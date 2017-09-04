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

    profile = c.sum(0)
    edge = find_edge(profile)
    
    I = int(round(T / float(dt)))
    i0 = edge + int(round(t0 / float(dt)))
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

from traits.api import HasTraits, Instance, Property, Range, Float, Int, Bool, Array, List, Str, Tuple, Enum, \
                                 on_trait_change, cached_property, DelegatesTo
from traitsui.api import View, Item, Tabbed, Group, HGroup, VGroup, VSplit, EnumEditor, TextEditor
from enable.api import ComponentEditor
from chaco.api import ArrayDataSource, LinePlot, LinearMapper, ArrayPlotData, Plot, Spectral, PlotLabel

from traitsui.file_dialog import save_file
from traitsui.menu import Action, Menu, MenuBar

import threading
import time
import logging

import fitting

from tools.utility import GetSetItemsHandler, GetSetItemsMixin

import measurements.pulsed as mp
import measurements.nmr as nmr
import measurements.opticalrabi as orabi
import measurements.nuclear_rabi as nr
import measurements.rabi as newrabi

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

class PulsedAna(HasTraits, GetSetItemsMixin):

    """
    Base class for a pulsed analysis. Provides calculation of spin state
    and plotting.
    Derive from this to create analysis tools for pulsed measurements.
    """

    # the measurement to analyze
    measurement = Instance(mp.Pulsed, factory=mp.Pulsed)
    
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

    def __init__(self):
        super(PulsedAna, self).__init__()
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
            self.line_data.set_data('index', np.arange(n))
        self.line_data.set_data('spin_state', y)

    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)
    
    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename)


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


class PulsedAnaDoubleTauRef(PulsedAnaTau):

    """
    Analysis of a pulsed measurement with a 'tau' as index-data,
    two subsequent pulsed sequences and bright / dark reference
    measurement at the end of the sequence.
    """

    measurement = Instance(mp.CPMG3pi2, factory=mp.CPMG3pi2)
    
    # overwrite __init__ such that change of 'run_sum' causes plot update 
    def __init__(self):
        super(PulsedAnaDoubleTauRef, self).__init__()
        self.on_trait_change(self._update_line_plot_value, 'run_sum') # ToDo: fix this

    # overwrite the line_plot such that the bright and dark state
    # as well as reference lines are plotted 
    def _create_line_plot(self):
        line_data = ArrayPlotData(index=np.array((0, 1)),
                                    first=np.array((0, 0)),
                                    second=np.array((0, 0)),
                                    bright=np.array((1, 1)),
                                    dark=np.array((0, 0)))
        plot = Plot(line_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('index', 'first'), color='blue', name='first')
        plot.plot(('index', 'second'), color='green', name='second')
        plot.plot(('index', 'bright'), color='red', name='bright')
        plot.plot(('index', 'dark'), color='black', name='dark')
        plot.index_axis.title = 'time [micro s]'
        plot.value_axis.title = 'spin state'
        self.line_data = line_data
        self.line_plot = plot
    
    # overwrite this one to provide splitting up of spin_state array
    # and setting of bright and dark state data 
    def _update_line_plot_value(self):
        y = self.spin_state
        n_ref = self.measurement.n_ref
        n = len(y) / 2 - n_ref
        first = y[:n]
        second = y[n:2 * n]
        bright = y[-2 * n_ref:-n_ref].mean() * self.run_sum
        dark = y[-n_ref:].mean() * self.run_sum
        if self.run_sum > 1:
            first = run_sum(first, self.run_sum)
            second = run_sum(second, self.run_sum)
        try:
            self.line_data.set_data('first', first)
            self.line_data.set_data('second', second)
            self.line_data.set_data('bright', bright * np.ones(n))
            self.line_data.set_data('dark', dark * np.ones(n))
        except:
            logging.getLogger().debug('Could not set data for spin_state plots.')
    

    

class PulsedAnaDoubleRef(PulsedAna):

    """
    Analysis of a pulsed measurement with two
    sequences and bright / dark reference
    measurement at the end of the sequence.
    """

    measurement = Instance(mp.Pulsed, factory=mp.Pi2PiCal)
    
    # overwrite the line_plot such that the bright and dark state reference lines are plotted 
    def _create_line_plot(self):
        line_data = ArrayPlotData(index=np.array((0, 1)),
                                    first=np.array((0, 0)),
                                    second=np.array((0, 0)),
                                    bright=np.array((1, 1)),
                                    dark=np.array((0, 0)))
        plot = Plot(line_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('index', 'first'), color='blue', name='first')
        plot.plot(('index', 'second'), color='green', name='second')
        plot.plot(('index', 'bright'), color='red', name='bright')
        plot.plot(('index', 'dark'), color='black', name='dark')
        plot.index_axis.title = 'pulse #'
        plot.value_axis.title = 'spin state'
        self.line_data = line_data
        self.line_plot = plot

    # overwrite this one to provide splitting up of spin_state array and setting of bright and dark state data 
    def _update_line_plot_value(self):
        y = self.spin_state
        n_ref = self.measurement.n_ref
        n = len(y) / 2 - n_ref
        old_index = self.line_data.get_data('index')
        if old_index is not None and len(old_index) != n:
            self.line_data.set_data('index', np.arange(n))
        self.line_data.set_data('first', y[:n])
        self.line_data.set_data('second', y[n:2 * n])
        self.line_data.set_data('bright', np.mean(y[2 * n:2 * n + n_ref]) * np.ones(n))
        self.line_data.set_data('dark', np.mean(y[2 * n + n_ref:]) * np.ones(n))


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


class PulseExtractAna(PulsedAnaTau):

    """
    Provides extraction of pulses from a Rabi measurement.
    """
        
    period = Float()
    contrast = Float()
    t_pi2 = Array()
    t_pi = Array()
    t_3pi2 = Array()
    t_2pi = Array()
    
    # add fit results to the get_set_items
    get_set_items = PulsedAnaTau.get_set_items + ['contrast', 'period', 't_pi2', 't_pi', 't_3pi2', 't_2pi']

    traits_view = View(VGroup(VGroup(Item(name='measurement', style='custom', show_label=False),
                                     Group(VGroup(HGroup(Item('contrast', style='readonly', format_str='%.1f'),
                                                         Item('period', style='readonly'),
                                                         ),
                                                  HGroup(Item('t_pi2', style='readonly'),
                                                         Item('t_pi', style='readonly'),
                                                         Item('t_3pi2', style='readonly'),
                                                         Item('t_2pi', style='readonly'),
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
                       title='Pulse Extraction',
                       menubar=menubar,
                       buttons=[], resizable=True, handler=PulsedAnaHandler)

    # overwrite __init__ to trigger update events
    def __init__(self):
        super(PulseExtractAna, self).__init__()
        self.on_trait_change(self._update_fit, 'spin_state', dispatch='ui')
    
    def _update_fit(self, y):
        # try to extract pulses from spin_state and tau.
        tau = self.measurement.tau
        try:
            f, r, p, tp = fitting.extract_pulses(y)
        except:
            logging.getLogger().debug('Failed to compute fit.')
            f = [0]
            r = [0]
            p = [0]
            tp = [0]
            
        pi2 = tau[f]
        pi = tau[p]
        three_pi2 = tau[r]
        two_pi = tau[tp]

        # compute some relevant parameters from the result
        mi = y.min()
        ma = y.max()
        contrast = 100 * (ma - mi) / ma
        # simple rough estimate of the period to avoid index out of range Error
        T = 4 * (pi[0] - pi2[0])
        
        # set respective attributes
        self.period = T
        self.contrast = contrast
        self.t_pi2 = pi2
        self.t_pi = pi
        self.t_3pi2 = three_pi2
        self.t_2pi = two_pi

        # create a summary of the result as a text string
        s = 'contrast: %.1f\n' % contrast
        s += 'period: %.2f ns\n' % T
        
        # markers in the plot that show the result of the pulse extraction
        self.line_data.set_data('pulse_indices', np.hstack((pi2, pi, three_pi2, two_pi)) * 1e-3)            
        self.line_data.set_data('pulse_values', np.hstack((y[f], y[p], y[r], y[tp])))
        self.line_plot.overlays[0].text = s

    # overwrite the line_plot to include fit and text label 
    def _create_line_plot(self):
        line_data = ArrayPlotData(index=np.array((0, 1)),
                                    spin_state=np.array((0, 0)),
                                    pulse_indices=np.array((0, 0)),
                                    pulse_values=np.array((0, 0)))
        plot = Plot(line_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('index', 'spin_state'), color='blue', name='spin_state')
        plot.plot(('pulse_indices', 'pulse_values'),
                  type='scatter',
                  marker='circle',
                  color='none',
                  outline_color='red',
                  line_width=1.0,
                  name='pulses')
        plot.index_axis.title = 'time [micro s]'
        plot.value_axis.title = 'spin state'
        plot.overlays.insert(0, PlotLabel(text=self.label_text, hjustify='left', vjustify='bottom', position=[64, 32]))
        self.line_data = line_data
        self.line_plot = plot


class PulseCalAna(PulsedAna):

    """Provides calculation of RMS from PulseCal  measurement."""
    
    measurement = Instance(mp.PulseCal, factory=mp.PulseCal)
    
    rms = Float()
    
    get_set_items = PulsedAna.get_set_items + ['rms']

    traits_view = View(VGroup(Item(name='measurement', style='custom', show_label=False),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     Item('run_sum'),
                                     Item('rms', style='readonly', format_str='%.1f'),
                                     ),
                              VSplit(Item('matrix_plot', show_label=False, width=500, height=300, resizable=True),
                                     Item('line_plot', show_label=False, width=500, height=300, resizable=True),
                                     Item('pulse_plot', show_label=False, width=500, height=300, resizable=True),
                                     ),
                              ),
                       title='Pulse Calibration Analysis',
                       menubar=menubar,
                       buttons=[], resizable=True, handler=PulsedAnaHandler)

    def __init__(self):
        super(PulseCalAna, self).__init__()
        self.on_trait_change(self._update_fit, 'spin_state', dispatch='ui')

    def _update_fit(self, s):
        """compute relative rms"""
        m = s.mean()
        rms = 2 * (np.sum((s - m) ** 2) / len(s)) ** 0.5 / float(s.max() - s.min())
        
        self.rms = rms
        self.line_plot.overlays[0].text = 'rms: %.2f' % rms

    # overwrite the line_plot such that the x-axis label is time 
    def _create_line_plot(self):
        line_data = ArrayPlotData(index=np.array((0, 1)), spin_state=np.array((0, 0)),)
        plot = Plot(line_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('index', 'spin_state'), color='blue', name='spin_state')
        plot.plot(('index', 'spin_state'), type='scatter', name='pulses')
        plot.index_axis.title = 'pulse #'
        plot.value_axis.title = 'spin state'
        plot.overlays.insert(0, PlotLabel(text='', hjustify='left', vjustify='bottom', position=[70, 40]))
        self.line_data = line_data
        self.line_plot = plot


#####################################################################
# old fitting code
#####################################################################

class PulsedFit(HasTraits, GetSetItemsMixin):

    """
    Base class for a pulsed fit. Provides calculation of normalized intensity.
    Derive from this to create fits for pulsed measurements.
    """

    measurement = Instance(mp.Pulsed, factory=mp.Pulsed)
    
    pulse = Array(value=np.array((0., 0.)))
    flank = Float(value=0.0)
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
        old_mesh = self.line_data.get_data('pulse_number')
        if old_mesh is not None and len(old_mesh) != len(self.spin_state):
            self.line_data.set_data('pulse_number', np.arange(len(self.spin_state)))
        self.line_data.set_data('spin_state', self.spin_state)

    traits_view = View(HGroup(Item('integration_width'),
                              Item('position_signal'),
                              Item('position_normalize'),
                              ),
                       title='Pulsed Fit',
                       )

    get_set_items = ['__doc__', 'integration_width', 'position_signal', 'position_normalize', 'pulse', 'flank', 'spin_state', 'spin_state_error']

    
class PulsedFitErrorBars(PulsedFit):

    """
    Adds error bars to PulsedFit.
    """

    # The following is a bit tricky. Data for plots is passed to the PulsedAnalyzer through these two attributes.
    # The first one is a standard chaco ArrayPlotData instance. It is very important to specify a proper initial instance,
    # otherwise the PulsedAnalyzer will not start. The line below specifies an initial instance through the factory and kw argument.
    line_data = Instance(ArrayPlotData,
                         factory=ArrayPlotData,
                         kw={'pulse_number':np.array((0, 1)),
                             'spin_state':np.array((0, 0)),
                             'spin_state_low':np.array((0, 0)),
                             'spin_state_high':np.array((0, 0))
                             }
                         )
    # The second line is a list that is interpreted to create appropriate plots through the chaco Plot.plot() method.
    # The list elements are dictionaries that are directly passed to the Plot.plot() command as keyword arguments through the **kwagrs expansion 
    plots = [{'data':('pulse_number', 'spin_state'), 'color':'blue', 'name':'pulsed'},
             {'data':('pulse_number', 'spin_state_low'), 'type':'scatter', 'marker':'dot', 'marker_size':1.0, 'marker_color':'black', 'name':'high'},
             {'data':('pulse_number', 'spin_state_high'), 'type':'scatter', 'marker':'dot', 'marker_size':1.0, 'marker_color':'black', 'name':'low'} ]
        
    def __init__(self):
        super(PulsedFitErrorBars, self).__init__()
        self.on_trait_change(self.update_plot_error_bars, 'spin_state_error', dispatch='ui')

    def update_plot_error_bars(self):
        self.line_data.set_data('spin_state_low', self.spin_state - self.spin_state_error)
        self.line_data.set_data('spin_state_high', self.spin_state + self.spin_state_error)

    traits_view = View(HGroup(Item('integration_width'),
                              Item('position_signal'),
                              Item('position_normalize'),
                              ),
                       title='Error Bar Fit',
                       )

    get_set_items = ['__doc__', 'integration_width', 'position_signal', 'position_normalize', 'pulse', 'flank', 'spin_state', 'spin_state_error']
    
class PulsedFitTau(PulsedFit):

    measurement = Instance(mp.Pulsed, factory=mp.Rabi)

    plots = [{'data':('tau', 'spin_state'), 'color':'blue', 'name':'rabi'}]
    
    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'spin_state':np.array((0, 0))})

    def __init__(self):
        super(PulsedFitTau, self).__init__()
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')

    def update_plot_spin_state(self):
        self.line_data.set_data('spin_state', self.spin_state)

    def update_plot_tau(self):
        if self.measurement is None:
            return
        self.line_data.set_data('tau', self.measurement.tau)

    traits_view = View(HGroup(Item('integration_width'),
                              Item('position_signal'),
                              Item('position_normalize'),
                              ),
                       title='Pulsed Fit Tau',
                       )

class PulsedFitTauRef(PulsedFitTau):

    """Provides plotting of measurements with time index and 2 ref points at the end of the sequence."""

    plots = [{'data':('tau', 'spin_state'), 'color':'blue', 'name':'rabi'},
             {'data':('tau', 'bright'), 'color':'red', 'name':'bright'},
             {'data':('tau', 'dark'), 'color':'black', 'name':'dark'}]

    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'spin_state':np.array((0, 0)),
                                                                    'bright':np.array((1, 1)),
                                                                    'dark':np.array((0, 0))})
    
    def update_plot_spin_state(self):
        n = len(self.spin_state) - 2
        self.line_data.set_data('spin_state', self.spin_state[:-2])
        self.line_data.set_data('bright', self.spin_state[-2] * np.ones(n))
        self.line_data.set_data('dark', self.spin_state[-1] * np.ones(n))

    def update_plot_tau(self):
        self.line_data.set_data('tau', self.measurement.tau)
    
    traits_view = View(HGroup(Item('integration_width'),
                              Item('position_signal'),
                              Item('position_normalize'),
                              ),
                       title='Tau with bright/dark Ref',
                       )

    get_set_items = ['integration_width', 'position_signal', 'position_normalize', 'pulse', 'flank', 'spin_state']

class DoubleFitTauRef(PulsedFit):

    """Provides plotting of double sequence measurements with time index and 2 ref points at the end of the sequence."""

    measurement = Instance(mp.Pulsed, factory=mp.CPMGxy)

    plots = [{'data':('tau', 'first'), 'color':'blue', 'name':'first'},
             {'data':('tau', 'second'), 'color':'green', 'name':'second'},
             {'data':('tau', 'bright'), 'color':'red', 'name':'bright'},
             {'data':('tau', 'dark'), 'color':'black', 'name':'dark'}]

    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'first':np.array((0, 0)),
                                                                    'second':np.array((0, 0)),
                                                                    'bright':np.array((1, 1)),
                                                                    'dark':np.array((0, 0))})
    def __init__(self):
        super(DoubleFitTauRef, self).__init__()
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')

    def update_plot_tau(self):
        self.line_data.set_data('tau', self.measurement.tau)

    def update_plot_spin_state(self):
        spin_state = self.spin_state
        n_ref = self.measurement.n_ref
        n = len(spin_state) / 2 - n_ref
        self.line_data.set_data('first', spin_state[:n])
        self.line_data.set_data('second', spin_state[n:-2 * n_ref])
        self.line_data.set_data('bright', np.mean(self.spin_state[-2 * n_ref:-n_ref]) * np.ones(n))
        self.line_data.set_data('dark', np.mean(self.spin_state[-n_ref:]) * np.ones(n))

    def update_plot_tau(self):
        self.line_data.set_data('tau', self.measurement.tau)
    
    traits_view = View(HGroup(Item('integration_width'),
                              Item('position_signal'),
                              Item('position_normalize'),
                              ),
                       title='Double Tau with bright/dark Ref',
                       )

    get_set_items = ['integration_width', 'position_signal', 'position_normalize', 'pulse', 'flank', 'spin_state']

class DoubleFitRef(PulsedFit):

    """Provides plotting of double sequence measurements with and bright / dark ref line."""

    measurement = Instance(mp.Pulsed, factory=mp.Pi2PiCal)

    plots = [{'data':('pulse_number', 'first'), 'color':'blue', 'name':'first'},
             {'data':('pulse_number', 'second'), 'color':'green', 'name':'second'},
             {'data':('pulse_number', 'bright'), 'color':'red', 'name':'bright'},
             {'data':('pulse_number', 'dark'), 'color':'black', 'name':'dark'}]

    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'pulse_number':np.array((0, 1)),
                                                                    'first':np.array((0, 0)),
                                                                    'second':np.array((0, 0)),
                                                                    'bright':np.array((1, 1)),
                                                                    'dark':np.array((0, 0))})
    
    def update_plot_spin_state(self):
        old_mesh = self.line_data.get_data('pulse_number')
        spin_state = self.spin_state
        n_ref = self.measurement.n_ref
        n = len(spin_state) / 2 - n_ref
        if old_mesh is not None and len(old_mesh) != n: # ToDo: there could be a tau mesh with length n
            self.line_data.set_data('pulse_number', np.arange(n))
        self.line_data.set_data('first', spin_state[:n])
        self.line_data.set_data('second', spin_state[n:-2 * n_ref])
        self.line_data.set_data('bright', np.mean(self.spin_state[-2 * n_ref:-n_ref]) * np.ones(n)) # ToDo: better use h_line
        self.line_data.set_data('dark', np.mean(self.spin_state[-n_ref:]) * np.ones(n)) # ToDo: better use h_line

    traits_view = View(HGroup(Item('integration_width'),
                              Item('position_signal'),
                              Item('position_normalize'),
                              ),
                       title='Double Sequence with bright/dark Ref',
                       )

    get_set_items = ['integration_width', 'position_signal', 'position_normalize', 'pulse', 'flank', 'spin_state']


"""
class RabiFit( PulsedFit ):

    Provides fits and plots for a Rabi measurement.

    measurement = Instance( Pulsed, factory=Rabi )
    fit_result = Property( depends_on='spin_state_error' )

    period          = Property( depends_on='fit_result', label='period' )
    offset          = Property( depends_on='fit_result', label='offset' )
    contrast        = Property( depends_on='fit_result', label='contrast' )
    q               = Property( depends_on='fit_result', label='q' )
    t_pi2           = Property( depends_on='fit_result', label='pi/2' )
    t_pi            = Property( depends_on='fit_result', label='pi' )
    t_3pi2          = Property( depends_on='fit_result', label='3pi/2' )

    def __init__(self):
        PulsedFit.__init__(self)
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')
        self.on_trait_change(self.update_plot_fit, 'fit_result', dispatch='ui')

    @cached_property
    def _get_fit_result(self):
        try:
            return fitting.fit_rabi( self.measurement.tau, self.spin_state, self.spin_state_error )
        except:
            return (np.NaN*np.zeros(4), np.NaN*np.zeros((4,4)), np.NaN, np.NaN)
    
    @cached_property
    def _get_period(self):
        return self.fit_result[0][1], abs(self.fit_result[1][1,1])**0.5

    @cached_property
    def _get_offset(self):
        return self.fit_result[0][2], abs(self.fit_result[1][2,2])**0.5

    @cached_property
    def _get_contrast(self):
        a = self.fit_result[0][0]
        c = self.fit_result[0][3]
        va = self.fit_result[1][0,0]
        vc = self.fit_result[1][3,3]
        return 200*a/(c+a), 200./c**2*(va*c**2+vc*a**2)**0.5

    @cached_property
    def _get_q(self):
        return self.fit_result[2]

    @cached_property
    def _get_t_pi2(self):
        a, T, x0, c = self.fit_result[0]
        return 0.25*T + x0, 0.25*abs(self.fit_result[1][1,1])**0.5 + abs(self.fit_result[1][2,2])**0.5

    @cached_property
    def _get_t_pi(self):
        a, T, x0, c = self.fit_result[0]
        return 0.5*T + x0, 0.25*abs(self.fit_result[1][1,1])**0.5 + abs(self.fit_result[1][2,2])**0.5
    
    @cached_property
    def _get_t_3pi2(self):
        a, T, x0, c = self.fit_result[0]
        return 0.75*T+x0, 0.75*abs(self.fit_result[1][1,1])**0.5 + abs(self.fit_result[1][2,2])**0.5
    
    plots = [{'data':('tau','spin_state'), 'color':'blue', 'name':'rabi'},
             {'data':('tau','fit'),         'color':'red',  'name':'cos fit'} ]
    
    line_data = Instance( ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0,1)),
                                                                    'spin_state':np.array((0,0)),
                                                                    'fit':np.array((0,0))} )

    def update_plot_spin_state(self):
        self.line_data.set_data('spin_state',self.spin_state)

    def update_plot_tau(self):
        self.line_data.set_data('tau',self.measurement.tau)

    def update_plot_fit(self):
        if self.fit_result[0][0] is not np.NaN:
            self.line_data.set_data('fit',fitting.Cosinus(*self.fit_result[0])(self.measurement.tau))
    
    traits_view = View(Tabbed(VGroup(HGroup(Item('contrast',  style='readonly', width=-100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.1f+-%.1f%%'%x)),
                                            Item('period',    style='readonly', width=-100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f'%x)),
                                            Item('offset',    style='readonly', width=-100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f'%x)),
                                            Item('q',         style='readonly', width=-100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e')%x)),
                                     ),
                                     HGroup(Item('t_pi2',     style='readonly', width=-100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f'%x)),
                                            Item('t_pi',      style='readonly', width=-100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f'%x)),
                                            Item('t_3pi2',    style='readonly', width=-100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f'%x)),
                                     ),
                                     label='fit_parameter'
                              ),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     label='settings'),),)

    get_set_items = PulsedFit.get_set_items + ['fit_result','contrast','period','offset','t_pi2','t_pi','t_3pi2']
"""

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
        self.line_data.set_data('tau', self.measurement.tau)

    def update_plot_fit(self):
        if self.fit_result[0][0] is not np.NaN:
            self.line_data.set_data('fit', fitting.Cosinus(*self.fit_result[0])(self.measurement.tau))            
    
    traits_view = View(Tabbed(VGroup(HGroup(Item('contrast', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.1f+-%.1f%%' % x)),
                                            Item('period', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('q', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x)),
                                     ),
                                     HGroup(Item('t_pi2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('t_pi', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('t_3pi2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
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

class PulseExtraction(PulsedFit):

    """Provides extraction of pulses from a Rabi measurement."""

    measurement = Instance(mp.Pulsed, factory=mp.Rabi)
        
    period = Float()
    contrast = Float()
    t_pi2 = Array()
    t_pi = Array()
    t_3pi2 = Array()
    t_2pi = Array()
    
    def __init__(self):
        super(PulseExtraction, self).__init__()
        self.on_trait_change(self.update_fit, 'spin_state', dispatch='ui')
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')

    def update_fit(self):
        # try to extract pulses from spin_state and tau.
        tau = self.measurement.tau
        y = self.spin_state
        f, r, p, tp = fitting.extract_pulses(y)
        pi2 = tau[f]
        pi = tau[p]
        three_pi2 = tau[r]
        two_pi = tau[tp]

        # compute some relevant parameters from the result
        mi = self.spin_state.min()
        ma = self.spin_state.max()
        contrast = 100 * (ma - mi) / ma
        # simple rough estimate of the period to avoid index out of range Error
        T = 4 * (pi[0] - pi2[0])
        
        # set respective attributes
        self.period = T
        self.contrast = contrast
        self.t_pi2 = pi2
        self.t_pi = pi
        self.t_3pi2 = three_pi2
        self.t_2pi = two_pi

        # create a summary of the result as a text string
        s = 'contrast: %.1f\n' % contrast
        s += 'period: %.2f ns\n' % T
        
        self.text = s
        
        # markers in the plot that show the result of the pulse extraction
        self.line_data.set_data('pulse_indices', np.hstack((pi2, pi, three_pi2, two_pi)))            
        self.line_data.set_data('pulse_values', np.hstack((y[f], y[p], y[r], y[tp])))

    plots = [{'data':('tau', 'spin_state'), 'color':'blue', 'name':'rabi'},
             {'data':('pulse_indices', 'pulse_values'), 'type':'scatter', 'marker':'circle', 'color':'none', 'outline_color':'red', 'line_width':1.0, 'name':'pulses'} ]
    
    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'spin_state':np.array((0, 0)),
                                                                    'pulse_indices':np.array((0, 0)),
                                                                    'pulse_values':np.array((0, 0))})

    def update_plot_spin_state(self):
        self.line_data.set_data('spin_state', self.spin_state)
    def update_plot_tau(self):
        self.line_data.set_data('tau', self.measurement.tau)
    
    traits_view = View(Tabbed(VGroup(HGroup(Item('contrast', style='readonly'),
                                            Item('period', style='readonly'),
                                     ),
                                     HGroup(Item('t_pi2', style='readonly'),
                                            Item('t_pi', style='readonly'),
                                            Item('t_3pi2', style='readonly'),
                                            Item('t_2pi', style='readonly'),
                                     ),
                                     label='fit_parameter'
                              ),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     label='settings'),
                              ),
                       title='Pulse Extraction',
                       )

    get_set_items = PulsedFit.get_set_items + ['contrast', 'period', 't_pi2', 't_pi', 't_3pi2', 't_2pi', 'text']

class PulseCalFit(PulsedFit):

    """Provides calculation of RMS from PulseCal  measurement."""

    measurement = Instance(mp.Pulsed, factory=mp.Rabi)
            
    rms = Float()
    
    text = Str('')
    
    

    plots = [ {'data':('pulse_number', 'spin_state'), 'color':'blue', 'name':'pulsed'},
              {'data':('pulse_number', 'spin_state'), 'type':'scatter'}, ]
    
    def __init__(self):
        super(PulseCalFit, self).__init__()
        self.on_trait_change(self.update_fit, 'spin_state', dispatch='ui')

    def update_fit(self):
        """compute relative rms"""
        s = self.spin_state
        m = s.mean()
        rms = 2 * (np.sum((s - m) ** 2) / len(s)) ** 0.5 / float(s.max() - s.min())
        
        self.rms = rms
        self.text = 'rms: %.2f' % rms
        
    get_set_items = PulsedFit.get_set_items + ['rms', 'text']

    traits_view = View(HGroup(Item('integration_width'),
                              Item('position_signal'),
                              Item('position_normalize'),
                              Item('rms', style='readonly'),
                              ),
                       title='Pulse Cal Fit',
                       )

class ExponentialFit(PulsedFit):

    """Provides an exponential fit."""

    measurement = Instance(mp.Pulsed, factory=mp.T1)
    fit_parameters = Property(trait=Array, depends_on='spin_state')
    T1 = Property(trait=Float, depends_on='fit_parameters')
    
    def __init__(self):
        super(ExponentialFit, self).__init__()
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')
        self.on_trait_change(self.update_plot_fit, 'fit_parameters', dispatch='ui')

    @cached_property
    def _get_fit_parameters(self):
        try:
            return fitting.fit(self.measurement.tau, self.spin_state, fitting.ExponentialZero, fitting.ExponentialZeroEstimator)
        except:
            return np.array((np.NaN, np.NaN, np.NaN))

    @cached_property
    def _get_T1(self):
        return self.fit_parameters[1]
    
    plots = [{'data':('tau', 'spin_state'), 'color':'blue', 'name':'rabi'},
             {'data':('tau', 'fit'), 'color':'red', 'name':'cos fit'} ]
    
    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'spin_state':np.array((0, 0)),
                                                                    'fit':np.array((0, 0))})

    def update_plot_spin_state(self):
        self.line_data.set_data('spin_state', self.spin_state)

    def update_plot_tau(self):
        self.line_data.set_data('tau', self.measurement.tau)

    def update_plot_fit(self):
        if self.fit_parameters[0] is not np.NaN:
            self.line_data.set_data('fit', fitting.ExponentialZero(*self.fit_parameters)(self.measurement.tau))
    
    traits_view = View(Tabbed(HGroup(Item('T1', style='readonly', width= -80),
                                     label='fit_parameter'),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     label='settings'),
                              ),
                       title='Exponential Fit',
                       )

    get_set_items = PulsedFit.get_set_items + ['fit_parameters', 'T1']

class HahnFit(PulsedFit):

    """Provides fits and plots for a Hahn-Echo measurement."""

    measurement = Instance(mp.Pulsed, factory=mp.Hahn)
    fit_result = Property(depends_on='spin_state_error')
    q = Property(depends_on='fit_result', label='q')
    T2 = Property(depends_on='fit_result')
    exponent = Property(depends_on='fit_result')
    
    def __init__(self):
        super(HahnFit, self).__init__()
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')
        self.on_trait_change(self.update_plot_fit, 'fit_result', dispatch='ui')

    @cached_property
    def _get_fit_result(self):
        try:
            return fitting.nonlinear_model(x, y, s, fitting.ExponentialPowerZero, fitting.ExponentialPowerZeroEstimator)
        except:
            return (np.NaN * np.zeros(4), np.NaN * np.zeros((4, 4)), np.NaN, np.NaN)
    
    @cached_property
    def _get_T2(self):
        return self.fit_result[0][1], abs(self.fit_result[1][1, 1]) ** 0.5 
    
    @cached_property
    def _get_exponent(self):
        return self.fit_result[0][2], abs(self.fit_result[1][2, 2]) ** 0.5 
    
    @cached_property
    def _get_q(self):
        return self.fit_result[2]

    plots = [{'data':('tau', 'spin_state'), 'color':'blue', 'name':'hahn'},
             {'data':('tau', 'fit'), 'color':'red', 'name':'exponential fit'} ]
    
    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'spin_state':np.array((0, 0)),
                                                                    'fit':np.array((0, 0))})

    def update_plot_spin_state(self):
        self.line_data.set_data('spin_state', self.spin_state)

    def update_plot_tau(self):
        self.line_data.set_data('tau', self.measurement.tau)

    def update_plot_fit(self):
        if self.fit_result[0][0] is not np.NaN:
            self.line_data.set_data('fit', fitting.ExponentialPowerZero(*self.fit_result[0])(self.measurement.tau))
    
    traits_view = View(Tabbed(HGroup(Item('T2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                     Item('exponent', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                     Item('q', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x)),
                                     label='fit_parameter'),
                              HGroup(Item('integration_width'),
                                     Item('position_signal'),
                                     Item('position_normalize'),
                                     label='settings'),
                              ),
                       title='Hahn Fit',
                       )

    get_set_items = PulsedFit.get_set_items + ['fit_result', 'T2']


class DoubleFit(PulsedFit):

    """Provides plotting for double sequence type measurements."""

    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'pulse_number':np.array((0, 1)),
                                                                    'first_sequence':np.array((0, 0)),
                                                                    'second_sequence':np.array((0, 0))})
    
    plots = [{'data':('pulse_number', 'first_sequence'), 'color':'blue', 'name':'first sequence'},
             {'data':('pulse_number', 'second_sequence'), 'color':'green', 'name':'second sequence'} ]
    
    def update_plot_spin_state(self):
        old_mesh = self.line_data.get_data('pulse_number')
        spin_state = self.spin_state
        n = len(spin_state) / 2
        if old_mesh is not None and len(old_mesh) != n:
            self.line_data.set_data('pulse_number', np.arange(n))
        self.line_data.set_data('first_sequence', spin_state[:n])
        self.line_data.set_data('second_sequence', spin_state[n:])
    
    traits_view = View(HGroup(Item('integration_width'),
                              Item('position_signal'),
                              Item('position_normalize'),
                              ),
                       title='Double Sequence Fit',
                       )

    get_set_items = ['integration_width', 'position_signal', 'position_normalize', 'pulse', 'flank', 'spin_state']

class DoubleFitTau(DoubleFit):

    """Provides plotting for double sequence type measurements with time index."""
    
    measurement = Instance(mp.Pulsed, factory=mp.FID3pi2)

    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'first_sequence':np.array((0, 0)),
                                                                    'second_sequence':np.array((0, 0))})

    plots = [{'data':('tau', 'first_sequence'), 'color':'blue', 'name':'first sequence'},
             {'data':('tau', 'second_sequence'), 'color':'green', 'name':'second sequence'} ]

    def __init__(self):
        DoubleFit.__init__(self)
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')

    def update_plot_spin_state(self):
        spin_state = self.spin_state
        n = len(spin_state) / 2
        self.line_data.set_data('first_sequence', spin_state[:n])
        self.line_data.set_data('second_sequence', spin_state[n:])

    def update_plot_tau(self):
        self.line_data.set_data('tau', self.measurement.tau)

class InterleavedDoubleTau(PulsedFit):

    """Provides plotting for interleaved double sequence type measurements with time bins."""

    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'first_sequence':np.array((0, 0)),
                                                                    'second_sequence':np.array((0, 0))})

    plots = [{'data':('tau', 'first_sequence'), 'color':'blue', 'name':'first sequence'},
             {'data':('tau', 'second_sequence'), 'color':'green', 'name':'second sequence'} ]

    measurement = Instance(mp.Pulsed, factory=mp.Rabi)

    def __init__(self):
        PulsedFit.__init__(self)
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')

    def update_plot_spin_state(self):
        old_mesh = self.line_data.get_data('pulse_number')
        spin_state = self.spin_state
        n = len(spin_state) / 2
        if old_mesh is not None and len(old_mesh) != n:
            self.line_data.set_data('pulse_number', np.arange(n))
        self.line_data.set_data('first_sequence', spin_state[::2])
        self.line_data.set_data('second_sequence', spin_state[1::2])
    
    def update_plot_tau(self):
        self.line_data.set_data('tau', self.measurement.tau)

    get_set_items = ['integration_width', 'position_signal', 'position_normalize', 'pulse', 'flank', 'spin_state']

    traits_view = View(HGroup(Item('integration_width'),
                              Item('position_signal'),
                              Item('position_normalize'),
                              ),
                       title='Interleaved Double Tau',
                       )

class Gaussian3pi2Fit(PulsedFit):

    """Provides Gaussian fits and plots for measurement with both pi/2 and 3pi/2 readout pulse."""

    measurement = Instance(mp.Pulsed, factory=mp.Hahn3pi2)

    fit_result_pi2 = Property(depends_on='spin_state_error')
    fit_result_3pi2 = Property(depends_on='spin_state_error')

    T2_pi2 = Property(depends_on='fit_result_pi2', label='T2 pi/2 [ns]')
    T2_3pi2 = Property(depends_on='fit_result_3pi2', label='T2 3pi/2 [ns]')

    q_pi2 = Property(depends_on='fit_result_pi2', label='q pi/2')
    q_3pi2 = Property(depends_on='fit_result_3pi2', label='q 3pi/2')

    def __init__(self):
        PulsedFit.__init__(self)
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')
        self.on_trait_change(self.update_plot_fit_pi2, 'fit_result_pi2', dispatch='ui')
        self.on_trait_change(self.update_plot_fit_3pi2, 'fit_result_3pi2', dispatch='ui')

    @cached_property
    def _get_fit_result_pi2(self):
        x = self.measurement.tau
        n = len(self.spin_state) / 2
        y = self.spin_state[:n]
        s = self.spin_state_error[:n]
        try:
            return fitting.nonlinear_model(x, y, s, fitting.GaussianZero, fitting.GaussianZeroEstimator)
        except: 
            return (np.NaN * np.zeros(3), np.NaN * np.zeros((3, 3)), np.NaN, np.NaN)  

    @cached_property
    def _get_fit_result_3pi2(self):
        x = self.measurement.tau
        n = len(self.spin_state) / 2
        y = self.spin_state[n:]
        s = self.spin_state_error[n:]
        try:
            return fitting.nonlinear_model(x, y, s, fitting.GaussianZero, fitting.GaussianZeroEstimator)
        except: 
            return (np.NaN * np.zeros(3), np.NaN * np.zeros((3, 3)), np.NaN, np.NaN)  

    @cached_property
    def _get_T2_pi2(self):
        return self.fit_result_pi2[0][1], abs(self.fit_result_pi2[1][1, 1]) ** 0.5 
    
    @cached_property
    def _get_T2_3pi2(self):
        return self.fit_result_3pi2[0][1], abs(self.fit_result_3pi2[1][1, 1]) ** 0.5 
    
    @cached_property
    def _get_q_pi2(self):
        return self.fit_result_pi2[2]

    @cached_property
    def _get_q_3pi2(self):
        return self.fit_result_3pi2[2]

    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'data_pi2':np.array((0, 0)),
                                                                    'data_3pi2':np.array((0, 0)),
                                                                    'fit_pi2':np.array((0, 0)),
                                                                    'fit_3pi2':np.array((0, 0)), })
    plots = [{'name':'data_pi2', 'data':('tau', 'data_pi2'), 'color':'blue'},
             {'name':'data_3pi2', 'data':('tau', 'data_3pi2'), 'color':'green'},
             {'name':'fit_pi2', 'data':('tau', 'fit_pi2'), 'color':'red'},
             {'name':'fit_3pi2', 'data':('tau', 'fit_3pi2'), 'color':'magenta'} ]

    def update_plot_tau(self):
        self.line_data.set_data('tau', self.measurement.tau)

    def update_plot_spin_state(self):
        spin_state = self.spin_state
        n = len(spin_state) / 2
        self.line_data.set_data('data_pi2', spin_state[:n])
        self.line_data.set_data('data_3pi2', spin_state[n:])
    
    def update_plot_fit_pi2(self):
        if self.fit_result_pi2[0][0] is not np.NaN:
            self.line_data.set_data('fit_pi2', fitting.GaussianZero(*self.fit_result_pi2[0])(self.measurement.tau))
    
    def update_plot_fit_3pi2(self):
        if self.fit_result_3pi2[0][0] is not np.NaN:
            self.line_data.set_data('fit_3pi2', fitting.GaussianZero(*self.fit_result_3pi2[0])(self.measurement.tau))
    
    traits_view = View(Tabbed(VGroup(HGroup(Item('T2_pi2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('q_pi2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x)),
                                     ),
                                     HGroup(Item('T2_3pi2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('q_3pi2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x)),
                                     ),
                                     label='fit results'
                              ),
                              VGroup(HGroup(Item('integration_width'),
                                            Item('position_signal'),
                                            Item('position_normalize'),
                                     ),
                                     label='settings'
                              ),
                       ),
                       title='Gaussian Fit 3pi2',
                  )

    get_set_items = PulsedFit.get_set_items + ['fit_result_pi2', 'fit_result_3pi2', 'T2_pi2', 'T2_3pi2']

class Hahn3pi2Fit(PulsedFit):

    """Provides fits and plots for Hahn-Echo measurement with both pi/2 and 3pi/2 readout pulse."""

    measurement = Instance(mp.Pulsed, factory=mp.Hahn3pi2)

    fit_result_pi2 = Property(depends_on='spin_state_error')
    fit_result_3pi2 = Property(depends_on='spin_state_error')

    T2_pi2 = Property(depends_on='fit_result_pi2', label='T2 pi/2 [ns]')
    T2_3pi2 = Property(depends_on='fit_result_3pi2', label='T2 3pi/2 [ns]')

    q_pi2 = Property(depends_on='fit_result_pi2', label='q pi/2')
    q_3pi2 = Property(depends_on='fit_result_3pi2', label='q 3pi/2')

    exponent_pi2 = Property(depends_on='fit_result_pi2', label='exponent pi/2')
    exponent_3pi2 = Property(depends_on='fit_result_3pi2', label='exponent 3pi/2')

    def __init__(self):
        PulsedFit.__init__(self)
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')
        self.on_trait_change(self.update_plot_fit_pi2, 'fit_result_pi2', dispatch='ui')
        self.on_trait_change(self.update_plot_fit_3pi2, 'fit_result_3pi2', dispatch='ui')

    @cached_property
    def _get_fit_result_pi2(self):
        x = self.measurement.tau
        n = len(self.spin_state) / 2
        y = self.spin_state[:n]
        s = self.spin_state_error[:n]
        try:
            return fitting.nonlinear_model(x, y, s, fitting.ExponentialPowerZero, fitting.ExponentialPowerZeroEstimator)
        except: 
            return (np.NaN * np.zeros(4), np.NaN * np.zeros((4, 4)), np.NaN, np.NaN)  

    @cached_property
    def _get_fit_result_3pi2(self):
        x = self.measurement.tau
        n = len(self.spin_state) / 2
        y = self.spin_state[n:]
        s = self.spin_state_error[n:]
        try:
            return fitting.nonlinear_model(x, y, s, fitting.ExponentialPowerZero, fitting.ExponentialPowerZeroEstimator)
        except: 
            return (np.NaN * np.zeros(4), np.NaN * np.zeros((4, 4)), np.NaN, np.NaN)  

    @cached_property
    def _get_T2_pi2(self):
        return self.fit_result_pi2[0][1], abs(self.fit_result_pi2[1][1, 1]) ** 0.5 
    
    @cached_property
    def _get_T2_3pi2(self):
        return self.fit_result_3pi2[0][1], abs(self.fit_result_3pi2[1][1, 1]) ** 0.5 
    
    @cached_property
    def _get_exponent_pi2(self):
        return self.fit_result_pi2[0][2], abs(self.fit_result_pi2[1][2, 2]) ** 0.5 
    
    @cached_property
    def _get_exponent_3pi2(self):
        return self.fit_result_3pi2[0][2], abs(self.fit_result_3pi2[1][2, 2]) ** 0.5 
    
    @cached_property
    def _get_q_pi2(self):
        return self.fit_result_pi2[2]

    @cached_property
    def _get_q_3pi2(self):
        return self.fit_result_3pi2[2]

    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'data_pi2':np.array((0, 0)),
                                                                    'data_3pi2':np.array((0, 0)),
                                                                    'fit_pi2':np.array((0, 0)),
                                                                    'fit_3pi2':np.array((0, 0)), })
    plots = [{'name':'data_pi2', 'data':('tau', 'data_pi2'), 'color':'blue'},
             {'name':'data_3pi2', 'data':('tau', 'data_3pi2'), 'color':'green'},
             {'name':'fit_pi2', 'data':('tau', 'fit_pi2'), 'color':'red'},
             {'name':'fit_3pi2', 'data':('tau', 'fit_3pi2'), 'color':'magenta'} ]

    def update_plot_tau(self):
        self.line_data.set_data('tau', self.measurement.tau)

    def update_plot_spin_state(self):
        spin_state = self.spin_state
        n = len(spin_state) / 2
        self.line_data.set_data('data_pi2', spin_state[:n])
        self.line_data.set_data('data_3pi2', spin_state[n:])
    
    def update_plot_fit_pi2(self):
        if self.fit_result_pi2[0][0] is not np.NaN:
            self.line_data.set_data('fit_pi2', fitting.ExponentialPowerZero(*self.fit_result_pi2[0])(self.measurement.tau))
    
    def update_plot_fit_3pi2(self):
        if self.fit_result_3pi2[0][0] is not np.NaN:
            self.line_data.set_data('fit_3pi2', fitting.ExponentialPowerZero(*self.fit_result_3pi2[0])(self.measurement.tau))
    
    traits_view = View(Tabbed(VGroup(HGroup(Item('T2_pi2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('exponent_pi2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('q_pi2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x)),
                                     ),
                                     HGroup(Item('T2_3pi2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('exponent_3pi2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('q_3pi2', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x)),
                                     ),
                                     label='fit results'
                              ),
                              VGroup(HGroup(Item('integration_width'),
                                            Item('position_signal'),
                                            Item('position_normalize'),
                                     ),
                                     label='settings'
                              ),
                       ),
                       title='Hahn-Echo Fit 3pi2',
                  )

    get_set_items = PulsedFit.get_set_items + ['fit_result_pi2', 'fit_result_3pi2', 'T2_pi2', 'T2_3pi2', 'exponent_pi2', 'exponent_3pi2']

class T1piFit(PulsedFit):

    """Provides fits and plots for T1 measurement with pi pulse. TODO: NOT YET IMPLEMENTED CORRECTLY"""

    measurement = Instance(mp.Pulsed, factory=mp.T1pi)

    fit_result_nopi = Property(depends_on='spin_state_error')
    fit_result_pi = Property(depends_on='spin_state_error')

    T1_nopi = Property(depends_on='fit_result_nopi', label='T1 nopi [ns]')
    T1_pi = Property(depends_on='fit_result_pi', label='T1 pi [ns]')

    q_nopi = Property(depends_on='fit_result_nopi', label='q nopi')
    q_pi = Property(depends_on='fit_result_pi', label='q pi')

    def __init__(self):
        PulsedFit.__init__(self)
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')
        self.on_trait_change(self.update_plot_fit_nopi, 'fit_result_nopi', dispatch='ui')
        self.on_trait_change(self.update_plot_fit_pi, 'fit_result_pi', dispatch='ui')

    @cached_property
    def _get_fit_result_nopi(self):
        x = self.measurement.tau
        n = len(self.spin_state) / 2
        y = self.spin_state[:n]
        s = self.spin_state_error[:n]
        try:
            return fitting.nonlinear_model(x, y, s, fitting.ExponentialZero, fitting.ExponentialZeroEstimator)
        except: 
            return (np.NaN * np.zeros(3), np.NaN * np.zeros((3, 3)), np.NaN, np.NaN)  

    @cached_property
    def _get_fit_result_pi(self):
        x = self.measurement.tau
        n = len(self.spin_state) / 2
        y = self.spin_state[n:]
        s = self.spin_state_error[n:]
        try:
            return fitting.nonlinear_model(x, y, s, fitting.ExponentialZero, fitting.ExponentialZeroEstimator)
        except: 
            return (np.NaN * np.zeros(3), np.NaN * np.zeros((3, 3)), np.NaN, np.NaN)  

    @cached_property
    def _get_T1_nopi(self):
        return self.fit_result_nopi[0][1], abs(self.fit_result_nopi[1][1, 1]) ** 0.5 
    
    @cached_property
    def _get_T1_pi(self):
        return self.fit_result_pi[0][1], abs(self.fit_result_pi[1][1, 1]) ** 0.5 
    
    @cached_property
    def _get_q_nopi(self):
        return self.fit_result_nopi[2]

    @cached_property
    def _get_q_pi(self):
        return self.fit_result_pi[2]

    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'data_nopi':np.array((0, 0)),
                                                                    'data_pi':np.array((0, 0)),
                                                                    'fit_nopi':np.array((0, 0)),
                                                                    'fit_pi':np.array((0, 0)), })
    plots = [{'name':'data_nopi', 'data':('tau', 'data_nopi'), 'color':'blue'},
             {'name':'data_pi', 'data':('tau', 'data_pi'), 'color':'green'},
             {'name':'fit_nopi', 'data':('tau', 'fit_nopi'), 'color':'red'},
             {'name':'fit_pi', 'data':('tau', 'fit_pi'), 'color':'magenta'} ]

    def update_plot_tau(self):
        self.line_data.set_data('tau', self.measurement.tau)

    def update_plot_spin_state(self):
        spin_state = self.spin_state
        n = len(spin_state) / 2
        self.line_data.set_data('data_nopi', spin_state[:n])
        self.line_data.set_data('data_pi', spin_state[n:])
    
    def update_plot_fit_nopi(self):
        if self.fit_result_nopi[0][0] is not np.NaN:
            self.line_data.set_data('fit_nopi', fitting.ExponentialZero(*self.fit_result_nopi[0])(self.measurement.tau))
    
    def update_plot_fit_pi(self):
        if self.fit_result_pi[0][0] is not np.NaN:
            self.line_data.set_data('fit_pi', fitting.ExponentialZero(*self.fit_result_pi[0])(self.measurement.tau))
    
    traits_view = View(Tabbed(VGroup(HGroup(Item('T1_nopi', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('q_nopi', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x)),
                                     ),
                                     HGroup(Item('T1_pi', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('q_pi', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x)),
                                     ),
                                     label='fit results'
                              ),
                              VGroup(HGroup(Item('integration_width'),
                                            Item('position_signal'),
                                            Item('position_normalize'),
                                     ),
                                     label='settings'
                              ),
                       ),
                  )

    get_set_items = PulsedFit.get_set_items + ['fit_result_nopi', 'fit_result_pi', 'T1_nopi', 'T1_pi']   
    """exponent_pi exponent_nopi """


    fit_result_nopi = Property(depends_on='spin_state_error')
    fit_result_pi = Property(depends_on='spin_state_error')

    T1_nopi = Property(depends_on='fit_result_nopi', label='T1 nopi [ns]')
    T1_pi = Property(depends_on='fit_result_pi', label='T1 pi [ns]')

    q_nopi = Property(depends_on='fit_result_nopi', label='q nopi')
    q_pi = Property(depends_on='fit_result_pi', label='q pi')

    def __init__(self):
        PulsedFit.__init__(self)
        self.on_trait_change(self.update_plot_tau, 'measurement.tau', dispatch='ui')
        self.on_trait_change(self.update_plot_fit_nopi, 'fit_result_nopi', dispatch='ui')
        self.on_trait_change(self.update_plot_fit_pi, 'fit_result_pi', dispatch='ui')

    @cached_property
    def _get_fit_result_nopi(self):
        x = self.measurement.tau
        n = len(self.spin_state) / 2
        y = self.spin_state[:n]
        s = self.spin_state_error[:n]
        try:
            return fitting.nonlinear_model(x, y, s, fitting.ExponentialZero, fitting.ExponentialZeroEstimator)
        except: 
            return (np.NaN * np.zeros(3), np.NaN * np.zeros((3, 3)), np.NaN, np.NaN)  

    @cached_property
    def _get_fit_result_pi(self):
        x = self.measurement.tau
        n = len(self.spin_state) / 2
        y = self.spin_state[n:]
        s = self.spin_state_error[n:]
        try:
            return fitting.nonlinear_model(x, y, s, fitting.ExponentialZero, fitting.ExponentialZeroEstimator)
        except: 
            return (np.NaN * np.zeros(3), np.NaN * np.zeros((3, 3)), np.NaN, np.NaN)  

    @cached_property
    def _get_T1_nopi(self):
        return self.fit_result_nopi[0][1], abs(self.fit_result_nopi[1][1, 1]) ** 0.5 
    
    @cached_property
    def _get_T1_pi(self):
        return self.fit_result_pi[0][1], abs(self.fit_result_pi[1][1, 1]) ** 0.5 
    
    @cached_property
    def _get_q_nopi(self):
        return self.fit_result_nopi[2]

    @cached_property
    def _get_q_pi(self):
        return self.fit_result_pi[2]

    line_data = Instance(ArrayPlotData, factory=ArrayPlotData, kw={'tau':np.array((0, 1)),
                                                                    'data_nopi':np.array((0, 0)),
                                                                    'data_pi':np.array((0, 0)),
                                                                    'fit_nopi':np.array((0, 0)),
                                                                    'fit_pi':np.array((0, 0)), })
    plots = [{'name':'data_nopi', 'data':('tau', 'data_nopi'), 'color':'blue'},
             {'name':'data_pi', 'data':('tau', 'data_pi'), 'color':'green'},
             {'name':'fit_nopi', 'data':('tau', 'fit_nopi'), 'color':'red'},
             {'name':'fit_pi', 'data':('tau', 'fit_pi'), 'color':'magenta'} ]

    def update_plot_tau(self):
        self.line_data.set_data('tau', self.measurement.tau)

    def update_plot_spin_state(self):
        spin_state = self.spin_state
        n = len(spin_state) / 2
        self.line_data.set_data('data_nopi', spin_state[:n])
        self.line_data.set_data('data_pi', spin_state[n:])
    
    def update_plot_fit_nopi(self):
        if self.fit_result_nopi[0][0] is not np.NaN:
            self.line_data.set_data('fit_nopi', fitting.ExponentialZero(*self.fit_result_nopi[0])(self.measurement.tau))
    
    def update_plot_fit_pi(self):
        if self.fit_result_pi[0][0] is not np.NaN:
            self.line_data.set_data('fit_pi', fitting.ExponentialZero(*self.fit_result_pi[0])(self.measurement.tau))
    
    traits_view = View(Tabbed(VGroup(HGroup(Item('T1_nopi', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('q_nopi', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x)),
                                     ),
                                     HGroup(Item('T1_pi', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f+-%.2f' % x)),
                                            Item('q_pi', style='readonly', width= -100, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:(' %.3f' if x >= 0.001 else ' %.2e') % x)),
                                     ),
                                     label='fit results'
                              ),
                              VGroup(HGroup(Item('integration_width'),
                                            Item('position_signal'),
                                            Item('position_normalize'),
                                     ),
                                     label='settings'
                              ),
                       ),
                  )

    get_set_items = PulsedFit.get_set_items + ['fit_result_nopi', 'fit_result_pi', 'T1_nopi', 'T1_pi']
    
    """ exponent_pi exponent_nopi"""


#########################################
# Pulsed Analyzer Tool
#########################################

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
            
    #def save_all(self, info):
    #    filename = save_file(title='Save All')
    #    if filename is '':
    #        return
    #    else:
    #        info.object.save_all(filename)
            info.object.save_line_plot(filename)
            
    # new measurements

    def new_t1_measurement(self, info):
        info.object.measurement = mp.T1()
        #if pulsed.measurement.state=='run':
        #    logging.getLogger().exception(str(RuntimeError('Measurement running. Stop it and try again!')))
        #    raise RuntimeError('Measurement running. Stop it and try again!')
    def new_rabi_measurement(self, info):
        """info.object.measurement = mp.Rabi()"""
        info.object.measurement = newrabi.Rabi()
    def new_nmr_measurement(self, info):
        info.object.measurement = nmr.NMR()
    def new_nr_measurement(self, info):
        info.object.measurement = nr.NuclearRabi()
    def new_hahn_measurement(self, info):
        info.object.measurement = mp.Hahn()
    def new_opticalrabi_measurement(self, info):
        info.object.measurement = orabi.OpticalRabi()
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
    
    # new fits 

    def new_pulsed_fit(self, info):
        info.object.fit = PulsedFit()
    def new_pulsed_fit_tau(self, info):
        info.object.fit = PulsedFitTau()
    def new_fit_tau_ref(self, info):
        info.object.fit = PulsedFitTauRef()
    def new_rabi_fit(self, info):
        info.object.fit = RabiFit()
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
        
    # Menu of the PulsedAnalyzer, only include tested measurements and Fits please!
    
    traits_view = View(HGroup(VGroup(Item(name='measurement', style='custom', show_label=False),
                                     Item('matrix_plot', editor=ComponentEditor(), show_label=False, width=500, height=300, resizable=True),
                                     ),
                              VGroup(Item(name='fit', style='custom', show_label=False),
                                     Tabbed(Item('line_plot', name='normalized intensity', editor=ComponentEditor(), show_label=False, width=500, height=300, resizable=True),
                                            Item('pulse_plot', name='pulse profile', editor=ComponentEditor(), show_label=False, width=500, height=300, resizable=True),
                                            ),
                                     ),
                              ),
                       menubar=MenuBar(Menu(Action(action='save', name='Save (.pyd or .pys)'),
                                              Action(action='load', name='Load (.pyd or .pys)'),
                                              Action(action='save_matrix_plot', name='Save Matrix Plot (.png)'),
                                              Action(action='save_line_plot', name='Save Line Plot (.png)'),
                                  #TODO            Action(action='save_all', name='Save Line Plot (.png) + .pys'),
                                              Action(action='_on_close', name='Quit'),
                                              name='File'),
                                         Menu(Action(action='new_t1_measurement', name='T1'),
                                              Action(action='new_t1pi_measurement', name='T1pi'),
                                              Action(action='new_rabi_measurement', name='Rabi'),
                                              Action(action='new_nmr_measurement', name='NMR'),
                                              Action(action='new_opticalrabi_measurement', name='OpticalRabi'),
                                              Action(action='new_nr_measurement', name='NuclearRabi'),
                                              Action(action='new_PulseCal_measurement', name='PulseCal'),
                                              Action(action='new_hahn_measurement', name='Hahn'),
                                              Action(action='new_fid3pi2_measurement', name='FID3pi2'),
                                              Action(action='new_hahn3pi2_measurement', name='Hahn3pi2'),
                                              Action(action='new_deer_measurement', name='DEER'),
                                              Action(action='new_deer3pi2_measurement', name='deer3pi2'),
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
                                              name='Measurement'),
                                         Menu(Action(action='new_pulsed_fit', name='Pulsed'),
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
                                              name='Fit'),
                                 ),
                       title='PulsedAnalyzer', buttons=[], resizable=True, handler=PulsedAnalyzerHandler)

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
