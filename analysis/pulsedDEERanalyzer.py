
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
                                 on_trait_change, cached_property, DelegatesTo, Any
from traitsui.api import View, Item, Tabbed, Group, HGroup, VGroup, VSplit, EnumEditor, TextEditor, InstanceEditor
from enable.api import ComponentEditor
from chaco.api import ArrayDataSource, LinePlot, LinearMapper, ArrayPlotData, Plot, Spectral, PlotLabel

from traitsui.file_dialog import save_file
from traitsui.menu import Action, Menu, MenuBar

import threading
import time
import logging

import fitting

from tools.utility import GetSetItemsHandler, GetSetItemsMixin

import measurements.DEER as dr

class PulsedFit(HasTraits, GetSetItemsMixin):

    """
    Base class for a pulsed fit. Provides calculation of normalized intensity.
    Derive from this to create fits for pulsed measurements.
    """

    measurement = Instance(dr.PulsedDEER, factory=dr.PulsedDEER)
    
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

class RabiFit_phase(PulsedFit):

    """Provides fits and plots for a Rabi measurement."""

    measurement = Instance(dr.PulsedDEER, factory=dr.Electron_Rabi)
    
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

    def new_rabi_measurement(self, info):
        """info.object.measurement = mp.Rabi()"""
        info.object.measurement = dr.Electron_Rabi()
    
    def new_deer_measurement(self, info):
        """info.object.measurement = mp.Rabi()"""
        info.object.measurement = dr.DEER()    
        
    # new fits 

    def new_pulsed_fit(self, info):
        info.object.fit = PulsedFit()
    def new_rabiphase_fit(self, info):
        info.object.fit = RabiFit_phase()
   

class PulsedAnalyzer(HasTraits, GetSetItemsMixin):
    
    """Fits a pulsed measurement with a pulsed fit.
    
    Example of usage:

        a = PulsedAnalyzer()
        a.measurement = Rabi
        a.fit = RabiFit()
        a.edit_traits()
        
    """
    
    measurement = Instance(dr.PulsedDEER, factory=dr.PulsedDEER)
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
                                  #TODO            Action(action='save_all', name='Save Line Plot (.png) + .pys'),
                                              Action(action='_on_close', name='Quit'),
                                              name='File'),
                                         Menu(Action(action='new_rabi_measurement', name='Electron Rabi'),
                                              Action(action='new_deer_measurement', name='DEER'),
                                              name='Measurement'),
                                         Menu(Action(action='new_pulsed_fit', name='Pulsed'),
                                              Action(action='new_rabiphase_fit', name='Rabi_phase'),
                                              name='Fit'),
                                 ),
                       title='Pulsed DEER Analyzer', buttons=[], resizable=True, handler=PulsedAnalyzerHandler)

    get_set_items = ['measurement', 'fit']
    get_set_order = ['measurement', 'fit']