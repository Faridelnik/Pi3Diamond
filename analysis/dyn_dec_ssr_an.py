"""
This module provides analysis of pulsed measurements of DDS + SSR.

The first part provides simple numeric functions.

The second part provides Trait GUIs
"""

import numpy as np
from fitting import find_edge, run_sum


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

# def spin_state(c, dt, T, t0=0.0, t1= -1.):
    
    # """
    # Compute the spin state from a 2D array of count data.
    
    # Parameters:
    
        # c    = count data
        # dt   = time step
        # t0   = beginning of integration window relative to the edge
        # t1   = None or beginning of integration window for normalization relative to edge
        # T    = width of integration window
        
    # Returns:
    
        # y       = 1D array that contains the spin state
        # profile = 1D array that contains the pulse profile
        # edge    = position of the edge that was found from the pulse profile
        
    # If t1<0, no normalization is performed. If t1>=0, each data point is divided by
    # the value from the second integration window and multiplied with the mean of
    # all normalization windows.
    # """

    # profile = c.sum(0)
    # edge = find_edge(profile)
    
    # I = int(round(T / float(dt)))
    # i0 = edge + int(round(t0 / float(dt)))
    # y = np.empty((c.shape[0],))
    # for i, slot in enumerate(c):
        # y[i] = slot[i0:i0 + I].sum()
    # if t1 >= 0:
        # i1 = edge + int(round(t1 / float(dt)))    
        # y1 = np.empty((c.shape[0],))
        # for i, slot in enumerate(c):
            # y1[i] = slot[i1:i1 + I].sum()
        # y = y / y1 * y1.mean()
    # return y, profile, edge


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

import measurements.dyn_decoupl_with_ssr as ds
import measurements.prog_gate_awg as pga
import measurements.pair_search as ps
import measurements.shallow_NV as sensing
import measurements.nmr as nmr
import measurements.singleshot as ss
#import measurements.opticalrabi as orabi
import measurements.nuclear_rabi as nr
import measurements.rabi as newrabi
from measurements.odmr import ODMR

class PulsedFit(HasTraits, GetSetItemsMixin):

    """
    Base class for a pulsed fit. Provides calculation of normalized intensity.
    Derive from this to create fits for pulsed measurements.
    """

    measurement = Instance(ds.Pulsed, factory=ds.Pulsed)
    
    x_tau = Array(value=np.array((0., 1.)))
    spin_state = Array(value=np.array((0., 0.)))
    spin_state_error = Array(value=np.array((0., 0.)))
    fit_parameters = Array(value=np.array((np.nan, np.nan, np.nan, np.nan)))
    
    normalized_counts=Array(value=np.array((0., 0.)))
    normalized_counts_error=Array(value=np.array((0., 0.)))
    free_evolution_time=Array(value=np.array((0., 1.)))
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
        
        contrast=self.measurement.rabi_contrast
        #c1=self.spin_state1
        #c2=self.spin_state2
        #l=(c1+c2)/2.
        #baseline=sum(l)/len(l)
        #C0_up=baseline/(1-0.01*contrast/2)
        #C0_down=C0_up*(1-0.01*contrast)
        #counts=c2-c1
        #self.normalized_counts=(counts)/(C0_up-C0_down)
        #self.normalized_counts_error=self.normalized_counts*(self.spin_state_error1/self.spin_state1)
        self.free_evolution_time=(2*self.x_tau+self.measurement.pi_1)
        
        self.normalized_counts=y
        

    line_data = Instance(ArrayPlotData,
                         factory=ArrayPlotData,
                         kw={'tau':np.array((0, 1)),
                             'spin_state':np.array((0, 0)),
                             #'spin_state2':np.array((0, 0)),
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
    plots = [ {'data':('tau', 'spin_state'), 'color':'blue','name':'pulsed'},
              #{'data':('tau', 'spin_state2'), 'color':'green', 'name':'pulsed2'},
              {'data':('x', 'y'), 'color':'black','name':'processed'},
            ]
        
    def update_plot_spin_state(self):
              
        old_mesh = self.x_tau
        #old_mesh = self.line_data.get_data('pulse_number')
        if old_mesh is not None and len(old_mesh) != len(self.spin_state):
            self.line_data.set_data('tau', np.arange(len(self.spin_state)))
            self.processed_plot_data.set_data('y', np.arange(len(self.spin_state)))
            #self.line_data2.set_data('pulse_number', np.arange(len(self.spin_state1)))
        self.line_data.set_data('spin_state', self.spin_state)
        #self.line_data.set_data('spin_state2', self.spin_state2)
        
        self.line_data.set_data('tau', old_mesh)
        #self.line_data2.set_data('spin_state', self.spin_state2)
        #self.line_data2.set_data('pulse_number', old_mesh)
        
        self.processed_plot_data.set_data('x', self.free_evolution_time)
        self.processed_plot_data.set_data('y', self.normalized_counts)
        

    traits_view = View(title='Pulsed Fit')

    get_set_items = ['__doc__', 'spin_state', 'spin_state_error']
    
class SSRAnalyzerHandler(GetSetItemsHandler):

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
            
    def new_odmr_fit_XY8(self, info):
        info.object.fit = OdmrFitXY8()
    
    def new_SSR_XY8_measurement(self, info):
        info.object.measurement = ds.XY8_with_SSR()

menubar = MenuBar(Menu(Action(action='save', name='Save (.pyd or .pys)'),
                   Action(action='load', name='Load (.pyd or .pys)'),
                   Action(action='save_matrix_plot', name='Save Matrix Plot (.png)'),
                   Action(action='save_line_plot', name='Save Line Plot (.png)'),
                   Action(action='save_all', name='Save All'),
                   Action(action='_on_close', name='Quit'),
                   name='File'
                   ),
              )
      
        
class SSRAnalyzer(HasTraits, GetSetItemsMixin):
    
    """Fits a pulsed measurement with a pulsed fit.
    
    Example of usage:

        a = PulsedAnalyzer()
        a.measurement = Rabi
        a.fit = RabiFit()
        a.edit_traits()
        
    """
    
    measurement = Instance(ds.Pulsed, factory=ds.Pulsed)
    fit = Instance(PulsedFit, factory=PulsedFit)
    
    #matrix_plot_data = Instance(ArrayPlotData)
    #pulse_plot_data = Instance(ArrayPlotData)
    
    #matrix_plot = Instance(Plot)
    #pulse_plot = Instance(Plot)
    line_plot = Instance(Plot)
    processed_plot = Instance(Plot)
    
    # line_data and processed_plot_data are provided by fit class

    def __init__(self):
        super(SSRAnalyzer, self).__init__()
        #self.on_trait_change(self.refresh_matrix_axis, 'measurement.time_bins, measurement.n_laser', dispatch='ui')
        #self.on_trait_change(self.refresh_matrix, 'measurement.count_data', dispatch='ui')
        #self.on_trait_change(self.refresh_pulse, 'fit.pulse', dispatch='ui')
        #self.on_trait_change(self.refresh_time_bins, 'measurement.time_bins', dispatch='ui')
        
        #self.on_trait_change(self.refresh_evol_time, 'fit.free_evolution_time', dispatch='ui')
        #self.on_trait_change(self.refresh_norm_counts, 'measurement.count_data, fit.normalized_counts', dispatch='ui')
        #self.on_trait_change(self.refresh_plot_fit, 'fit.fit_parameters', dispatch='ui')
    
        #self.on_trait_change(self.refresh_flank, 'fit.flank', dispatch='ui')
        
    def _measurement_changed(self, new):
        self.fit = PulsedFit()
    
    # def _matrix_plot_data_default(self):
        # return ArrayPlotData(image=np.zeros((self.measurement.n_laser, self.measurement.n_bins)))
        
    # def _pulse_plot_data_default(self):
        # return ArrayPlotData(x=self.measurement.time_bins, y=self.fit.pulse)
        
    def _processed_plot_data_default(self): 
        return ArrayPlotData(x=self.fit.free_evolution_time, y=self.fit.normalized_counts, fit=np.array((0., 0.)))
    
    # def _matrix_plot_default(self):
        # plot = Plot(self.matrix_plot_data, width=500, height=500, resizable='hv', padding=8, padding_left=48, padding_bottom=36)
        # plot.index_axis.title = 'time [ns]'
        # plot.value_axis.title = 'laser pulse'
        # plot.img_plot('image',
                      # xbounds=(self.measurement.time_bins[0], self.measurement.time_bins[-1]),
                      # ybounds=(0, self.measurement.n_laser),
                      # colormap=Spectral)[0]
        # return plot
        
    # def _pulse_plot_default(self):
        # plot = Plot(self.pulse_plot_data, padding=8, padding_left=64, padding_bottom=36)
        # plot.plot(('x', 'y'), style='line', color='blue', name='data')
        # edge_marker = LinePlot(index=ArrayDataSource(np.array((0, 0))),
                               # value=ArrayDataSource(np.array((0, 1e9))),
                               # color='red',
                               # index_mapper=LinearMapper(range=plot.index_range),
                               # value_mapper=LinearMapper(range=plot.value_range),
                               # name='marker')
        # plot.add(edge_marker)
        # plot.index_axis.title = 'time [ns]'
        # plot.value_axis.title = 'intensity'
        # return plot
        
    def _line_plot_default(self):
        plot = Plot(self.fit.line_data, padding=8, padding_left=64, padding_bottom=36)
        for item in self.fit.plots[0:1]:
            plot.plot(**item)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'spin state'
        return plot
        
    def _processed_plot_default(self):
        plot = Plot(self.fit.processed_plot_data, padding=8, padding_left=64, padding_bottom=36)
        for item in self.fit.plots[1:]:
            plot.plot(**item)
        plot.index_axis.title = 'free evolution time [ns]'
        plot.value_axis.title = 'spin state'
        return plot

    #def refresh_matrix_axis(self):
        #self.matrix_plot.components[0].index.set_data((self.measurement.time_bins[0], self.measurement.time_bins[-1]), (0.0, float(self.measurement.n_laser)))
        
    #def refresh_matrix(self):
        #s = self.measurement.count_data.shape
        #if not s[0] * s[1] > 1000000:
            #self.matrix_plot_data.set_data('image', self.measurement.count_data)

    #def refresh_pulse(self):
        #self.pulse_plot_data.set_data('y', self.fit.pulse)
    
    #def refresh_time_bins(self):
        #self.pulse_plot_data.set_data('x', self.measurement.time_bins)
    
    #def refresh_flank(self):
        #self.pulse_plot.components[1].index.set_data(np.array((self.fit.flank, self.fit.flank)))
        
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
        for key in plot.plots.keys()[0:1]:
            plot.delplot(key)
        for key in plot2.plots.keys()[1:]:
            plot2.delplot(key)   
        
        # set new data source
        plot.data = self.fit.line_data
        plot2.data = self.fit.processed_plot_data
                
        # make new plots
        for item in self.fit.plots[0:1]:
            plot.plot(**item)
            
        for item in self.fit.plots[1:]:
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
    
    #def save_matrix_plot(self, filename):
        #self.save_figure(self.matrix_plot, filename)
    
    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename)
        
    def save_processed_plot(self, filename):
        self.save_figure(self.processed_plot, filename)
        
    def save_all(self, filename):
        self.save_line_plot(filename + '_LinePlot.png')
        #self.save_matrix_plot(filename + '_MatrixPlot.png')
        self.save_processed_plot(filename + '_ProcPlot.png')
        self.save(filename + '.pyd')
        
    # Menu of the PulsedAnalyzer, only include tested measurements and Fits please!
    
    traits_view = View(HGroup(VGroup(Item(name='measurement', style='custom', show_label=False),
                                     Item('line_plot', name='normalized intensity', editor=ComponentEditor(), show_label=False, width=300, height=300, resizable=True), 
                                     ),
                              VGroup(Item(name='fit', style='custom', show_label=False),
                                     Tabbed(#Item('matrix_plot', editor=ComponentEditor(), show_label=False, width=300, height=300, resizable=True),
                                            #Item('pulse_plot', name='pulse profile', editor=ComponentEditor(), show_label=False, width=300, height=300, resizable=True),
                                            Item('processed_plot', name='normalized intensity2', editor=ComponentEditor(), show_label=False, width=300, height=300, resizable=True), 
                                            ),
                                     ),
                              ),
                       menubar=MenuBar(Menu(Action(action='save', name='Save (.pyd or .pys)'),
                                              Action(action='load', name='Load (.pyd or .pys)'),
                                              #Action(action='save_matrix_plot', name='Save Matrix Plot (.png)'),
                                              Action(action='save_line_plot', name='Save Line Plot (.png)'),
                                              Action(action='save_all', name='Save All'),
                                              Action(action='_on_close', name='Quit'),
                                              name='File'),
                                       Menu(Action(action='new_SSR_XY8_measurement', name='XY8 with SSR'),
                                              name='Measurement'),
                                       Menu(Action(action='new_odmr_fit_XY8', name='fit lorentzians'), 
                                              name='Fit'
                                              ), 
                                 ),
                       title='SSRAnalyzer', buttons=[], resizable=True, handler=SSRAnalyzerHandler)

    get_set_items = ['measurement', 'fit']
    get_set_order = ['measurement', 'fit'] 
    
class OdmrFitXY8(PulsedFit):

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

    get_set_items = PulsedFit.get_set_items + ['fit_parameters', 'fit_frequencies', 'fit_line_width', 'fit_contrast',  'text', 'fit_times']         
    
 