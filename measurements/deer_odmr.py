import numpy as np

from traits.api import SingletonHasTraits, Trait, Instance, Property, String, Range, Float, Int, Bool, Array, Enum, Button, on_trait_change, cached_property, Code, List, NO_COMPARE
from traitsui.api import View, Item, HGroup, VGroup, VSplit, Tabbed, EnumEditor, TextEditor, Group
from enable.api import Component, ComponentEditor
from chaco.api import ArrayPlotData, Plot, Spectral, PlotLabel

from traitsui.file_dialog import save_file
from traitsui.menu import Action, Menu, MenuBar

import time
import threading
import logging

from tools.emod import ManagedJob

import hardware.api as ha

from analysis import fitting
from hardware.api import Microwave, Microwave_HMC 
from tools.utility import GetSetItemsHandler, GetSetItemsMixin

MW = Microwave()
MW_HMC = Microwave_HMC() 

class ODMRHandler(GetSetItemsHandler):

    def saveLinePlot(self, info):
        filename = save_file(title='Save Line Plot')
        if filename is '':
            return
        else:
            if filename.find('.png') == -1:
                filename = filename + '.png'
            info.object.save_line_plot(filename)

    def saveMatrixPlot(self, info):
        filename = save_file(title='Save Matrix Plot')
        if filename is '':
            return
        else:
            if filename.find('.png') == -1:
                filename = filename + '.png'
            info.object.save_matrix_plot(filename)
    
    def saveAll(self, info):
        filename = save_file(title='Save All')
        if filename is '':
            return
        else:
            info.object.save_all(filename)


class ODMR(ManagedJob, GetSetItemsMixin):
    """Provides ODMR measurements."""

    # starting and stopping
    keep_data = Bool(False) # helper variable to decide whether to keep existing data
    resubmit_button = Button(label='resubmit', desc='Submits the measurement to the job manager. Tries to keep previously acquired data. Behaves like a normal submit if sequence or time bins have changed since previous run.')    
    
    # measurement parameters
    power = Range(low= -100., high=25., value= -8, desc='Power [dBm]', label='Power [dBm]', mode='text', auto_set=False, enter_set=True)
    frequency_begin = Range(low=1, high=20e9, value=2.85e9, desc='Start Frequency [Hz]', label='Begin [Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.4e'))
    frequency_end = Range(low=1, high=20e9, value=2.90e9, desc='Stop Frequency [Hz]', label='End [Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.4e'))
    frequency_delta = Range(low=1e-3, high=20e9, value=1e6, desc='frequency step [Hz]', label='Delta [Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.4e'))
    t_pi_electrons = Range(low=1., high=100000., value=1000., desc='length of pi pulse [ns]', label='pi [ns]', mode='text', auto_set=False, enter_set=True)
    
    frequency_NV = Range(low=1, high=20e9, value=2.85e9, desc='NV frequency [Hz]', label='NV freq [Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.4e'))
    power_NV = Range(low= -100., high=25., value= -8, desc='Power NV [dBm]', label='Power NV [dBm]', mode='text', auto_set=False, enter_set=True)
    t_pi_NV = Range(low=1., high=100000., value=1000., desc='length of pi pulse NV [ns]', label='pi NV [ns]', mode='text', auto_set=False, enter_set=True)
    t_pi12_NV = Range(low=1., high=100000., value=1000., desc='length of pi half pulse [ns]', label='pi12 NV [ns]', mode='text', auto_set=False, enter_set=True)
    time_Hahn = Range(low=1., high=100000., value=1000., desc='time Hahn [ns]', label='time Hahn [ns]', mode='text', auto_set=False, enter_set=True)
    
    laser = Range(low=1., high=10000., value=300., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
    wait = Range(low=1., high=10000., value=1000., desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)
    seconds_per_point = Range(low=20e-3, high=1, value=20e-3, desc='Seconds per point', label='Seconds per point', mode='text', auto_set=False, enter_set=True)
    stop_time = Range(low=1., value=np.inf, desc='Time after which the experiment stops by itself [s]', label='Stop time [s]', mode='text', auto_set=False, enter_set=True)
    n_lines = Range (low=1, high=10000, value=50, desc='Number of lines in Matrix', label='Matrix lines', mode='text', auto_set=False, enter_set=True)
    
    # control data fitting
    perform_fit = Bool(False, label='perform fit')
    number_of_resonances = Trait('auto', String('auto', auto_set=False, enter_set=True), Int(10000., desc='Number of Lorentzians used in fit', label='N', auto_set=False, enter_set=True))
    threshold = Range(low= -99, high=99., value= -50., desc='Threshold for detection of resonances [%]. The sign of the threshold specifies whether the resonances are negative or positive.', label='threshold [%]', mode='text', auto_set=False, enter_set=True)
    
    # fit result
    fit_parameters = Array(value=np.array((np.nan, np.nan, np.nan, np.nan)))
    fit_frequencies = Array(value=np.array((np.nan,)), label='frequency [Hz]') 
    fit_line_width = Array(value=np.array((np.nan,)), label='line_width [Hz]') 
    fit_contrast = Array(value=np.array((np.nan,)), label='contrast [%]')

    # measurement data    
    frequency = Array()
    counts = Array()
    counts_matrix = Array()
    run_time = Float(value=0.0, desc='Run time [s]', label='Run time [s]')

    # plotting
    line_label = Instance(PlotLabel)
    line_data = Instance(ArrayPlotData)
    matrix_data = Instance(ArrayPlotData)
    line_plot = Instance(Plot, editor=ComponentEditor())
    matrix_plot = Instance(Plot, editor=ComponentEditor())
    
    #added by Farida, Magnetic field calculation
    central_dip_position = Float(value=0.0, desc='Centr freq [GHz]', label='Centr freq [GHz]', editor=TextEditor(auto_set=False, enter_set=False, evaluate=float, format_str='%.4e'))
    magnetic_field = Float(value=0.0, desc='Magnetic field [G]', label='Magnetic field [G]', editor=TextEditor(auto_set=False, enter_set=False, evaluate=float, format_str='%.1f'))
    crossing = Bool(False, label='crossing')
    

    def __init__(self):
        super(ODMR, self).__init__()
        self._create_line_plot()
        self._create_matrix_plot()
        self.on_trait_change(self._update_line_data_index, 'frequency', dispatch='ui')
        self.on_trait_change(self._update_line_data_value, 'counts', dispatch='ui')
        self.on_trait_change(self._update_line_data_fit, 'fit_parameters', dispatch='ui')
        self.on_trait_change(self._update_matrix_data_value, 'counts_matrix', dispatch='ui')
        self.on_trait_change(self._update_matrix_data_index, 'n_lines,frequency', dispatch='ui')
        self.on_trait_change(self._update_fit, 'counts,perform_fit,number_of_resonances,threshold', dispatch='ui')
        
    def _counts_matrix_default(self):
        return np.zeros((self.n_lines, len(self.frequency)))

    def _frequency_default(self):
        
        return np.arange(self.frequency_begin, self.frequency_end + self.frequency_delta, self.frequency_delta)

    def _counts_default(self):
        return np.zeros(self.frequency.shape)

    # data acquisition

    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""
        
        frequency = np.arange(self.frequency_begin, self.frequency_end + self.frequency_delta, self.frequency_delta)

        if not self.keep_data or np.any(frequency != self.frequency):
            self.frequency = frequency
            self.counts = np.zeros(frequency.shape)
            self.run_time = 0.0

        self.keep_data = False # when job manager stops and starts the job, data should be kept. Only new submission should clear data.

    def _run(self):
                
        try:
            self.state = 'run'
            self.apply_parameters()

            if self.run_time >= self.stop_time:
                self.state = 'done'
                return

            ha.PulseGenerator().Sequence([(['hmc_trigger'], self.t_pi12_NV), ([       ], self.time_Hahn), (['hmc_trigger'], self.t_pi_NV), (['mw'], self.t_pi_electrons), ([       ], self.time_Hahn-self.t_pi_electrons), (['hmc_trigger'], self.t_pi12_NV), (['laser', 'trigger'], self.laser), ([       ], self.wait)])
                        
            n = len(self.frequency)
            
            MW_HMC.setFrequency(self.frequency_NV)
            MW_HMC.setPower(self.power_NV)

            MW.setPower(self.power)
            MW.initSweep(self.frequency, self.power * np.ones(self.frequency.shape))
            
            ha.Counter().configure(n, self.seconds_per_point, DutyCycle=0.8)
            time.sleep(0.5)

            while self.run_time < self.stop_time:
                start_time = time.time()
                if threading.currentThread().stop_request.isSet():
                    break
                ha.Microwave().resetListPos()
                counts = ha.Counter().run()/1e3                
                self.run_time += time.time() - start_time
                self.counts += counts
                self.counts_matrix = np.vstack((counts, self.counts_matrix[:-1, :]))
                self.trait_property_changed('counts', self.counts)
                    
            if self.run_time < self.stop_time:
                self.state = 'idle'            
            else:
                self.state = 'done'
            
            ha.Microwave().setOutput(None, self.frequency_begin)
            ha.Microwave_HMC().Off()
            ha.PulseGenerator().Light()
            ha.Counter().clear()
        except:
            logging.getLogger().exception('Error in odmr.')
            self.state = 'error'
        finally:
            ha.Microwave().setOutput(None, self.frequency_begin)
            ha.Microwave_HMC().Off()

    # fitting
    def _update_fit(self):
        if self.perform_fit:
            N = self.number_of_resonances 
            if N != 'auto':
                N = int(N)
            try:
                p = fitting.fit_multiple_lorentzians(self.frequency, self.counts, N, threshold=self.threshold * 0.01)
            except Exception:
                logging.getLogger().debug('ODMR fit failed.', exc_info=True)
                p = np.nan * np.empty(4)
        else:
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
    
    
    # plotting
        
    def _create_line_plot(self):
        line_data = ArrayPlotData(frequency=np.array((0., 1.)), counts=np.array((0., 0.)), fit=np.array((0., 0.))) 
        line_plot = Plot(line_data, padding=8, padding_left=64, padding_bottom=32)
        line_plot.plot(('frequency', 'counts'), style='line', color='blue')
        line_plot.index_axis.title = 'Frequency [MHz]'
        line_plot.value_axis.title = 'Fluorescence/k'
        line_label = PlotLabel(text='', hjustify='left', vjustify='bottom', position=[64, 128])
        line_plot.overlays.append(line_label)
        self.line_label = line_label
        self.line_data = line_data
        self.line_plot = line_plot
        
    def _create_matrix_plot(self):
        matrix_data = ArrayPlotData(image=np.zeros((2, 2)))
        matrix_plot = Plot(matrix_data, padding=8, padding_left=64, padding_bottom=32)
        matrix_plot.index_axis.title = 'Frequency [MHz]'
        matrix_plot.value_axis.title = 'line #'
        matrix_plot.img_plot('image',
                             xbounds=(self.frequency[0], self.frequency[-1]),
                             ybounds=(0, self.n_lines),
                             colormap=Spectral)
        self.matrix_data = matrix_data
        self.matrix_plot = matrix_plot

    def _perform_fit_changed(self, new):
        plot = self.line_plot
        if new:
            plot.plot(('frequency', 'fit'), style='line', color='red', name='fit')
            self.line_label.visible = True
        else:
            plot.delplot('fit')
            self.line_label.visible = False
        plot.request_redraw()

    def _update_line_data_index(self):
        self.line_data.set_data('frequency', self.frequency * 1e-6)
        self.counts_matrix = self._counts_matrix_default()

    def _update_line_data_value(self):
        self.line_data.set_data('counts', self.counts)

    def _update_line_data_fit(self):
        if not np.isnan(self.fit_parameters[0]):            
            self.line_data.set_data('fit', fitting.NLorentzians(*self.fit_parameters)(self.frequency))
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
            for i, fi in enumerate(f):
                s += 'f %i: %.6e Hz, HWHM %.3e Hz, contrast %.1f%%\n' % (i + 1, fi, w[i], contrast[i])
            self.line_label.text = s

    def _update_matrix_data_value(self):
        self.matrix_data.set_data('image', self.counts_matrix)

    def _update_matrix_data_index(self):
        if self.n_lines > self.counts_matrix.shape[0]:
            self.counts_matrix = np.vstack((self.counts_matrix, np.zeros((self.n_lines - self.counts_matrix.shape[0], self.counts_matrix.shape[1]))))
        else:
            self.counts_matrix = self.counts_matrix[:self.n_lines]
        self.matrix_plot.components[0].index.set_data((self.frequency[0] * 1e-6, self.frequency[-1] * 1e-6), (0.0, float(self.n_lines)))

    # saving data
    
    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename + '_ODMR_Line_Plot.png')

    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)
    
    def save_all(self, filename):
        self.save_line_plot(filename)
        self.save_matrix_plot(filename + '_ODMR_Matrix_Plot.png')
        self.save(filename + '_ODMR.pyd')

    # react to GUI events

    def submit(self):
        """Submit the job to the JobManager."""
        self.keep_data = False
        ManagedJob.submit(self)
        print threading.currentThread().getName()

    def resubmit(self):
        """Submit the job to the JobManager."""
        self.keep_data = True
        ManagedJob.submit(self)

    def _resubmit_button_fired(self):
        """React to start button. Submit the Job."""
        self.resubmit()      

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority', enabled_when='state != "run"'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f'),
                                     Item('stop_time'),
                                     ),
                              
                                     HGroup(Item('power', width= -40, enabled_when='state != "run"'),
                                            Item('frequency_begin', width= -80, enabled_when='state != "run"'),
                                            Item('frequency_end', width= -80, enabled_when='state != "run"'),
                                            Item('frequency_delta', width= -80, enabled_when='state != "run"'),
                                            Item('t_pi_electrons', width= -40, enabled_when='state != "run"')                                            
                                            ), 
                                     HGroup(Item('frequency_NV', width= -80, enabled_when='state != "run"'),
                                            Item('power_NV', width= -40, enabled_when='state != "run"'),
                                            Item('t_pi_NV', width= -40, enabled_when='state != "run"'),
                                            Item('t_pi12_NV', width= -40, enabled_when='state != "run"'),
                                            Item('time_Hahn', width= -40, enabled_when='state != "run"')                                            
                                            ), 
                                     HGroup(Item('seconds_per_point', width= -40, enabled_when='state != "run"'),
                                            Item('laser', width= -50, enabled_when='state != "run"'),
                                            Item('wait', width= -50, enabled_when='state != "run"'),
                                            ),
                                     HGroup(Item('perform_fit'),
                                            Item('number_of_resonances', width= -60),
                                            Item('threshold', width= -60),
                                            Item('n_lines', width= -60),
                                            ),
                                     HGroup(Item('fit_contrast', width= -90,style='readonly'),
                                            Item('fit_line_width', width= -90,style='readonly'),
                                            Item('fit_frequencies', width= -90,style='readonly'),
                         
                                            ),
                                    HGroup( Item('crossing', enabled_when='state != "run"'),
                                            Item('central_dip_position', style='readonly'),
                                            Item('magnetic_field', style='readonly'),
                                            ),
                                     
                              VSplit(Item('line_plot', show_label=False, resizable=True),
                                     Item('matrix_plot', show_label=False, resizable=True),
                                    ),
                              ),
                       menubar=MenuBar(Menu(Action(action='saveLinePlot', name='SaveLinePlot (.png)'),
                                              Action(action='saveMatrixPlot', name='SaveMatrixPlot (.png)'),
                                              Action(action='save', name='Save (.pyd or .pys)'),
                                              Action(action='saveAll', name='Save All (.png+.pyd)'),
                                              Action(action='export', name='Export as Ascii (.asc)'),
                                              Action(action='export', name='Export as Ascii (.asc)'),
                                              Action(action='load', name='Load'),
                                              Action(action='_on_close', name='Quit'),
                                              name='File')),
                       title='DEER ODMR', width=900, height=800, buttons=[], resizable=True, handler=ODMRHandler
                       )

    get_set_items = ['frequency', 'counts', 'counts_matrix', 'perform_fit', 'run_time',
                     'power', 'frequency_begin', 'frequency_end', 'frequency_delta', 't_pi_electrons',
                     'frequency_NV', 'power_NV', 't_pi_NV', 't_pi12_NV',
                     'laser', 'wait', 
                     'seconds_per_point', 'stop_time', 'n_lines',
                     'number_of_resonances', 'threshold',
                     't_pi_electrons', 'time_Hahn',
                     'fit_parameters', 'fit_contrast', 'fit_line_width', 'fit_frequencies',   
                     '__doc__']

#if __name__ == '__main__':

 #   logging.getLogger().addHandler(logging.StreamHandler())
  #  logging.getLogger().setLevel(logging.DEBUG)
   # logging.getLogger().info('Starting logger.')

    #from tools.emod import JobManager
    #JobManager().start()

    #o = ODMR()
    #o.edit_traits()
    
    
