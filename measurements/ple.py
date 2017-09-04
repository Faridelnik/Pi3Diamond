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

from tools.utility import GetSetItemsHandler, GetSetItemsMixin

class PLEHandler(GetSetItemsHandler):

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
            
    def saveAsText(self, info):
        filename = save_file(title='Save data as text file')
        if filename is '':
            return
        else:
            if filename.find('.txt') == -1:
                filename = filename + '.txt'
                info.object.save_as_text(filename)
                
    def saveAll(self, info):
        filename = save_file(title='Save All')
        if filename is '':
            return
        else:
            info.object.save_all(filename)


class PLE(ManagedJob, GetSetItemsMixin):
    """Provides PLE measurements."""

    keep_data = Bool(False) # helper variable to decide whether to keep existing data

    resubmit_button = Button(label='resubmit', desc='Submits the measurement to the job manager. Tries to keep previously acquired data. Behaves like a normal submit if sequence or time bins have changed since previous run.')    

    mw = Enum('off', 'on', desc='switching MW on and off', label='MW')
    mw_frequency = Range(low=1, high=20e9, value=2.8770e9, desc='microwave frequency', label='MW frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    mw_power = Range(low= -100., high=25., value= -5.0, desc='microwave power', label='MW power [dBm]', mode='text', auto_set=False, enter_set=True)

    green = Enum('off', 'on', 'pulse', desc='Green laser control', label='Green')
    light = Enum('light', 'night', desc='Status control', label='Light')
    off_mw = Enum('off', 'on', desc='MW control when not running', label='MW when not running')
    off_red = Enum('off', 'on', desc='red laser control when not running', label='red when not running')
    off_green = Enum('off', 'on', desc='Green laser control when not running', label='Green for red when not running')
    
    green_length = Range(low=1e-3, high=10., value=0.5, desc='TTL On [s]', label='Green laser [s]', mode='text', auto_set=False, enter_set=True)

    switch = Enum('fix wave', 'scan wave', desc='switching from large scan to small scan', label='scan mode')
    
    red = Enum('off', 'on', desc='switching Red laser on and off', label='Red', auto_set=False, enter_set=True)
    wavelength = Range(low=636.0, high=639.0, value=636.86, desc='Wavelength[nm]', label='Red Wavelength [nm]', auto_set=False, enter_set=True)
    current = Range(low=0.0, high=80.0, value=0., desc='Current of red [mA]', label='Red Current [mA]', auto_set=False, enter_set=True)
    detuning = Range(low= -45.0, high=45.0, value=0., desc='detuning', label='Red detuning [GHz]', auto_set=False, enter_set=True)
    go_detuning = Bool(False, label='Go to detuning')
    lock_box = Bool(False, label='lockbox mode')
     
    red_monitor = Enum('on', 'off', desc='red laser status', label='Red Status')
    wavelength_monitor = Range(low=0.0, high=639.0, value=0.0, desc='Wavelength[nm]', label='Red Wavelength [nm]')
    current_monitor = Range(low=0.0, high=150.0, value=0.0, desc='Current of red [mA]', label='Red Current [mA]')
    power_monitor = Range(low=0.0, high=20.0, value=0.0, desc='power of red [mW]', label='Red Power [mW]')
    detuning_monitor = Range(low= -50., high=50., value=0., desc='detuning', label='detuning freq [GHz]')

    scan_begin = Range(low=636., high=639., value=636.5, desc='Start Wavelength [nm]', label='Start Wavelength [nm]')
    scan_end = Range(low=636., high=639., value=637.5, desc='Stop Wavelength [nm]', label='Stop Wavelength [nm]')
    scan_rate = Range(low=0.01, high=1., value=1.0e-2, desc='Scan rate[nm/s]', label='Scan Rate [nm/s]')

    detuning_begin = Range(low= -45.0, high=45.0, value= -45.0, desc='Start detuning [GHz]', label='Begin detuning [GHz]')
    detuning_end = Range(low= -45.0, high=45.0, value=45.0, desc='Stop detuning [GHz]', label='End detuning [GHz]')
    number_points = Range(low=1, high=1e5, value=1000, desc='number of points', label='number of points')
    #detuning_delta = Range(low=0.0, high=45.0, value=0.10, desc='detuning step[GHz]', label='Delta detuning [GHz]')
    seconds_per_point = Range(low=1.0e-4, high=1, value=2.0e-3, desc='Seconds per point', label='Seconds per point', mode='text', auto_set=False, enter_set=True)

    n_lines = Range (low=1, high=10000, value=50, desc='Number of lines in Matrix', label='Matrix lines', mode='text', auto_set=False, enter_set=True)
    
    stop_time = Range(low=1., value=np.inf, desc='Time after which the experiment stops by itself [s]', label='Stop time [s]', mode='text', auto_set=False, enter_set=True)
    run_time = Float(value=0.0, desc='Run time [s]', label='Run time [s]')

    detuning_mesh = Array()
    counts = Array()
    single_line_counts = Array()
    counts_matrix = Array()

    # plotting
    line_data = Instance(ArrayPlotData)
    single_line_data = Instance(ArrayPlotData)
    matrix_data = Instance(ArrayPlotData)
    line_plot = Instance(Plot, editor=ComponentEditor())
    single_line_plot = Instance(Plot, editor=ComponentEditor())
    matrix_plot = Instance(Plot, editor=ComponentEditor())

    def __init__(self):
        super(PLE, self).__init__()
        self._create_line_plot()
        self._create_single_line_plot()
        self._create_matrix_plot()
        self.on_trait_change(self._update_index, 'detuning_mesh', dispatch='ui')
        self.on_trait_change(self._update_line_data_value, 'counts', dispatch='ui')
        self.on_trait_change(self._update_single_line_data_value, 'single_line_counts', dispatch='ui')
        self.on_trait_change(self._update_matrix_data_value, 'counts_matrix', dispatch='ui')
        self.on_trait_change(self._update_matrix_data_index, 'n_lines,detuning', dispatch='ui')
        self.red_monitor = ha.LaserRed().check_status()
        self.red = self.red_monitor
        ha.LaserRed().set_output(self.current, self.wavelength)
        self.wavelength_monitor = ha.LaserRed().get_wavelength() 
        self.current_monitor = ha.LaserRed().get_current() 
        self.power_monitor = ha.LaserRed().get_power() 
        self.detuning_monitor = ha.LaserRed().get_detuning()
        
        
    def submit(self):
        """Submit the job to the JobManager."""
        self.keep_data = False
        ManagedJob.submit(self)

    def resubmit(self):
        """Submit the job to the JobManager."""
        self.keep_data = True
        ManagedJob.submit(self)

    def _resubmit_button_fired(self):
        """React to start button. Submit the Job."""
        if self.switch == 'scan wave':
                ha.LaserRed().resume_scan()
        self.resubmit() 
    
    @on_trait_change('red')
    def _open_laser(self):
        if self.red == 'on':
            ha.LaserRed().on()
            time.sleep(7.0)
            self.red_monitor = ha.LaserRed().check_status()
        elif self.red == 'off':
            ha.LaserRed().off()
            time.sleep(1.0)
            self.red_monitor = ha.LaserRed().check_status()

    @on_trait_change('off_mw')
    def _off_mw_control(self):
        if self.state == 'idle':
            if self.off_mw == 'on':
                ha.Microwave().setOutput(self.mw_power, self.mw_frequency)
            else:
                ha.Microwave().setOutput(None, self.mw_frequency)
    
    @on_trait_change('off_green,off_red,off_mw')
    def _off_control(self):
        if self.state == 'idle' and self.light == 'night' and self.off_red == 'on':
            if self.off_green == 'on' and self.off_mw == 'on':
                ha.PulseGenerator().Continuous(['mw','mw_x', 'red', 'green'])
            elif self.off_green == 'off' and self.off_mw == 'on':
                ha.PulseGenerator().Continuous(['mw','mw_x', 'red'])
            elif self.off_green == 'on' and self.off_mw == 'off':
                ha.PulseGenerator().Continuous(['red', 'green'])    
            elif self.off_green == 'off' and self.off_mw == 'off':
                ha.PulseGenerator().Continuous(['red'])
        elif self.state == 'idle' and self.off_red == 'off' and self.light == 'light':
            ha.PulseGenerator().Light()
        elif self.state == 'idle' and self.off_red == 'off' and self.light == 'night':
            ha.PulseGenerator().Night()
        
    @on_trait_change('light')
    def _set_day(self):
        if self.state == 'idle':
            if self.light == 'light':
                ha.PulseGenerator().Light()
            elif self.off_red == 'on':
                if self.off_green == 'on' and self.off_mw == 'on':
                    ha.PulseGenerator().Continuous(['mw','mw_x', 'red', 'green'])
                elif self.off_green == 'off' and self.off_mw == 'on':
                    ha.PulseGenerator().Continuous(['mw','mw_x', 'red'])
                elif self.off_green == 'on' and self.off_mw == 'off':
                    ha.PulseGenerator().Continuous(['red', 'green'])    
                elif self.off_green == 'off' and self.off_mw == 'off':
                    ha.PulseGenerator().Continuous(['red'])
            else:
                ha.PulseGenerator().Night()
        
    @on_trait_change('current,wavelength')
    def _set_laser(self):
        """React to set laser button. Submit the Job."""
        ha.LaserRed().set_output(self.current, self.wavelength)
        self.wavelength_monitor = ha.LaserRed().get_wavelength() 
        self.current_monitor = ha.LaserRed().get_current() 
        self.power_monitor = ha.LaserRed().get_power() 
    
    @on_trait_change('detuning,go_detuning')
    def _detuning_changed(self, detuning):
        """React to set laser button. Submit the Job."""
        if self.go_detuning:
            ha.LaserRed().set_detuning(detuning)
            self.detuning_monitor = ha.LaserRed().get_detuning()

    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""
        step = (self.detuning_end - self.detuning_begin) / float(self.number_points)
        detuning_mesh = np.arange(self.detuning_begin, self.detuning_end, step)
        
        if not self.keep_data or np.any(detuning_mesh != self.detuning_mesh):
            self.counts = np.zeros(detuning_mesh.shape)
            self.run_time = 0.0

        self.detuning_mesh = detuning_mesh
        self.keep_data = True # when job manager stops and starts the job, data should be kept. Only new submission should clear data.

    def _run(self):
                
        try:
            self.state = 'run'
            self.apply_parameters()

            if self.run_time >= self.stop_time:
                self.state = 'done'
                return

            n = len(self.detuning_mesh)
            
            self.red_monitor = ha.LaserRed().check_status()
            if self.red_monitor == 'off':
                ha.LaserRed().on()
                time.sleep(7.0)
                self.red_monitor = ha.LaserRed().check_status()
            #ha.LaserRed().set_output(self.current, self.wavelength)
            ha.Microwave().setOutput(self.mw_power, self.mw_frequency)
            time.sleep(0.5)
            
            if self.switch == 'scan wave':
                ha.LaserRed().scan(self.scan_begin, self.scan_end, self.scan_rate)
                self.wavelength_monitor = ha.LaserRed().get_wavelength() 
                self.detuning_monitor = ha.LaserRed().get_detuning()
                
            while self.run_time < self.stop_time:

                start_time = time.time()
                if threading.currentThread().stop_request.isSet():
                    break

                if self.green == 'pulse':
                    if self.mw == 'on':
                        ha.PulseGenerator().Continuous(['mw', 'mw_x','red', 'green'])
                    else:
                        ha.PulseGenerator().Continuous(['red', 'green'])
                
                time.sleep(self.green_length)

                if self.mw == 'on' and self.green == 'on':
                    ha.PulseGenerator().Continuous(['green', 'mw','mw_x', 'red'])
                elif self.mw == 'off' and self.green == 'on':
                    ha.PulseGenerator().Continuous(['green', 'red'])
                elif self.mw == 'on' and (self.green == 'off' or self.green == 'pulse'):
                    ha.PulseGenerator().Continuous(['mw', 'mw_x','red'])
                elif self.mw == 'off' and (self.green == 'off' or self.green == 'pulse'):
                    ha.PulseGenerator().Continuous(['red'])
                
                if self.lock_box:
                    #voltage = ha.LaserRed()._detuning_to_voltage(self.detuning_end)
                    step = (self.detuning_begin - self.detuning_end) / float(self.number_points)
                    detuning_mesh = np.arange(self.detuning_end, self.detuning_begin, step)
                    counts = ha.LaserRed().piezo_scan(detuning_mesh, self.seconds_per_point)
                    junk = ha.LaserRed().piezo_scan(self.detuning_mesh, self.seconds_per_point)
                    self.single_line_counts = counts
                    self.counts += counts
                    self.trait_property_changed('counts', self.counts)
                    self.counts_matrix = np.vstack((counts, self.counts_matrix[:-1, :]))
                    time.sleep(0.1)
                else:
                    
                    counts = ha.LaserRed().piezo_scan(self.detuning_mesh, self.seconds_per_point)
                    
                    #ha.LaserRed().set_detuning(self.detuning_begin)
                    #time.sleep(0.1)
                    
                    self.run_time += time.time() - start_time
                    self.single_line_counts = counts
                    self.counts += counts
                    self.trait_property_changed('counts', self.counts)
                    self.counts_matrix = np.vstack((counts, self.counts_matrix[:-1, :]))
                    
                    """ return to origin
                    """
                    voltage = ha.LaserRed()._detuning_to_voltage(self.detuning_end)
                    junks = ha.LaserRed().ni_task.line(np.arange(voltage, -3.0, ha.LaserRed()._detuning_to_voltage(-0.1)), 0.001)
                    ha.LaserRed().ni_task.point(-3.0)
                    
                    voltage = ha.LaserRed()._detuning_to_voltage(self.detuning_begin)
                    if voltage > -2.90:
                        junks = ha.LaserRed().ni_task.line(np.arange(-3.0, voltage, ha.LaserRed()._detuning_to_voltage(0.1)), 0.001)
                        ha.LaserRed().ni_task.point(voltage)
                    time.sleep(0.1)
                    
                    self.wavelength_monitor = ha.LaserRed().get_wavelength() 
                    self.detuning_monitor = ha.LaserRed().get_detuning()
  
                      
            if self.run_time < self.stop_time:
                self.state = 'idle'            
            else:
                self.state = 'done'
            if self.switch == 'scan wave':
                ha.LaserRed().stop_scan()
            
            ha.LaserRed().set_detuning(self.detuning)
            #ha.Microwave().setOutput(None, self.mw_frequency)
            if self.light == 'light':
                ha.PulseGenerator().Light()
                ha.Microwave().setOutput(None, self.mw_frequency)
            elif self.off_red == 'on':
                if self.off_green == 'on' and self.off_mw == 'on':
                    ha.PulseGenerator().Continuous(['mw', 'mw_x','red', 'green'])
                elif self.off_green == 'off' and self.off_mw == 'on':
                    ha.PulseGenerator().Continuous(['mw','mw_x', 'red'])
                elif self.off_green == 'on' and self.off_mw == 'off':
                    ha.Microwave().setOutput(None, self.mw_frequency)
                    ha.PulseGenerator().Continuous(['red', 'green'])    
                elif self.off_green == 'off' and self.off_mw == 'off':
                    ha.Microwave().setOutput(None, self.mw_frequency)
                    ha.PulseGenerator().Continuous(['red'])
            else:
                ha.Microwave().setOutput(None, self.mw_frequency)
                ha.PulseGenerator().Night()
        
        except:
            logging.getLogger().exception('Error in PLE.')
            self.state = 'error'            
     
    # plotting    
    def _create_line_plot(self):
        line_data = ArrayPlotData(frequency=np.array((0., 1.)), counts=np.array((0., 0.)), fit=np.array((0., 0.))) 
        line_plot = Plot(line_data, padding=8, padding_left=64, padding_bottom=32)
        line_plot.plot(('frequency', 'counts'), style='line', color='blue')
        line_plot.index_axis.title = 'Detuning [GHz]'
        line_plot.value_axis.title = 'Fluorescence counts'
        self.line_data = line_data
        self.line_plot = line_plot
        
    def _create_single_line_plot(self):
        line_data = ArrayPlotData(frequency=np.array((0., 1.)), counts=np.array((0., 0.)), fit=np.array((0., 0.))) 
        line_plot = Plot(line_data, padding=8, padding_left=64, padding_bottom=32)
        line_plot.plot(('frequency', 'counts'), style='line', color='blue')
        line_plot.index_axis.title = 'Detuning [GHz]'
        line_plot.value_axis.title = 'Fluorescence counts'
        self.single_line_data = line_data
        self.single_line_plot = line_plot
        
    def _create_matrix_plot(self):
        matrix_data = ArrayPlotData(image=np.zeros((2, 2)))
        matrix_plot = Plot(matrix_data, padding=8, padding_left=64, padding_bottom=32)
        matrix_plot.index_axis.title = 'Detuning [GHz]'
        matrix_plot.value_axis.title = 'line #'
        matrix_plot.img_plot('image',
                             xbounds=(self.detuning_mesh[0], self.detuning_mesh[-1]),
                             ybounds=(0, self.n_lines),
                             colormap=Spectral)
        self.matrix_data = matrix_data
        self.matrix_plot = matrix_plot


    def _counts_matrix_default(self):
        return np.zeros((self.n_lines, len(self.detuning_mesh)))
    
    @on_trait_change('detuning_begin,detuning_end,number_points')
    def _detuning_mesh_(self):
        step = (self.detuning_end - self.detuning_begin) / float(self.number_points)
        self.detuning_mesh = np.arange(self.detuning_begin, self.detuning_end, step)
        self.counts = np.zeros(self.detuning_mesh.shape)
        self.single_line_counts = np.zeros(self.detuning_mesh.shape)
        self.counts_matrix = np.zeros((self.n_lines, len(self.detuning_mesh)))
        self.line_data.set_data('frequency', self.detuning_mesh)
        self.single_line_data.set_data('frequency', self.detuning_mesh)

    def _detuning_mesh_default(self):
        step = (self.detuning_end - self.detuning_begin) / float(self.number_points)
        return np.arange(self.detuning_begin, self.detuning_end, step)
   
    def _counts_default(self):
        return np.zeros(self.detuning_mesh.shape)

    def _single_line_counts_default(self):
        return np.zeros(self.detuning_mesh.shape)

    def _update_index(self):
        self.line_data.set_data('frequency', self.detuning_mesh)
        self.single_line_data.set_data('frequency', self.detuning_mesh)
        self.counts_matrix = self._counts_matrix_default()

    def _update_line_data_value(self):
        self.line_data.set_data('counts', self.counts)

    def _update_single_line_data_value(self):
        self.single_line_data.set_data('counts', self.single_line_counts)

    def _update_matrix_data_value(self):
        self.matrix_data.set_data('image', self.counts_matrix)

    def _update_matrix_data_index(self):
        if self.n_lines > self.counts_matrix.shape[0]:
            self.counts_matrix = np.vstack((self.counts_matrix, np.zeros((self.n_lines - self.counts_matrix.shape[0], self.counts_matrix.shape[1]))))
        else:
            self.counts_matrix = self.counts_matrix[:self.n_lines]
        self.matrix_plot.components[0].index.set_data((self.detuning_mesh[0], self.detuning_mesh[-1]), (0.0, float(self.n_lines)))

    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename)

    def save_single_line_plot(self, filename):
        self.save_figure(self.single_line_plot, filename)

    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)
    
    def save_as_text(self, filename):
        matsave = np.vstack((self.detuning_mesh, self.counts_matrix))
        np.savetxt(filename, matsave)
        
    def save_all(self, filename):
        self.save_line_plot(filename + '_PLE_Line_Plot.png')
        self.save_single_line_plot(filename + '_PLE_SingleLine_Plot.png')
        self.save_matrix_plot(filename + '_PLE_Matrix_Plot.png')
        self.save_as_text(filename + '_PLE_.txt')
        self.save(filename + '_PLE.pys')

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('n_lines'),
                                     Item('run_time', style='readonly', format_str='%.f'),
                                     ),
                              Group(HGroup(Item('mw', style='custom'),
                                           Item('mw_frequency', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x)),
                                           Item('mw_power', width= -40, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                                           Item('green', style='custom'),
                                           Item('green_length', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x)),
                                           ),
                                    HGroup(Item('light', style='custom'),
                                           Item('off_mw', style='custom'),
                                           Item('off_red', style='custom'),
                                           Item('off_green', style='custom'),
                                           Item('lock_box',style='custom'),
                                           ),
                                    HGroup(Item('red', style='custom', enabled_when='state == "idle"'),
                                           Item('red_monitor', width= -20, style='readonly'),
                                           Item('wavelength', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                                           Item('wavelength_monitor', width= -80, style='readonly'),
                                           Item('current', width= -40, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                                           Item('current_monitor', width= -40, style='readonly'),
                                           Item('power_monitor', width= -40, style='readonly'),
                                           Item('detuning', width= -40, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                                           Item('detuning_monitor', width= -40, style='readonly'),
                                           Item('go_detuning'),
                                           ),
                                    HGroup(Item('switch', style='custom', enabled_when='state == "idle"'),
                                           Item('scan_begin', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                                           Item('scan_end', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                                           Item('scan_rate', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                                           Item('detuning_begin', width= -40, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                                           Item('detuning_end', width= -40, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                                           Item('number_points', width= -40, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                                           #Item('detuning_delta', width= -40, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                                           Item('seconds_per_point', width= -40, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                                           ),
                                    ),
                              VSplit(Item('matrix_plot', show_label=False, resizable=True),
                                     Item('single_line_plot', show_label=False, resizable=True),
                                     Item('line_plot', show_label=False, resizable=True),
                                     ),
                              ),
                       menubar=MenuBar(Menu(Action(action='saveLinePlot', name='SaveLinePlot (.png)'),
                                              Action(action='saveMatrixPlot', name='SaveMatrixPlot (.png)'),
                                              Action(action='save', name='Save (.pyd or .pys)'),
                                              Action(action='saveAsText', name='SaveAsText (.txt)'),
                                              Action(action='saveAll', name='Save All (.png+.pys)'),
                                              Action(action='export', name='Export as Ascii (.asc)'),
                                              Action(action='load', name='Load'),
                                              Action(action='_on_close', name='Quit'),
                                              name='File')),
                       title='PLE', width=1200, height=1000, buttons=[], resizable=True, handler=PLEHandler
                       )

    get_set_items = ['mw', 'mw_frequency', 'mw_power', 'wavelength', 'current', 'detuning', 'number_points', 'detuning_begin', 'detuning_end', 'detuning_mesh', 'red',
                   'green_length', 'scan_begin', 'scan_end', 'scan_rate', 'seconds_per_point', 'stop_time', 'n_lines',
                   'counts', 'counts_matrix', 'run_time', '__doc__' ]
    get_set_order = ['detuning', 'n_lines']


if __name__ == '__main__':

    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().info('Starting logger.')

    from tools.emod import JobManager
    JobManager().start()

    ple = PLE()
    ple.edit_traits()
    
    
