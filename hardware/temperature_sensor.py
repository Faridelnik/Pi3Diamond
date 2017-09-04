# located in hardware

import serial
import numpy as np

from traits.api import SingletonHasTraits, Trait, Instance, Property, String, Range, Float, Int, Bool, Array, Enum, Button, on_trait_change, cached_property, Code, List, NO_COMPARE, Tuple
from traitsui.api import View, Item, HGroup, VGroup, VSplit, Tabbed, EnumEditor, TextEditor, Group
from enable.api import Component, ComponentEditor
from chaco.api import ArrayPlotData, Plot, Spectral, PlotLabel

from traitsui.file_dialog import save_file
from traitsui.menu import Action, Menu, MenuBar

import time
import threading
import logging

from hardware import coil
from tools.emod import ManagedJob

import hardware.api as ha

from analysis import fitting

from tools.utility import GetSetItemsHandler, GetSetItemsMixin

ser = serial.Serial('COM7', 9600)
power_supply = coil.Coil('z')


class SensorHandler(GetSetItemsHandler):

    def saveTemperaturePlot(self, info):
        filename = save_file(title='Save Temperature Plot')
        if filename is '':
            return
        else:
            info.object.save_temperature_plot(filename)
            
    def init(self, info):
        info.object.start()

class Temperature_Sensor(ManagedJob, GetSetItemsMixin):

    keep_data = Bool(False) # helper variable to decide whether to keep existing data

    temperature = Array()
    vremya = Array()
    sleep_time = Range(low= 0., high=3600., value= 300., desc='periodicity of measurements', label='interval [s]', mode='text', auto_set=False, enter_set=True)
       
    plot_data = Instance(ArrayPlotData)
    temperature_plot=Instance(Plot, editor=ComponentEditor())
    
    voltage = Range(low = 0., high = 12., value = 12.0, desc='Voltage [V]', label='Voltage [V]', mode='text', auto_set=False, enter_set=True)
    current = Range(low = 0., high = 2., value = 3.0, desc='Current [A]', label='Current [A]', mode='text', auto_set=False, enter_set=True)
    limiting_voltage = Range(low = 0., high = 70., value = 20.0, desc='Voltage [V]', label='limiting voltage [V]', mode='text', auto_set=False, enter_set=True)
    limiting_current = Range(low = 0., high = 10., value = 4.0, desc='Current [A]', label='limiting current [A]', mode='text', auto_set=False, enter_set=True)
    
    
    set_limiting_current_button = Button(label='set limiting curent', desc='set limiting current')
    set_limiting_voltage_button = Button(label='set limiting curent', desc='set limiting current')
    set_current_button = Button(label='set current', desc='set current')
    set_voltage_button = Button(label='set voltage', desc='set voltage')
    stop_heating_button = Button(label='swich heating off', desc='swich off heating')
    
    perform_fit=Bool(False, label='perform fit')
    fit_result = Array(value=np.array((np.nan, np.nan, np.nan, np.nan)))
    
    def __init__(self):
        super(Temperature_Sensor, self).__init__()
        self._create_temperature_plot()
        self.on_trait_change(self._update_plot_data, 'temperature, perform_fit', dispatch='ui')
        self.on_trait_change(self._update_fit, 'temperature, perform_fit', dispatch='ui')
        self.on_trait_change(self._update_plot_fit, 'fit_result', dispatch='ui')
        
    def _run(self):
    
        logging.getLogger().debug("trying run")
        
        try:
            self.state='run'  
            
            k=0

            while True:
                for i in range(10):
                
                    buffer = ser.readline()            
                    
                    if len(buffer)==49:
                        temperature = float(buffer[40:45])
                        c=time.strftime('%H:%M:%S')
                        vremya=float(c[0:2])+float(c[3:5])/60+float(c[6:8])/3600
                        
                        self.temperature = np.append(self.temperature, temperature)
                        self.vremya = np.append(self.vremya, vremya)
                        
                time.sleep(self.sleep_time)
                              
                if threading.currentThread().stop_request.isSet():
                    break
                    
            self.state = 'done' 
        except:
            logging.getLogger().exception('Error in temperature sensor')
            self.state = 'error'
            
    # plotting ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            
    def _update_plot_data(self):
        self.plot_data.set_data('time', self.vremya)
        self.plot_data.set_data('temperature', self.temperature)      

    def _create_temperature_plot(self):
        plot_data = ArrayPlotData(time = np.array((0., 1.)), temperature = np.array((0., 40.)), fit=np.array((0., 0.)))
        plot = Plot(plot_data, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('time','temperature'), color='green')
        plot.index_axis.title = 'time [h]'
        plot.value_axis.title = 'temperature [C]'
        
        self.plot_data = plot_data
        self.temperature_plot = plot
        return self.temperature_plot   
        
    def save_temperature_plot(self, filename):
        self.save_figure(self.temperature_plot, filename + '.png' )   
        self.save(filename +'TPlot_'+ '_V=' + string.replace(str(self.voltage), '.', 'd') + '.pyd' )
        
    # fitting---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _perform_fit_changed(self, new):
        plot = self.temperature_plot
        if new:
            plot.plot(('time', 'fit'), style='line', color='red', name='fit')
        else:
            plot.delplot('fit')
        plot.request_redraw()
        
    def _update_fit(self):
        if self.perform_fit:
           
            try:
                p = fitting.fit_exp_raise(self.vremya, self.temperature, 0.5)
            except Exception:
                logging.getLogger().debug('temperature fit failed.', exc_info=True)
                p = np.nan * np.empty(4)
        else:
            p = np.nan * np.empty(4)
        self.fit_result = p
        
    def _update_plot_fit(self):
        if not np.isnan(self.fit_result[0]):            
            self.plot_data.set_data('fit', fitting.ExponentialTemperatureSaturation(*self.fit_result[0])(self.vremya))
            #self.decay_time=self.fit_result[0][0]*1e-3 Tsat, a, tau, T0
    
    # heating system--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
  
    def _set_limiting_current_button_fired(self):
        power_supply._set_limiting_current(self.limiting_current)
        
    def _set_limiting_voltage_button_fired(self):
        power_supply._set_limiting_voltage(self.limiting_voltage)
        
    def _set_current_button_fired(self):
        power_supply._set_current(self.current)
        
    def _set_voltage_button_fired(self):
        power_supply._set_voltage(self.voltage)
        
    def _stop_heating_button_fired(self):
        power_supply._set_current(0)
        power_supply._set_voltage(0)
        
        
    def submit(self):
        """Submit the job to the JobManager."""
        self.keep_data = False
        ManagedJob.submit(self)
        print threading.currentThread().getName()
                
    traits_view =  View(VGroup ( HGroup (Item('submit_button', show_label=False),
                                          Item('remove_button', show_label=False),
                                          Item('state', style='readonly'),
                                          Item('sleep_time', width= -70, format_str='%.f'),
                                          Item('current', width= -40),
                                          Item('set_current_button', show_label=False),
                                          Item('voltage', width= -40),
                                          Item('set_voltage_button', show_label=False),
                                          Item('limiting_current', width= -40),
                                          Item('set_limiting_current_button', show_label=False),
                                          Item('limiting_voltage', width= -40),
                                          Item('set_limiting_voltage_button', show_label=False),
                                          Item('stop_heating_button', show_label=False),
                                          Item('perform_fit')
                                          ),
                                                                      
                                 HGroup(Item('temperature_plot', show_label=False, resizable=True)),
                                ),
                                                                                                
                       menubar=MenuBar(Menu(Action(action='saveTemperaturePlot', name='Save Temperature Plot'),
                                            Action(action='load', name='Load'),
                                              name='File')),
                       title='Temperature control', width=1380, height=750, buttons=[], resizable=True, handler=SensorHandler
                       )
                     
    get_set_items = ['temperature', 'vremya', '__doc__']            