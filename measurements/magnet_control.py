import numpy as np

from traits.api import SingletonHasTraits, Trait, Instance, Property, String, Range, Float, Int, Bool, Array, Enum, Button, on_trait_change, cached_property, Code, List, NO_COMPARE
from traitsui.api import View, Item, HGroup, VGroup, VSplit, Tabbed, EnumEditor, TextEditor, Group
from enable.api import Component, ComponentEditor
from chaco.api import ArrayPlotData, Plot, Spectral, PlotLabel,HPlotContainer, jet,ColorBar, LinearMapper, CMapImagePlot

from traitsui.file_dialog import save_file
from traitsui.menu import Action, Menu, MenuBar

import time
import threading
import logging
import string

from tools.emod import ManagedJob

import hardware.api as ha
import hardware.SMC_controller as smc

from analysis import fitting

from hardware.api import Scanner   
counter_tr= Scanner()

from tools.utility import GetSetItemsHandler, GetSetItemsMixin

Magnet=smc.SMC()
          
class MagnetHandler(GetSetItemsHandler):

    def saveFluorescenceXPlot(self, info):
        filename = save_file(title='Save Fluorescence X Plot')
        if filename is '':
            return
        else:
            # if filename.find('.png') == -1:
                # filename = filename + '.png'
            info.object.save_fluor_x_plot(filename)
            
    def saveFluorescenceYPlot(self, info):
        filename = save_file(title='Save Fluorescence Y Plot')
        if filename is '':
            return
        else:
            # if filename.find('.png') == -1:
                # filename = filename + '.png'
            info.object.save_fluor_y_plot(filename)

    def saveFluorescence2DPlot(self, info):
        filename = save_file(title='Save Fluorescence 2D Plot')
        if filename is '':
            return
        else:
            # if filename.find('.png') == -1:
                # filename = filename + '.png'
            info.object.save_fluor_2D_plot(filename)
            
    def saveODMRXPlot(self, info):
        filename = save_file(title='Save Matrix Plot')
        if filename is '':
            return
        else:
            # if filename.find('.png') == -1:
                # filename = filename + '.png'
            info.object.save_fluor_x_plot_ODMR(filename)
    
    def saveODMRYPlot(self, info):
        filename = save_file(title='Save Matrix Plot')
        if filename is '':
            return
        else:
            # if filename.find('.png') == -1:
                # filename = filename + '.png'
            info.object.save_fluor_y_plot_ODMR(filename)
            
    def saveODMR2DPlot(self, info):
        filename = save_file(title='Save Matrix Plot')
        if filename is '':
            return
        else:
            # if filename.find('.png') == -1:
                # filename = filename + '.png'
            info.object.save_fluor_2D_plot_ODMR(filename)


class Magnet_alignment(ManagedJob, GetSetItemsMixin):
    """ This window is realized the following algorithm for the alignment of the magnetic field along NV with high accuracy.
    1. Choose the axis, define the scanning range, number of points and acquisition time
    2. Acquire the fluorescence counts for each magnet position, fit to Gaussian, find maximum, change axis
    3. Move to the position of the maximum, run odmr around it (decrease the scale)
    4. Find ODMR dips and calculate the contrast ratio (for HF coupling)
    5. Put the point to the contrast ratio/position plot
  
    Change the axis, repeat 1-5... Eventually we should find the maximum at contrast ratio/position plot (2D)""" 
      
    #General part
    # Parameters for point to point move
    resubmit_button = Button(label='resubmit', desc='Submits the measurement to the job manager. Tries to keep previously acquired data. Behaves like a normal submit if sequence or time bins have changed since previous run.')
    set_X_position = Range(low= 0., high=23., value= 0, desc='X [mm]', label='X [mm]', mode='text', auto_set=False, enter_set=True)
    set_Y_position = Range(low= 0., high=23., value= 0, desc='Y [mm]', label='Y [mm]', mode='text', auto_set=False, enter_set=True)
    set_Z_position = Range(low= 0., high=20., value= 0, desc='Z [mm]', label='Z [mm]', mode='text', auto_set=False, enter_set=True)
    
    get_X_position = Float(value = 0.0,  desc='X [mm]', label='X [mm]', mode='text', auto_set=False, enter_set=False)
    get_Y_position = Float(value = 0.0,  desc='Y [mm]', label='Y [mm]', mode='text', auto_set=False, enter_set=False)
    get_Z_position = Float(value = 0.0,  desc='Z [mm]', label='Z [mm]', mode='text', auto_set=False, enter_set=False)
     
    get_button = Button(label='Current_position', desc='get current position') 
    apply_button = Button(label='Apply_position', desc='apply current position to set position') 
    set_button = Button(label='Set position', desc='move to position')
     
    choosed_align = Enum('fluoresence', 'odmr',
                     label='choosed align',
                     desc='method for align magnetic field',
                     editor=EnumEditor(values={'fluoresence':'1:fluoresence','odmr':'2:odmr',},cols=2),)
                     
                     
    choosed_scan =  Enum('fx_scan', 'fy_scan','f2D_scan',
                     label='choosed scan',
                     desc='scanning direction for align magnetic field',
                     editor=EnumEditor(values={'fx_scan':'1:fx_scan','fy_scan':'2:fy_scan', 'f2D_scan':'3:f2D_scan'},cols=3),)        
                     
    fitting_func = Enum('gaussian', 'loretzian',
                     label='fit',
                     desc='fitting function',
                     editor=EnumEditor(values={'gaussian':'1:gaussian','loretzian':'2:loretzian'},cols=2),)  
    
    #Fluorescence alignment part
    scanningX = Array( ) 
    scanningY = Array( ) 
    data_xy = Array( )
    data_x = Array( )
    data_y = Array( )
    
    periodic_focus = Bool(False, label='periodic_focus')
    Npoint_auto_focus = Int(low= 1, high=40, value= 10, desc='auto focus per Npoint', label='Npoint_auto_focus', mode='text', auto_set=False, enter_set=True)
    
    Acquisition_time = Range(low= 0., high=25., value= 1, desc='acquisition time [s]', label='Acq. time [s]', mode='text', auto_set=False, enter_set=True)
     
    start_positionX = Range(low= 0., high=24., value= 0., desc='X start [mm]', label='X start [mm]', mode='text', auto_set=False, enter_set=True) 
    end_positionX = Range(low= 0., high=24., value= 10., desc='X end [mm]', label='X end [mm]', mode='text', auto_set=False, enter_set=True) 
    stepX = Range(low= 2e-4, high=25., value= 0.3,desc='movement step X [mm]', label='step X [mm]', mode='text', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.4f'))
     
    Number_of_scanpointsX = Int(value= 10., desc='Number of points for scanning X', label='Nx', mode='text', auto_set=False, enter_set=True)  
     
    start_positionY = Range(low= 0., high=24., value= 0., desc='Y start [mm]', label='Y start [mm]', mode='text', auto_set=False, enter_set=True) 
    end_positionY = Range(low= 0., high=24., value= 10., desc='Y end [mm]', label='Y end [mm]', mode='text', auto_set=False, enter_set=True) 
    stepY = Range(low= 2e-4, high=24., value= 0.3,desc='movement step Y [mm]', label='step Y [mm]', mode='text', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.4f'))
     
    Number_of_scanpointsY = Int(value= 10., desc='Number of points for scanning Y', label='Ny', mode='text', auto_set=False, enter_set=True)
     
    perform_fit=Bool(False, label='perform fit')
     
    Xmax = Float(value= 0., desc='max X [mm]', label='Xmax [mm]', mode='text',  auto_set=False, enter_set=True)
    Ymax = Float(value= 0., desc='max Y[mm]',  label='Ymax [mm]', mode='text',  auto_set=False, enter_set=True)
      
    plot_data_image = Instance( ArrayPlotData )
    plot_data_x_line  = Instance( ArrayPlotData )
    plot_data_y_line  = Instance( ArrayPlotData )
    fluor_x_plot=Instance(Plot, editor=ComponentEditor())
    fluor_y_plot=Instance(Plot, editor=ComponentEditor())
    fluor_2D_plot=Instance(HPlotContainer, editor=ComponentEditor())
    image_plot  = Instance( CMapImagePlot )
    
    run_time = Float(value=0.0, desc='Run time [s]', label='Run time [s]')
    expected_duration = Float(value=0.0, desc='expected_duration[s]', label='expected_duration[s]')
    
    line_label = Instance(PlotLabel)
    line_data = Instance(ArrayPlotData)
    
    
    #===========================================================================================================================================================================================================================
    
    #ODMR alignment part
    
    """Provides ODMR measurements."""
   
    # measurement parameters

    t_pi = Range(low=1., high=100000., value=1000., desc='length of pi pulse [ns]', label='pi [ns]', mode='text', auto_set=False, enter_set=True)
    laser = Range(low=1., high=10000., value=300., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
    wait = Range(low=1., high=10000., value=1000., desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)
    pulsed = Bool(False, label='pulsed')
    power_p = Range(low= -100., high=25., value= -20, desc='Power Pmode [dBm]', label='Power[dBm]', mode='text', auto_set=False, enter_set=True)
    frequency_begin_p = Range(low=1, high=20e9, value=2.87e9, desc='Start Frequency Pmode[Hz]', label='Begin[Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    frequency_end_p = Range(low=1, high=20e9, value=2.88e9, desc='Stop Frequency Pmode[Hz]', label='End[Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    frequency_delta_p = Range(low=1e-3, high=20e9, value=1e5, desc='frequency step Pmode[Hz]', label='Delta[Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    seconds_per_point = Range(low=20e-3, high=1, value=20e-3, desc='Seconds per point', label='Seconds per point', mode='text', auto_set=False, enter_set=True)
    stop_time = Range(low=1., value=np.inf, desc='Time after which the experiment stops by itself [s]', label='Stop time [s]', mode='text', auto_set=False, enter_set=True)
    odmr_threshold = Range(low= -99, high=99., value= -50., desc='Threshold for detection of resonances [%]. The sign of the threshold specifies whether the resonances are negative or positive.', label='threshold [%]', mode='text', auto_set=False, enter_set=True)
    fit_threshold = Range(low= -99, high=99., value= 20., desc='Threshold for detection of resonances [%]. The sign of the threshold specifies whether the resonances are negative or positive.', label='threshold [%]', mode='text', auto_set=False, enter_set=True)
    line_width_threshold = Range(low= 0, high=1e7, value= 5e5, desc='Threshold for detection of resonances [%].', label='line_width_threshold [Hz]', editor=TextEditor(auto_set=False,enter_set=True, evaluate=float, format_str='%e'))
    number_of_resonances = Trait('auto', String('auto', auto_set=False, enter_set=True), Int(10000., desc='Number of Lorentzians used in fit', label='N', auto_set=False, enter_set=True)) 
    nlines =  Range(low= 1, high=600, value=100, desc='nlines for counts matrix', label='nlines', mode='text', auto_set=False, enter_set=True)

    matrix_lines =  Range(low=0, high=min(nlines, 500), value=10, desc='matrix lines', label='matrix_lines', mode='spinner',  auto_set=False, enter_set=False)
    x_point =  Float(value= 0, desc='x scanning point', label='x_point ', mode='text',  auto_set=False, enter_set=False)
    y_point =  Float(value= 0, desc='y scanning point', label='y_point ', mode='text',  auto_set=False, enter_set=False)
    #mline =  Int(low= 1, high=50, value= 10, desc='point in another dimension', label='mline', mode='text', auto_set=False, enter_set=True)
    #plot_button = Button(label='Plot button', desc='array odmr plot')
    
    #ODMR data
    data_xy_ODMR = Array( )
    data_x_ODMR = Array( )
    data_y_ODMR = Array( )
    
    perform_ODMR_fit=Bool(False, label='perform fit')
    
    #ODMR plots
    counts_matrix_ODMR = Array()
    matrix_data_ODMR = Instance(ArrayPlotData)
    matrix_plot_ODMR = Instance(Plot, editor=ComponentEditor())
    #n_lines = Range (low=1, high=10000, value=50, desc='Number of lines in Matrix', label='Matrix lines', mode='text', auto_set=False, enter_set=True)
    
    plot_data_image_ODMR = Instance( ArrayPlotData )
    plot_data_x_line_ODMR  = Instance( ArrayPlotData )
    plot_data_y_line_ODMR  = Instance( ArrayPlotData )
    fluor_x_plot_ODMR =Instance(Plot, editor=ComponentEditor())
    fluor_y_plot_ODMR =Instance(Plot, editor=ComponentEditor())
    fluor_2D_plot_ODMR =Instance(HPlotContainer, editor=ComponentEditor())
    image_plot_ODMR  = Instance( CMapImagePlot )
    
    plot_data_multiline_ODMR = Instance( ArrayPlotData )
    multi_plot_ODMR =Instance(Plot, editor=ComponentEditor())
    
    line_label_ODMR = Instance(PlotLabel)
    line_data_ODMR = Instance(ArrayPlotData)    
    
    fit_parameters = Array(value=np.array((np.nan, np.nan, np.nan, np.nan)))
    fit_centers = Array(value=np.array((np.nan,)), label='center_position [Hz]') 
    fit_line_width = Array(value=np.array((np.nan,)), label='uncertanity [Hz]') 
    fit_contrast = Array(value=np.array((np.nan,)), label='contrast [%]')
    

    def __init__(self, odmr, auto_focus):
        super(Magnet_alignment, self).__init__()
        
        #Fluorescence part
        self.odmr = odmr
        self.auto_focus = auto_focus
        self._create_fluor_y_plot()
        self._create_fluor_x_plot()
        self.on_trait_change(self._update_plot_data_x_line, 'scanningX, data_x, run_time', dispatch='ui')
        self.on_trait_change(self._update_plot_data_y_line, 'scanningY, data_y ,run_time', dispatch='ui')
        self.on_trait_change(self._update_plot_image, 'scanningX,scanningY,data_xy, run_time', dispatch='ui')
        
        self.on_trait_change(self._update_line_data_fit, 'perform_fit, fitting_func,fit_parameters, run_time', dispatch='ui')
        self.on_trait_change(self._update_fit, 'fit, perform_fit,run_time', dispatch='ui')
        self.on_trait_change(self._update_position, 'perform_fit, fitting_func,fit_parameters, run_time', dispatch='ui')
        self.on_trait_change(self._update_point, 'matrix_lines', dispatch='ui')
        self.on_trait_change(self._update_plot_data_multiline_ODMR, 'matrix_lines', dispatch='ui')
        
        
        #ODMR part
        
        self._create_fluor_y_plot_ODMR()
        self._create_fluor_x_plot_ODMR()
        self._create_matrix_plot_ODMR()
        self._create_multi_plot_ODMR()
        
        #self.on_trait_change(self._update_counts_matrix_shape, 'frequency_begin_p', dispatch='ui')
        self.on_trait_change(self._update_plot_data_x_line_ODMR, 'scanningX, data_x_ODMR, run_time', dispatch='ui')
        self.on_trait_change(self._update_plot_data_y_line_ODMR, 'scanningY, data_y_ODMR ,run_time', dispatch='ui')
        self.on_trait_change(self._update_plot_image_ODMR, 'scanningX,scanningY,data_xy_ODMR, run_time', dispatch='ui')

        self.on_trait_change(self._update_matrix_data_value, 'counts_matrix_ODMR', dispatch='ui')
        self.on_trait_change(self._update_matrix_data_index, 'run_time', dispatch='ui')
        
        
    def _set_button_fired(self):
         Magnet.move_absolute(1, self.set_X_position)
         Magnet.move_absolute(2, self.set_Y_position)
         Magnet.move_absolute(3, self.set_Z_position)
         
    def _get_button_fired(self):
         self.get_X_position = float(Magnet.get_current_position(1)[3:-2])
         self.get_Y_position = float(Magnet.get_current_position(2)[3:-2])
         self.get_Z_position = float(Magnet.get_current_position(3)[3:-2])    
         
    def _apply_button_fired(self):
         self.set_X_position = self.get_X_position
         self.set_Y_position = self.get_Y_position
         self.set_Z_position = self.get_Z_position  
         
    def _scanningX_default(self):
         return np.arange(self.start_positionX, self.end_positionX, self.stepX)
         
    def _scanningY_default(self):
         return np.arange(self.start_positionY, self.end_positionY, self.stepY)
                  
    # default data     
    
    def _data_x_default(self):
        return np.zeros(self.scanningX.shape)     
        
    def _data_y_default(self):
        return np.zeros(self.scanningY.shape)       
    
    def _data_xy_default(self):
        return np.zeros((self.scanningX.shape[0],self.scanningY.shape[0]))      
        
    def _plot_data_image_default(self):
        return ArrayPlotData(image=np.zeros((2,2)))  
    
    def _data_x_ODMR_default(self):
        return np.zeros(self.scanningX.shape)     
        
    def _data_y_ODMR_default(self):
        return np.zeros(self.scanningY.shape)       
    
    def _data_xy_ODMR_default(self):
        return np.zeros((self.scanningX.shape[0],self.scanningY.shape[0]))      
        
    def _plot_data_image_ODMR_default(self):
        return ArrayPlotData(image=np.zeros((2,2)))     
    
    def _counts_default(self):
        return np.zeros(self.frequency.shape)        
        
        
    def apply_parameters(self):
                
        self.odmr.pulsed = True
        self.odmr.perform_fit = True
        self.odmr.power_p = self.power_p
        self.odmr.t_pi = self.t_pi
        self.odmr.stop_time = self.stop_time 
        
        self.odmr.frequency_begin_p = self.frequency_begin_p
        self.odmr.frequency_end_p = self.frequency_end_p
        self.odmr.frequency_delta_p = self.frequency_delta_p
        self.odmr.number_of_resonances = 2
        self.odmr.threshold = self.odmr_threshold
        
        #self.counts_matrix_ODMR = np.zeros((self.nlines, len(self.odmr.frequency)))
        
    # data acquisition
        
    def _run(self):
        
        logging.getLogger().debug("trying run.")
        
        
        try:
            self.state='run'
            start_time = time.time()
            
            
            self.stepX = int(self.stepX / 2e-4) * 2e-4
            self.stepY = int(self.stepY / 2e-4) * 2e-4
                         
            self.scanningX = np.arange(self.start_positionX, self.end_positionX, self.stepX)
            self.scanningY = np.arange(self.start_positionY, self.end_positionY, self.stepY)
            
            self.Number_of_scanpointsX = int(len(self.scanningX))
            self.Number_of_scanpointsY = int(len(self.scanningY))
            
            
            scanningX_reverse = self.scanningX[::-1]
            
            
            if self.choosed_align == 'fluoresence':
            
                if self.choosed_scan == 'fx_scan':
                    
                    self.data_x = np.zeros(self.Number_of_scanpointsX)
                    motion_time = float(Magnet.get_motiontime_relativemove(1, self.stepY)[3:-2])
                    
                    self.expected_duration = self.Number_of_scanpointsX * (self.Acquisition_time + motion_time)
                    
                    Magnet.move_absolute(1, self.start_positionX)
                    time.sleep(15)
                    self._get_button_fired()
                    
                    for xn, xpos in enumerate(self.scanningX):
                        if threading.currentThread().stop_request.isSet():
                            break
                            Magnet.stop_motion(1)
                        self.run_time = time.time() - start_time
                        
                        Magnet.move_absolute(1, xpos)
                        time.sleep(motion_time) 
                        self.get_X_position = xpos
                       
                
                        counter_tr.startCounter()
                
                        time.sleep(self.Acquisition_time)
                 
                        self.data_x[xn] = counter_tr.Count()
                        
                        counter_tr.stopCounter()
                        
                        self.trait_property_changed('data_x', self.data_x)
                        
                        if self.periodic_focus is True:
                            if xn % self.Npoint_auto_focus == 0:
                                self.auto_focus.start()
                                time.sleep(10.0)
                        
                    self.state = 'done'    
                    
                    
                elif self.choosed_scan == 'fy_scan':
                    self.data_y = np.zeros(self.Number_of_scanpointsY)
                    motion_time = float(Magnet.get_motiontime_relativemove(1, self.stepY)[3:-2])
                    Magnet.move_absolute(2, self.start_positionY)
                    time.sleep(10)
                    self._get_button_fired()
                    
                    self.expected_duration = self.Number_of_scanpointsY * (self.Acquisition_time + motion_time)
                    time.sleep(3)
                    
                    for yn, ypos in enumerate(self.scanningY):
                        if threading.currentThread().stop_request.isSet():
                            break
                            Magnet.stop_motion(2)
                        self.run_time = time.time() - start_time
                        Magnet.move_absolute(2, ypos)
                        time.sleep(motion_time) 
                        self.get_Y_position = ypos
                        
                        counter_tr.startCounter()
                        time.sleep(self.Acquisition_time)
                        self.data_y[yn] = counter_tr.Count()
                        counter_tr.stopCounter()
                        self.trait_property_changed('data_y', self.data_y)
                        
                        if self.periodic_focus is True:
                            if yn % self.Npoint_auto_focus == 0:
                                self.auto_focus.start()
                                time.sleep(10.0)
                        
                    self.state = 'done'        
                    
                    
                elif self.choosed_scan == 'f2D_scan':
                
                    Magnet.move_absolute(1, self.start_positionX)
                    
                    Magnet.move_absolute(2, self.start_positionY)
                    
                    time.sleep(15)
                    self._get_button_fired()
                    
                    self.data_xy=np.zeros((self.Number_of_scanpointsX,self.Number_of_scanpointsY))
                    motion_time_x = float(Magnet.get_motiontime_relativemove(1, self.stepX)[3:-2])
                    motion_time_y = float(Magnet.get_motiontime_relativemove(1, self.stepY)[3:-2])
                    self.expected_duration = self.Number_of_scanpointsX * self.Number_of_scanpointsY * (self.Acquisition_time + motion_time_x + motion_time_y)
                    
                    for yn , ypos in enumerate(self.scanningY):
                        Magnet.move_absolute(2, ypos)
                        time.sleep(10)
                        #self._get_button_fired()
                        self.get_Y_position = ypos
                        
                        if yn%2 == 0:
                            for xn, xpos in enumerate(self.scanningX):
                                if threading.currentThread().stop_request.isSet():
                                    break
                                    Magnet.stop_motion(1)
                                    Magnet.stop_motion(2)
                                self.run_time = time.time() - start_time
                                Magnet.move_absolute(1, xpos)
                                time.sleep(motion_time_x) 
                                
                                self.get_X_position = xpos
                                
                                
                                counter_tr.startCounter()
                                time.sleep(self.Acquisition_time)
                                self.data_xy[xn, yn] = counter_tr.Count()
                            
                                counter_tr.stopCounter()
                                self.trait_property_changed('data_xy', self.data_xy)
                                
                                if self.periodic_focus is True:
                                    if xn % self.Npoint_auto_focus == 0:
                                        self.auto_focus.start()
                                        time.sleep(10.0)
         
                        else:
                            for xn, xpos in enumerate(scanningX_reverse):
                                if threading.currentThread().stop_request.isSet():
                                    break
                                    Magnet.stop_motion(1)
                                    Magnet.stop_motion(2)
                                xn=self.Number_of_scanpointsX - xn - 1
                                self.run_time = time.time() - start_time
                                Magnet.move_absolute(1, xpos)
                                
                                time.sleep(motion_time_x)
                                
                                self.get_X_position = xpos                         
                            
                                counter_tr.startCounter()
                                time.sleep(self.Acquisition_time)
                                self.data_xy[xn, yn] = counter_tr.Count()
                                counter_tr.stopCounter()
                                
                                if self.periodic_focus is True:
                                    if xn % self.Npoint_auto_focus == 0:
                                        self.auto_focus.start()
                                        time.sleep(10.0)
                                
                                self.trait_property_changed('data_xy', self.data_xy)
                                
                        if threading.currentThread().stop_request.isSet():
                            break   
                            
                    self.state = 'done'                
    #============================================================================================================================================================================================================================
            elif self.choosed_align == 'odmr':
               
                self.apply_parameters()
                
                if self.choosed_scan == 'fx_scan':
                    self.data_x_ODMR = np.zeros(self.Number_of_scanpointsX)
                    motion_time = float(Magnet.get_motiontime_relativemove(1, self.stepY)[3:-2])
                    self.expected_duration = self.Number_of_scanpointsX * (self.stop_time + motion_time)
                    Magnet.move_absolute(1, self.start_positionX)
                    
                    time.sleep(10)
                    self._get_button_fired()
                    
                    for xn, xpos in enumerate(self.scanningX):
                        if threading.currentThread().stop_request.isSet():
                            break
                            Magnet.stop_motion(1)
                            self.odmr.stop()   
                        self.run_time = time.time() - start_time
                        
                        Magnet.move_absolute(1, xpos)
                        time.sleep(motion_time) 
                        self.get_X_position = xpos
                               

                        self.odmr.start()   

                        while self.odmr.state != 'done':
                            #print threading.currentThread().getName()

                            threading.currentThread().stop_request.wait(1.0)
                            if threading.currentThread().stop_request.isSet():
                                 break
                                 
              
                        self.odmr.remove()   
                        self.odmr.run_time =  0 
                        self.odmr.state =  'idle' 
                        if self.counts_matrix_ODMR.shape[1] != self.odmr.counts.shape[0]:
                           self.counts_matrix_ODMR = self._counts_matrix_ODMR_default()
                        self.counts_matrix_ODMR = np.vstack((self.odmr.counts, self.counts_matrix_ODMR[:-1, :]))
                        
                        if np.isnan(self.odmr.fit_frequencies[0]) is True : 
                            self.data_x_ODMR[xn] = 50
                        elif len(self.odmr.fit_frequencies) > 2 or  self.odmr.fit_contrast[0] > 40: 
                            self.data_x_ODMR[xn] = 50
                        else:
                            if self.odmr.fit_line_width[0] > self.line_width_threshold:
                                self.data_x_ODMR[xn] = 50
                            else:
                                self.data_x_ODMR[xn] = self.odmr.fit_contrast[0]    
                       # elif self.odmr.fit_contrast[0] > self.odmr.fit_contrast[1]:
                        #    self.data_x_ODMR[xn] = self.odmr.fit_contrast[1]/self.odmr.fit_contrast[0]
                        #elif self.odmr.fit_contrast[0] < self.odmr.fit_contrast[1]:
                         #   self.data_x_ODMR[xn] = self.odmr.fit_contrast[0]/self.odmr.fit_contrast[1]    
                        if self.periodic_focus is True:
                            if xn % self.Npoint_auto_focus == 0:
                                self.auto_focus.start()
                                time.sleep(10.0)
                        
                        self.trait_property_changed('data_x_ODMR', self.data_x_ODMR)
                        
                    self.state = 'done'    
                    
                    
                elif self.choosed_scan == 'fy_scan':
                    self.data_y_ODMR = np.zeros(self.Number_of_scanpointsY)
                    motion_time = float(Magnet.get_motiontime_relativemove(1, self.stepY)[3:-2])
                    Magnet.move_absolute(2, self.start_positionY)
                    time.sleep(10)
                    self._get_button_fired()
                    
                    self.expected_duration = self.Number_of_scanpointsY * (self.stop_time + motion_time)
                    for yn, ypos in enumerate(self.scanningY):
                        if threading.currentThread().stop_request.isSet():
                            break
                            Magnet.stop_motion(2)
                            self.odmr.stop()   
                        self.run_time = time.time() - start_time
                        Magnet.move_absolute(2, ypos)
                        
                        time.sleep(motion_time) 
                       
                        self.get_Y_position = ypos
                        self.odmr.start()   

                        while self.odmr.state != 'done':
                            #print threading.currentThread().getName()

                            threading.currentThread().stop_request.wait(1.0)
                            if threading.currentThread().stop_request.isSet():
                                 break
                                 
              
                        self.odmr.remove()   
                        self.odmr.run_time =  0 
                        self.odmr.state =  'idle' 
                        
                        if self.counts_matrix_ODMR.shape[1] != self.odmr.counts.shape[0]:
                           self.counts_matrix_ODMR = self._counts_matrix_ODMR_default()
                        self.counts_matrix_ODMR = np.vstack((self.odmr.counts, self.counts_matrix_ODMR[:-1, :]))
                        
                        if np.isnan(self.odmr.fit_frequencies[0]) is True : 
                            self.data_y_ODMR[yn] = 50
                        elif len(self.odmr.fit_frequencies) > 2 or  self.odmr.fit_contrast[0] > 40:  
                            self.data_y_ODMR[yn] = 50
                        else:
                            if self.odmr.fit_line_width[0] > self.line_width_threshold:
                                self.data_y_ODMR[yn] = 50
                            else:
                                self.data_y_ODMR[yn] = self.odmr.fit_contrast[0]    
                            
                        if self.periodic_focus is True:
                            if yn % self.Npoint_auto_focus == 0:
                                self.auto_focus.start()
                                time.sleep(10.0)    
                        self.trait_property_changed('data_y_ODMR', self.data_y_ODMR)
                        
                    self.state = 'done'        
                    
                    
                elif self.choosed_scan == 'f2D_scan':
                
                    Magnet.move_absolute(1, self.start_positionX)

                    Magnet.move_absolute(2, self.start_positionY)
                    
                    time.sleep(10)
                    self._get_button_fired()
                    
                    self.data_xy_ODMR=np.zeros((self.Number_of_scanpointsX,self.Number_of_scanpointsY))
                    motion_time_x = float(Magnet.get_motiontime_relativemove(1, self.stepX)[3:-2])
                    motion_time_y = float(Magnet.get_motiontime_relativemove(1, self.stepY)[3:-2])
                    self.expected_duration = self.Number_of_scanpointsX * self.Number_of_scanpointsY * (self.stop_time + motion_time_x + motion_time_y)        
                    for yn , ypos in enumerate(self.scanningY):
                        Magnet.move_absolute(2, ypos)
                        
                        self.get_Y_position = ypos
                        if yn%2 == 0:
                            for xn, xpos in enumerate(self.scanningX):
                                if threading.currentThread().stop_request.isSet():
                                    break
                                    Magnet.stop_motion(1)
                                    Magnet.stop_motion(2)
                                    self.odmr.stop()   
                                self.run_time = time.time() - start_time
                                Magnet.move_absolute(1, xpos)
                                time.sleep(motion_time_x) 
                                self.get_X_position = xpos
                                
                                
                                self.odmr.start()   

                                while self.odmr.state != 'done':
                                    #print threading.currentThread().getName()

                                    threading.currentThread().stop_request.wait(1.0)
                                    if threading.currentThread().stop_request.isSet():
                                         break
                                         
                      
                                self.odmr.remove()   
                                self.odmr.run_time =  0 
                                self.odmr.state =  'idle' 
                                
                                if self.counts_matrix_ODMR.shape[1] != self.odmr.counts.shape[0]:
                                    self.counts_matrix_ODMR = self._counts_matrix_ODMR_default()
                                self.counts_matrix_ODMR = np.vstack((self.odmr.counts, self.counts_matrix_ODMR[:-1, :]))
                                
                                if np.isnan(self.odmr.fit_frequencies[0]) is True : 
                                    self.data_xy_ODMR[xn,yn] = 50
                                elif len(self.odmr.fit_frequencies) > 2 or  self.odmr.fit_contrast[0] > 40: 
                                    self.data_xy_ODMR[xn,yn] = 50
                                else:
                                    if self.odmr.fit_line_width[0] > self.line_width_threshold:
                                        self.data_xy_ODMR[xn,yn] = 50
                                    else:
                                        self.data_xy_ODMR[xn,yn] = self.odmr.fit_contrast[0]    
                                if self.periodic_focus is True:
                                    if xn % self.Npoint_auto_focus == 0:
                                        self.auto_focus.start()
                                        time.sleep(10.0)    
                                self.trait_property_changed('data_xy_ODMR', self.data_xy_ODMR)
         
                        else:
                            for xn, xpos in enumerate(scanningX_reverse):
                                if threading.currentThread().stop_request.isSet():
                                    break
                                    Magnet.stop_motion(1)
                                    Magnet.stop_motion(2)
                                    self.odmr.stop()   
                                xn=self.Number_of_scanpointsX - xn - 1
                                self.run_time = time.time() - start_time
                                Magnet.move_absolute(1, xpos)
                                time.sleep(motion_time_x)
                                self.get_X_position = xpos                      
                                
                                self.odmr.start()   

                                while self.odmr.state != 'done':
                                    #print threading.currentThread().getName()

                                    threading.currentThread().stop_request.wait(1.0)
                                    if threading.currentThread().stop_request.isSet():
                                         break
                                         
                      
                                self.odmr.remove()   
                                self.odmr.run_time =  0 
                                self.odmr.state =  'idle'    

                                if self.counts_matrix_ODMR.shape[1] != self.odmr.counts.shape[0]:
                                    self.counts_matrix_ODMR = self._counts_matrix_ODMR_default()
                                self.counts_matrix_ODMR = np.vstack((self.odmr.counts, self.counts_matrix_ODMR[:-1, :]))
                                
                                if np.isnan(self.odmr.fit_frequencies[0]) is True : 
                                    self.data_xy_ODMR[xn,yn] = 50
                                elif len(self.odmr.fit_frequencies) > 2 or  self.odmr.fit_contrast[0] > 40: 
                                    self.data_xy_ODMR[xn,yn] = 50
                                else:
                                    if self.odmr.fit_line_width[0] > self.line_width_threshold:
                                        self.data_xy_ODMR[xn,yn] = 50
                                    else:
                                        self.data_xy_ODMR[xn,yn] = self.odmr.fit_contrast[0]    
                                    
                                if self.periodic_focus is True:
                                    if xn % self.Npoint_auto_focus == 0:
                                        self.auto_focus.start()
                                        time.sleep(10.0)    
                                self.trait_property_changed('data_xy_ODMR', self.data_xy_ODMR)
                                
                        if threading.currentThread().stop_request.isSet():
                            break   
                    self.state = 'done'          
               
    #========================================================================================================================================================================================================================               
            
        except:
            logging.getLogger().exception('Error in magnet alignment')
            self.state = 'error'
 
    # plotting
   
    def _update_plot_data_x_line(self):
        if self.choosed_align == 'fluoresence':
            self.plot_data_x_line.set_data('scanningX', self.scanningX)
            self.plot_data_x_line.set_data('data_x', self.data_x)
        

    def _create_fluor_x_plot(self):
        plot_data_x_line = ArrayPlotData(scanningX=np.array((0., 1.)), data_x=np.array((0., 0.)), fit=np.array((0., 0.)) )
        plot = Plot(plot_data_x_line, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('scanningX','data_x'), color='blue')
        plot.index_axis.title = 'x scan [mm]'
        plot.value_axis.title = 'Fluorescence [ counts / s ]'
        
        #line_label = PlotLabel(text='', hjustify='left', vjustify='bottom', position=[64, 128])
        #plot.overlays.append(line_label)
        #self.line_label = line_label

        self.plot_data_x_line = plot_data_x_line
        self.fluor_x_plot = plot
        return self.fluor_x_plot    
        
    def _update_plot_data_y_line(self):
        if self.choosed_align == 'fluoresence':
            self.plot_data_y_line.set_data('scanningY', self.scanningY)
            self.plot_data_y_line.set_data('data_y', self.data_y)
        
    def _create_fluor_y_plot(self):
        plot_data_y_line = ArrayPlotData(scanningY=np.array((0., 1.)), data_y=np.array((0., 0.)),  fit=np.array((0., 0.)) )
        plot = Plot(plot_data_y_line, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('scanningY','data_y'), color='blue')
        plot.index_axis.title = 'y scan [mm]'
        plot.value_axis.title = 'Fluorescence [ counts / s ]'
        #line_label = PlotLabel(text='', hjustify='left', vjustify='bottom', position=[64, 128])
        #plot.overlays.append(line_label)
        #self.line_label = line_label
        
        self.plot_data_y_line = plot_data_y_line
        self.fluor_y_plot = plot
        return self.fluor_y_plot  
        
        
    def _plot_data_image_default(self):
        return ArrayPlotData(image=np.zeros((2,2)))
        
    def _update_plot_image(self):
        self.plot_data_image.set_data('image', self.data_xy)
        
    def _image_plot_default(self):
        return self.fluor_2D_plot.components[0].plots['image'][0]     
        
    def _fluor_2D_plot_default(self):
        plot = Plot(self.plot_data_image, width=180, height=180, padding=3, padding_left=48, padding_bottom=32)
        plot.img_plot('image', colormap=jet, name='image')
        plot.aspect_ratio=1
        plot.value_mapper.domain_limits = (self.start_positionX, self.end_positionX)
        plot.index_mapper.domain_limits = (self.start_positionY, self.end_positionY)

        container = HPlotContainer()
        image = plot.plots['image'][0]
        colormap = image.color_mapper
        colorbar = ColorBar(index_mapper=LinearMapper(range=colormap.range),
                            color_mapper=colormap,
                            plot=plot,
                            orientation='v',
                            resizable='v',
                            width=20,
                            height=200,
                            padding=8,
                            padding_left=20)
        container = HPlotContainer()
        container.add(plot)
        container.add(colorbar)
        
        return container
        
    #ODMR curves plotting========================================================================================================================================================================================================
    #def _update_counts_matrix_shape(self):
     #   self.counts_matrix_ODMR = self._counts_matrix_ODMR_default()
        
    def _update_plot_data_x_line_ODMR(self):
        if self.choosed_align == 'odmr':
            self.plot_data_x_line_ODMR.set_data('scanningX', self.scanningX)
            self.plot_data_x_line_ODMR.set_data('data_x_ODMR', self.data_x_ODMR)
        
    def _create_fluor_x_plot_ODMR(self):
        plot_data_x_line_ODMR = ArrayPlotData(scanningX=np.array((0., 1.)), data_x_ODMR=np.array((0., 0.)),  fit=np.array((0., 0.)) )
        plot = Plot(plot_data_x_line_ODMR, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('scanningX','data_x_ODMR'), color='blue')
        plot.index_axis.title = 'x ODMR scan [mm]'
        plot.value_axis.title = 'Mutual contrast, %'
        line_label = PlotLabel(text='', hjustify='left', vjustify='bottom', position=[64, 128])
        plot.overlays.append(line_label)
        self.line_label = line_label
        self.plot_data_x_line_ODMR = plot_data_x_line_ODMR
        self.fluor_x_plot_ODMR = plot
        return self.fluor_x_plot_ODMR   
        
        
             
    def _update_plot_data_y_line_ODMR(self):
        if self.choosed_align == 'odmr':
            self.plot_data_y_line_ODMR.set_data('scanningY', self.scanningY)
            self.plot_data_y_line_ODMR.set_data('data_y_ODMR', self.data_y_ODMR)
        
    def _create_fluor_y_plot_ODMR(self):
        plot_data_y_line_ODMR = ArrayPlotData(scanningY=np.array((0., 1.)), data_y_ODMR=np.array((0., 0.)), fit=np.array((0., 0.)) )
        plot = Plot(plot_data_y_line_ODMR, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('scanningY','data_y_ODMR'), color='blue')
        plot.index_axis.title = 'y ODMR scan [mm]'
        plot.value_axis.title = 'Mutual contrast, %'
        #line_label = PlotLabel(text='', hjustify='left', vjustify='bottom', position=[64, 128])
        #plot.overlays.append(line_label)
        #self.line_label = line_label
        self.plot_data_y_line_ODMR = plot_data_y_line_ODMR
        self.fluor_y_plot_ODMR = plot
        return self.fluor_y_plot_ODMR
        
    def _update_plot_data_multiline_ODMR(self):
        self.plot_data_multiline_ODMR.set_data('x', np.arange(self.frequency_begin_p, self.frequency_end_p + self.frequency_delta_p, self.frequency_delta_p))
        #ind =(self.Number_of_scanpointsY - self.matrix_y - 1) * self.Number_of_scanpointsX + (self.Number_of_scanpointsX - self.matrix_x - 1)
        self.plot_data_multiline_ODMR.set_data('y', self.counts_matrix_ODMR[self.matrix_lines, : ])   
        
    def _update_point(self):
        y1 = self.matrix_lines / self.Number_of_scanpointsX
        x1 = self.matrix_lines % self.Number_of_scanpointsX
        scanningX = np.arange(self.start_positionX, self.end_positionX, self.stepX)
        scanningY = np.arange(self.start_positionY, self.end_positionY, self.stepY)
        self.x_point = scanningX[self.Number_of_scanpointsX - x1 - 1]
        self.y_point = scanningY[self.Number_of_scanpointsY - y1 - 1]
    
    def _create_multi_plot_ODMR(self):
        plot_data_multiline_ODMR = ArrayPlotData(x=np.array((0., 1.)), y=np.array((0., 0.)))
        plot = Plot(plot_data_multiline_ODMR, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('x','y'), color='blue')
        #x_key = 'x'
        #plot_data_multiline_ODMR.set_data(x_key, self.odmr.frequency)
        '''
        if self.choosed_plot == 'xscan':
            for nk in np.arange(self.n_start,self.n_end,1):
                #x = np.array(track['xw'])
                #y = np.array(track['yw'])
                #x_key = 'x'+str(nk)
                y_key = 'y'+str(nk)
                
                nk = self.Number_of_scanpointsX * (self.mline - 1) + nk
                plot_data_multiline_ODMR.set_data(y_key, self.counts_matrix_ODMR[nk,:])
                plot.plot((x_key, y_key), color='blue')
        
        elif self.choosed_plot == 'yscan':
                
            for nk in np.arange(self.n_start,self.n_end,1):
                #x = np.array(track['xw'])
                #y = np.array(track['yw'])
                #x_key = 'x'+str(nk)
                y_key = 'y'+str(nk)
                
                nk = self.Number_of_scanpointsX * nk + self.mline
                plot_data_multiline_ODMR.set_data(y_key, self.counts_matrix_ODMR[nk,:])
                plot.plot((x_key, y_key), color='blue')
                '''
        plot.index_axis.title = 'Frequency [MHz]'
        self.multi_plot_ODMR = plot
        self.plot_data_multiline_ODMR = plot_data_multiline_ODMR


        
    def _plot_data_image_ODMR_default(self):
        return ArrayPlotData(image=np.zeros((10, 10)))
        
    def _update_plot_image_ODMR(self):
        self.plot_data_image_ODMR.set_data('image', self.data_xy_ODMR)
        
    def _image_plot_ODMR_default(self):
        return self.fluor_2D_plot_ODMR.components[0].plots['image'][0]     
        
    def _fluor_2D_plot_ODMR_default(self):
        plot = Plot(self.plot_data_image_ODMR, width=180, height=180, padding=3, padding_left=48, padding_bottom=32)
        plot.img_plot('image', colormap=jet, name='image', xbounds = (self.scanningX[0],self.scanningX[-1],self.Number_of_scanpointsX),ybounds = (self.scanningY[0],self.scanningY[-1],self.Number_of_scanpointsY))
        plot.aspect_ratio=1
        plot.value_mapper.domain_limits = (self.start_positionX, self.end_positionX)
        plot.index_mapper.domain_limits = (self.start_positionY, self.end_positionY)

        container = HPlotContainer()
        image = plot.plots['image'][0]
        colormap = image.color_mapper
        colorbar = ColorBar(index_mapper=LinearMapper(range=colormap.range),
                            color_mapper=colormap,
                            plot=plot,
                            orientation='v',
                            resizable='v',
                            width=20,
                            height=200,
                            padding=8,
                            padding_left=20)
        container = HPlotContainer()
        container.add(plot)
        container.add(colorbar)
        
        return container
        
    #ODMR tools==================================================================================================================================================================================================================    

    def _counts_matrix_ODMR_default(self):
        return np.zeros((int(self.nlines), len(self.odmr.frequency)))

        
    def _create_matrix_plot_ODMR(self):
        matrix_data = ArrayPlotData(image=np.zeros((2, 2)))
        matrix_plot = Plot(matrix_data, padding=8, padding_left=64, padding_bottom=32)
        matrix_plot.index_axis.title = 'Frequency [MHz]'
        matrix_plot.value_axis.title = 'line #'
        matrix_plot.img_plot('image',
                             xbounds=(self.odmr.frequency[0]/1e6, self.odmr.frequency[-1]/1e6),
                             ybounds=(0, int(self.nlines)),
                             colormap=Spectral)
        self.matrix_data_ODMR = matrix_data
        self.matrix_plot_ODMR = matrix_plot
        
    def _update_matrix_data_value(self):
        self.matrix_data_ODMR.set_data('image', self.counts_matrix_ODMR )

    def _update_matrix_data_index(self):
        if int(self.nlines) > self.counts_matrix_ODMR .shape[0]:
            self.counts_matrix_ODMR = np.vstack((self.counts_matrix_ODMR , np.zeros((int(self.nlines) - self.counts_matrix_ODMR .shape[0], self.counts_matrix_ODMR .shape[1]))))
        else:
            self.counts_matrix_ODMR  = self.counts_matrix_ODMR [:int(self.nlines)]
        self.matrix_plot_ODMR.components[0].index.set_data((self.odmr.frequency[0] * 1e-6, self.odmr.frequency[-1] * 1e-6), (0.0, float(int(self.nlines))))
        
    #fitting---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    
    def _update_position(self):
        if not np.isnan(self.fit_parameters[0]):
            if self.choosed_scan == 'fx_scan': 
                self.Xmax = float(self.fit_parameters[1::3].max())

            elif self.choosed_scan == 'fy_scan': 
                self.Ymax = float(self.fit_parameters[1::3].max())   
            
    def _perform_fit_changed(self, new):
    
       
        if (self.choosed_align == 'fluoresence') & (self.choosed_scan == 'fx_scan'):
            plot = self.fluor_x_plot
            x_name = self.plot_data_x_line.list_data()[0]
        elif (self.choosed_align == 'fluoresence') & (self.choosed_scan == 'fy_scan'):
            plot = self.fluor_y_plot 
            x_name = self.plot_data_y_line.list_data()[1]
            #popt = fitting.fit_gaussian(self.scanningY, self.data_y)
            #plot.plot(self.scanningY, fitting.Gaussian(self.scanningY,*popt),'ro:', style='line', color='red')
        elif (self.choosed_align == 'odmr') & (self.choosed_scan == 'fx_scan'):
            plot = self.fluor_x_plot_ODMR
            x_name = self.plot_data_x_line_ODMR.list_data()[0]
            #popt = fitting.fit_gaussian(self.scanningX, self.data_x_ODMR)
            #plot.plot(self.scanningX, fitting.Gaussian(self.scanningX,*popt),'ro:', style='line', color='red')
        elif (self.choosed_align == 'odmr') & (self.choosed_scan == 'fy_scan'):
            plot = self.fluor_y_plot_ODMR
            x_name = self.plot_data_y_line_ODMR.list_data()[0]
            #popt = fitting.fit_gaussian(self.scanningY, self.data_y_ODMR)
            #plot.plot(self.scanningY, fitting.Gaussian(self.scanningY,*popt),'ro:', style='line', color='red')
                
        if new:
            plot.plot((x_name, 'fit'), style='line', color='red', name='fit')
            #self.line_label.visible = True
        else:
            plot.delplot('fit')
            #self.line_label.visible = False
        plot.request_redraw()
            
            
    def _update_fit(self):
        if self.perform_fit:
            N = self.number_of_resonances 
            if N != 'auto':
                N = int(N)
                
            if self.fitting_func == 'loretzian':
                fit_func = fitting.fit_multiple_lorentzians
            elif self.fitting_func == 'gaussian':
                fit_func = fitting.fit_multiple_gaussian
                
            if (self.choosed_align == 'fluoresence') & (self.choosed_scan == 'fx_scan'):
                self.fit_x = self.scanningX
                self.counts = self.data_x
                
            elif (self.choosed_align == 'fluoresence') & (self.choosed_scan == 'fy_scan'):
                self.fit_x = self.scanningY
                self.counts = self.data_y 
                #popt = fitting.fit_gaussian(self.scanningY, self.data_y)
                #plot.plot(self.scanningY, fitting.Gaussian(self.scanningY,*popt),'ro:', style='line', color='red')
            elif (self.choosed_align == 'odmr') & (self.choosed_scan == 'fx_scan'):
                self.fit_x = self.scanningX
                self.counts = self.data_x_ODMR 
                #popt = fitting.fit_gaussian(self.scanningX, self.data_x_ODMR)
                #plot.plot(self.scanningX, fitting.Gaussian(self.scanningX,*popt),'ro:', style='line', color='red')
            elif (self.choosed_align == 'odmr') & (self.choosed_scan == 'fy_scan'):
                self.fit_x = self.scanningY
                self.counts = self.data_y_ODMR 

            
            try:
                p = fit_func(self.fit_x, self.counts, N, threshold=self.fit_threshold * 0.01)
            except Exception:
                logging.getLogger().debug('fit failed.', exc_info=True)
                p = np.nan * np.empty(4)
        else:
            p = np.nan * np.empty(4)
            
        self.fit_parameters = p
        self.fit_centers = p[1::3]
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
            
    def _update_line_data_fit(self):
        if not np.isnan(self.fit_parameters[0]):       
            if self.fitting_func == 'loretzian':  
                fit_func = fitting.NLorentzians
            elif self.fitting_func == 'gaussian':
                fit_func = fitting.NGaussian
                
            if (self.choosed_align == 'fluoresence') & (self.choosed_scan == 'fx_scan'):
                #self.plot_data_x_line.set_data('fit_x', self.fit_x)
                self.plot_data_x_line.set_data('fit', fit_func(*self.fit_parameters)(self.fit_x))
                
            elif (self.choosed_align == 'fluoresence') & (self.choosed_scan == 'fy_scan'):
                #self.plot_data_y_line.set_data('fit_x', self.fit_x)
                self.plot_data_y_line.set_data('fit', fit_func(*self.fit_parameters)(self.fit_x))
                #popt = fitting.fit_gaussian(self.scanningY, self.data_y)
                #plot.plot(self.scanningY, fitting.Gaussian(self.scanningY,*popt),'ro:', style='line', color='red')
            elif (self.choosed_align == 'odmr') & (self.choosed_scan == 'fx_scan'):
                #self.plot_data_x_line_ODMR.set_data('fit_x', self.fit_x)
                self.plot_data_x_line_ODMR.set_data('fit', fit_func(*self.fit_parameters)(self.fit_x))
                #popt = fitting.fit_gaussian(self.scanningX, self.data_x_ODMR)
                #plot.plot(self.scanningX, fitting.Gaussian(self.scanningX,*popt),'ro:', style='line', color='red')
            elif (self.choosed_align == 'odmr') & (self.choosed_scan == 'fy_scan'):
                #self.plot_data_y_line_ODMR.set_data('fit_x', self.fit_x)
                self.plot_data_y_line_ODMR.set_data('fit', fit_func(*self.fit_parameters)(self.fit_x))
                
 
                
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
                s += 'f %i: %.6e mm, HWHM %.3e mm, contrast %.1f%%\n' % (i + 1, fi, w[i], contrast[i])
            #self.line_label.text = s        
       
    # saving data================================================================================================================================================================================================================
        
              
    def save_fluor_x_plot(self, filename):
        self.save_figure(self.fluor_x_plot, filename + 'FluorXPlot_'+ '_Z=' + string.replace(str(self.set_Z_position), '.', 'd') + '_Y=' + string.replace(str(self.set_Y_position), '.', 'd') + '.png' )
        self.save(filename +'FluorXPlot_'+ '_Z=' + string.replace(str(self.set_Z_position), '.', 'd') + '_Y=' + string.replace(str(self.set_Y_position), '.', 'd') + '.pyd' )
        
    def save_fluor_y_plot(self, filename):
        self.save_figure(self.fluor_y_plot, filename + 'FluorYPlot_' + '_Z=' + string.replace(str(self.set_Z_position), '.', 'd') + '_X=' + string.replace(str(self.set_X_position), '.', 'd') + '.png')
        self.save(filename + 'FluorYPlot_' + '_Z=' + string.replace(str(self.set_Z_position), '.', 'd') + '_X=' + string.replace(str(self.set_X_position), '.', 'd') + '.pyd')
        
    def save_fluor_2D_plot(self, filename):
        self.save_figure(self.fluor_2D_plot, filename + 'Fluor2DPlot_' + '_Z=' + string.replace(str(self.set_Z_position), '.', 'd') + '.png')
        self.save(filename + 'Fluor2Dlot_' + '_Z=' + string.replace(str(self.set_Z_position), '.', 'd') + '.pyd')

    def save_fluor_x_plot_ODMR(self, filename):
        self.save_figure(self.fluor_x_plot_ODMR, filename + 'ODMRXPlot_' + '_Z=' + string.replace(str(self.set_Z_position), '.', 'd') + '_Y=' + string.replace(str(self.set_Y_position), '.', 'd') + '.png')
        self.save_figure(self.matrix_plot_ODMR, filename + 'Matrix_X_ODMR.png')
        self.save(filename + 'ODMRXPlot_' + '_Z=' + string.replace(str(self.set_Z_position), '.', 'd') + '_Y=' + string.replace(str(self.set_Y_position), '.', 'd') + '.pyd')
        
    def save_fluor_y_plot_ODMR(self, filename):
        self.save_figure(self.fluor_y_plot_ODMR, filename + 'FluorYPlot_' + '_Z=' + string.replace(str(self.set_Z_position), '.', 'd') + '_X=' + string.replace(str(self.set_X_position), '.', 'd') + '.png')
        self.save_figure(self.matrix_plot_ODMR, filename + 'Matrix_Y_ODMR.png')
        self.save(filename + 'ODMRYPlot_' + '_Z=' + string.replace(str(self.set_Z_position), '.', 'd') + '_Y=' + string.replace(str(self.set_Y_position), '.', 'd') + '.pyd')
        
    def save_fluor_2D_plot_ODMR(self, filename):
        self.save_figure(self.fluor_2D_plot_ODMR, filename + 'FluorXPlot_'+ '_Z=' + string.replace(str(self.set_Z_position), '.', 'd') + '.png')
        self.save_figure(self.matrix_plot_ODMR, filename + '_Matrix_2D_ODMR.png') 
        self.save(filename + 'ODMR2DPlot_'+  '_Z=' + str(self.set_Z_position) + '_Y=' + string.replace(str(self.set_Z_position), '.', 'd') + '.pyd')
        
    # def save_all(self, filename):
    
        # save_fluor_x_plot(filename + '_FluorXPlot.png')
        # save_fluor_y_plot(filename + '_FluorYPlot.png')
        # save_fluor_2D_plot(filename + '_ODMR_Line_Plot.png')
        # self.save(filename + '_Fluoresc_Alignm.pyd')

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
        
    traits_view =  View(HGroup( VGroup ( HGroup (Item('submit_button', show_label=False),
                                          Item('remove_button', show_label=False),
                                          Item('resubmit_button', show_label=False),
                                          Item('state', style='readonly'),
                                          Item('run_time', style='readonly', width= -70, format_str='%.f'),
                                          Item('expected_duration', style='readonly', format_str='%.f'),
                                          ),
                                  HGroup(Item('get_X_position', width= -60, style='readonly', format_str='%f'),
                                         Item('get_Y_position', width= -60,  style='readonly', format_str='%f'),
                                         Item('get_Z_position', width= -60,  style='readonly', format_str='%f'),
                                         Item('get_button', show_label=False),
                                         Item('apply_button', show_label=False),
                                        ),
                                                   
                                  HGroup(Item('set_X_position', width= -60, enabled_when='state != "run"'),
                                         Item('set_Y_position', width= -60, enabled_when='state != "run"'),
                                         Item('set_Z_position', width= -60, enabled_when='state != "run"'),
                                         Item('set_button', show_label=False),
                                        ),
                                        
                                  HGroup(Item('Acquisition_time', width= -50, enabled_when='state != "run"'),
                                         Item('periodic_focus', show_label=True, enabled_when='state != "run"'),
                                         Item('Npoint_auto_focus', width= -60, enabled_when='state != "run"'),
                                        ),
                                        
                                  HGroup(Item('choosed_align', style='custom', show_label=False, enabled_when='state != "run"'),
                                         Item('choosed_scan',  style='custom', show_label=False, enabled_when='state != "run"'),
                                        ),
                                       
                                  HGroup(Item('start_positionX', width= -70, enabled_when='state != "run"'),
                                         Item('end_positionX', width= -70, enabled_when='state != "run"'),
                                         Item('stepX', width= -70, enabled_when='state != "run"'),
                                         Item('Number_of_scanpointsX', style='readonly'),
                                        ),
                                        
                                  HGroup(Item('start_positionY', width= -70, enabled_when='state != "run"'),
                                         Item('end_positionY', width= -70, enabled_when='state != "run"'),
                                         Item('stepY', width= -70, enabled_when='state != "run"'),
                                         Item('Number_of_scanpointsY', style='readonly'),
                                        ),  
                                        
                                  HGroup(
                                         Item('perform_fit'),
                                         Item('fitting_func', style='custom', show_label=False),
                                         Item('number_of_resonances', width= -50),
                                         Item('fit_threshold', width= -50),
                                        
                                        ),
                                        
                                  HGroup(
                                         Item('Xmax', style='readonly', format_str='%.4f'),
                                         Item('Ymax', style='readonly', format_str='%.4f'),
                                        ),      
                                     
                                  VSplit(HGroup(Item('fluor_x_plot', show_label=False, resizable=True),
                                                Item('fluor_y_plot', show_label=False, resizable=True)), 
                                         Item('fluor_2D_plot', show_label=False, resizable=True), label='fluorescence scan'
                                        ),
                                ),
                                            
                         VGroup (HGroup ( Item('stop_time'),
                                          Item('odmr_threshold', width= -60),
                                          Item('line_width_threshold', width= -60),
                                        ),
                                 HGroup(
                                                Item('power_p', width= -40, enabled_when='state != "run"'),
                                                Item('t_pi', width= -50, enabled_when='state != "run"'),
                                                ),
                                          HGroup(Item('frequency_begin_p', width= -80, enabled_when='state != "run"'),
                                                Item('frequency_end_p', width= -80, enabled_when='state != "run"'),
                                                Item('frequency_delta_p', width= -80, enabled_when='state != "run"'),

                                                ),
                                 HGroup (Item('seconds_per_point', width= -40, enabled_when='state != "run"'),
                                         Item('laser', width= -50, enabled_when='state != "run"'),
                                         Item('wait', width= -50, enabled_when='state != "run"'),
                                         Item('nlines', width= -50, enabled_when='state != "run"'),
                                        ),
                                        
                                HGroup (#Item('choosed_plot', width= -40, enabled_when='state != "run"'),
                                         Item('matrix_lines', width= 70,),
                                         Item('x_point', width= 70,style='readonly'),
                                         Item('y_point', width= 70,style='readonly'),
                                         ),
                                       
                                

                                 VSplit(HGroup(Item('fluor_x_plot_ODMR', show_label=False, resizable=True),
                                               Item('fluor_y_plot_ODMR', show_label=False, resizable=True),), 
                                        HGroup(Item('fluor_2D_plot_ODMR', show_label=False, resizable=True), 
                                               Item('matrix_plot_ODMR', show_label=False, resizable=True),),
                                        HGroup(Item('multi_plot_ODMR', show_label=False, resizable=True), ),  label='ODMR scan'
                                              
                                       ),),),
                                                    
                       menubar=MenuBar(Menu(Action(action='saveFluorescenceXPlot', name='Save Fluor X Scan'),
                                             Action(action='saveFluorescenceYPlot', name='Save Fluor Y Scan'),
                                             Action(action='saveFluorescence2DPlot', name='Save Fluor 2D Scan'),
                                             Action(action='saveODMRXPlot', name='Save ODMR X Scan'),
                                             Action(action='saveODMRYPlot', name='Save ODMR Y Scan'),
                                             Action(action='saveODMR2DPlot', name='Save ODMR 2D Scan'),
                                             Action(action='load', name='Load'),
                                             name='File')),
                       title='Magnet Control', width=1380, height=750, buttons=[], resizable=True, handler=MagnetHandler
                       )
                     
    get_set_items = ['set_X_position', 'set_Y_position', 'set_Z_position', 'run_time', 'Number_of_scanpointsX','Number_of_scanpointsY',
                     'start_positionX', 'end_positionX', 'stepX',
                     'start_positionY', 'end_positionY', 'stepY','matrix_lines','x_point','y_point',
                     'Acquisition_time', 'perform_fit', 'Xmax', 'Ymax','data_x','data_y','data_xy','data_x_ODMR','data_y_ODMR','data_xy_ODMR','counts_matrix_ODMR',
                     'choosed_align', 'choosed_scan','stop_time','odmr_threshold','fit_threshold','power_p','t_pi','frequency_begin_p','frequency_end_p','line_width_threshold',
                     'frequency_delta_p','seconds_per_point','laser','wait','fit_parameters','fit_centers','fit_contrast','fit_line_width','fitting_func','number_of_resonances','nlines',
                     '__doc__']
    
    
