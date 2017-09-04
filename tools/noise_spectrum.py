import numpy as np
from traitsui.file_dialog import save_file
from traitsui.menu import Action, Menu, MenuBar
#import peakutils
from traits.api import HasTraits, Trait, Instance, Property, String, Range, Float, Int, Bool, Array, Enum, Button, on_trait_change, cached_property, Code, List, NO_COMPARE, File
from traitsui.api import View, Item, HGroup, VGroup, VSplit, Tabbed, EnumEditor, TextEditor, Group, FileEditor, ButtonEditor
from enable.api import Component, ComponentEditor
from chaco.api import ArrayPlotData, Plot, Spectral, PlotLabel,HPlotContainer, jet,ColorBar, LinearMapper, CMapImagePlot

from numpy import trapz
from scipy.fftpack import fft

from scipy.optimize import leastsq
import time

import cPickle
import string

from analysis import fitting
import scipy.optimize, scipy.stats
from scipy.optimize import curve_fit

from tools.utility import GetSetItemsHandler, GetSetItemsMixin

class DistanceHandler(GetSetItemsHandler):

    def Save_All(self, info):
        filename = save_file(title='Save all')
        if filename is '':
            return
        else:
            # if filename.find('.png') == -1:
                # filename = filename + '.png'
            info.object.save_all(filename)

class Noise_Spectrum(HasTraits, GetSetItemsMixin):

    counts=Array( )
    counts2=Array( )
    hi=Array()
    time=Array( )
    normalized_counts=Array()
    nu=Array()
    S=Array()
    FFx=Array()
    FFy=Array()
    tau=Array() #pulse spacing
    T=Array() # The whole evolution time
    frequency=Array()
    
    line_label = Instance(PlotLabel)
    line_data = Instance(ArrayPlotData)
    
    myfile=File(exists=True)
    
    rabi_contrast=Float(value=30.0, desc='Rabi contrast', label='contrast', mode='text', auto_set=True, enter_set=True)
    n_FFT=Range(low=2.**10, high=3.0e+6, value=1.0e+06, desc='NUMBER OF POINTS FOR FOURIER TRANSFORM', label='N FFT', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.2e'))
    
    pulse_spacing=Float(value=0., desc='pulse_spacing', label='pulse spacing [ns]')
    pi_pulse=Float(value=0., desc='pi pulse', label='pi pulse')
    N=Range(low=0, high=1000, value=6, desc='number of repetitions', label='XY8-N', mode='text', auto_set=True, enter_set=True)
         
    perform_fit=Bool(False, label='perform fit')
    filter_low=Float(value=1.5, desc='filter', label='filter low [MHz]')
    filter_high=Float(value=3.5, desc='filter', label='filter high [MHz]')
    threshold=Float(value=1.0, desc='filter', label='threshold')
        
    norm_button=Button(label='filter function', desc='normalize')
    import_data_button = Button(label='import data', desc='import xy8')
    calculate_noise_spectrum_button=Button(label='calculate spectrum', desc='calculate noise spectrum')
    filter_function_button = Button(label='Filter Function Fourier Transform', desc='calculate filter function')
    check_button=Button(label='check', desc='do integration')
    remove_peaks_button=Button(label='remove_peaks', desc='remove_peaks')
    remove_point_button=Button(label='remove_point', desc='remove_point')
    next_point_button=Button(label='next_point', desc='next_point')
    
    error_approximation=Float(value=0.0, desc='error %', label='error %')
    
    plot_data_xy8_line  = Instance(ArrayPlotData)
    xy8_plot=Instance(Plot, editor=ComponentEditor())
    
    plot_data_normxy8_line  = Instance( ArrayPlotData )
    normxy8_plot=Instance(Plot, editor=ComponentEditor())
    
    plot_data_filter_function  = Instance( ArrayPlotData )
    filter_function_plot=Instance(Plot, editor=ComponentEditor())
    
    plot_data_spin_noise  = Instance( ArrayPlotData )
    spin_noise_plot=Instance(Plot, editor=ComponentEditor())
   
    N_tau =  Range(low=0, high=50, value=6, desc='N tau', label='n tau', mode='spinner',  auto_set=False, enter_set=False)
    i=Int(value=-1, desc='index', label='index')
        
    fit_parameters = Array(value=np.array((np.nan, np.nan)))
    
    delta=Float(value=0., desc='delta [MHz]', label='delta [MHz]')
    tc=Float(value=0., desc='corr [ns]', label='tc [ns]')
    
    sequence = Enum('XY8', 'XY16',
                     label='sequence',
                     desc='sequence',
                     editor=EnumEditor(values={'XY8':'1:XY8','XY16':'2:XY16'},cols=2),)
    
    def __init__(self):
        super(Noise_Spectrum, self).__init__()                  

        self._create_xy8_plot()      
        self._create_normxy8_plot()  
        self._create_filter_function_plot()
        self._create_spin_noise_plot()
        self.on_trait_change(self._update_line_data_fit, 'perform_fit, fitting_func,fit_parameters', dispatch='ui')
        self.on_trait_change(self._update_fit, 'fit, perform_fit', dispatch='ui')
        #self.on_trait_change(self._update_plot, 'alpha, line_width, n_FFT', dispatch='ui')       
            
    def _counts_default(self):
        return np.zeros(self.pdawg3.fit.x_tau.shape[0])   

    def _counts2_default(self):
        return np.zeros(self.pdawg3.fit.x_tau.shape[0])
        
    def _normalized_counts_default(self):
        return np.zeros(self.pdawg3.fit.x_tau.shape[0])

    def _time_default(self):
        return np.zeros(self.pdawg3.fit.x_tau.shape[0])        
        
    def _create_normxy8_plot(self):
        plot_data_normxy8_line = ArrayPlotData(normalized_counts=np.array((0., 1.)), time=np.array((0., 0.)), fit=np.array((0., 0.)) )
        plot = Plot(plot_data_normxy8_line, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('time', 'normalized_counts'), color='green', line_width = 2)
        plot.index_axis.title = 't_prime [ns]'
        plot.value_axis.title = 'f(t, t_prime)'
        #plot.title='normalized counts'
        
        self.plot_data_normxy8_line = plot_data_normxy8_line
        self.normxy8_plot = plot
        return self.normxy8_plot   
        
    def _create_xy8_plot(self):
        plot_data_xy8_line = ArrayPlotData(counts2=np.array((0., 1.)), time=np.array((0., 0.)), fit=np.array((0., 0.)), time1=np.array((0., 0.)),  point_x=np.array((0., 0.)), point_y=np.array((0., 0.)))
        plot = Plot(plot_data_xy8_line, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('time', 'counts2'), color='green', line_width = 2)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'counts'
        #plot.title='counts'
        self.plot_data_xy8_line = plot_data_xy8_line
        self.xy8_plot = plot
        return self.xy8_plot
        
    def _create_filter_function_plot(self):
        plot_data_filter_function = ArrayPlotData(freq=np.array((0., 1.)), value=np.array((0., 0.)), fit=np.array((0., 0.)) )
        plot = Plot(plot_data_filter_function, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('freq', 'value'), color='red', type='line', line_width = 3)
        plot.index_axis.title = 'frequency [MHz]'
        plot.value_axis.title = 'Filter Function Fourier Transform'
        #plot.title='Fourier transform of filter function'
        
        self.plot_data_filter_function = plot_data_filter_function
        self.filter_function_plot = plot
        return self.filter_function_plot
        
    def _create_spin_noise_plot(self):
        plot_data_spin_noise = ArrayPlotData(value=np.array((0., 1.)), time=np.array((0., 0.)), fit=np.array((0., 0.)), peak_free=np.array((0., 0.)), f=np.array((0., 0.)), fit_x=np.array((0., 0.)))
        plot = Plot(plot_data_spin_noise, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('time', 'value'), color='red', line_width = 2, type='scatter')#, index_scale = 'log', value_scale = 'log')
        plot.index_axis.title = 'frequency [MHz]'
        plot.value_axis.title = 'noise spectrum [nT^2/Hz]'
        #plot.title='noise spectrum'
        line_label = PlotLabel(text='', hjustify='left', vjustify='top', position=[50, 100])
        plot.overlays.append(line_label)
        self.line_label = line_label
        
        self.plot_data_spin_noise = plot_data_spin_noise
        self.spin_noise_plot = plot
        return self.spin_noise_plot
    
    def _import_data_button_fired(self):
              
        File1=open(self.myfile,'r')

        File2=cPickle.load(File1)

        #File2.keys()
         
        if self.sequence == 'XY8':  
                self.N= File2['measurement']['pulse_num']
        elif self.sequence == 'XY16':
                self.N= File2['measurement']['pulse_num']*2
        
        self.counts=File2['fit']['spin_state1']
        self.counts2= File2['fit']['spin_state2']
        self.time= File2['measurement']['tau']
        self.pi_pulse=File2['measurement']['pi_1']
        
        self.rabi_contrast=File2['measurement']['rabi_contrast']
        #self.rabi_contrast=40
        counts=self.counts2-self.counts
        l=(self.counts+self.counts2)/2.
        self.baseline=sum(l)/len(l)
        
        C0_up=self.baseline/(1-0.01*self.rabi_contrast/2)
        
        C0_down=C0_up*(1-0.01*self.rabi_contrast)
        
        counts=self.counts2-self.counts
        
        self.normalized_counts=(counts)/(C0_up-C0_down)
        self.tau=(2*self.time+self.pi_pulse)*1e-9 # in seconds 
                
        self.plot_data_xy8_line.set_data('time', self.tau*1e+9)
        self.plot_data_xy8_line.set_data('counts2', self.normalized_counts)
                
        T, v = self._filter_function(self.tau[self.N_tau])
                
        self.plot_data_normxy8_line.set_data('time', T*1e+9)
        self.plot_data_normxy8_line.set_data('normalized_counts', v)
        
        self.i=-1
        
    def _next_point_button_fired(self):
    
        self.i=self.i+1
    
        plot = self.xy8_plot      
        plot.plot(('point_x', 'point_y'), color='red', line_width=3, type='scatter')
        self.plot_data_xy8_line.set_data('point_x', np.array([self.tau[self.i]*1e+9]))
        self.plot_data_xy8_line.set_data('point_y', np.array([self.normalized_counts[self.i]]))    
        
    def _remove_point_button_fired(self):
    
        self.normalized_counts=np.delete(self.normalized_counts, self.i)
        self.tau=np.delete(self.tau, self.i)
        
        self.i=self.i-1
        
    def _calculate_noise_spectrum_button_fired(self):
        
        self.T=self.tau*8*self.N+0.5*self.pi_pulse*1e-9        
        S=-np.log(self.normalized_counts)/self.T
        
        self.S=np.concatenate((self.S, S), axis=0)
        
        frequency=1e+6*1e-9*0.5e+3/self.tau #(in Hz)
        
        self.frequency=np.concatenate((self.frequency, frequency), axis=0)
           
        self.plot_data_spin_noise.set_data('value', self.S)
        self.plot_data_spin_noise.set_data('time', self.frequency*1e-6)
        
    def _remove_peaks_button_fired(self):
    
        filter_low=self.filter_low*1e+6
        filter_high=self.filter_high*1e+6
        
        S=self.S
        
        fit=self.plot_data_spin_noise['fit']
        f=self.frequency
        
        a=S[np.where(f<filter_low)]
        b=S[np.where(np.logical_and(f>=filter_low, f<=filter_high))]
        c=S[np.where(f>filter_high)]
        
        a1=f[np.where(f<filter_low)]
        b1=f[np.where(np.logical_and(f>=filter_low, f<=filter_high))]
        c1=f[np.where(f>filter_high)]
    
        j=[]
        
        for i in range(len(b)-1):
            if np.abs(b[i]-fit[i])/fit[i]>self.threshold: 
                j.append(i)          
            
        b=np.delete(b, j)
        b1=np.delete(b1, j)
                
        self.S=np.concatenate((a, b, c), axis=0)
        self.frequency=np.concatenate((a1, b1, c1), axis=0)
        
        plot = self.spin_noise_plot      
        plot.plot(('f', 'peak_free'), color='green', line_width=2, type='scatter')
        self.plot_data_spin_noise.set_data('peak_free', self.S)
        self.plot_data_spin_noise.set_data('f', self.frequency*1e-6)
            
    def _filter_function(self, tau):
     
        #generate filter function
        dt = 1e-9
        n = int(tau / dt)
            
        v = np.zeros(8*self.N*n)
       
        T=np.linspace(0, dt*n*8*self.N, num=8*self.N*n)
        v[:n/2]=1
        k=n/2+1
        for j in range(8*self.N-1):
            v[(n/2+j*n):(n/2+j*n+n)]=(-1)**(j+1)
            k=k+1
        v[8*self.N*n-n/2:8*self.N*n]=np.ones((n/2,), dtype=np.int)    
        return T, v
        
    def _fourier_transform(self, tau):
    
        T, v = self._filter_function(tau)
            
        g=int(self.n_FFT)
       
        signalFFT=np.fft.fft(v, g)
        
        yf=(np.abs(signalFFT)**2)*(1e-9)/(8*self.N)
        xf = np.fft.fftfreq(g, 1e-9)   

        self.FFy=yf[0:g] #take only real part
        self.FFx=xf[0:g]
        
        f1=(1/(2*self.tau[0]))*1.1 #bigger
        f2=(1/(2*self.tau[-1]))*0.5 #smaller
         
        yf1=self.FFy[np.where(self.FFx<=f1)]
        xf1=self.FFx[np.where(self.FFx<=f1)]  

        self.FFy=self.FFy[np.where(xf1>=f2)]
        self.FFx=self.FFx[np.where(xf1>=f2)]    
                
        
    def _filter_function_button_fired(self):
    
        self._fourier_transform(self.tau[self.N_tau])
     
        self.plot_data_filter_function.set_data('value', self.FFy) 
        self.plot_data_filter_function.set_data('freq', self.FFx*1e-6)
        self.pulse_spacing=self.tau[self.N_tau]*1e+9
        
    def _check_button_fired(self):
    
        #we have calculated noise spectrum
        
        #for each free evolution time we perform Fourier transform and integrate from 0 to inf with fit of the noise spectrum
        
        hi=[]
                
        self._fourier_transform(self.tau[0])
                
        FFx=np.zeros((len(self.tau), len(self.FFx)))
        FFy=np.zeros((len(self.tau), len(self.FFx)))
                
        for i in range(len(self.tau)):
            self._fourier_transform(self.tau[i]) # now we have self.FFx and self.FFy
            FFx[i][:]=self.FFx
            FFy[i][:]=self.FFy**2
            
        d=self.FFx[1]-self.FFx[0]
        
        S=fitting.LorentzianNoise(self.FFx, *self.fit_parameters) # fit for intergration
        
        g=2*np.pi*2.8*10 #Hz/n       
        #self.S=2*hi/(g**2*self.T)  # noise spectrum
    
        for i in range(len(self.tau)):
            Int = trapz(2*S*FFy[i][:]/g**2, dx=d) # integration
           
            hi.append(Int)
                        
        hi=[-x for x in hi]
            
        calculated_counts=np.exp(hi)          
        self.plot_data_xy8_line.set_data('fit', calculated_counts)
        self.plot_data_xy8_line.set_data('time1', self.tau*1e+9)
        plot = self.xy8_plot
        plot.plot(('time1', 'fit'), color='purple', line_width = 2)
        
        self.error_approximation=100*np.sum(np.abs(self.normalized_counts-calculated_counts)/calculated_counts)/len(self.normalized_counts) 
               
     #fitting---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _rearrange_data(self):

        S=[]
        f=[]
        
        N=len(self.S)
        
        for k in range(N):  
            
            idx=np.argmin(self.frequency)
            S.append(self.S[idx])
            f.append(self.frequency[idx])
            self.frequency=np.delete(self.frequency, idx)
            self.S=np.delete(self.S, idx)
                       
        self.S=np.asarray(S)
        self.frequency=np.asarray(f)
                             
    def _perform_fit_changed(self, new):
    
        plot = self.spin_noise_plot
                        
        if new:
            plot.plot(('fit_x', 'fit'), style='line', color='blue', name='fit', line_width=1)
            self.line_label.visible = True
        else:
            plot.delplot('fit')
            self.line_label.visible = False
            self.fit_parameters=np.array((np.nan, np.nan))
        plot.request_redraw()
            
    def _update_fit(self):
        if self.perform_fit:
            
            self._rearrange_data()
        
            popt,pcov = curve_fit(fitting.LorentzianNoise, self.frequency, self.S, p0=[self.delta*1e+6, self.tc*1e-9])
            self.fit_parameters = popt
        else:
            p = np.array((np.nan, np.nan))
            self.fit_parameters = p
        
    def _update_line_data_fit(self):
        if not np.isnan(self.fit_parameters[0]):       
            
            self.plot_data_spin_noise.set_data('fit_x', self.frequency*1e-6)
            self.plot_data_spin_noise.set_data('fit', fitting.LorentzianNoise(self.frequency, *self.fit_parameters))
            self.delta=self.fit_parameters[0]*1e-6
            self.tc=self.fit_parameters[1]*1e+9
                  
    def save_all(self, filename):
        self.save_figure(self.xy8_plot, filename + 'XY8-' + str(self.N) + '_counts' + '.png' )
        self.save_figure(self.normxy8_plot, filename + 'XY8-' + str(self.N) + '_normalized_counts' + '.png' )
        self.save_figure(self.spin_noise_plot, filename + 'XY8-' + str(self.N) + '_noise_spectrum_z=' + string.replace(str(self.z)[0:4], '.', 'd') + 'nm_B=' + string.replace(str(self.Brms)[0:4], '.', 'd') + 'nT.png' )
        self.save(filename + 'XY8-' + str(self.N) +'_distance' + '.pyd' )       
        self.save_figure(self.second_method_plot, filename + 'XY8-' + str(self.N) + '_noise_spectrum_z=' + string.replace(str(self.z2)[0:4], '.', 'd') + 'nm_B=' + string.replace(str(self.Brms2)[0:4], '.', 'd') + 'nT.png' )
        
    traits_view =  View( VGroup( HGroup( VGroup( HGroup( Item('myfile', show_label=False),
                                                         Item('import_data_button', show_label=False),
                                                         Item('next_point_button', show_label=False),
                                                         Item('remove_point_button', show_label=False)
                                                       ),
                                                 HGroup( Item('xy8_plot', show_label=False, resizable=True),
                                                       ),
                                                 HGroup( Item('filter_function_button', show_label=False)
                                                       ),
                                                 HGroup(                                         
                                                         Item('N_tau', width= -40),
                                                         Item('n_FFT', width= -70),
                                                         Item('pulse_spacing', width= -40, format_str='%.1f')
                                                       ),
                                                 HGroup( Item('filter_function_plot', show_label=False, resizable=True),
                                                       ),
                                               ),
                                    
                                         VGroup( HGroup( 
                                                         Item('rabi_contrast', width= -40),
                                                         Item('N', width= -40),
                                                        ),
                                                                    
                                                 HGroup( Item('normxy8_plot', show_label=False, resizable=True)
                                                       ),
                                                 HGroup( Item('sequence', style='custom', show_label=False), 
                                                         Item('calculate_noise_spectrum_button', show_label=False),
                                                         Item('check_button', show_label=False)
                                                       ),
                                                 HGroup(Item('perform_fit'),
                                                        #Item('filter_low', width= -40, format_str='%.1f'),
                                                        #Item('filter_high', width= -40, format_str='%.1f'),
                                                        #Item('threshold', width= -40, format_str='%.1f'),
                                                        #Item('remove_peaks_button', show_label=False)
                                                        Item('delta', width= -40, format_str='%.2f'),
                                                        Item('tc', width= -40, format_str='%.1f'),
                                                        Item('error_approximation', width= -40, format_str='%.1f')
                                                       ),
                                                 HGroup( Item('spin_noise_plot', show_label=False, resizable=True)
                                                       ),
                                               ), 
                                       ),
                               ),
                               
                         menubar=MenuBar( Menu( Action(action='Save_All', name='Save all'),
                                                Action(action='load', name='Load'),
                                                name='File'
                                               )
                                        ),
                             
                     title='Noise spectrum', width=1200, height=800, buttons=[], resizable=True, handler=DistanceHandler)
                     
    get_set_items = ['N_tau', 'pulse_spacing', 'rabi_contrast', 'N', 'S', 'time', 'counts', 'counts2',
                     'z', 'Brms', 'normalized_counts','tau', 'fit_parameters','fit_centers','fit_contrast','fit_line_width',
                     'fitting_func', 'n_FFT',
                     '__doc__']