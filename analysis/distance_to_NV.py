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

class Distance_to_NV(HasTraits, GetSetItemsMixin):

    counts=Array( )
    counts2=Array( )
    time=Array( )
    normalized_counts=Array()
    nu=Array()
    S=Array()
    FFx=Array()
    FFy=Array()
    tau=Array() #pulse spacing
    
    line_label = Instance(PlotLabel)
    line_data = Instance(ArrayPlotData)
    
    myfile=File(exists=True)
    
    substance =  Enum('immersion oil, H1 signal', 'hBN, B11 signal',
                     label='substance',
                     desc='choose the nucleus to calculate larmor frequency',
                     editor=EnumEditor(values={'immersion oil, H1 signal':'1:immersion oil, H1 signal','hBN, B11 signal':'2:hBN, B11 signal'},cols=8),)  
    
    rabi_contrast=Range(low=0., high=100., value=30., desc='full contrast', label='full contrast [%]', auto_set=False, enter_set=True)
    z=Float(value=0., desc='distance to NV [nm]', label='distance to NV [nm]')
    frequencies=Float(value=0., desc='frequencies [MHz]', label='frequencies [MHz]')
    z2=Float(value=0., desc='distance to NV [nm]', label='distance to NV [nm]')
    fit_threshold = Range(low= -99, high=99., value= 50., desc='Threshold for detection of resonances [%]. The sign of the threshold specifies whether the resonances are negative or positive.', label='threshold [%]', mode='text', auto_set=False, enter_set=True)
    n_FFT=Range(low=2.**10, high=2.**21, value=2.0e+6, desc='NUMBER OF POINTS FOR FOURIER TRANSFORM', label='N FFT', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.2e'))
    
    
    alpha=Range(low=0., high=1000., value=1., desc='fitting paramenter', label='alpha', auto_set=False, enter_set=True, mode='spinner')
    pulse_spacing=Float(value=0., desc='pulse_spacing', label='pulse spacing [ns]')
    pi_pulse=Float(value=0., desc='pi pulse', label='pi pulse')
    baseline=Float(value = 0., desc='baseline', label='baseline')
    Brms=Float(value = 0., desc='Brms', label='Brms [nT]')
    Brms2=Float(value = 0., desc='Brms', label='Brms [nT]')
    N=Int(value = 0, desc='number of repetitions', label='XY8-N')
    Magnetic_field=Range(low=0., high=3000., value=330., desc='magnetic field', label='Magnetic field [G]', auto_set=False, enter_set=True)
      
    perform_fit=Bool(False, label='perform fit')
    number_of_resonances=Trait('auto', String('auto', auto_set=False, enter_set=True), Int(10000., desc='Number of Lorentzians used in fit', label='N', auto_set=False, enter_set=True))
    
    norm_button=Button(label='normalize', desc='normalize')
    import_data_button = Button(label='import data', desc='import xy8')
    calculate_noise_spectrum_button=Button(label='calculate spectrum', desc='calculate noise spectrum')
    distance_to_NV_button = Button(label='calculate distance', desc='calculate distance to NV')
    filter_function_button = Button(label='Filter Function Fourier Transform', desc='calculate filter function')
    distance_from_envelope_button = Button(label='calculate distance from envelope', desc='calculate distance from envelope')
    
    plot_data_xy8_line  = Instance(ArrayPlotData)
    xy8_plot=Instance(Plot, editor=ComponentEditor())
    
    plot_data_normxy8_line  = Instance( ArrayPlotData )
    normxy8_plot=Instance(Plot, editor=ComponentEditor())
    
    plot_data_filter_function  = Instance( ArrayPlotData )
    filter_function_plot=Instance(Plot, editor=ComponentEditor())
    
    plot_data_spin_noise  = Instance( ArrayPlotData )
    spin_noise_plot=Instance(Plot, editor=ComponentEditor())
    
    plot_data_second_method  = Instance( ArrayPlotData )
    second_method_plot=Instance(Plot, editor=ComponentEditor())
    
    N_tau =  Range(low=0, high=50, value=6, desc='N tau', label='n tau', mode='spinner',  auto_set=False, enter_set=False)
        
    line_width=Float(value=0., desc='line_width', label='linewidth [kHz]')
    
    fit_parameters = Array(value=np.array((np.nan, np.nan, np.nan, np.nan)))
    fit_centers = Array(value=np.array((np.nan,)), label='center_position [Hz]') 
    fit_line_width = Array(value=np.array((np.nan,)), label='uncertanity [Hz]') 
    fit_contrast = Array(value=np.array((np.nan,)), label='contrast [%]')
    
    fitting_func = Enum('gaussian', 'loretzian',
                     label='fit',
                     desc='fitting function',
                     editor=EnumEditor(values={'gaussian':'1:gaussian','loretzian':'2:loretzian'},cols=2),)
    
    def __init__(self):
        super(Distance_to_NV, self).__init__()                  

        self._create_xy8_plot()      
        self._create_normxy8_plot()  
        self._create_filter_function_plot()
        self._create_spin_noise_plot()
        self._create_second_method_plot()
        self.on_trait_change(self._update_line_data_fit, 'perform_fit, fitting_func,fit_parameters', dispatch='ui')
        self.on_trait_change(self._update_fit, 'fit, perform_fit', dispatch='ui')
        self.on_trait_change(self._update_plot, 'alpha, line_width, n_FFT', dispatch='ui')       
            
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
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'normalized counts'
        #plot.title='normalized counts'
        
        self.plot_data_normxy8_line = plot_data_normxy8_line
        self.normxy8_plot = plot
        return self.normxy8_plot   
        
        
    def _create_xy8_plot(self):
        plot_data_xy8_line = ArrayPlotData(counts2=np.array((0., 1.)), time=np.array((0., 0.)), fit=np.array((0., 0.)) )
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
        plot_data_spin_noise = ArrayPlotData(value=np.array((0., 1.)), time=np.array((0., 0.)), fit=np.array((0., 0.)) )
        plot = Plot(plot_data_spin_noise, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('time', 'value'), color='green', line_width = 2)
        plot.index_axis.title = 'frequency [MHz]'
        plot.value_axis.title = 'noise spectrum [nT^2/Hz]'
        #plot.title='noise spectrum'
        line_label = PlotLabel(text='', hjustify='left', vjustify='top', position=[50, 100])
        plot.overlays.append(line_label)
        self.line_label = line_label
        
        self.plot_data_spin_noise = plot_data_spin_noise
        self.spin_noise_plot = plot
        return self.spin_noise_plot
    
    def _create_second_method_plot(self):
        plot_data_second_method = ArrayPlotData(value=np.array((0., 1.)), time=np.array((0., 0.)), fit=np.array((0., 0.)) )
        plot = Plot(plot_data_second_method, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('time', 'value'), color='green', line_width = 2)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'Normalized Contrast'
        #plot.title='noise spectrum'
        line_label = PlotLabel(text='', hjustify='left', vjustify='top', position=[64, 280])
        plot.overlays.append(line_label)
        self.line_label = line_label
        
        self.plot_data_second_method= plot_data_second_method
        self.second_method_plot = plot
        return self.second_method_plot
    
    def _import_data_button_fired(self):
              
        File1=open(self.myfile,'r')

        File2=cPickle.load(File1)

        #File2.keys()
               
        self.N= File2['measurement']['pulse_num']
        self.counts=File2['fit']['spin_state1']
        self.counts2= File2['fit']['spin_state2']
        self.time= File2['measurement']['tau']
        self.pi_pulse=File2['measurement']['pi_1']
        
        self.rabi_contrast=File2['measurement']['rabi_contrast']
        #self.rabi_contrast=40
        counts=self.counts2-self.counts
        
        self.plot_data_xy8_line.set_data('time', self.time)
        self.plot_data_xy8_line.set_data('counts2', counts) 

        self.tau=(2*self.time+self.pi_pulse)*1e-9 # in seconds   

        self._norm_button_fired()  
        self._calculate_noise_spectrum_button_fired()
        self.perform_fit=True     
        #self._distance_to_NV_button_fired()       
        #self._distance_from_envelope_button_fired()
       
        
    def _norm_button_fired(self):
        
        l=(self.counts+self.counts2)/2.
        self.baseline=sum(l)/len(l)
        
        C0_up=self.baseline/(1-0.01*self.rabi_contrast/2)
        
        C0_down=C0_up*(1-0.01*self.rabi_contrast)
        
        counts=self.counts2-self.counts
        
        self.normalized_counts=(counts)/(C0_up-C0_down)
        
        self.plot_data_normxy8_line.set_data('time', self.tau*1e+9)
        self.plot_data_normxy8_line.set_data('normalized_counts', self.normalized_counts)
        
        
    def _calculate_noise_spectrum_button_fired(self):
 
        # Noise spectrum 
        
        g=2*np.pi*2.8*1e+10 #Hz/T
        hi=-np.log(self.normalized_counts)
        self.S=self.alpha*2*hi/(g**2*8*self.N*self.tau)  # noise spectrum
           
        self.plot_data_spin_noise.set_data('value', self.S*1e18)
        self.plot_data_spin_noise.set_data('time', (1/(2*self.tau))*1e-6)         
        
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
        
    def _fourier_transform(self, N_tau):
        
        T, v = self._filter_function(self.tau[N_tau])
            
        g=int(self.n_FFT)
       
        signalFFT=np.fft.fft(v, g)
        
        yf=(np.abs(signalFFT)**2)*(1e-9)/(8*self.N)
        #yf = (np.abs(signalFFT)**1)/(8*self.N)
        #xf[:, m] = np.linspace(0, 1.0/(tau[m]/n), 8*self.N*n)
        xf = np.fft.fftfreq(g, 1e-9)   

        self.FFy=yf[0:g] #take only real part
        self.FFx=xf[0:g]
        
        f1=(1/(2*self.tau[0]))*1.03 #bigger
        f2=(1/(2*self.tau[-1]))*0.97 #smaller
         
        yf1=self.FFy[np.where(self.FFx<=f1)]
        xf1=self.FFx[np.where(self.FFx<=f1)]  

        self.FFy=self.FFy[np.where(xf1>=f2)]
        self.FFx=self.FFx[np.where(xf1>=f2)]       
        
    def _filter_function_button_fired(self):
    
        self._fourier_transform(self.N_tau)
     
        self.plot_data_filter_function.set_data('value', self.FFy) 
        self.plot_data_filter_function.set_data('freq', self.FFx*1e-6)
        self.pulse_spacing=self.tau[self.N_tau]*1e+9
        
    def _distance_to_NV_button_fired(self):
    
        rho_H = 5*1e+28 # m^(-3), density of protons
        rho_B11 = 2.1898552552552544e+28  # m^(-3), density of B11
        
        mu_p= 1.41060674333*1e-26 # proton magneton, J/T
        g_B11=85.847004*1e+6/(2*np.pi) # Hz/T
        hbar=1.054571800e-34 #J*s
        mu_B11=hbar*g_B11*2*np.pi # for central transition
        
        if self.substance == 'immersion oil, H1 signal':
            rho=rho_H   
            mu=mu_p
        elif self.substance == 'hBN, B11 signal':
            rho=rho_B11
            mu=mu_B11
    
        g=2*np.pi*2.8*1e+10 #rad/s/T            
        mu0=4*np.pi*1e-7 # vacuum permeability, H/m or T*m/A
                
        freq = 1/(2*self.tau)  # in Hz
                
        self._fourier_transform(0) 
        
        fit_x=self.FFx[np.where(self.FFx<=freq[0]*1.03)]
        fit_x=fit_x[np.where(fit_x>=freq[-1]*0.97)]
        
        d=fit_x[1]-fit_x[0]
        
        if self.fitting_func == 'loretzian':  
                fit_func = fitting.NLorentzians
        elif self.fitting_func == 'gaussian':
                fit_func = fitting.NGaussian 
        
        self.fit_parameters[2]=self.line_width*1e-3
        S=fit_func(*self.fit_parameters)(fit_x*1e-6) # fit for intergration
        
        base=S[0]
        
        Int = trapz((S-base), dx=d) # integration
        
        self.Brms=np.sqrt(Int)
        
        if self.substance == 'immersion oil, H1 signal':
            self.z=np.power(rho*((0.05*mu0*mu/self.Brms*1e9)**2), 1/3.)
        elif self.substance == 'hBN, B11 signal':
            C1=np.sqrt(0.654786)/(4*np.pi)
            self.z=np.power(rho*((C1*mu0*mu/self.Brms*1e9)**2), 1/3.)
        
        self.z=self.z*1e+9
        
        x1 = fit_x*1e-6                      #baseline
        y1 = [base.item()]*len(S)
        
        x_key1 = 'x1'
        y_key1 = 'y1'
        self.plot_data_spin_noise.set_data(x_key1, x1)
        self.plot_data_spin_noise.set_data(y_key1, y1)
        self.spin_noise_plot.plot((x_key1, y_key1), color='red', line_width = 1)
        
        x2 = fit_x*1e-6           # integrated fit
        y2 = S
        x_key2 = 'x2'
        y_key2 = 'y2'
        self.plot_data_spin_noise.set_data(x_key2, x2)
        self.plot_data_spin_noise.set_data(y_key2, y2)
        self.spin_noise_plot.plot((x_key2, y_key2), color='red', line_width = 1)
        
        # convolution
        
        Sum=np.ones(len(self.tau))
        for i in range(len(self.tau)):
            self._fourier_transform(i)
            
            #self.FFy=self.FFy[np.where(self.FFx<=freq[0]*1.05)]
            #fit_x=self.FFx[np.where(self.FFx<=freq[0]*1.05)]
            
            #self.FFy=self.FFy[np.where(fit_x>=freq[-1]*0.95)]
            
            S0=0
            for j in range(len(S)):             # j correspond to frequency
                S0=S0+S[j]*(self.FFy[j]**2)*1e-18
            Sum[i]=S0  
        
        dt = 1e-9        
        hi=((g**2)/2.)*Sum*dt/(8*self.N)
        
        calculated_counts=np.exp(-hi)   
       
        x3 =self.tau*1e+9
        y3 = calculated_counts
        x_key3 = 'x3'
        y_key3 = 'y3'
        self.plot_data_normxy8_line.set_data(x_key3, x3)
        self.plot_data_normxy8_line.set_data(y_key3, y3)
        self.normxy8_plot.plot((x_key3, y_key3), color='red', line_width = 1)      
        
    def _distance_from_envelope_button_fired(self):
    
        rho_H = 5*1e+28 # m^(-3), density of protons
        rho_B11 = 2.69105105105105e+28  # m^(-3), density of B11
        
        
        mu_p= 1.41060674333*1e-26 # proton magneton, J/T
        g_B11=85.84*1e+6 # Hz/T
        hbar=1.054571800e-34 #J*s
        mu_B11=hbar*g_B11 # for central transition
        
        if self.substance == 'immersion oil, H1 signal':
            rho=rho_H   
            mu=mu_p
        elif self.substance == 'hBN, B11 signal':
            rho=rho_B11
            mu=mu_B11
    
        def K(tau, T2star):
            tau=self.tau
            N=8*self.N
            g_H1=42.576 #gyromagnetic ratio, MHz/T 
            larmor_frequency=g_H1*self.Magnetic_field/10000 #Larmor frequency in MHz
            
            #return N*tau*np.sinc((N*tau/2)*(2*np.pi*larmor_frequency*1e+6 - np.pi/tau))
            return (2*T2star**2/(1+T2star**2*(2*np.pi*larmor_frequency*1e+6 - np.pi/tau)**2)**2)*\
                   (np.exp(-N*tau/T2star)*((1-T2star**2*(2*np.pi*larmor_frequency*1e+6 - np.pi/tau)**2)*\
                   np.cos(N*tau*(2*np.pi*larmor_frequency*1e+6 - np.pi/tau))-\
                   2*T2star*(2*np.pi*larmor_frequency*1e+6 - np.pi/tau)*np.sin(N*tau*(2*np.pi*larmor_frequency*1e+6 - np.pi/tau)))+\
                   (N*tau/T2star)*(1+T2star**2*(2*np.pi*larmor_frequency*1e+6 - np.pi/tau)**2) + T2star**2*(2*np.pi*larmor_frequency*1e+6 - np.pi/tau)**2-1)
    
        def func(tau, Brms, T2star):
            
            gamma_e=1.76e11 #rad/s/T
            return np.exp(-2*(gamma_e*Brms/np.pi)**2*K(tau, T2star))  
            
        def residuals(tau, tpl, S):
            Brms=tpl[0]
            T2star=tpl[1]
            return S-func(tau, Brms, T2star)        
            
        S=self.normalized_counts/self.normalized_counts[0]
       
        self.plot_data_second_method.set_data('time', self.tau*1e9)
        self.plot_data_second_method.set_data('value', S)
        T2star=60*1e-6
        
        popt,pcov = curve_fit(func, self.tau, S, p0=[self.Brms*1e-9, T2star])  
        
        tpl=(self.Brms*1e-9, T2star)
        
        #popt, pcov = leastsq(residuals, tpl[:], args=(self.tau, S))
        
        self.Brms2=popt[0]*1e+9
      
        fit=func(self.tau, *popt)

        self.plot_data_second_method.set_data('fit', fit)
        self.second_method_plot.plot(('time', 'fit'), color='blue', line_width = 1)   

        gamma_n=2.68*1e8 # rad/s/T
        mu0=4*np.pi*1e-7 # vacuum permeability, H/m or T*m/A
        h_planck = 1.054571800*1e-34 # J*s
        
        self.z2=np.power((5*np.pi*rho/96)*(mu0*h_planck*gamma_n/(4*np.pi*popt[0]))**2, 1./3)*1e+9
                                   
     #fitting---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
              
    def _perform_fit_changed(self, new):
    
        plot = self.spin_noise_plot
        x_name = self.plot_data_spin_noise.list_data()[2]
                        
        if new:
            plot.plot(('time', 'fit'), style='line', color='blue', name='fit', line_width=1)
            self.line_label.visible = True
            self.line_width=self.fit_parameters[2]
        else:
            plot.delplot('fit')
            self.line_label.visible = False
        plot.request_redraw()
            
            
    def _update_fit(self):
        if self.perform_fit:
        
            if self.fitting_func == 'loretzian':
                fit_func = fitting.fit_multiple_lorentzians
            elif self.fitting_func == 'gaussian':
                fit_func = fitting.fit_multiple_gaussian
                    
            N = self.number_of_resonances # number of peaks
            fit_x = (1/(2*self.tau))*1e-6
            self.counts = self.S*1e18
            p = fit_func(fit_x, self.counts, N, threshold=self.fit_threshold * 0.01)
            
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
        self.line_width=self.fit_parameters[2]*1e+3   
        #self.frequencies=self.centers[1]*1e+6
            
    def _update_line_data_fit(self):
        if not np.isnan(self.fit_parameters[0]):       
            
            if self.fitting_func == 'loretzian':  
                fit_func = fitting.NLorentzians
            elif self.fitting_func == 'gaussian':
                fit_func = fitting.NGaussian                
                               
            self.plot_data_spin_noise.set_data('fit', fit_func(*self.fit_parameters)((1/(2*self.tau))*1e-6))
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
                s += '%.2f MHz, LW %.2f kHz, contrast %.1f%%\n' % (fi, w[i]*1e+3, contrast[i])
            self.line_label.text = s  
            self.line_width=self.fit_parameters[2]*1e+3
            
    def _update_plot(self):
        self._distance_to_NV_button_fired()
        # self._ditance_button_fired()
        # self._ditance_button_fired()
        # self._ditance_button_fired()
            
    def save_all(self, filename):
        self.save_figure(self.xy8_plot, filename + 'XY8-' + str(self.N) + '_counts' + '.png' )
        self.save_figure(self.normxy8_plot, filename + 'XY8-' + str(self.N) + '_normalized_counts' + '.png' )
        self.save_figure(self.spin_noise_plot, filename + 'XY8-' + str(self.N) + '_noise_spectrum_z=' + string.replace(str(self.z)[0:4], '.', 'd') + 'nm_B=' + string.replace(str(self.Brms)[0:4], '.', 'd') + 'nT.png' )
        self.save(filename + 'XY8-' + str(self.N) +'_distance' + '.pyd' )       
        self.save_figure(self.second_method_plot, filename + 'XY8-' + str(self.N) + '_noise_spectrum_z=' + string.replace(str(self.z2)[0:4], '.', 'd') + 'nm_B=' + string.replace(str(self.Brms2)[0:4], '.', 'd') + 'nT.png' )
        
    traits_view =  View( VGroup( HGroup( VGroup( HGroup( Item('myfile', show_label=False),
                                                         Item('import_data_button', show_label=False),
                                                         Item('substance', style='custom', show_label=False)
                                                       ),
                                                 HGroup( Item('xy8_plot', show_label=False, resizable=True),
                                                       ),
                                                 HGroup( Item('filter_function_button', show_label=False)
                                                       ),
                                                 HGroup(                                          
                                                         Item('N_tau', width= -40),
                                                         Item('pulse_spacing', width= -40, format_str='%.1f')
                                                       ),
                                                 HGroup( Item('filter_function_plot', show_label=False, resizable=True),
                                                       ),
                                               ),
                                    
                                         VGroup( HGroup( Item('rabi_contrast', width= -40),
                                                         Item('N', width= -40),
                                                         Item('norm_button', show_label=False)
                                                       ),
                                                                    
                                                 HGroup( Item('normxy8_plot', show_label=False, resizable=True)
                                                       ),
                                                 HGroup( Item('calculate_noise_spectrum_button', show_label=False),
                                                       ),
                                                 HGroup( Item('fit_threshold', width= -40),
                                                         Item('perform_fit'),
                                                         Item('number_of_resonances', width= -60),
                                                 
                                                       ),
                                                 HGroup( Item('spin_noise_plot', show_label=False, resizable=True)
                                                       ),
                                               ), 
                                       ),
                                 HGroup( 
                                         Item('alpha', width= -60, format_str='%.2f'),
                                         Item('n_FFT', width= -70),
                                         Item('line_width', width= -60, format_str='%.2f'),
                                         Item('fitting_func', style='custom', show_label=False),                                         
                                         Item('distance_to_NV_button', show_label=False),
                                         Item('z', width= -60, style='readonly', format_str='%.2f'),
                                         Item('Brms', width= -60, style='readonly', format_str='%.0f')
                                       )
                               ),
                               
                          Group( HGroup( Item('Magnetic_field', width= -60),
                                         Item('distance_from_envelope_button', show_label=False)),
                                 HGroup( Item('z2', width= -60, style='readonly', format_str='%.2f'),       
                                         Item('Brms2', width= -60, style='readonly', format_str='%.0f')), 
                                         
                                 HGroup( Item('second_method_plot', show_label=False, resizable=True)
                                                       )),
                               
                         menubar=MenuBar( Menu( Action(action='Save_All', name='Save all'),
                                                Action(action='load', name='Load'),
                                                name='File'
                                               )
                                        ),
                             
                     title='NV depth Toby', width=1200, height=800, buttons=[], resizable=True, handler=DistanceHandler)
                     
    get_set_items = ['N_tau', 'pulse_spacing', 'rabi_contrast', 'N', 'alpha','line_width', 'S', 'time', 'counts', 'counts2',
                     'z', 'Brms', 'normalized_counts','tau', 'fit_parameters','fit_centers','fit_contrast','fit_line_width',
                     'fitting_func', 'n_FFT',
                     '__doc__']