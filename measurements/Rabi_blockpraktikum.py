# coding: utf-8

import numpy as np
import threading
import time
import logging
from analysis import fitting

from traits.api import HasTraits, Trait, Instance, Property, Float, Range, Int,\
                       Bool, Array, String, Str, Enum, Button, on_trait_change, cached_property, DelegatesTo
from traitsui.api import View, Item, Group, HGroup, VGroup, VSplit, Tabbed, EnumEditor, TextEditor, Action, Menu, MenuBar

from enable.api import ComponentEditor, Component
from chaco.api import CMapImagePlot, ArrayPlotData, DataRange1D, LinePlot, ArrayDataSource,\
                      Spectral, gray, ColorBar, LinearMapper, DataLabel, PlotLabel      
import chaco.api

from tools.emod import ManagedJob

#customized zoom tool to keep aspect ratio #
from tools.chaco_addons import SavePlot as Plot, SaveHPlotContainer as HPlotContainer, SaveTool
from tools.utility import GetSetItemsHandler

import hardware.api as ha
from tools.utility import singleton

from hardware.microwave_sources import SMIQ

from hardware.api import PulseGenerator, FastComtec, Microwave

@singleton
def Microwave2():
    from microwave_sources import SMIQ
    return SMIQ(visa_address='GPIB0::28')

PG = PulseGenerator()
mw = Microwave()
mw2 = mw
#mw2 = Microwave2()
FC = FastComtec()


# @singleton
# def PulseBlaster():
    # from hardware.pulse_blaster_blockpraktikum import PulseBlaster as pbb
    # return pbb

# PG = PulseBlaster()
# PG.channel_map = {'laser':2,'pumping':2, 'mw':1, 'microwave':1, 'trigger':3, 'SequenceTrigger':0, 'awgTrigger':0, 'Orange':4, 'csd':5}


def sequence_remove_zeros(sequence):
    return filter(lambda x: x[1]!=0.0, sequence)

def sequence_union(s1, s2):
    """
    Return the union of two pulse sequences s1 and s2.
    """
    # make sure that s1 is the longer sequence and s2 is merged into it
    if sequence_length(s1) < sequence_length(s2):
        sp = s2
        s2 = s1
        s1 = sp
    s = []
    c1, dt1 = s1.pop(0)
    c2, dt2 = s2.pop(0)
    while True:
        if dt1 < dt2:
            s.append( (list( set(c1) | set(c2) ), dt1) )
            dt2 -= dt1
            try:
                c1, dt1 = s1.pop(0)
            except:
                break
        elif dt2 < dt1:
            s.append( (list( set(c1) | set(c2) ), dt2) )
            dt1 -= dt2
            try:
                c2, dt2 = s2.pop(0)
            except:
                c2 = []
                dt2 = np.inf
        else:
            s.append( (list( set(c1) | set(c2) ), dt1) )
            try:
                c1, dt1 = s1.pop(0)
            except:
                break
            try:
                c2, dt2 = s2.pop(0)
            except:
                c2 = []
                dt2 = np.inf            
    return s

def sequence_defrag(sequence):
    '''
    Takes a sequence list and merges all consecutive instructions which have the same channel list.
    E.g. [ (['ch1'],t1), (['ch1'],t2) ] is merged into [ (['ch1'], t1+t2) ]
    '''
    seq = sequence_remove_zeros(sequence)

    i = 0
    while i < len(seq):
        c0,t0 = seq[i]
        while i + 1 < len(seq) and set(seq[i][0]) == set(seq[i+1][0]):
            c,t = seq.pop(i+1)
            t0 += t
        seq[i] = (c0,t0)
        i += 1    # apparently, using loop counters is not considered very 'pythonic' and a for loop could be used as well [Ben Blanks answer on https://stackoverflow.com/questions/864603/python-while-loop-condition-evaluation]
    return seq    # Actually, sequence is changed in-place, so no return is needed. All / most other operations on sequences do return s.th., though, so this will make for a more consistent appearance in the main file.

def find_laser_pulses(sequence):
    n = 0
    prev = []
    for channels, t in sequence:
        if 'trigger' in channels and not 'trigger' in prev:
            n += 1
        prev = channels
    return n

def spin_state(c, dt, T, t0=0.0, t1=-1.):
    
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
    edge = fitting.find_edge(profile)
    
    I = int(round(T/float(dt)))
    i0 = edge + int(round(t0/float(dt)))
    y = np.empty((c.shape[0],))
    for i, slot in enumerate(c):
        y[i] = slot[i0:i0+I].sum()
    if t1 >= 0:
        i1 = edge + int(round(t1/float(dt)))    
        y1 = np.empty((c.shape[0],))
        for i, slot in enumerate(c):
            y1[i] = slot[i1:i1+I].sum()
        y = y/y1*y1.mean()
    return y, profile, edge    

    
    
class Rabi( ManagedJob, GetSetItemsHandler ):
    """Provides wide field ODMR measurements."""
    
    keep_data = Bool(False) # helper variable to decide whether to keep existing data    
    resubmit_button = Button(label='resubmit', desc='Submits the measurement to the job manager. Tries to keep previously acquired data. Behaves like a normal submit if sequence or time bins have changed since previous run.')    

    # measurement parameters
    frequency   = Range(low=1., high=20e9, value=2.87e9, desc='MW Frequency [Hz]', label='Frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power       = Range(low=-100., high=0., value=-20, desc='Power [dBm]', label='Power [dBm]', mode='text', auto_set=False, enter_set=True)
    tau_start   = Float(default_value=12.,    desc='Start duration [ns]',    label='t_min [ns]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float))
    tau_end     = Float(default_value=200.,    desc='End duration [ns]',    label='t_max [ns]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float))
    tau_delta   = Float(default_value=2.,    desc='Duration step [ns]',    label='delta_t [ns]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float))
    t_laser     = Range(low=1.5, high=10000., value=5000., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
    record_length = Range(low=100, high=100000., value=3000, desc='length of acquisition record [ns]', label='record length [ns]', mode='text', auto_set=False, enter_set=True)
    bin_width   = Range(low=0.1, high=1000., value=2, desc='data bin width [ns]', label='bin width [ns]', mode='text', auto_set=False, enter_set=True)
    t_wait      = Range(low=1.5, high=30000., value=1000., desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)
    mw_source   = Enum( 'mw', 'mw2',   desc='switch between microwave sources',     label='axis', editor=EnumEditor(cols=3, values={'mw':'1:MW1', 'mw2':'2:MW2'}) )
    sweeps      = Range(low=1, high=1e10, value=1e6, desc='number of sweeps of the sequence', label='Sweeps', mode='text', auto_set=False, enter_set=True)
    n_lines     = Range (low=1, high=10000, value=10, desc='Number of lines in Matrix', label='Matrix lines', mode='text', auto_set=False, enter_set=True)
    stop_time   = Range(low=1., value=np.inf, desc='Time after which the experiment stops by itself [s]', label='Stop time [s]', mode='text', auto_set=False, enter_set=True)
    decay_read  = Float(default_value=0.,    desc='time to let the system decay before laser pulse [ns]',       label='decay read [ns]',        mode='text', auto_set=False, enter_set=True)
    aom_delay   = Float(default_value=0.,    desc='If set to a value other than 0.0, the aom triggers are applied\nearlier by the specified value. Use with care!', label='aom delay [ns]', mode='text', auto_set=False, enter_set=True)
    
    n_laser = Int()
    
    # control data fitting
    perform_fit = Bool(True, label='perform fit')
    
    # fit result
    fit_parameters = Array(value=np.array((np.nan, np.nan, np.nan, np.nan)))
    contrast = Float(format_str='%.1f')
    period = Float()
    t_pi2 = Float()
    t_pi = Float()
    t_3pi2 = Float()
    t_2pi = Float()

    # measurement data
    tau = Array()
    count_data = Array()# value=np.zeros((2,2)) )    # is it necessary, to give it an initial value?
    counts_matrix = Array()
    run_time = Float(value=0.0, desc='Run time [s]', label='Run time [s]')
    elapsed_sweeps = Int()
    
    # analyzed data
    pulse               = Array(value=np.array((0.,0.)))
    edge                = Float(value=0.0)
    spin_state          = Array(value=np.array((0.,0.)))
    
    # parameters for calculating spin state
    integration_width   = Float(default_value=200.,   desc='width of integration window [ns]',                     label='integr. width [ns]', mode='text', auto_set=False, enter_set=True)
    position_signal     = Float(default_value=0.,     desc='position of signal window relative to edge [ns]',         label='pos. signal [ns]', mode='text',   auto_set=False, enter_set=True)
    position_normalize  = Float(default_value=-1.,    desc='position of normalization window relative to edge [ns]. If negative, no normalization is performed',  label='pos. norm. [ns]', mode='text',    auto_set=False, enter_set=True)

    # plotting
    line_label  = Instance( PlotLabel )
    line_data   = Instance( ArrayPlotData )
    matrix_data = Instance( ArrayPlotData )
    line_plot   = Instance( Plot, editor=ComponentEditor() )
    matrix_plot = Instance( Plot, editor=ComponentEditor() )
    pulse_plot  = Instance( Plot, editor=ComponentEditor() )
    
    def __init__(self, **kwargs): # microwave, pulse_generator, counter, **kwargs): # the devices are hard-coded in this script
        super(Rabi, self).__init__(**kwargs)
        #self.microwave = microwave
        #self.pulse_generator = pulse_generator
        #self.counter = counter
        self.tau = np.arange(self.tau_start, self.tau_end+self.tau_delta, self.tau_delta)
        self._create_line_plot()
        self._create_matrix_plot()
        self._create_pulse_plot()
        self.on_trait_change(self._update_line_data_index,      'tau',                  dispatch='ui')
        self.on_trait_change(self._update_line_data_value,      'count_data',               dispatch='ui')
        # # self.on_trait_change(self._update_line_data_fit,      'fit_parameters',       dispatch='ui')
        self.on_trait_change(self._update_matrix_data_value,    'count_data',           dispatch='ui')
        self.on_trait_change(self._update_matrix_data_index,    'time_bins',                  dispatch='ui')
        self.on_trait_change(self._update_pulse_index,          'time_bins',            dispatch='ui')
        self.on_trait_change(self._update_pulse_value,          'pulse',                dispatch='ui')
        self.on_trait_change(self._on_edge_change,              'edge',                 dispatch='ui')
        self.on_trait_change(self._update_fit,              'count_data,perform_fit',   dispatch='ui')
        self.on_trait_change(self._compute_spin_state, 'count_data,integration_width,position_signal,position_normalize', dispatch='ui')
        
        
    def _counts_matrix_default(self):
        return np.zeros( (self.n_lines, self.n_laser) )    
    
    def _counts_default(self):
        return np.zeros(self.tau.shape)

    # data acquisition
    def generate_sequence(self):
        tau = self.tau
        laser = self.t_laser
        wait = self.t_wait
        decay_read = self.decay_read
        aom_delay = self.aom_delay
        mw = self.mw_source
        if aom_delay == 0.0:
            sequence = [ (['laser'],laser) ]
            for t in tau:
                sequence += [ ([],wait), ([mw],t), ([],decay_read),(['laser','trigger'],laser) ]
        else:
            s1 = [ (['laser'],laser) ]
            s2 = [ ([],aom_delay+laser) ]
            for t in tau:
                s1 += [ ([], wait+t+decay_read), (['laser'], laser) ]
                s2 += [ ([], wait), ([mw],t), ([],decay_read),(['trigger'],laser) ]
            s1 = sequence_remove_zeros(s1)            
            s2 = sequence_remove_zeros(s2)            
            sequence = sequence_union(s1,s2)
        sequence = sequence_defrag(sequence)
        return sequence        

    # def generate_sequence(self):
        # tau = self.tau
        # laser = self.laser
        # wait = self.wait
        # sequence = []
        # for t in tau:
            # sequence.append(  (['mw'               ] , t       )  )
            # sequence.append(  (['laser', 'trigger' ] , laser   )  )
            # sequence.append(  ([                   ] , wait    )  )
        # return sequence
        
    def apply_parameters(self):
        """Apply the current parameters."""
        n_bins = int(self.record_length / self.bin_width)
        time_bins = self.bin_width * np.arange(n_bins)
        

        self.tau = np.arange(self.tau_start, self.tau_end+self.tau_delta, self.tau_delta)
        sequence = self.generate_sequence()
        self.n_laser = find_laser_pulses(sequence)
        
        FC.Configure(self.record_length, self.bin_width, self.n_laser)

        if self.keep_data and sequence == self.sequence and np.all(time_bins == self.time_bins): # if the sequence and time_bins are the same as previous, keep existing data
            self.old_count_data = self.count_data.copy()
        else:
            self.old_count_data = np.zeros_like(FC.GetData())
        
        self.sequence = sequence
        self.time_bins = time_bins
        self.n_bins = n_bins
        
        self.MW_source = {'mw':mw, 'mw2':mw2}[self.mw_source]
        
    def _run(self):
                
        try:
            self.state='run'

            self.apply_parameters()
            PG.High([])
            FC.SetCycles(np.inf)
            FC.SetTime(np.inf)
            FC.SetDelay(0)
            FC.SetLevel(0.6, 0.6)
            self.MW_source.setFrequency(self.frequency)
            self.MW_source.setPower(self.power)
            FC.Start()
            time.sleep(2.0)

            self.run_time = 0
            elapsed_time = 0
            elapsed_sweeps = 0
            #previous_sweeps = 0
            start_time = time.time()

            PG.Sequence(self.sequence, loop=True)
            time.sleep(0.1)
            
            while elapsed_time < self.stop_time and elapsed_sweeps < self.sweeps:
               self.thread.stop_request.wait(1.0)
               if self.thread.stop_request.isSet():
                  logging.getLogger().debug('Caught stop signal. Exiting.')
                  break
               elapsed_time = time.time() - start_time
               self.run_time = elapsed_time
               runtime, cycles = FC.GetState()
               sweeps = cycles / FC.GetData().shape[0]
               self.elapsed_sweeps = int(sweeps) #previous_sweeps + sweeps
               self.count_data = self.old_count_data + FC.GetData()
               # self.trait_property_changed('counts', self.counts)
        except:
            logging.getLogger().exception('Error in Rabi_blockpraktikum.')
            self.state = 'error'
        finally:
            FC.Halt()
            PG.High(['laser'])
            self.MW_source.Off()
        self.state='done'

    def _compute_spin_state(self):
        y, profile, edge = spin_state(c=self.count_data,
                                      dt=self.bin_width,
                                      T=self.integration_width,
                                      t0=self.position_signal,
                                      t1=self.position_normalize,
                                      )
        self.spin_state = y
        self.pulse = profile
        self.edge = self.time_bins[edge] 
        self.trait_property_changed('pulse', self.pulse)
        self.trait_property_changed('edge', self.edge)
        
    # fitting
    def _update_fit(self):
        # try to extract pulses from counts and tau.
        tau = self.tau
        y = self.spin_state
        try:
            f,r,p,tp=fitting.extract_pulses(y)
        except:
            logging.getLogger().debug('Failed to compute fit.')
            f=[0]
            r=[0]
            p=[0]
            tp=[0]
            
        pi2         = tau[f]
        pi          = tau[p]
        three_pi2   = tau[r]
        two_pi      = tau[tp]

        # compute some relevant parameters from the result
        mi = y.min()
        ma = y.max()
        contrast =  100*(ma-mi)/ma
        # simple rough estimate of the period to avoid index out of range Error
        T = 4*(pi[0]-pi2[0])
        
        # set respective attributes
        self.period = T
        self.contrast = contrast
        self.t_pi2 = pi2[0]
        self.t_pi = pi[0]
        self.t_3pi2 = three_pi2[0]
        self.t_2pi = two_pi[0]

        # create a summary of the result as a text string
        s = 'contrast: %.1f\n'%contrast
        s += 'pi/2: %.2f ns\n'%pi2[0]
        s += 'pi: %.2f ns\n'%pi[0]
        s += '3pi/2: %.2f ns\n'%three_pi2[0]
        s += '2pi: %.2f ns\n'%two_pi[0]
        
        # markers in the plot that show the result of the pulse extraction
        self.line_data.set_data('pulse_indices',np.hstack((pi2, pi, three_pi2, two_pi)))            
        self.line_data.set_data('pulse_values',np.hstack((y[f], y[p], y[r], y[tp])))
        self.line_plot.overlays[0].text = s

    def _create_line_plot(self):
        line_data   = ArrayPlotData(index=np.array((0,1)),
                                    counts=np.array((0,0)),
                                    pulse_indices=np.array((0,0)),
                                    pulse_values=np.array((0,0)))
        plot = Plot(line_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('index','counts'), color='blue', name='counts')
        plot.plot(('pulse_indices','pulse_values'),
                  type='scatter',
                  marker='circle',
                  color='none',
                  outline_color='red',
                  line_width=1.0,
                  name='pulses')
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'spin state'
        #plot.overlays.insert(0, PlotLabel(text=self.label_text, hjustify='left', vjustify='bottom', position=[64,32]) )
        self.line_data = line_data
        self.line_plot = plot
        
    def _create_matrix_plot(self):
        matrix_data = ArrayPlotData(image=np.zeros((2,2)))
        matrix_plot = Plot(matrix_data, padding=8, padding_left=64, padding_bottom=32)
        matrix_plot.index_axis.title = 'time bins'
        matrix_plot.value_axis.title = 'tau'
        matrix_plot.img_plot('image',
                             xbounds=(self.tau[0],self.tau[-1]),
                             ybounds=(0,self.n_lines),
                             colormap=Spectral)
        matrix_plot.tools.append(SaveTool(matrix_plot))
        self.matrix_data = matrix_data
        self.matrix_plot = matrix_plot
        
    def _create_pulse_plot(self):
        pulse_data  = ArrayPlotData(x=np.array((0.,100)),y=np.array((0,1)))
        plot = Plot(pulse_data, padding=8, padding_left=64, padding_bottom=36)    
        line = plot.plot(('x','y'), style='line', color='blue', name='data')[0]
        plot.index_axis.title = 'time bins'
        plot.value_axis.title = 'intensity'
        edge_marker = LinePlot(index = ArrayDataSource(np.array((0,0))),
                               value = ArrayDataSource(np.array((0,1e9))),
                               color = 'red',
                               index_mapper = LinearMapper(range=plot.index_range),
                               value_mapper = LinearMapper(range=plot.value_range),
                               name='marker')
        plot.add(edge_marker)
        plot.tools.append(SaveTool(plot))
        self.pulse_data = pulse_data
        self.pulse_plot = plot
            
    def _on_edge_change(self):
        y = self.edge
        self.pulse_plot.components[1].index.set_data(np.array((y,y)))
    
    def _perform_fit_changed(self,new):
        plot = self.line_plot
        if new:
            plot.plot(('frequency','fit'), style='line', color='red', name='fit')
            self.line_label.visible=True
        else:
            plot.delplot('fit')
            self.line_label.visible=False
        plot.request_redraw()

    def _update_line_data_index(self):
        self.line_data.set_data('index', self.tau)
        #self.counts_matrix = self._counts_matrix_default()

    def _update_line_data_value(self):
        self.line_data.set_data('counts', self.spin_state)

    def _update_line_data_fit(self):
        pass
        # if not np.isnan(self.fit_parameters[0]):            
            # self.line_data.set_data('fit', n_lorentzians(*self.fit_parameters)(self.frequency))
            # p = self.fit_parameters
            # f = p[1::3]
            # w = p[2::3]
            # N = len(p)/3
            # contrast = np.empty(N)
            # c = p[0]
            # pp=p[1:].reshape((N,3))
            # for i,pi in enumerate(pp):
                # a = pi[2]
                # g = pi[1]
                # A = np.abs(a/(np.pi*g))
                # if a > 0:
                    # contrast[i] = 100*A/(A+c)
                # else:
                    # contrast[i] = 100*A/c
            # s = ''
            # for i, fi in enumerate(f):
                # s += 'f %i: %.2f MHz (%.1f G), HWHM %.1f MHz, contrast %.1f%%\n'%(i+1, fi/1e6, np.abs(2870-fi/1e6)/2.8, w[i]/1e6, contrast[i])
            # self.line_label.text = s

    def _update_matrix_data_index(self):
        self.matrix_plot.components[0].index.set_data((self.time_bins[0], self.time_bins[-1]),(0.0,float(self.n_laser)))
            
    def _update_matrix_data_value(self):
        self.matrix_data.set_data('image', self.count_data)
                
    def _update_pulse_index(self):
        self.pulse_data.set_data('x', self.time_bins)        
    
    def _update_pulse_value(self):
        self.pulse_data.set_data('y', self.pulse)

    # saving data
    
    def save_all(self, filename):
        self.line_plot.save(filename+'_Rabi_Line_Plot.png')
        self.matrix_plot.save(filename+'_Rabi_Matrix_Plot.png')
        self.save(filename+'_Rabi.pys')

    # react to GUI events

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
        self.resubmit() 

    traits_view = View(VGroup(HGroup(Item('submit_button',   show_label=False),
                                     Item('remove_button',   show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority', enabled_when='state != "run"'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly',format_str='%.f'),
                                     Item('stop_time'),
                                     Item('elapsed_sweeps', style='readonly')
                                     ),
                              #HGroup(Item('filename',springy=True),
                                     #Item('save_button', show_label=False),
                                     #Item('load_button', show_label=False)
                                     #),
                              VGroup(HGroup(Item('power', width=-40, enabled_when='state != "run"'),
                                            Item('frequency', width=-80, enabled_when='state != "run"'),
                                            Item('mw_source',width=-60,style='custom',enabled_when='state != "run"'),
                                            Item('t_laser', width=-50, enabled_when='state != "run"'),
                                            Item('record_length', width=-50, enabled_when='state != "run"'),
                                            Item('bin_width', width=-50, enabled_when='state != "run"')
                                            ),
                                     HGroup(Item('t_wait', width=-50, enabled_when='state != "run"'),
                                            Item('decay_read', width=-50, enabled_when='state != "run"'),
                                            Item('aom_delay', width=-50, enabled_when='state != "run"'),
                                            ),
                                     HGroup(Item('perform_fit'),
                                            Item('sweeps'),
                                            Item('stop_time'),
                                            Item('n_lines', width=-60),
                                            ),
                                     HGroup(Item('tau_start', width=-80, enabled_when='state != "run"'),
                                            Item('tau_end', width=-80, enabled_when='state != "run"'),
                                            Item('tau_delta', width=-80, enabled_when='state != "run"'),
                                            ),
                                     HGroup(Item('contrast', width=-60, style='readonly'),
                                            Item('period', width=-60, style='readonly'),
                                            Item('t_pi2', width=-60, style='readonly'),
                                            Item('t_pi', width=-60, style='readonly'),
                                            Item('t_3pi2', width=-60, style='readonly'),
                                            Item('t_2pi', width=-60, style='readonly'),
                                            ),
                                     ),
                              VSplit(Item('line_plot', show_label=False, resizable=True),
                                     Item('pulse_plot', show_label=False, width=500, height=-300, resizable=True),
                                     Item('matrix_plot', show_label=False, resizable=True)
                                     ),
                              ),
                       title='Rabi Blockpraktikum', width=900, height=800, buttons=[], resizable=True
                       )

    get_set_items = ['frequency', 'counts', 'counts_matrix', 'n_lines', 'mw_source',
                     'fit_parameters', 'fit_contrast', 'fit_line_width', 'fit_frequencies', 'axis',
                     'perform_fit', 'run_time', 'stop_time'
                     'power', 'tau_start', 'tau_end', 'tau_delta',
                     't_laser', 'wait', 
                     'stop_time', 'n_lines',
                     '__doc__']

if __name__ == '__main__':

    pass    
    






