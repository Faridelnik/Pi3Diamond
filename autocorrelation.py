
from traits.api             import Range, Array, Instance, Enum, on_trait_change, Bool, Button, Float
from traitsui.api           import View, Item, HGroup #, EnumEditor
from enable.api             import ComponentEditor
from chaco.api              import ArrayPlotData, Plot 
from traitsui.menu          import Action, Menu, MenuBar
from traitsui.file_dialog   import save_file

from threading              import currentThread
import logging
import time

import numpy as np

from hardware.api           import TimeTagger

from tools.emod             import ManagedJob

from tools.utility          import GetSetItemsHandler, GetSetItemsMixin

class AutocorrelationHandler( GetSetItemsHandler ):
    
    def save_plot(self, info):
        filename = save_file(title='Save Plot')
        if filename is '':
            return
        else:
            if filename.find('.png')==-1:
                filename=filename+'.png'
            info.object.save_plot(filename)
    
class Autocorrelation( ManagedJob, GetSetItemsMixin):
    
    window_size = Range(low=1., high=10000., value=500., desc='window_size in the time domain  [ns]', label='window_size', mode='text', auto_set=False, enter_set=True)
    bin_width   = Range(low=0.01, high=1000., value=1., desc='bin width  [ns]', label='bin width', mode='text', auto_set=False, enter_set=True)
    chan1 = Enum(0,1,2,3,4,5,6,7, desc="the trigger channel", label="Channel 1")
    chan2 = Enum(0,1,2,3,4,5,6,7, desc="the signal channel", label="Channel 2")
    normalize = Bool(False, desc="normalize autocorrelation", label="normalize")

    counts = Array()
    time_bins = Array()
    
    keep_data = Bool(False) # helper variable to decide whether to keep existing data
    resubmit_button = Button(label='resubmit', desc='Submits the measurement to the job manager. Tries to keep previously acquired data. Behaves like a normal submit if sequence or time bins have changed since previous run.')
    run_time = Float(value=0.0, label='run time [s]')
    stop_time = Range(low=1., value=np.inf, desc='Time after which the experiment stops by itself [s]', label='Stop time [s]', mode='text', auto_set=False, enter_set=True)
    
    plot = Instance( Plot )
    plot_data = Instance( ArrayPlotData )
    
    get_set_items=['window_size', 'bin_width', 'chan1', 'chan2', 'normalize', 'counts', 'time_bins', 'norm_factor']
    
    def __init__(self):
        super(Autocorrelation, self).__init__()
        self._create_plot()
        
    def submit(self):
        """Submit the job to the JobManager."""
        self.keep_data = False
        self.run_time = 0.0
        ManagedJob.submit(self)

    def resubmit(self):
        """Submit the job to the JobManager."""
        self.keep_data = True
        ManagedJob.submit(self)

    def _resubmit_button_fired(self):
        """React to start button. Submit the Job."""
        self.resubmit() 

    @on_trait_change('chan1,chan2,window_size,bin_width')
    def _create_threads(self):
        self.p1 = TimeTagger.Pulsed(int(np.round(self.window_size/self.bin_width)),int(np.round(self.bin_width*1000)),1,self.chan1,self.chan2)
        self.p2 = TimeTagger.Pulsed(int(np.round(self.window_size/self.bin_width)),int(np.round(self.bin_width*1000)),1,self.chan2,self.chan1) 
        self.run_time = 0.0
    
    def _run(self):
        """Acquire data."""
        
        try: # run the acquisition
            self.state='run'
    
            if self.run_time >= self.stop_time:
                logging.getLogger().debug('Runtime larger than stop_time. Returning')
                self.state='done'
                return
    
            n_bins = int(np.round(self.window_size/self.bin_width))
            time_bins = self.bin_width*np.arange(-n_bins+1,n_bins)
            
            if self.keep_data:
                try:
                    self.p1.start()
                    self.p2.start()
                except:
                    self._create_threads()
            else:
                self._create_threads()
            
            start_time = time.time()
            
            self.time_bins = time_bins
            self.n_bins = n_bins
            self.keep_data = True # when job manager stops and starts the job, data should be kept. Only new submission should clear data.
    
            while self.run_time < self.stop_time:
                data1 = self.p1.getData()
                data2 = self.p2.getData()
                current_time = time.time()
                self.run_time += current_time - start_time
                start_time = current_time
                data = np.append(np.append(data1[0][-1:0:-1], max(data1[0][0],data2[0][0])),data2[0][1:])
                
                self.counts = data
                
                try:
                    self.norm_factor = self.run_time/(self.bin_width*1e-9*self.p1.getCounts()*self.p2.getCounts())
                except:
                    self.norm_factor = 1.0
                
                self.thread.stop_request.wait(1.0)
                if self.thread.stop_request.isSet():
                    logging.getLogger().debug('Caught stop signal. Exiting.')
                    break
    
            self.p1.stop()
            self.p2.stop()
    
            if self.run_time < self.stop_time:
                self.state = 'idle'
            else:
                self.state='done'

        except: # if anything fails, recover
            logging.getLogger().exception('Error in autocorrelation.')
            self.state='error'

    def save_plot(self, filename):
        self.save_figure(self.plot, filename)
    
    def _counts_default(self):
        return np.zeros((int(np.round(self.window_size/self.bin_width))*2-1,))
    
    def _time_bins_default(self):
        return self.bin_width*np.arange(-int(np.round(self.window_size/self.bin_width))+1,int(np.round(self.window_size/self.bin_width)))
    
    def _time_bins_changed(self):
        self.plot_data.set_data('t', self.time_bins)

    def _counts_changed(self):
        if self.normalize:
            self.plot_data.set_data('y', self.counts*self.norm_factor)
        else:
            self.plot_data.set_data('y', self.counts)
    
    def _chan1_default(self):
        return 0
    
    def _chan2_default(self):
        return 1 
        
    def _window_size_changed(self):
        self.counts = self._counts_default()
        self.time_bins = self._time_bins_default()
        
    def _create_plot(self):
        data = ArrayPlotData(t=self.time_bins, y=self.counts)
        plot = Plot(data, width=500, height=500, resizable='hv')
        plot.plot(('t','y'), type='line', color='blue')
        self.plot_data=data
        self.plot=plot
    
    traits_view = View(HGroup(Item('submit_button',   show_label=False),
                              Item('remove_button',   show_label=False),
                              Item('resubmit_button', show_label=False),
                              Item('priority'),
                              Item('state', style='readonly'),
                              Item('run_time', style='readonly',format_str='%.f'),
                              Item('stop_time'),
                              ),
        Item('plot', editor=ComponentEditor(), show_label=False),
        HGroup(
            Item('window_size'),
            Item('bin_width'),
            Item('chan1'),
            Item('chan2'),
            Item('normalize'),
            ),
        menubar = MenuBar(
            Menu(
                Action(action='save', name='Save (.pyd or .pys)'),
                Action(action='load', name='Load'),
                Action(action='save_plot', name='Save Plot (.png)'),
                Action(action='_on_close', name='Quit'),
            name='File')),
        title='Autocorrelation', width=900, height=800, buttons=[], resizable=True,
        handler=AutocorrelationHandler)

if __name__ == '__main__':
    
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().info('Starting logger.')
    
    from tools.emod import JobManager
    JobManager().start()
    
    a = Autocorrelation()    
    a.edit_traits()
    