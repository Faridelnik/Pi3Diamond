
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

from hardware.api           import TimeTagger, RotationStage, PowerMeter

from tools.emod             import ManagedJob

from tools.utility          import GetSetItemsHandler, GetSetItemsMixin

class PolarizationHandler( GetSetItemsHandler ):
    
    def save_plot(self, info):
        filename = save_file(title='Save Plot')
        if filename is '':
            return
        else:
            if filename.find('.png')==-1:
                filename=filename+'.png'
            info.object.save_plot(filename)
    
class Polarization( ManagedJob, GetSetItemsMixin):
    """
    Record a polarization curve.
    
    written by: helmut.fedder@gmail.com
    last modified: 2012-08-17
    """
    
    seconds_per_point = Range(low=1e-4, high=100., value=1., desc='integration time for one point', label='seconds per point', mode='text', auto_set=False, enter_set=True)
    angle_step = Range(low=1e-3, high=100., value=1., desc='angular step', label='angle step', mode='text', auto_set=False, enter_set=True)

    angle = Array()
    intensity = Array()
    power = Array()
    
    plot = Instance( Plot )
    plot_data = Instance( ArrayPlotData )
    
    get_set_items=['__doc__', 'seconds_per_point', 'angle_step', 'angle', 'intensity', 'power']
    
    def __init__(self):
        super(Polarization, self).__init__()
        self._create_plot()
        self.on_trait_change(self._update_index,    'angle',        dispatch='ui')
        self.on_trait_change(self._update_value,    'intensity',    dispatch='ui')
        
    def _run(self):
        """Acquire data."""
        
        try: # run the acquisition
            self.state='run'
    
            RotationStage().go_home()
            
            self.angle = np.array(())
            self.intensity = np.array(())
            self.power = np.array(())
    
            c1 = TimeTagger.Countrate(0)
            c2 = TimeTagger.Countrate(1)
            
            for phi in np.arange(0.,360.,self.angle_step):
                RotationStage().set_angle(phi)
                c1.clear()
                c2.clear()
                self.thread.stop_request.wait(self.seconds_per_point)
                if self.thread.stop_request.isSet():
                    logging.getLogger().debug('Caught stop signal. Exiting.')
                    self.state = 'idle'
                    break            
                self.angle=np.append(self.angle,phi)
                self.intensity=np.append(self.intensity,c1.getData() + c2.getData())
                self.power=np.append(self.power,PowerMeter().getPower())
            else:
                self.state='done'

        except: # if anything fails, recover
            logging.getLogger().exception('Error in polarization.')
            self.state='error'
        finally:
            del c1
            del c2

    def _create_plot(self):
        plot_data = ArrayPlotData(angle=np.array(()), intensity=np.array(()),)
        plot = Plot(plot_data, padding=8, padding_left=64, padding_bottom=64)
        plot.plot(('angle','intensity'), color='blue')
        plot.index_axis.title = 'angle [deg]'
        plot.value_axis.title = 'intensity [count/s]'
        self.plot_data = plot_data
        self.plot = plot

    def _update_index(self,new):
        self.plot_data.set_data('angle',new)
        
    def _update_value(self,new):
        self.plot_data.set_data('intensity',new)

    def save_plot(self, filename):
        self.save_figure(self.plot, filename)
        
    traits_view = View(HGroup(Item('submit_button',   show_label=False),
                              Item('remove_button',   show_label=False),
                              Item('priority'),
                              Item('state', style='readonly'),
                              ),
                       HGroup(Item('seconds_per_point'),
                              Item('angle_step'),
                              ),
                       Item('plot', editor=ComponentEditor(), show_label=False),
                       menubar = MenuBar(
                                         Menu(Action(action='save', name='Save (.pyd or .pys)'),
                                              Action(action='load', name='Load'),
                                              Action(action='save_plot', name='Save Plot (.png)'),
                                              Action(action='_on_close', name='Quit'),
                                              name='File')),
                       title='Polarization', width=640, height=640, buttons=[], resizable=True,
                       handler=PolarizationHandler)

if __name__ == '__main__':
    
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().info('Starting logger.')
    
    from tools.emod import JobManager
    JobManager().start()
    
    p = Polarization()    
    p.edit_traits()
    