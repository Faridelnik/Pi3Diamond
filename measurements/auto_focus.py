"""
Auto focus tool.
"""

import numpy as np

from traits.api       import SingletonHasTraits, Instance, Range, Bool, Array, Str, Enum, Button, on_trait_change, Trait
from traitsui.api     import View, Item, Group, HGroup, VGroup, VSplit, Tabbed, EnumEditor, Action, Menu, MenuBar, TextEditor
from enable.api       import ComponentEditor
from chaco.api        import HPlotContainer, Plot, PlotAxis, CMapImagePlot, ColorBar, LinearMapper, ArrayPlotData, jet

# date and time tick marks
from chaco.scales.api import CalendarScaleSystem
from chaco.scales_tick_generator import ScalesTickGenerator

import threading
import time
import logging

from tools.emod import ManagedJob
from tools.cron import CronDaemon, CronEvent

from tools.utility import GetSetItemsHandler, GetSetItemsMixin, StoppableThread, warning

from hardware.api import Scanner
import hardware.api as ha
from measurements.confocal import Confocal
from measurements.odmr import ODMR
from measurements.counter_trace import CounterTrace   #to stop time trace when autofocusing

time_trace = CounterTrace() #to stop time trace when autofocusing
scanner = Scanner() 

pg = ha.PulseGenerator()

class AutoFocusHandler( GetSetItemsHandler ):
    """Provides target menu."""
    
    def remove_all_targets(self,info):
        info.object.remove_all_targets()

    def forget_drift(self,info):
        info.object.forget_drift()


class AutoFocus( ManagedJob, GetSetItemsMixin ):

    # overwrite default priority from ManagedJob (default 0)
    priority = 10

    confocal = Instance( Confocal )
    odmr = Instance( ODMR )
    #counter_trace = Instance ( CounterTrace )

    size_xy                 = Range(low=0.5, high=10., value=0.8,   desc='Size of XY Scan',                 label='Size XY [micron]',           mode='slider',  auto_set=False, enter_set=True)
    size_z                  = Range(low=0.5, high=10., value=1.5,    desc='Size of Z Scan',                  label='Size Z [micron]',            mode='slider',  auto_set=False, enter_set=True)
    step_xy                 = Range(low=0.01, high=10., value=0.04,  desc='Step of XY Scan',                 label='Step XY [micron]',           mode='slider',  auto_set=False, enter_set=True)
    step_z                  = Range(low=0.01, high=10., value=0.15,  desc='Step of Z Scan',                  label='Step Z [micron]',            mode='slider',  auto_set=False, enter_set=True)
    seconds_per_point_xy    = Range(low=1e-3, high=10, value=0.03,  desc='Seconds per point for XY Scan',   label='seconds per point XY [s]',   mode='text',    auto_set=False, enter_set=True)
    seconds_per_point_z     = Range(low=1e-3, high=10, value=0.05,   desc='Seconds per point for Z Scan',    label='seconds per point Z  [s]',    mode='text',    auto_set=False, enter_set=True)

    fit_method_xy = Enum('Maximum', 'Gaussian', desc='Fit Method for XY Scan',    label='XY Fit Method')
    fit_method_z  = Enum('Maximum', 'Gaussian', desc='Fit Method for Z Scan',     label='Z Fit Method')

    X = Array(value=np.array((0.,1.)) )
    Y = Array(value=np.array((0.,1.)) )
    Z = Array(value=np.array((-1.,1.)) )

    data_xy = Array( )
    data_z = Array( value=np.array((0,0)) )

    targets         = Instance( {}.__class__, factory={}.__class__ ) # Dict traits are no good for pickling, therefore we have to do it with an ordinary dictionary and take care about the notification manually 
    target_list     = Instance( list, factory=list, args=([None],) ) # list of targets that are selectable in current_target editor
    current_target  = Enum(values='target_list')
        
    drift               = Array( value=np.array(((0,0,0,),)) )
    drift_time          = Array( value=np.array((0,)) )
    current_drift       = Array( value=np.array((0,0,0)) )

    focus_interval    = Range(low=1, high=6000, value=10, desc='Time interval between automatic focus events', label='Interval [m]', auto_set=False, enter_set=True)
    periodic_focus    = Bool(False, label='Periodic focusing')
    periodic_freq_feedback  = Bool(False, label='Periodic freq feedback')
    threshold = Range(low=1, high=6000, value=1000, desc='ignore oil junks', label='threshold kcs/s', auto_set=False, enter_set=True)

    target_name = Str(label='name', desc='name to use when adding or removing targets')
    add_target_button       = Button(label='Add Target', desc='add target with given name')
    remove_current_target_button    = Button(label='Remove Current', desc='remove current target')
    next_target_button      = Button(label='Next Target', desc='switch to next available target')
    undo_button       = Button(label='undo', desc='undo the movement of the stage')
    
    previous_state = Instance( () )
    
    plot_data_image = Instance( ArrayPlotData )
    plot_data_line  = Instance( ArrayPlotData )
    plot_data_drift = Instance( ArrayPlotData )
    figure_image    = Instance( HPlotContainer, editor=ComponentEditor() )
    figure_line     = Instance( Plot, editor=ComponentEditor() )
    figure_drift    = Instance( Plot, editor=ComponentEditor() )
    image_plot      = Instance( CMapImagePlot )

    def __init__(self, confocal, odmr):
        super(AutoFocus, self).__init__()
        self.confocal = confocal
        self.odmr = odmr
        #self.counter_trace = counter_trace
        self.on_trait_change(self.update_plot_image, 'data_xy', dispatch='ui')
        self.on_trait_change(self.update_plot_line_value, 'data_z', dispatch='ui')
        self.on_trait_change(self.update_plot_line_index, 'Z', dispatch='ui')
        self.on_trait_change(self.update_plot_drift_value, 'drift', dispatch='ui')
        self.on_trait_change(self.update_plot_drift_index, 'drift_time', dispatch='ui')
        

    @on_trait_change('next_target_button')
    def next_target(self):
        """Convenience method to switch to the next available target."""
        keys = self.targets.keys()
        key = self.current_target
        if len(keys) == 0:
            logging.getLogger().info('No target available. Add a target and try again!')
        elif not key in keys:
            self.current_target = keys[0]
        else:
            self.current_target = keys[(keys.index(self.current_target)+1)%len(keys)]

    def _targets_changed(self, name, old, new):
        l = new.keys() + [None]      # rebuild target_list for Enum trait
        l.sort()
        self.target_list = l
        self._draw_targets()    # redraw target labels

    def _current_target_changed(self):
        self._draw_targets()    # redraw target labels

    def _draw_targets(self):
        c = self.confocal
        c.remove_all_labels()
        c.show_labels=True
        for key, coordinates in self.targets.iteritems():
            if key == self.current_target:
                c.set_label(key, coordinates, marker_color='red')
            else:
                c.set_label(key, coordinates)

    def _periodic_focus_changed(self, new):
        if not new and hasattr(self, 'cron_event'):
            CronDaemon().remove(self.cron_event)
        if new:
            self.cron_event = CronEvent(self.submit, min=range(0,60,self.focus_interval))
            CronDaemon().register(self.cron_event)

    def fit_xy(self):
        if self.fit_method_xy == 'Maximum':
            index = self.data_xy.argmax()
            xp = self.X[index%len(self.X)]
            yp = self.Y[index/len(self.X)]
            self.XYFitParameters = [xp, yp]
            self.xfit = xp
            self.yfit = yp
            return xp, yp
        else:
            print 'Not Implemented! Fix Me!'

    def fit_z(self):
        if self.fit_method_z == 'Maximum':
            zp = self.Z[self.data_z.argmax()]
            self.zfit = zp
            return zp
        else:
            print 'Not Implemented! Fix Me!'

    def add_target(self, key, coordinates=None):
        if coordinates is None:
            c = self.confocal
            coordinates = np.array((c.x,c.y,c.z))
        if self.targets == {}:
            self.forget_drift()
        if self.targets.has_key(key):
            if warning('A target with this name already exists.\nOverwriting will move all targets.\nDo you want to continue?'):
                self.current_drift = coordinates - self.targets[key]
                self.forget_drift()
            else:
                return
        else:
            coordinates = coordinates - self.current_drift
            self.targets[key] = coordinates
        self.trait_property_changed('targets', self.targets)    # trigger event such that Enum is updated and Labels are redrawn
        self.confocal.show_labels=True

    def remove_target(self, key):
        if not key in self.targets:
            logging.getLogger().info('Target cannot be removed. Target does not exist.')
            return
        self.targets.pop(key)        # remove target from dictionary
        self.trait_property_changed('targets', self.targets)    # trigger event such that Enum is updated and Labels are redrawn
        
    def remove_all_targets(self):
        self.targets = {}

    def forget_drift(self):
        targets = self.targets
        # reset coordinates of all targets according to current drift
        for key in targets:
            targets[key] += self.current_drift
        # trigger event such that target labels are redrawn
        self.trait_property_changed('targets', self.targets)
        # set current_drift to 0 and clear plot
        self.current_drift = np.array((0., 0., 0.))
        self.drift_time = np.array((time.time(),))
        self.drift = np.array(((0,0,0),))
        
    def _add_target_button_fired(self):
        self.add_target( self.target_name )
        
    def _remove_current_target_button_fired(self):
        self.remove_target( self.current_target )

    def _run(self):
        
        logging.getLogger().debug("trying run.")
        
        try:
            self.state='run'
            #ha.PulseGenerator().Light()
            
           
            if self.current_target is None:
                self.focus()
                if np.amax(self.data_xy)<10:
                    self._undo_button_fired()
                    pg.Light()
                    self.focus()
                # if self.periodic_freq_feedback: 
                    # self.odmr.submit()
                   
            else: # focus target
                coordinates = self.targets[self.current_target]
                confocal = self.confocal
                confocal.x, confocal.y, confocal.z = coordinates + self.current_drift
                current_coordinates = self.focus()
                self.current_drift = current_coordinates - coordinates  
                self.drift = np.append(self.drift, (self.current_drift,), axis=0)
                self.drift_time = np.append(self.drift_time, time.time())
                logging.getLogger().debug('Drift: %.2f, %.2f, %.2f'%tuple(self.current_drift))
                # if self.periodic_freq_feedback: 
                    # self.odmr.submit()
                    
        finally:
            self.state = 'idle'

    def focus(self):
            """
            Focuses around current position in x, y, and z-direction.
            """
            xp = self.confocal.x
            yp = self.confocal.y
            zp = self.confocal.z
            self.previous_state = ((xp,yp,zp), self.current_target)
            ##+scanner.getXRange()[1]
            safety = 0 #distance to keep from the ends of scan range
            xmin = np.clip(xp-0.5*self.size_xy, scanner.getXRange()[0]+safety, scanner.getXRange()[1]-safety)
            xmax = np.clip(xp+0.5*self.size_xy, scanner.getXRange()[0]+safety, scanner.getXRange()[1]-safety)
            ymin = np.clip(yp-0.5*self.size_xy, scanner.getYRange()[0]+safety, scanner.getYRange()[1]-safety)
            ymax = np.clip(yp+0.5*self.size_xy, scanner.getYRange()[0]+safety, scanner.getYRange()[1]-safety)
            
            X = np.arange(xmin, xmax, self.step_xy)
            Y = np.arange(ymin, ymax, self.step_xy)

            self.X = X
            self.Y = Y

            XP = X[::-1]

            self.data_xy=np.zeros((len(Y),len(X)))
            #self.image_plot.index.set_data(X, Y)  
                        
            for i,y in enumerate(Y):
                if threading.current_thread().stop_request.isSet():
                    self.confocal.x = xp
                    self.confocal.y = yp
                    self.confocal.z = zp
                    return xp, yp, zp                    
                if i%2 != 0:
                    XL = XP
                else:
                    XL = X
                YL = y * np.ones(X.shape)
                ZL = zp * np.ones(X.shape)
                Line = np.vstack( (XL, YL, ZL) )
                
                c = scanner.scanLine(Line, self.seconds_per_point_xy)/1e3
                if i%2 == 0:
                    self.data_xy[i,:] = c[:]
                else:
                    self.data_xy[i,:] = c[-1::-1]
                
                self.trait_property_changed('data_xy', self.data_xy)
                
            for i in range(self.data_xy.shape[0]):
                for j in range(self.data_xy.shape[1]):
                    if self.data_xy[i][j]>self.threshold:
                        self.data_xy[i][j]=0
            
            xp, yp = self.fit_xy()
                        
            self.confocal.x = xp
            self.confocal.y = yp

            Z = np.hstack( ( np.arange(zp, zp-0.5*self.size_z, -self.step_z),
                                np.arange(zp-0.5*self.size_z, zp+0.5*self.size_z, self.step_z),
                                np.arange(zp+0.5*self.size_z, zp, -self.step_z) ) )
            Z = np.clip(Z, scanner.getZRange()[0]+safety, scanner.getZRange()[1]-safety)

            X = xp * np.ones(Z.shape)
            Y = yp * np.ones(Z.shape)

            if not threading.current_thread().stop_request.isSet():
                Line = np.vstack( (X, Y, Z) )
                data_z = scanner.scanLine(Line, self.seconds_per_point_z)/1e3

                self.Z = Z
                self.data_z = data_z
                
            for i in range(self.data_z.shape[0]):
                if self.data_z[i]>self.threshold:
                        self.data_z[i]=0

                zp = self.fit_z()

            self.confocal.z = zp

            logging.getLogger().info('Focus: %.2f, %.2f, %.2f' %(xp, yp, zp))
            return xp, yp, zp

    def undo(self):
        if self.previous_state is not None:
            coordinates, target = self.previous_state
            self.confocal.x, self.confocal.y, self.confocal.z = coordinates
            if target is not None:
                self.drift_time = np.delete(self.drift_time, -1)
                self.current_drift = self.drift[-2]
                self.drift = np.delete(self.drift, -1, axis=0)
            self.previous_state = None
        else:
            logging.getLogger().info('Can undo only once.')

    def _undo_button_fired(self):
        self.remove()
        self.undo()
    
    def _plot_data_image_default(self):
        return ArrayPlotData(image=np.zeros((2,2)))
    def _plot_data_line_default(self):
        return ArrayPlotData(x=self.Z, y=self.data_z)
    def _plot_data_drift_default(self):
        return ArrayPlotData(t=self.drift_time, x=self.drift[:,0], y=self.drift[:,1], z=self.drift[:,2])

    def _figure_image_default(self):
        plot = Plot(self.plot_data_image, width=180, height=180, padding=3, padding_left=48, padding_bottom=32)
        plot.img_plot('image', colormap=jet, name='image')
        plot.aspect_ratio=1
        #plot.value_mapper.domain_limits = (scanner.getYRange()[0],scanner.getYRange()[1])
        #plot.index_mapper.domain_limits = (scanner.getXRange()[0],scanner.getXRange()[1])
        plot.value_mapper.domain_limits = (0,self.size_xy)
        plot.index_mapper.domain_limits = (0,self.size_xy)

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
    def _figure_line_default(self):
        plot = Plot(self.plot_data_line, width=70, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('x','y'), color='blue')
        plot.index_axis.title = 'z [um]'
        plot.value_axis.title = 'Fluorescence [ k / s ]'
        return plot
    def _figure_drift_default(self):
        plot = Plot(self.plot_data_drift, width=70, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('t','x'), type='line', color='blue', name='x')
        plot.plot(('t','y'), type='line', color='red', name='y')
        plot.plot(('t','z'), type='line', color='green', name='z')
        bottom_axis = PlotAxis(plot,
                               orientation="bottom",
                               tick_generator=ScalesTickGenerator(scale=CalendarScaleSystem()))
        plot.index_axis=bottom_axis
        plot.index_axis.title = 'time'
        plot.value_axis.title = 'drift [um]'
        plot.legend.visible=True
        return plot        

    def _image_plot_default(self):
        return self.figure_image.components[0].plots['image'][0]

    def update_plot_image(self):
        self.plot_data_image.set_data('image', self.data_xy)
    def update_plot_line_value(self):
        self.plot_data_line.set_data('y', self.data_z)        
    def update_plot_line_index(self):
        self.plot_data_line.set_data('x', self.Z)
    def update_plot_drift_value(self):
        if len(self.drift) == 1:
            self.plot_data_drift.set_data('x', np.array(()))
            self.plot_data_drift.set_data('y', np.array(()))
            self.plot_data_drift.set_data('z', np.array(()))            
        else:
            self.plot_data_drift.set_data('x', self.drift[:,0])
            self.plot_data_drift.set_data('y', self.drift[:,1])
            self.plot_data_drift.set_data('z', self.drift[:,2])
    def update_plot_drift_index(self):
        if len(self.drift_time) == 0:
            self.plot_data_drift.set_data('t', np.array(()))
        else:
            self.plot_data_drift.set_data('t', self.drift_time - self.drift_time[0])

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('undo_button', show_label=False),
                                     ),
                              Group(VGroup(HGroup(Item('target_name'),
                                                  Item('add_target_button', show_label=False),
                                                  ),
                                           HGroup(Item('current_target'),
                                                  Item('next_target_button', show_label=False),
                                                  Item('remove_current_target_button', show_label=False),
                                                  ),
                                           HGroup(Item('periodic_focus'),
                                                  Item('focus_interval', enabled_when='not periodic_focus'),
                                                  ),
                                           HGroup(Item('periodic_freq_feedback'),
                                                  Item('threshold')
                                                  ),       
                                           label='tracking',
                                           ),
                                    VGroup(Item('size_xy'),
                                           Item('step_xy'),
                                           Item('size_z'),
                                           Item('step_z'),
                                           HGroup(Item('seconds_per_point_xy'),
                                                  Item('seconds_per_point_z'),
                                                  ),
                                           label='Settings',
                                           springy=True,
                                           ),
                                    layout='tabbed'
                                    ),
                              VSplit(Item('figure_image', show_label=False, resizable=True)),
                              HGroup(Item('figure_line', show_label=False, resizable=True),
                                     Item('figure_drift', show_label=False, resizable=True),
                                    ),
                              ),
                       menubar = MenuBar(Menu(Action(action='save', name='Save (.pyd or .pys)'),
                                              Action(action='load', name='Load'),
                                              Action(action='_on_close', name='Quit'),
                                              name='File'),
                                         Menu(Action(action='remove_all_targets', name='Remove All'),
                                              Action(action='forget_drift', name='Forget Drift'),
                                              name='Target'),),
                       title='Auto Focus', width=500, height=700, buttons=[], resizable=True,
                       handler=AutoFocusHandler)

    get_set_items=['confocal','targets','current_target','current_drift','drift','drift_time','periodic_focus',
                   'size_xy', 'size_z', 'step_xy', 'step_z', 'seconds_per_point_xy', 'seconds_per_point_z',
                   'data_xy', 'data_z', 'X', 'Y', 'Z', 'focus_interval', 'threshold' ]
    get_set_order=['confocal','targets']

    
# testing

if __name__ == '__main__':
    
    logging.getLogger().setLevel(logging.DEBUG)
    
    from emod import JobManager
    
    JobManager().start()

    from cron import CronDaemon
    
    CronDaemon().start()
    
    c = Confocal()
    c.edit_traits()
    a = AutoFocus(c)
    a.edit_traits()
        