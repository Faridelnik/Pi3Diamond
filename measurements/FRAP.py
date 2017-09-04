from tools.utility import edit_singleton
from datetime import date
import os
from traits.api import SingletonHasTraits, Trait, Instance, Property, String, Range, Float, Int, Bool, Array, Enum, Button, on_trait_change, cached_property, Code, List, NO_COMPARE, Tuple
from traitsui.api import View, Item, HGroup, VGroup, VSplit, Tabbed, EnumEditor, TextEditor, Group, Action, Menu, MenuBar
from enable.api import Component, ComponentEditor
from chaco.api import ArrayPlotData, Plot, Spectral, PlotLabel, OverlayPlotContainer, HPlotContainer
import imp
import numpy as np
import string
import time
import threading
import hardware.SMC_controller as smc
import hardware.api as ha
pg = ha.PulseGenerator()
from hardware.api import Scanner
scanner = Scanner()

from tools.emod import ManagedJob
from tools.utility import GetSetItemsHandler, GetSetItemsMixin, GetSetSaveImageHandler

import hardware.api as ha


class FluorescenceRecoveryAfterPhotobleaching(ManagedJob, GetSetItemsMixin):

    plot = Instance(OverlayPlotContainer)
    plot_data = Instance(ArrayPlotData)
    time1 = Array()
    intensity = Array()
    distance = Array()

    def __init__(self, confocal):
        super(FluorescenceRecoveryAfterPhotobleaching, self).__init__()     
        
        self._plot_default()
        self.confocal=confocal
        
    def _plot_default(self):
        plot_data = ArrayPlotData(distance = np.array((0., 1.)), i1 = np.array((0., 1.)), i2=np.array((0., 1.)), i3 = np.array((0., 1.)), i4 = np.array((0., 1.)), i5 = np.array((0., 1.)))
        plot = Plot(plot_data, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('distance','i1'), color='green')
        plot.index_axis.title = 'distance'
        plot.value_axis.title = 'intensity'
        self.plot_data = plot_data
        #self.plot = plot
        
        #line2=Plot(plot_data, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('distance','i2'), color='red')
        
        #line3=Plot(plot_data, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('distance','i3'), color='blue')
        
        #line4=Plot(plot_data, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('distance','i4'), color='magenta')
        
        #line5=Plot(plot_data, width=50, height=40, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('distance','i5'), color='cyan')
        
        # container=OverlayPlotContainer(plot, line2, line3, line4, line5)
        
        # return container  
        return plot
        
    def _run(self):
        
        file_name ='D:/data/protonNMR/FRAP/bare diamond/21.02.17/0.1 mg per ml 30 min incubation/3' 
        
        print file_name
        
        os.path.exists(file_name)
        
        pg.Night()
        r=101
        
        self.intensity=np.zeros((r))
    
        self.confocal.resolution = r
        self.confocal.seconds_per_point = 0.001
        
        self.confocal.x1=15
        self.confocal.x2=35
        self.confocal.y1=15
        self.confocal.y2=35
        
        self.distance = np.linspace(self.confocal.x1, self.confocal.x2, self.confocal.resolution)        
        self.confocal.slider=-8.34
        
        # pre-bleached image-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        pg.Light()
        self.confocal.submit()                     
        time.sleep(15)
        c=time.strftime('%H:%M:%S')
        time1=float(c[0:2])+float(c[3:5])/60+float(c[6:8])/3600
        self.time1 = np.append(self.time1, time1)
        m=self.confocal.image
        y,x = np.ogrid[-r/2+1:r-r/2, -r/2+1:r-r/2]
        intensity=np.zeros(r/2+1)
                
        for R in range(1, r/2+2):
            mask = (x*x + y*y <= R*R) & (x*x + y*y >= (R-1)*(R-1))
            array = np.zeros((r, r))
            array[mask]=1
            intensity[R-1]=(m*array).sum()/array.sum()
            
        k=intensity[::-1]    
        self.intensity = np.vstack((self.intensity, np.concatenate((k[:-1], intensity), axis=0)))        
        pg.Night()
        
        time.sleep(3)
        #self.confocal.center_cursor()
        file_nv = file_name + '/pre-bleached'
        self.confocal.save_image(file_nv + '.png')
        self.confocal.save(file_nv +'.pyd')
            
        #bleaching-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        pg.Light
        self.confocal.slider=-5     
        pg.Light()
        time.sleep(5)
        
        # tracking the recovery process--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.confocal.slider=-8.34
        
        for i in range(5):
        
            pg.Light()
            self.confocal.submit()                     
            time.sleep(15)
            pg.Night()
            c=time.strftime('%H:%M:%S')
            time1=float(c[0:2])+float(c[3:5])/60+float(c[6:8])/3600
            self.time1 = np.append(self.time1, time1)
            m=self.confocal.image
            intensity=np.zeros(r/2+1)
                
            for R in range(1, r/2+2):
                mask = (x*x + y*y <= R*R) & (x*x + y*y >= (R-1)*(R-1))
                array = np.zeros((r, r))
                array[mask]=1
                intensity[R-1]=(m*array).sum()/array.sum()
                
            k=intensity[::-1]    
            self.intensity = np.vstack((self.intensity, np.concatenate((k[:-1], intensity), axis=0)))          
            
            time.sleep(15)
            
            file_nv = file_name + '/recovery' + str(i) 
            self.confocal.save_image(file_nv + '.png')
            self.confocal.save(file_nv +'.pyd')
            
            
        self.plot_data.set_data('distance', self.distance)
        self.plot_data.set_data('i1', self.intensity[1])
        self.plot_data.set_data('i2', self.intensity[2])
        self.plot_data.set_data('i3', self.intensity[3])
        self.plot_data.set_data('i4', self.intensity[4])
        self.plot_data.set_data('i5', self.intensity[-1])
                                
        print 'finish'
        
    def save_image(self, filename=None):
        self.save_figure(self.plot, filename)
            
    traits_view=View(VGroup ( HGroup (Item('submit_button', show_label=False),
                                     ),
                                                                      
                             HGroup(Item('plot',editor=ComponentEditor(), show_label=False)
                                   ) 
                            ),
                     menubar = MenuBar(Menu(Action(action='save_image', name='Save Image (.png)'),
                                            Action(action='save', name='Save (.pyd or .pys)'),
                                            Action(action='load', name='Load'),
                                            Action(action='_on_close', name='Quit'),
                                            name='File'),),
                     title='FRAP', width=500, height=500, buttons=[], resizable=True, x=0, y=0,
                     handler=GetSetSaveImageHandler
                     )
                     
    get_set_items = ['time1', 'intensity', 'distance', '__doc__']