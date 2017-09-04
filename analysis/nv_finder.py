import numpy
from scipy import ndimage

# Enthought library imports
from traits.api import HasTraits, Instance, Property, Range, Float, Array,\
                                 Tuple, Button, on_trait_change, cached_property
from traitsui.api import View, Item, HGroup, Tabbed
from enable.api import ComponentEditor, Component

from chaco.api import Plot, ArrayPlotData, ColorBar, LinearMapper, gray, HPlotContainer
from chaco.tools.api import ZoomTool

from tools.emod import ManagedJob
import logging

class NVFinder( ManagedJob, HasTraits ):

    submit_button = Button(label='correct targets', desc='Performs a refocus for all targets in auto focus (without recording the drift).')
    remove_button = Button(label='abort', desc='Stop the running refocus measurement.')

    Sigma = Range(low=0.01, high=10., value=0.1, desc='Sigma of Gaussian for smoothing [micron]', label='sigma [micron]', mode='slider', auto_set = False, enter_set = True)
    Threshold = Range(low=0, high=1000000, value=50000, desc='Threshold [counts/s]', label='threshold [counts/s]', auto_set = False, enter_set = True)
    Margin = Range(low=0.0, high=100., value=4., desc='Margin [micron]', label='margin [micron]', auto_set = False, enter_set = True)

    RawPlot = Instance( Component )
    SmoothPlot = Instance( Component )
    RegionsPlot = Instance( Component )

    X = Array(dtype=numpy.float)
    Y = Array(dtype=numpy.float)
    z = Float()
    Raw = Array(dtype=numpy.float)
    Smooth = Property( trait=Array(), depends_on='Raw,Sigma' )
    Thresh = Property( trait=Array(), depends_on='Raw,Smooth,Sigma,Threshold' )
    RegionsAndLabels = Property( trait=Tuple(Array(),Array()), depends_on='Raw,Smooth,Thresh,Sigma,Threshold' )
    Positions = Property( trait=Array(), depends_on='Raw,X,Y,Sigma,Threshold' )
    XPositions = Property( trait=Array() )
    YPositions = Property( trait=Array() )

    ImportData = Button()
    ExportTargets = Button()

    traits_view = View( HGroup(Item('submit_button',   show_label=False),
                               Item('remove_button',   show_label=False),
                               Item('priority', enabled_when='state != "run"'),
                               Item('state', style='readonly'),
                               ),
                 Tabbed( Item('RawPlot', editor=ComponentEditor(), show_label=False, resizable=True),
                         Item('SmoothPlot', editor=ComponentEditor(), show_label=False, resizable=True),
                         Item('RegionsPlot', editor=ComponentEditor(), show_label=False, resizable=True) ),
                 HGroup( Item('ImportData', show_label=False), Item('ExportTargets', show_label=False), Item('Margin') ),
                 Item('Sigma'),
                 Item('Threshold'),
                 title='NV Finder', width=800, height=700, buttons=['OK'], resizable=True)                 
    
    def __init__(self, confocal, auto_focus):
        super(NVFinder, self).__init__()
        self.confocal=confocal
        self.auto_focus=auto_focus
    
    def _run(self):
        
        try: # refocus all targets
            self.state='run'
            
            af = self.auto_focus
            confocal = af.confocal
            
            af.periodic_focus=False
            af.forget_drift()
            af.current_target=None
            for target in af.targets.iterkeys():
                if threading.current_thread().stop_request.isSet():
                    break
                coordinates = af.targets[target]
                confocal.x, confocal.y, confocal.z = coordinates
                corrected_coordinates = af.focus()
                af.targets[target] = corrected_coordinates
                af.trait_property_changed('targets', af.targets)
                logging.getLogger().debug('NV finder: auto focus target '+str(target)+': %.2f, %.2f, %.2f'%tuple(corrected_coordinates))
            self.state='idle'                        
    
        except: # if anything fails, recover
            logging.getLogger().exception('Error in NV finder.')
            self.state='error'
    
    
    @cached_property
    def _get_RegionsAndLabels(self):
        s = [[1,1,1], [1,1,1], [1,1,1]]
        regions, labels = ndimage.label(self.Thresh, s)
        return regions, labels

    @cached_property
    def _get_Positions(self):
        positions = []
        for i in range(1,self.RegionsAndLabels[1]+1):
            y, x = ndimage.center_of_mass(  (self.RegionsAndLabels[0] == i).astype(int)  )
            if y < 0:
                y = 0
            if y >= len(self.Y):
                y = len(self.Y) - 1
            if x < 0:
                x = 0
            if x >= len(self.X):
                x = len(self.X) - 1
            positions.append((self.Y[int(y)],self.X[int(x)]))
        return numpy.array(positions)
        
    def _get_XPositions(self):
        if len(self.Positions) == 0:
            return numpy.array(())
        else:
            return self.Positions[:,1]
        
    def _get_YPositions(self):
        if len(self.Positions) == 0:
            return numpy.array(())
        else:
            return self.Positions[:,0]
        
    @cached_property
    def _get_Thresh(self):
        return (self.Smooth > self.Threshold).astype(int)

    @cached_property
    def _get_Smooth(self):
        return ndimage.gaussian_filter(self.Raw, int(self.Sigma/(self.X[1]-self.X[0])))

    def _Raw_default(self):
        return self.confocal.image
        #return numpy.asarray(Image.open('original.png'))[:,:,0]
 
    def _X_default(self):
        return self.confocal.X
        #return numpy.arange(self.Raw.shape[1])/100.
 
    def _Y_default(self):
        return self.confocal.Y
        #return numpy.arange(self.Raw.shape[0])/100.

    def _z_default(self):
        return self.confocal.z
        #return 0.0
 
    def _RawPlot_default(self):
    
        plotdata = ArrayPlotData(imagedata=self.Raw, x=self.XPositions, y=self.YPositions)
        plot = Plot(plotdata, width=500, height=500, resizable='hv')
        RawImage = plot.img_plot('imagedata',  colormap=gray, xbounds=(self.X[0],self.X[-1]), ybounds=(self.Y[0],self.Y[-1]),)[0]                                   
        RawImage.x_mapper.domain_limits = (self.X[0],self.X[-1])
        RawImage.y_mapper.domain_limits = (self.Y[0],self.Y[-1])
        RawImage.overlays.append(ZoomTool(RawImage))
        scatterplot = plot.plot( ('x', 'y'), type='scatter', marker='plus', color='yellow')
        colormap = RawImage.color_mapper
        colorbar = ColorBar(index_mapper=LinearMapper(range=colormap.range),
                            color_mapper=colormap,
                            plot=plot,
                            orientation='v',
                            resizable='v',
                            width=10,
                           padding=20)
        colorbar.padding_top = plot.padding_top
        colorbar.padding_bottom = plot.padding_bottom

        self.RawData = plotdata
        self.RawImage = RawImage

        container = HPlotContainer(padding=20, fill_padding=True, use_backbuffer=True)
        container.add(colorbar)
        container.add(plot)

        return container

    def _SmoothPlot_default(self):
    
        plotdata = ArrayPlotData(imagedata=self.Smooth)
        plot = Plot(plotdata, width=500, height=500, resizable='hv')
        SmoothImage = plot.img_plot('imagedata',  colormap=gray, xbounds=(self.X[0],self.X[-1]), ybounds=(self.Y[0],self.Y[-1]),)[0]
        SmoothImage.x_mapper.domain_limits = (self.X[0],self.X[-1])
        SmoothImage.y_mapper.domain_limits = (self.Y[0],self.Y[-1])
        SmoothImage.overlays.append(ZoomTool(SmoothImage))
                                   
        colormap = SmoothImage.color_mapper
        colorbar = ColorBar(index_mapper=LinearMapper(range=colormap.range),
                            color_mapper=colormap,
                            plot=plot,
                            orientation='v',
                            resizable='v',
                            width=10,
                           padding=20)
        colorbar.padding_top = plot.padding_top
        colorbar.padding_bottom = plot.padding_bottom

        self.SmoothImage = SmoothImage

        container = HPlotContainer(padding=20, fill_padding=True, use_backbuffer=True)
        container.add(colorbar)
        container.add(plot)

        return container
                
    def _RegionsPlot_default(self):
    
        plotdata = ArrayPlotData(imagedata=self.RegionsAndLabels[0], x=self.XPositions, y=self.YPositions)
        plot = Plot(plotdata, width=500, height=500, resizable='hv')
        RegionsImage = plot.img_plot('imagedata',  colormap=gray, xbounds=(self.X[0],self.X[-1]), ybounds=(self.Y[0],self.Y[-1]),)[0]
        RegionsImage.x_mapper.domain_limits = (self.X[0],self.X[-1])
        RegionsImage.y_mapper.domain_limits = (self.Y[0],self.Y[-1])
        RegionsImage.overlays.append(ZoomTool(RegionsImage))
                                   
        scatterplot = plot.plot( ('x', 'y'), type='scatter', marker='plus', color='yellow')
                                   
        colormap = RegionsImage.color_mapper
        colorbar = ColorBar(index_mapper=LinearMapper(range=colormap.range),
                            color_mapper=colormap,
                            plot=plot,
                            orientation='v',
                            resizable='v',
                            width=10,
                           padding=20)
        colorbar.padding_top = plot.padding_top
        colorbar.padding_bottom = plot.padding_bottom

        self.RegionsData = plotdata
        self.RegionsImage = RegionsImage

        container = HPlotContainer(padding=20, fill_padding=True, use_backbuffer=True)
        container.add(colorbar)
        container.add(plot)

        return container
        
    def _Sigma_changed(self):
        self._get_Positions()

    def _Threshold_changed(self):
        self._get_Positions()

    # automatic update of plots

    def _Smooth_changed(self):
        self.SmoothImage.value.set_data(self.Smooth)

    def _RegionsAndLabels_changed(self):
        self.RegionsData.set_data('imagedata',self.RegionsAndLabels[0])

    def _Positions_changed(self):
        self.RegionsData.set_data('x',self.XPositions)
        self.RegionsData.set_data('y',self.YPositions)
        self.RawData.set_data('x',self.XPositions)
        self.RawData.set_data('y',self.YPositions)
        
    @on_trait_change( 'ImportData' )
    def Import(self):
        self.Raw = self._Raw_default()
        self.X = self._X_default()
        self.Y = self._Y_default()
        self.z = self._z_default()
        self.RawImage.index.set_data(self.X, self.Y)
        self.SmoothImage.index.set_data(self.X, self.Y)
        self.RegionsImage.index.set_data(self.X, self.Y)
        self.RawData.set_data('imagedata', self.Raw)
        self._get_Positions()

    @on_trait_change( 'ExportTargets' )
    def Export(self):
        z = self.z
        for i, pos in enumerate(self.Positions):
            y, x = pos
            if x > self.X[0] + self.Margin and x < self.X[-1] - self.Margin and y > self.Y[0] + self.Margin and y < self.Y[-1] - self.Margin:
                self.auto_focus.add_target( str(i), np.array((x, y, z)) )

if __name__ == '__main__':
    
    nv_finder=NVFinder(confocal,auto_focus)
    nv_finder.edit_traits()
    
    