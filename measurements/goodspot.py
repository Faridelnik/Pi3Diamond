
import numpy
import pickle
from scipy import ndimage, optimize
from chaco.api import jet
#f=open('test.pys','rb')
#bwimage=pickle.load(f)
#f.close()    
#threshold=50000

#bwimage=[[0,0,0,0,1,0,0],
#         [1,0,0,1,1,1,0],
#         [1,1,0,0,1,0,0],
#         [1,0,0,0,0,0,0]]
#bwimage=bwimage[151:200,150:200]

#Centroids=find_centroids(2,bwimage) 

#SpotfinderBasicFunc provides the basic functions for finding targets based on thresholding 
class GoodspotBasic():    
    
    def twoD_Gaussian(self, (x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        x, y = numpy.meshgrid(x,y)
        xo = float(xo)
        yo = float(yo)    
        a = (numpy.cos(theta)**2)/(2*sigma_x**2) + (numpy.sin(theta)**2)/(2*sigma_y**2)
        b = -(numpy.sin(2*theta))/(4*sigma_x**2) + (numpy.sin(2*theta))/(4*sigma_y**2)
        c = (numpy.sin(theta)**2)/(2*sigma_x**2) + (numpy.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*numpy.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
        return g.ravel()
    
    def find_bwimage(self,image,Param):
        threshold=Param
        bwimage=numpy.maximum(numpy.round(image-threshold),numpy.zeros(numpy.shape(image)))

        return bwimage
    
from tools.chaco_addons import SavePlot as Plot, SaveHPlotContainer as HPlotContainer, SaveTool, AspectZoomTool
from traits.api import HasTraits, Trait, Instance,Any, Property, Float, Range,Int,\
                       Bool, Array, String, Str, Enum, Button, on_trait_change, cached_property, DelegatesTo
from traitsui.api import View, Item, Group, HGroup, VGroup,VSplit, Tabbed, EnumEditor, TextEditor, Action, Menu, MenuBar,InstanceEditor

from enable.api import ComponentEditor, Component
from chaco.api import CMapImagePlot, ArrayPlotData, DataRange1D,\
                      Spectral, ColorBar, LinearMapper, DataLabel, PlotLabel,gray 
import chaco.api

from tools.emod import ManagedJob

from traitsui.file_dialog import save_file
from tools.utility import GetSetItemsHandler, GetSetItemsMixin,GetSetSaveImageHandler

# Spotfinder is the gui for finding spots
class Goodspot(ManagedJob, GetSetItemsMixin, GoodspotBasic):    
    
    X = Array()
    Y = Array()
    Xfit = Array()
    Yfit = Array()
    image = Array()
    image_forfit = Array()
    image2 = Array()
    fitimage= Array()
    label_text = Str('')
    
    resolution=[[0.1],[0.1]]
    refpoint=[[0],[0]]
    
    # plots
    plot_data           = Instance( ArrayPlotData )
    scan_plot           = Instance( CMapImagePlot )
    figure              = Instance( Plot )
    fitplot_data           = Instance( ArrayPlotData )
    fitscan_plot           = Instance( CMapImagePlot )
    fitfigure              = Instance( Plot )
    
    figure_container    = Instance( HPlotContainer, editor=ComponentEditor() )
    confocal = Any(editor=InstanceEditor)
   
    ImportImage    = Button(label='import image')
    Getfitimage     = Button(label='get fitimage')
    ExportTag=Str('nv')
    ExportButton=Button('Export to auto_focus')
    Centroids=list()
    Centroids1=list()
    IntWeighting=Bool(True)
    off_diagonal_account = Bool(True)
    labels = dict()
    scaler=[[1],[1]]
    
    
    def __init__(self,confocal,auto_focus,**kwargs):
        super(Goodspot, self).__init__(**kwargs)
        self.confocal=confocal
        self.auto_focus=auto_focus
        self.X = numpy.linspace(self.confocal.scanner.getXRange()[0], self.confocal.scanner.getXRange()[-1], self.confocal.resolution+1)
        self.Y = numpy.linspace(self.confocal.scanner.getYRange()[0], self.confocal.scanner.getYRange()[-1], self.confocal.resolution+1)
        self.image = numpy.zeros((len(self.X), len(self.Y)))
        dim = min(self.image.shape)
        self.fitimage = numpy.zeros((dim, dim))
        self.Xfit = self.X[0:dim]
        self.Yfit = self.Y[0:dim]
        self._create_plot()
        self.on_trait_change(self.update_image_plot, 'image', dispatch='ui')
        self.on_trait_change(self._on_label_text_change, 'label_text', dispatch='ui')
        
        self.on_trait_change(self.redraw_image, 'confocal.thresh_high', dispatch='ui')
        self.on_trait_change(self.redraw_image, 'confocal.thresh_low', dispatch='ui')
        self.on_trait_change(self.redraw_image, 'confocal.thresh_high', dispatch='ui')
        self.on_trait_change(self.redraw_image, 'confocal.thresh_low', dispatch='ui')
        self.on_trait_change(self.set_mesh_and_aspect_ratio, 'X,Y,Xfit,Yfit', dispatch='ui')
        
       
            
    def _ImportImage_fired(self):
        # the long road for extracting the zoom parameters ...
        Dim=numpy.shape(self.confocal.image)
        ImagePosX=[self.confocal.X[0],self.confocal.X[-1]]
        ImagePosY=[self.confocal.Y[0],self.confocal.Y[-1]]
        ImageRangeX=self.confocal.figure.index_range.get()
        ImageRangeX=[ImageRangeX['_low_setting'],ImageRangeX['_high_setting']]
        ImageRangeY=self.confocal.figure.value_range.get()
        ImageRangeY=[ImageRangeY['_low_setting'],ImageRangeY['_high_setting']]
        
        FullRangeX=self.confocal.scanner.getXRange()
        FullRangeY=self.confocal.scanner.getYRange()
        resolution=[(ImagePosY[1]-ImagePosY[0])/Dim[0],(ImagePosX[1]-ImagePosX[0])/Dim[1]]

        RangeX=numpy.round(numpy.asarray([ImageRangeX[0]-ImagePosX[0],ImageRangeX[1]-ImagePosX[0]])/resolution[1])
        RangeY=numpy.round(numpy.asarray([ImageRangeY[0]-ImagePosY[0],ImageRangeY[1]-ImagePosY[0]])/resolution[0])
        
        self.scaler[0]=FullRangeX[1]/resolution[1]
        
        
        self.scaler[1]=FullRangeY[1]/resolution[0]
        self.resolution = resolution
        self.refpoint=[ImagePosY[0],ImagePosX[0]]
           
        # import only the part of the image of the zoom
        self.image=self.confocal.image[RangeY[0]:RangeY[1]+1,RangeX[0]:RangeX[1]+1]
        #self.image2=self.confocal.image
        self.update_mesh()
        self.redraw_image()
            
    def _Getfitimage_fired(self): 
        offset = 20.0
        amplitude = 70.0
        y0=numpy.mean(self.Yfit)
        x0=numpy.mean(self.Xfit)
        sigma_y = 0.2
        sigma_x = 0.2
        theta = 0.0
        initial_guess = (amplitude,y0,x0,sigma_y,sigma_x,theta,offset)
        popt, pcov = optimize.curve_fit(self.twoD_Gaussian, (self.Yfit, self.Xfit), self.image_forfit.ravel(), initial_guess)
        fit_data = self.twoD_Gaussian((self.Yfit,self.Xfit), *popt)
        self.fitparemeter = popt
        #print(fit_data)
        self.fitimage = fit_data.reshape(len(self.Yfit),len(self.Xfit))
        self.fitplot_data.set_data('fitimage', self.fitimage)
        
        s = 'amp: %.1f Kcounts\n' % popt[0]
        s += 'sigmay: %.3f and sigmax: %.3f micrometer\n' % (popt[3], popt[4])
        s += 'theta: %.1f degree\n' % float(popt[5]*180/3.1415)
        self.label_text = s


    def redraw_image(self):
        self.scan_plot.value_range.high_setting=self.confocal.thresh_high
        self.scan_plot.value_range.low_setting=self.confocal.thresh_low
        self.scan_plot.request_redraw()
         
    def _create_plot(self):
        plot_data = ArrayPlotData(image=self.image)
        plot = Plot(plot_data, width=500, height=500, resizable='hv', aspect_ratio=1.0, padding=8, padding_left=32, padding_bottom=32)
        plot.img_plot('image',  colormap=jet, xbounds=(self.X[0],self.X[-1]), ybounds=(self.Y[0],self.Y[-1]), name='image')
        image = plot.plots['image'][0]
        fitplot_data = ArrayPlotData(fitimage=self.fitimage)
        fitplot = Plot(fitplot_data, width=500, height=500, resizable='hv', aspect_ratio=1.0, padding=8, padding_left=32, padding_bottom=32)
        fitplot.img_plot('fitimage',  colormap=jet, xbounds=(self.Xfit[0],self.Xfit[-1]), ybounds=(self.Yfit[0],self.Yfit[-1]), name='fitimage')
        fitplot.overlays.insert(0, PlotLabel(text=self.label_text, hjustify='right', vjustify='bottom', position=[880, 590]))
        fitimage = fitplot.plots['fitimage'][0]
       
        image.x_mapper.domain_limits = (self.confocal.scanner.getXRange()[0],self.confocal.scanner.getXRange()[1])
        image.y_mapper.domain_limits = (self.confocal.scanner.getYRange()[0],self.confocal.scanner.getYRange()[1])
        fitimage.x_mapper.domain_limits = (self.confocal.scanner.getXRange()[0],self.confocal.scanner.getXRange()[1])
        fitimage.y_mapper.domain_limits = (self.confocal.scanner.getYRange()[0],self.confocal.scanner.getYRange()[1])
        colormap = image.color_mapper
        colorbar = ColorBar(index_mapper=LinearMapper(range=colormap.range),
                            color_mapper=colormap,
                            plot=plot,
                            orientation='v',
                            resizable='v',
                            width=16,
                            height=320,
                            padding=8,
                            padding_left=32)
        container = HPlotContainer()
        container.add(plot)
        container.add(colorbar)
        container.add(fitplot)
        
        container.tools.append(SaveTool(container))

        self.plot_data = plot_data
        self.scan_plot = image
        self.figure = plot
        self.fitplot_data = fitplot_data
        self.fitscan_plot = fitimage
        self.fitfigure = fitplot
        
        self.figure_container = container
        
    def _on_label_text_change(self):
        self.fitfigure.overlays[0].text = self.label_text        
    
    def update_image_plot(self):
        self.plot_data.set_data('image', self.image)      
    
    def update_mesh(self):
        
        Info=self.confocal.figure.index_range.get()
        
        x1=Info['_low_setting']
        x2=Info['_high_setting']
        
        Info=self.confocal.figure.value_range.get()
        y1=Info['_low_setting']
        y2=Info['_high_setting']
        Dim=numpy.shape(self.image)
        self.Y = numpy.linspace(y1,y2,Dim[0])
        self.X = numpy.linspace(x1,x2,Dim[1])
        if Dim[0]>Dim[1]:
            self.Yfit = self.Y[0:Dim[1]]
            self.Xfit = numpy.linspace(x1,x2,Dim[1])
            self.image_forfit = self.image[:,0:Dim[1]]
        else:
            self.Xfit = self.X[0:Dim[0]]
            self.Yfit = numpy.linspace(y1,y2,Dim[0])
            self.image_forfit = self.image[0:Dim[0],:]
       
        
    def set_mesh_and_aspect_ratio(self):
        self.scan_plot.index.set_data(self.X,self.Y)
        self.fitscan_plot.index.set_data(self.Xfit,self.Yfit)
       
        x1=self.X[0]
        x2=self.X[-1]
        y1=self.Y[0]
        y2=self.Y[-1]
        x1fit=self.Xfit[0]
        x2fit=self.Xfit[-1]
        y1fit=self.Yfit[0]
        y2fit=self.Yfit[-1]
    
        self.figure.aspect_ratio = (x2-x1) / float((y2-y1))
        self.figure.index_range.low = x1
        self.figure.index_range.high = x2
        self.figure.value_range.low = y1
        self.figure.value_range.high = y2
        
        self.fitfigure.aspect_ratio = 1
        self.fitfigure.index_range.low = x1fit
        self.fitfigure.index_range.high = x2fit
        self.fitfigure.value_range.low = y1fit
        self.fitfigure.value_range.high = y2fit    

        
    def save_image(self, filename=None):
        self.save_figure(self.figure_container, filename)
        
    IAN=View(Item('ImportImage', show_label=False, resizable=True)) 
    traits_view = View(VSplit(HGroup(Item('ImportImage', show_label=False, resizable=True),
                                     Item('Getfitimage', show_label=False, resizable=True)                        
                                     ),
                              Item('figure_container', show_label=False, resizable=True,height=600)),
                              menubar = MenuBar(Menu(Action(action='save_image', name='Save Image (.png)'),
                              Action(action='saveMatrixPlot', name='SaveMatrixPlot (.png)'),
                              name='File')),
                       title='Goodspot', width=1080, height=800, buttons=[], resizable=True, x=0, y=0, handler=GetSetSaveImageHandler
                       
                       )
