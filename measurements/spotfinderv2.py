
import numpy
import pickle
from scipy import ndimage
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
class SpotfinderBasicFunc():    
    
    def find_bwimage(self,image,Param,Mode='simple',Median=10):
        if Mode=='simple':
            threshold=Param
            bwimage=numpy.maximum(numpy.round(image-threshold),numpy.zeros(numpy.shape(image)))
        if Mode=='advanced':
            Faktor=Param
            MedianImage=ndimage.median_filter(image,Median)
            threshold=Faktor*numpy.mean(MedianImage)
            image=image-MedianImage
            bwimage=numpy.maximum(numpy.round(image-threshold),numpy.zeros(numpy.shape(image)))
        bwimage=numpy.minimum(bwimage,numpy.ones(numpy.shape(image)))    
        return bwimage
        
    def find_maximage(self,image,Param,Median=10):
        Centroids1 = list()
        data_max = ndimage.filters.maximum_filter(image, Median)
        maxima = (image == data_max)
        data_min = ndimage.filters.minimum_filter(image, Median)
        threshold=Param
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        self.m1=image
        self.m2=data_max
        x, y = [], []
        for dy,dx in slices:
            x_center = (dx.start + dx.stop - 1)/2
            #x.append(x_center)
            y_center = (dy.start + dy.stop - 1)/2    
            #y.append(y_center)
            Centroids1.append([y_center,x_center])
            print(Centroids1)
        return Centroids1
       
    def find_centroids(self,bwimage,MinArea=3,MaxArea=200,ShapeRatio=None,MinInt=None,MaxInt=None,image=None,intweighting=False):
        bwimage2=bwimage
        # create a label frame and sublabel frames
        LabelIndex=0
        LabelIndex_n=0
        Dim=numpy.shape(bwimage)
        LabelImage=numpy.zeros(Dim)
        #LabelImage_n=numpy.zeros(Dim)
        #SubImg=numpy.zeros([2,2])
        #SubImgL=numpy.zeros([2,2])
        SubImg=numpy.zeros([3,3])
        SubImgL=numpy.zeros([3,3])

        # create the label frame for marking individual spots with a label
        for i in range(Dim[0]-3):
            for j in range(Dim[1]-3):

                SubImg[0,0]=bwimage[i,j]
                SubImg[0,1]=bwimage[i,j+1]
                SubImg[0,2]=bwimage[i,j+2]
               
                SubImg[1,0]=bwimage[i+1,j]
                SubImg[1,1]=bwimage[i+1,j+1]
                SubImg[1,2]=bwimage[i+1,j+2]
                
                SubImg[2,0]=bwimage[i+2,j]
                SubImg[2,1]=bwimage[i+2,j+1]
                SubImg[2,2]=bwimage[i+2,j+2]
               
                
                SubImgL[0,0]=LabelImage[i,j]
                SubImgL[0,1]=LabelImage[i,j+1]
                SubImgL[0,2]=LabelImage[i,j+2]
                
                SubImgL[1,0]=LabelImage[i+1,j]
                SubImgL[1,1]=LabelImage[i+1,j+1]
                SubImgL[1,2]=LabelImage[i+1,j+2]
                
                SubImgL[2,0]=LabelImage[i+2,j]
                SubImgL[2,1]=LabelImage[i+2,j+1]
                SubImgL[2,2]=LabelImage[i+2,j+2]
                
        
                if numpy.sum(SubImg)>0:
                    LabelIndexM=numpy.max(SubImgL)
                    if LabelIndexM ==0:
                        LabelIndex+=1
                        Label=LabelIndex
                    else:
                        Label=LabelIndexM
                    #for k,l in [[0,0],[0,1],[1,0],[1,1]]:
                    for k,l in [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]:
                        if int(SubImg[k,l])>0 & int(SubImgL[k,l])==0:
                            SubImgL[k,l]=Label

               
                LabelImage[i,j]=SubImgL[0,0]
                LabelImage[i,j+1]=SubImgL[0,1]
                LabelImage[i,j+2]=SubImgL[0,2]
                LabelImage[i+1,j]=SubImgL[1,0]
                LabelImage[i+1,j+1]=SubImgL[1,1]
                LabelImage[i+1,j+2]=SubImgL[1,2]
                LabelImage[i+2,j]=SubImgL[2,0]
                LabelImage[i+2,j+1]=SubImgL[2,1]
                LabelImage[i+2,j+2]=SubImgL[2,2]
             
                # delete areas that are to small or to big, relabel, and calculate Centroids    
        LabelImage2=LabelImage
        LabelIndex2=1
        Centroids=list()
        if image is not None:
            WeightedImage=bwimage*image
        
        self.matrix1 = LabelImage
        for ID in range(1,LabelIndex+1):
            Coord=list()
            PreC=numpy.argwhere(LabelImage==ID)
            yCoord=0
            xCoord=0
            TargetIntW=0
            if len(PreC) >0:
                for C in PreC:
                    Coord.append(list(C))
                    
                DimC0=list()
                DimC1=list()
                for C in Coord:
                    DimC0.append(C[0])   
                    DimC1.append(C[1])
                DimC0=numpy.asarray(DimC0)
                DimC1=numpy.asarray(DimC1)
                width = numpy.max(DimC0)-numpy.min(DimC0)
                length = numpy.max(DimC1)-numpy.min(DimC1)
                
                # creating the conditions for being a real target
                Condition=width<MinArea or width>MaxArea or length<MinArea or length>MaxArea or abs(width-length) > 3
                
                if image is not None:  
                    TargetInt=0
                    for C in Coord:
                        TargetInt+=(WeightedImage[numpy.round(C[0]),numpy.round(C[1])])
                    TargetInt=TargetInt/len(Coord)
                    if MaxInt is not None:
                        Condition=Condition or TargetInt <= MinInt 
                    if MinInt is not None:
                        Condition=Condition or TargetInt >= MaxInt
                    
                    
                if ShapeRatio is not None: 
                    # to avoid errors by zero devision the target the min max distriction is used
                    DimC0= float(numpy.maximum(numpy.minimum(float(numpy.max(DimC0)-numpy.min(DimC0)),100),1./100))
                    DimC1= float(numpy.maximum(numpy.minimum(float(numpy.max(DimC1)-numpy.min(DimC1)),100),1./100))
                    TargetShape=DimC0/DimC1
                    Condition = Condition or TargetShape <= 1/ShapeRatio or TargetShape >= ShapeRatio
                    
                #apply condition on spot    
                if Condition:
                    for C in Coord:
                        bwimage2[C[0],C[1]]=0
                        LabelImage2[C[0],C[1]]=0
                else:
                    for C in Coord:
                        bwimage2[C[0],C[1]]=1
                        LabelImage2[C[0],C[1]]=LabelIndex2
                        if intweighting is False:
                            yCoord+=C[0]
                            xCoord+=C[1]
                        else:
                            yCoord+=C[0]*image[numpy.round(C[0]),numpy.round(C[1])]
                            xCoord+=C[1]*image[numpy.round(C[0]),numpy.round(C[1])]
                            TargetIntW+=(image[numpy.round(C[0]),numpy.round(C[1])])
                    if intweighting is False:
                        yCoord=float(yCoord)/len(Coord)
                        xCoord=float(xCoord)/len(Coord)
                    else:
                        yCoord=float(yCoord)/TargetIntW
                        xCoord=float(xCoord)/TargetIntW
                        
                    LabelIndex2+=1
                    Centroids.append([yCoord,xCoord]) 
        
        return bwimage2,Centroids
        


from tools.chaco_addons import SavePlot as Plot, SaveHPlotContainer as HPlotContainer, SaveTool, AspectZoomTool
from traits.api import HasTraits, Trait, Instance,Any, Property, Float, Range,Int,\
                       Bool, Array, String, Str, Enum, Button, on_trait_change, cached_property, DelegatesTo
from traitsui.api import View, Item, Group, HGroup, VGroup,VSplit, Tabbed, EnumEditor, TextEditor, Action, Menu, MenuBar,InstanceEditor

from enable.api import ComponentEditor, Component
from chaco.api import CMapImagePlot, ArrayPlotData, DataRange1D,\
                      Spectral, ColorBar, LinearMapper, DataLabel, PlotLabel,gray 
import chaco.api

from tools.emod import ManagedJob
from tools.utility import GetSetItemsMixin

# Spotfinder is the gui for finding spots
class Spotfinder(ManagedJob, GetSetItemsMixin, SpotfinderBasicFunc):    
    
    X = Array()
    Y = Array()
    image = Array()
    image2 = Array()
    bwimage= Array()
    resolution=[[0.1],[0.1]]
    refpoint=[[0],[0]]
    
    # plots
    plot_data           = Instance( ArrayPlotData )
    scan_plot           = Instance( CMapImagePlot )
    figure              = Instance( Plot )
    bwplot_data           = Instance( ArrayPlotData )
    bwscan_plot           = Instance( CMapImagePlot )
    bwfigure              = Instance( Plot )
    
    figure_container    = Instance( HPlotContainer, editor=ComponentEditor() )
    confocal = Any(editor=InstanceEditor)
   
    ImportImage    = Button(label='import image')
    GetBWimage     = Button(label='get bwimage')
    ProcessBWimage     = Button(label='process bwimage',desc='clean image (minarea maxarea) and calculate centroids')
    MinArea=Int(value=10,label='minarea')
    MaxArea=Int(value=10,label='maxarea')
    BWimageMeth    = Enum('simple','advanced')
    BWimageMedian   =Int(value=10,label='median size')
    ImageThreshold = Float(value=50000,desc='This value sets the threshold for the bw image',label='Threshold')
    BWimageMedian1   =Int(value=10,label='median size')
    ImageThreshold1 = Float(value=50000,desc='This value sets the threshold for the bw image',label='Threshold')
    SpotMaxInt = Float(value=5000000,desc='This value sets the threshold for maximum allowed mean spot intensity',label='max int')
    SpotMinInt = Float(value=0,desc='This value sets the threshold for minimum allowed mean spot intensity',label='min int')
    SpotShapeRatio=Float(value=2,desc='This value sets the threshold for minimum allowed mean spot intensity',label='shape ratio')
    ExportTag=Str('nv')
    ExportButton=Button('Export to auto_focus')
    Centroids=list()
    Centroids1=list()
    IntWeighting=Bool(True)
    labels = dict()
    scaler=[[1],[1]]
    
    
    def __init__(self,confocal,auto_focus,**kwargs):
        super(Spotfinder, self).__init__(**kwargs)
        self.confocal=confocal
        self.auto_focus=auto_focus
        self.X = numpy.linspace(self.confocal.scanner.getXRange()[0], self.confocal.scanner.getXRange()[-1], self.confocal.resolution+1)
        self.Y = numpy.linspace(self.confocal.scanner.getYRange()[0], self.confocal.scanner.getYRange()[-1], self.confocal.resolution+1)
        self.image = numpy.zeros((len(self.X), len(self.Y)))
        self.bwimage = numpy.zeros((len(self.X), len(self.Y)))
        
        self._create_plot()
        self.on_trait_change(self.update_image_plot, 'image', dispatch='ui')
        
        self.on_trait_change(self.redraw_image, 'confocal.thresh_high', dispatch='ui')
        self.on_trait_change(self.redraw_image, 'confocal.thresh_low', dispatch='ui')
        self.on_trait_change(self.redraw_image, 'confocal.thresh_high', dispatch='ui')
        self.on_trait_change(self.redraw_image, 'confocal.thresh_low', dispatch='ui')
        self.on_trait_change(self.set_mesh_and_aspect_ratio, 'X,Y', dispatch='ui')
        
        #startup values
        self.BWimageMedian=10
        self.MinArea=3
        self.MaxArea=100
        
    def _ExportButton_fired(self):
        
        if len(self.labels) is not 0:
            for label in self.labels.keys():
                self.auto_focus.target_name=self.ExportTag+label
                self.confocal.x=self.labels[label][1]
                self.confocal.y=self.labels[label][0]
                #self.confocal.z=self.labels[label][2]
                self.auto_focus._add_target_button_fired()
        else:
            print 'no tartgets to export'
            
            
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
        if self.BWimageMeth == 'simple':
            self.ImageThreshold=numpy.median(self.image)*1.5
        if self.BWimageMeth == 'advanced':   
            self.ImageThreshold=1.5
            
        self.SpotMaxInt=numpy.max(self.image)
        self.SpotMinInt=numpy.min(self.image)
        self.SpotShapeRatio=2
            
    def _GetBWimage_fired(self):   
        self.bwimage=self.find_bwimage(self.image, Param=self.ImageThreshold, Mode=self.BWimageMeth, Median=self.BWimageMedian)
        self.bwplot_data.set_data('bwimage', self.bwimage)
        self.remove_all_targets()
    def _ProcessBWimage_fired(self):
        self._GetBWimage_fired()
        [self.bwimage,self.Centroids]=self.find_centroids(self.bwimage,MinArea=self.MinArea,MaxArea=self.MaxArea,MinInt=self.SpotMinInt,MaxInt=self.SpotMaxInt,ShapeRatio=self.SpotShapeRatio,image=self.image,intweighting=self.IntWeighting)
        self.image_handle = self.bwimage + numpy.min(self.image)
        self.bwplot_data.set_data('bwimage', self.image_handle)
        self.Centroids1=self.find_maximage(self.image,Param=self.ImageThreshold1, Median=self.BWimageMedian1)
        for ID,coords in enumerate(self.Centroids1):
            self.add_target(str(ID), coords)
      
    def redraw_image(self):
        self.scan_plot.value_range.high_setting=self.confocal.thresh_high
        self.scan_plot.value_range.low_setting=self.confocal.thresh_low
        self.scan_plot.request_redraw()
         
    def _create_plot(self):
        plot_data = ArrayPlotData(image=self.image)
        plot = Plot(plot_data, width=500, height=500, resizable='hv', aspect_ratio=1.0, padding=8, padding_left=32, padding_bottom=32)
        plot.img_plot('image',  colormap=gray, xbounds=(self.X[0],self.X[-1]), ybounds=(self.Y[0],self.Y[-1]), name='image')
        image = plot.plots['image'][0]
        bwplot_data = ArrayPlotData(bwimage=self.bwimage)
        bwplot = Plot(bwplot_data, width=500, height=500, resizable='hv', aspect_ratio=1.0, padding=8, padding_left=32, padding_bottom=32)
        bwplot.img_plot('bwimage',  colormap=gray, xbounds=(self.X[0],self.X[-1]), ybounds=(self.Y[0],self.Y[-1]), name='bwimage')
        bwimage = bwplot.plots['bwimage'][0]
       
        image.x_mapper.domain_limits = (self.confocal.scanner.getXRange()[0],self.confocal.scanner.getXRange()[1])
        image.y_mapper.domain_limits = (self.confocal.scanner.getYRange()[0],self.confocal.scanner.getYRange()[1])
        bwimage.x_mapper.domain_limits = (self.confocal.scanner.getXRange()[0],self.confocal.scanner.getXRange()[1])
        bwimage.y_mapper.domain_limits = (self.confocal.scanner.getYRange()[0],self.confocal.scanner.getYRange()[1])
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
        container.add(bwplot)
        
        container.tools.append(SaveTool(container))

        self.plot_data = plot_data
        self.scan_plot = image
        self.figure = plot
        self.bwplot_data = bwplot_data
        self.bwscan_plot = bwimage
        self.bwfigure = bwplot
        
        self.figure_container = container
    
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
   
    def set_mesh_and_aspect_ratio(self):
        self.scan_plot.index.set_data(self.X,self.Y)
        self.bwscan_plot.index.set_data(self.X,self.Y)
       
        x1=self.X[0]
        x2=self.X[-1]
        y1=self.Y[0]
        y2=self.Y[-1]
    
        self.figure.aspect_ratio = (x2-x1) / float((y2-y1))
        self.figure.index_range.low = x1
        self.figure.index_range.high = x2
        self.figure.value_range.low = y1
        self.figure.value_range.high = y2
        
        self.bwfigure.aspect_ratio = (x2-x1) / float((y2-y1))
        self.bwfigure.index_range.low = x1
        self.bwfigure.index_range.high = x2
        self.bwfigure.value_range.low = y1
        self.bwfigure.value_range.high = y2
        
    def add_target(self, key, coordinates=None):
        Info=self.confocal.figure.index_range.get()
        coordinates2=[[],[]]
        x1=Info['_low_setting']
        Info=self.confocal.figure.value_range.get()
        y1=Info['_low_setting']
       
        coordinates[0]=coordinates[0]*self.resolution[0]+y1
        coordinates[1]=coordinates[1]*self.resolution[1]+x1
        
        
        plot = self.scan_plot

        point = (coordinates[1],coordinates[0])

        defaults = {'component':plot,
                    'data_point':point,
                    'label_format':key,
                    'label_position':'top right',
                    'bgcolor':'transparent',
                    'text_color':'red',
                    'border_visible':False,
                    'padding_bottom':8,
                    'marker':'cross',
                    'marker_color':'red',
                    'marker_line_color':'red',
                    'marker_line_width':1.4,
                    'marker_size':6,
                    'arrow_visible':False,
                    'clip_to_plot':False,
                    'visible':True}

        
        self.labels[key] = coordinates
        label = DataLabel(**defaults)

        plot.overlays.append(label)
       
        plot.request_redraw()
        self.labels[key] = coordinates
        
    def remove_all_targets(self):
         
        plot = self.scan_plot
        new_overlays = []
        for item in plot.overlays:
            if not ( isinstance(item, DataLabel) and item.label_format in self.labels ) :
                 new_overlays.append(item)
        plot.overlays = new_overlays
        plot.request_redraw()
        self.labels.clear()
        
    IAN=View(Item('ImportImage', show_label=False, resizable=True)) 
    traits_view = View(VSplit(HGroup(Item('ImportImage', show_label=False, resizable=True),
                                     Item('GetBWimage', show_label=False, resizable=True),
                                     Item('BWimageMeth', show_label=True, resizable=True),
                                     Item('ImageThreshold', show_label=True, resizable=True ,width=-30),
                                     Item('BWimageMedian', show_label=True, resizable=True,width=-30),
                                     Item('ImageThreshold1', show_label=True, resizable=True,width=-30),
                                     Item('BWimageMedian1', show_label=True, resizable=True,width=-30)
                                     
                                     ),
                              HGroup(Item('ProcessBWimage', show_label=False, resizable=True),
                                     Item('MinArea', show_label=True, resizable=True),
                                     Item('MaxArea', show_label=True, resizable=True),
                                     Item('SpotMinInt', show_label=True, resizable=True),
                                     Item('SpotMaxInt', show_label=True, resizable=True),
                                     Item('SpotShapeRatio', show_label=True, resizable=True),
                                     Item('IntWeighting')
                                     ),
                              HGroup(Item('ExportButton', show_label=False, resizable=True),
                              Item('ExportTag', show_label=False, resizable=True)
                              ),
                              Item('figure_container', show_label=False, resizable=True,height=600)
                       ),
                       title='SpotFinder', width=1080, height=800, buttons=[], resizable=True, x=0, y=0
                       
                       )
