import time
import os
import cPickle

# Enthought library imports
from traits.api import Float, HasPrivateTraits, Str, Tuple
from traitsui.api import Handler, View, Item, OKButton, CancelButton
from traitsui.file_dialog import open_file, save_file

from chaco.api import PlotGraphicsContext
from chaco.tools.simple_zoom import SimpleZoom 

import logging

from data_toolbox import writeDictToFile

import threading

def timestamp():
    """Returns the current time as a human readable string."""
    return time.strftime('%y-%m-%d_%Hh%Mm%S', time.localtime())

class Singleton(type):
    """
    Singleton using metaclass.
    
    Usage:
    
    class Myclass( MyBaseClass )
        __metaclass__ = Singleton
    
    Taken from stackoverflow.com.
    http://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

#class Singleton(object):
#    """
#    Singleton overwriting __new__.
#
#    Cons: multiple inheritance
#          __new__ could be overwritten
#          __init__ is called upon every 'instantiation'
#    """
#    def __new__(cls, *a, **k):
#        if not hasattr(cls, '_inst'):
#            cls._inst = super(Singleton, cls).__new__(cls)
#        return cls._inst

class History(object):
    """History of length 'length'."""
    def __init__(self, length):
        self.length = length
        self.items = [ ]
        self.i = 0

    def get(self):
        return self.items[self.i]

    def back(self):
        if self.i != 0:
            self.i = self.i - 1
        return self.items[self.i]

    def forward(self):
        if self.i != len(self.items) - 1:
            self.i = self.i + 1
        return self.items[self.i]

    def put(self, item):
        while self.i < len(self.items) - 1:
            self.items.pop()
        if self.i == self.length - 1:
            self.items.pop(0)
        self.items.append(item)
        self.i = len(self.items) - 1

    def setlength(self, length):
        while len(self.items) > length:
            self.items.pop(0)
            self.i = self.i - 1
        self.length = length


class StoppableThread( threading.Thread ):
    """
    A thread that can be stopped.
    
    Parameters:
        target:    callable that will be execute by the thread
        name:      string that will be used as a name for the thread
    
    Methods:
        stop():    stop the thread
        
    Use threading.currentThread().stop_request.isSet()
    or threading.currentThread().stop_request.wait([timeout])
    in your target callable to react to a stop request.
    """
    
    def __init__(self, target=None, name=None):
        threading.Thread.__init__(self, target=target, name=name)
        self.stop_request = threading.Event()
        
    def stop(self, timeout=10.):
        name = str(self)
        logging.getLogger().debug('attempt to stop thread '+name)
        if threading.currentThread() is self:
            logging.getLogger().debug('Thread '+name+' attempted to stop itself. Ignoring stop request...')
            return
        elif not self.is_alive():
            logging.getLogger().debug('Thread '+name+' is not running. Continuing...')
            return
        self.stop_request.set()
        self.join(timeout)
        if self.is_alive():
            logging.getLogger().warning('Thread '+name+' failed to join after '+str(timeout)+' s. Continuing anyway...')


class Warning( HasPrivateTraits ):
    """Traits warning string."""
    warning = Str
    
def warning( warning='', buttons=[OKButton, CancelButton] ):
    """Traits popup box that displays a warning string."""    
    w = Warning( warning=warning )
    
    ui = w.edit_traits( view=View( Item('warning', show_label=False, style='readonly'),
                                   buttons=buttons,
                                   width=400, height=150,
                                   kind='modal' ) )
    
    return ui.result



class GetSetItemsMixin:
    """
    Provides save, load, save figure methods. Useful with HasTraits models.
    Data is stored in a dictionary with keys that are strings and identical to
    class attribute names. To save, pass a list of strings that denote attribute names.
    Load methods accept a filename. The dictionary is read from file and attributes
    on the class are set (if necessary created) according to the dictionary content. 
    """

    get_set_items = [] # Put class members that will be saved upon calling 'save' here.
    # BIG FAT WARNING: do not include List() traits here. This will cause inclusion of the entire class definition  during pickling
    # and will result in completely uncontrolled behavior. Normal [] lists are OK.

    _file_mode_map = {'asc':'U', 'bin':'b'}
    _pickle_mode_map = {'asc':0, 'bin':1}
    
    def set_items(self, d):
        # In order to set items in the order in which they appear
        # in the get_set_items, we first iterate through the get_set_items
        # and check whether there are corresponding values in the dictionary.
        for key in self.get_set_items:
            try:
                if key in d:
                    val = d[key]
                    attr = getattr(self, key)
                    if isinstance(val,dict) and isinstance(attr, GetSetItemsMixin): # iterate to the instance
                        attr.set_items(val)
                    else:
                        setattr(self, key, val)
            except:
                logging.getLogger().warning("failed to set item '"+key+"'")

    def get_items(self, keys=None):
        if keys is None:
            keys = self.get_set_items
        d = {}
        for key in keys:
            attr = getattr(self, key)
            if isinstance(attr, GetSetItemsMixin): # iterate to the instance
                d[key] = attr.get_items()
            else:
                d[key] = attr
        return d

    def save(self, filename):
        """detects the format of the savefile and saves it according to the file-ending. .txt and .asc result in an ascii sav,
        .pyd in a pickled python save with mode='asc' and .pys in a pickled python file with mode='bin'"""
        writeDictToFile(self.get_items(),filename)
            
    def copy_items(self, keys):
        d = {}
        for key in keys:
            item = getattr(self, key)
            if hasattr(item,'copy'):
                d[key] = item.copy()
            else:
                d[key] = item
        return d            
    
    def load(self, filename=None):
        if os.access(filename, os.F_OK):
            logging.getLogger().debug('attempting to restore state of '+self.__str__()+' from '+filename+'...')
            if filename.find('.txt')!=-1 or filename.find('.asc')!=-1:
                logging.getLogger().warning('Cannot import from Ascii-File')
            else: 
                try:
                    self.set_items(cPickle.load(open(filename,'r')))
                    logging.getLogger().debug('state of '+self.__str__()+' restored.')
                except:
                    try:
                        self.set_items(cPickle.load(open(filename,'rb')))
                        logging.getLogger().debug('state of '+self.__str__()+' restored.')
                    except:
                        try:
                            self.set_items(cPickle.load(open(filename,'rU')))
                            logging.getLogger().debug('state of '+self.__str__()+' restored.')
                        except:
                            logging.getLogger().debug('failed to restore state of '+self.__str__()+'.')  
                    
    def save_figure(self, figure, filename):
        """
        Saves a figure as graphics file, e.g. .png.
        
        Example of usage:
        
            plot = my_instance.line_plot
            filename = 'foo.png'
            my_instance.save_figure(plot, filename)
        """
        gc = PlotGraphicsContext(figure.outer_bounds, dpi=72)
        gc.render_component(figure)
        gc.save(filename)

class GetSettableHistory(History,GetSetItemsMixin):
    """
    Implements a history that can be pickled and unpickled
    in a generic way using GetSetItems. When this class is used,
    the data attached to the history is saved instead of
    the history object, which otherwise would require the definition
    of the history class to be present when unpickling the file.
    """
    get_set_items=['items','length','i']

class GetSetItemsHandler( Handler ):
    """Handles save and load actions."""
        
    def save(self,info):
        filename = save_file(title='Save')
        if filename is '':
            return
        else:
            info.object.save(filename)

    def export(self, info):
        filename = save_file(title='Export to Ascii')
        if filename is '':
            return
        if filename.find('.txt')==-1 or filename.find('.asc')==-1:
            filename=filename+'.asc'
            info.object.save(filename)
        else:
            info.object.save(filename)

    def load(self, info):
        filename = open_file(title='Load')
        if filename is '':
            return
        else:
            info.object.load(filename)

    """
    def _on_close(self, info):
        return Handler._on_close(self, info)
    """
    
    def closed(self,info, is_ok):
        try:
            thread = info.object.thread
            thread.stop()
        except AttributeError, e:
            pass
    
class GetSetSaveImageHandler( GetSetItemsHandler ):

    """Provides handling of image save action."""

    def save_image(self, info):
        filename = save_file(title='Save Image')
        if filename is '':
            return
        else:
            if filename.find('.png')==-1:
                filename=filename+'.png'
            info.object.save_image(filename)    

class AspectZoomTool(SimpleZoom):

    box = Tuple()

    def _do_zoom(self):
        """ Does the zoom operation.
        """
        # Sets the bounds on the component using _cur_stack_index
        low, high = self._current_state()
        orig_low, orig_high = self._history[0]
    
        if self._history_index == 0:
            if self.tool_mode == "range":
                mapper = self._get_mapper()
                mapper.range.low_setting = self._orig_low_setting
                mapper.range.high_setting = self._orig_high_setting
            else:
                x_range = self.component.x_mapper.range
                y_range = self.component.y_mapper.range
                x_range.low_setting, y_range.low_setting = \
                    self._orig_low_setting
                x_range.high_setting, y_range.high_setting = \
                    self._orig_high_setting

                # resetting the ranges will allow 'auto' to pick the values
                x_range.reset()
                y_range.reset()
               
        else:   
            if self.tool_mode == "range":
                mapper = self._get_mapper()
                if self._zoom_limit_reached(orig_low, orig_high, low, high, mapper):
                    self._pop_state()
                    return
                mapper.range.low = low
                mapper.range.high = high
            else:
                for ndx in (0, 1):
                    mapper = (self.component.x_mapper, self.component.y_mapper)[ndx]
                    if self._zoom_limit_reached(orig_low[ndx], orig_high[ndx],
                                                low[ndx], high[ndx], mapper):
                        # pop _current_state off the stack and leave the actual
                        # bounds unmodified.
                        self._pop_state()
                        return
                x_range = self.component.x_mapper.range
                y_range = self.component.y_mapper.range
                x_range.low, y_range.low = low
                x_range.high, y_range.high = high

        plot = self.component.container
        plot.aspect_ratio = (x_range.high - x_range.low) / (y_range.high - y_range.low)
        
        self.box=(low[0],low[1],high[0],high[1])
        
        self.component.request_redraw()
        return

def edit_traits(old_func):
    """
    Takes care about calling of 'edit_traits' for a factory function that
    returns a SingletonHasTraits instance.
    
    Example of usage:
    
    @edit_traits
    def factory_func():
        my_instance = MySingletonHasTraits() 
        return my_instance
    """
    def func():
        if not hasattr(func,'ui') or func.ui.result:
            func.ui = old_func().edit_traits()
        return old_func()
    return func

def edit_singleton(old_func):
    """
    Emulates singleton behavior for a factory function that returns a
    HasTraits instance. Also takes care about proper calling of 'edit_traits'.
    
    Example of usage:
    
    @singleton
    def factory_func():
        my_instance = MyHasTraits() 
        return my_instance
    
    a = factory_func()
    b = factory_func()
    a is b

    Result without decorator: False
    Result with singleton decorator: True
    """
    def func():
        if not hasattr(func,'instance'):
            func.instance = old_func()
            func.ui = func.instance.edit_traits()
        if func.ui.result:
            func.ui = func.instance.edit_traits()
        return func.instance
    return func

def singleton(old_func):
    """
    Emulates singleton behavior for a factory function.
    
    Example of usage:
    
    @singleton
    def factory_func():
        my_instance = MyHasTraits() 
        return my_instance
    
    a = factory_func()
    b = factory_func()
    a is b

    Result without decorator: False
    Result with singleton decorator: True
    """
    def func():
        if not hasattr(func,'instance'):
            func.instance = old_func()
        return func.instance
    return func


if __name__ is '__main__':
    pass
