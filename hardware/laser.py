import numpy as np

from traits.api import SingletonHasTraits, Range
from traitsui.api import View, Item

from hardware.nidaq import AOTask

class Laser( SingletonHasTraits ):

    def __init__(self, AO_channel='/Dev1/ao3', voltage_range=(0.,5.)):
        SingletonHasTraits.__init__(self)
        self.AOTask = AOTask(Channels=AO_channel, range=voltage_range)
        self.AOTask.Write(np.array((float(voltage_range[0]),)))
        self.add_trait('voltage', Range(low=float(voltage_range[0]), high=float(voltage_range[1]), value=float(voltage_range[0]), desc='output voltage', label='Voltage [V]'))
        self.on_trait_change(self.write_voltage, 'voltage')
        
    def write_voltage(self):
        self.AOTask.Write(np.array((self.voltage,)))

    view = View(Item('voltage'),
                title='Laser', width=400, buttons=[], resizable=True)

