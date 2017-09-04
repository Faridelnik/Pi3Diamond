# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:09:18 2015

@author: Ali.Eftekhari
"""

from motion_controller_main import MotionControllers

class CONEXSMC(MotionControllers):
    def __init__(self):
        super(CONEXSMC,self).__init__()
        
    def get_hysteresis(self,axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"BH?")
        
    def set_hysteresis(self,axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"BH"+self.value)          
        
    def get_driver_voltage(self,axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"DV?")

    def set_driver_voltage(self,axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"DV"+self.value)           

    def get_lowpass_filter(self,axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"FD?")
        
    def set_lowpass_filter(self,axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"FD"+self.value)           
        
    def get_friction_compensation(self,axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"FF?")
        
    def set_friction_compensation(self,axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"FF"+self.value)         
        
    def get_home_search_type(self,axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"HT?")
        
    def set_home_search_type(self,axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"HT"+self.value)         
               
    def get_jerktime(self,axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"JR?")
        
    def set_jerktime(self,axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"JR"+self.value)     
                       
    def get_enable_disable_state(self,axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"MM?")
        
    def set_enable_disable_state(self,axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"MM"+self.value)
        
    def home(self,axis):
        self.axis=str(axis)
        self.send(self.axis+"OR")
        
    def get_homeserach_timeout(self,axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"OT?")
        
    def set_homeserach_timeout(self,axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"OT"+self.value)         
               
    def enter_configuration(self,axis):
        self.axis=str(axis)
        return self.send(self.axis+"PW1")

    def leave_configuration(self,axis):
        self.axis=str(axis)
        return self.send(self.axis+"PW0")
        
    def move_absolute_axisnumber(self,axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"SE"+self.value)

    def get_motiontime_relativemove(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        self.send(self.axis+"PT"+self.value)
        return self.read()
      
    def get_controller_loopstate(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"SC?")

    def close_controllerloop(self, axis):
        self.axis=str(axis)
        return self.send(self.axis+"SC1")
        
    def open_controllerloop(self, axis):
        self.axis=str(axis)
        return self.send(self.axis+"SC0")
        
    def configure_simultaneous_startmove(self,number_of_axis):
        for axis in range(number_of_axis):
            axis=str(axis+1)
            self.set_controller_addrees(axis)
            absolute_position=raw_input("What is the absolute position for axis number "+axis+" ?")
            self.move_absolute_axisnumber(axis,absolute_position)
        self.send("SE")
        return self.read()
            
    def get_setpoint_position(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"TH")

    def enter_trackingmode(self, axis):
        self.axis=str(axis)
        return self.send(self.axis+"TK1") 

    def leave_trackingmode(self, axis):
        self.axis=str(axis)
        return self.send(self.axis+"TK0") 

    def get_configuration_parameters(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"ZT?")

    def get_homeserach_velocity(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"OH?")
        
    def set_homeserach_velocity(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"OH"+self.value)  