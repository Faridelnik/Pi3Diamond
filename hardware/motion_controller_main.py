# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 08:23:17 2015

@author: Ali.Eftekhari
"""
"""Install the latest version of PYVISA from https://pypi.python.org/pypi/PyVISA"""
"""install PYVISA using pip that is "pip install pyvisa" """
import visa

"""MotionController class has the common functions/methods for SMC, CONEX and ESP controllers  """
class MotionControllers(object):
    def __init__(self, visa_address = u'ASRL3::INSTR'):
        """ default backend for ResourceManager is visa32.dll and is located at C:\Windows\system32\visa32.dll"""
        self.rm=visa.ResourceManager()
        #self.list_of_devices = self.rm.list_resources()
        #number_of_devices=len (self.list_of_devices)
        #for device in range (number_of_devices):
            #self.device_key=self.list_of_devices[device]
            #self.connect=self.rm.open_resource(self.device_key)
      
        self.connect=self.rm.open_resource(visa_address)     
        
    def send(self, command):
        self.command=command
        self.connect.write(self.command)         
        
    def write_read(self, command):
        self.command=command
        self.connect.write(self.command)
        return (self.connect.read())
        
    def read(self):
        print (self.connect.read())        
        

    def get_acceleration(self,axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"AC?")
        
    def set_acceleration(self,axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"AC"+self.value)        

    def get_backlash(self,axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"BA?")
        
    def set_backlash(self,axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"BA"+self.value)          

    def get_following_error(self,axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"FE?")
        
    def set_following_error(self,axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"FE"+self.value) 

    def get_stage_modelnumber(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"ID?")
        
    def set_stage_modelnumber(self,axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"ID"+self.value)

    def get_derivative_gain(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"KD?")
        
    def set_derivative_gain(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"KD"+self.value)     

    def get_integral_gain(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"KI?")
        
    def set_integral_gain(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"KI"+self.value)     

    def get_proportional_gain(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"KP?")
        
    def set_proportional_gain(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"KP"+self.value) 

    def move_absolute(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"PA"+self.value)
        
    def move_relative(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"PR"+self.value)

    def get_motor_currentlimit(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"QI?")
        
    def set_motor_currentlimit(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"QI"+self.value)

    def set_controller_addrees(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"SA"+self.value)
    
    def get_controller_addrees(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"SA?")

    def reset_controller(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"RS")

    def stop_motion(self, axis):
        self.axis=str(axis)
        self.send(self.axis+"ST")

    def get_negative_softwarelimit(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"SL?")
        
    def set_negative_softwarelimit(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"SL"+self.value) 

    def get_positive_softwarelimit(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"SR?")
        
    def set_positive_softwarelimit(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"SR"+self.value) 

    def get_encodercount(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"SU?")
        
    def set_encodercount(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"SU"+self.value) 

    def get_error_message(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"TB")

    def get_error_code(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"TE")

    def get_current_position(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"TP")

    def get_positioner_error(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"TS")

    def get_velocity(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"VA?")
        
    def set_velocity(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"VA"+self.value) 
        
    def get_firmware_version(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"VE")

    def save_settings(self):
        self.send("SM")







            
