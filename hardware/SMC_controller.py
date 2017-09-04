# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:44:52 2015

@author: Ali.Eftekhari
"""
import visa

class SMC(object):
    def __init__(self):
        super(SMC,self).__init__()
        device_key= u'ASRL3::INSTR'
        self.rm=visa.ResourceManager()
        self.connect=self.rm.open_resource(device_key, baud_rate=57600, timeout=2000, data_bits=8, write_termination='\r\n')
        
        
    def send(self, command):
        self.command=command
        self.connect.write(self.command)         
        
    def write_read(self, command):
        self.command=command
        self.connect.write(self.command)
        return (self.connect.read())
        
    def read(self):
        return(self.connect.read())
        #print (self.connect.read())        
        
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


            
    
        
    def get_base_velocity(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"VB?")
        
    def set_base_velocity(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"VB"+self.value) 
        
    def get_microstep(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"FR?")
        
    def set_microstep(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"FR"+self.value)

    def leave_jogging(self, axis):
        self.axis=str(axis)
        return self.send(self.axis+"JD") 
        
    def leave_keypad(self, axis):
        self.axis=str(axis)
        return self.send(self.axis+"JM0")

    def enter_keypad(self, axis):
        self.axis=str(axis)
        return self.send(self.axis+"JM1")

    def get_TTL_output(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"SB?")
        
    def set_TTL_output(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"SB"+self.value)

    def get_ESP_configuration(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"ZX?")
        
    def set_ESP_configuration(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"ZX"+self.value)
        
    def get_velocity_feedforward(self, axis):
        self.axis=str(axis)
        return self.write_read(self.axis+"KV?")
        
    def set_velocity_feedforward(self, axis,value):
        self.axis=str(axis)
        self.value=str(value)
        return self.send(self.axis+"KV"+self.value)    
        
 