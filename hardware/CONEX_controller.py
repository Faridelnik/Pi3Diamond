# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:33:37 2015

@author: Ali.Eftekhari
"""
from CONEX_SMC_common import CONEXSMC

class CONEX(CONEXSMC):
    def __init__(self):
        super(CONEX,self).__init__()
        device_key=raw_input("What is the COM port associated with CONEX controller? ")
        self.connect=self.rm.open_resource(device_key, baud_rate=921600, timeout=2000, data_bits=8, write_termination='\r\n')

    def get_velocity_feedforward(self):
        return self.write_read("1KV?")
        
    def set_velocity_feedforward(self,value):
        self.value=str(value)
        return self.send("1KV"+self.value)    
