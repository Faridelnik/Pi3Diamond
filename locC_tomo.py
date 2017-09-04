
from tools.utility import edit_singleton
from datetime import date
import os

from traits.api import Bool
import imp
import numpy as np
import time
import threading

from tools.emod import ManagedJob
from tools.utility import GetSetItemsHandler, GetSetItemsMixin
from hardware.waveform import *
import hardware.api as ha

class process_tomo(ManagedJob):

    def __init__(self,prog):
        super(process_tomo, self).__init__()     
        self.prog = prog
        self.file_pos = 'D:/data/ProgGate/tomo/cnot'
        os.path.exists(self.file_pos)
        self.p_init_i = []
        self.p_init_q = []
        
    def _Do_tomo(self):
        zero = Idle(1)
    
        from measurements.prog_gate_awg import EspinTomo_LoC1
        self.prog.measurement = EspinTomo_LoC1()
       
        self.prog.measurement.sweeps = 5.0e5
        self.prog.measurement.tau1 = 660
        
        istate_seq_file = 'D:/data/ProgGate/tomo/seq/diag'
        datfile = istate_seq_file + '/initial_seq.py'
        
        sampling = 1.2e9
        
        f1 = (self.prog.measurement.freq - self.prog.measurement.freq_center)/sampling
        f2 = (self.prog.measurement.freq_4 - self.prog.measurement.freq_center)/sampling
        f3= (self.prog.measurement.freq_2 - self.prog.measurement.freq_center)/sampling
        f4 = (self.prog.measurement.freq_3 - self.prog.measurement.freq_center)/sampling
        p = {}
        fileHandle = open (datfile) 
        #read the cotend of the file
        datfilelines=fileHandle.read()
        exec datfilelines 
        fileHandle.close()
        
        # gate
        datfile1 = istate_seq_file + '/cnot.py'
        fileHandle = open (datfile1) 
        #read the cotend of the file
        datfilelines=fileHandle.read()
        exec datfilelines 
        fileHandle.close()
        
        cnot_gate_i = va_vb_i + [Idle(12600 * 1.2)] + pi_i + [Idle(12600 * 1.2)] + ua_ub_i
        cnot_gate_q = va_vb_q + [Idle(12600 * 1.2)] + pi_q + [Idle(12600 * 1.2)] + ua_ub_q
        
        #cnot_gate_i = [Idle(1)]
        #cnot_gate_q = [Idle(1)]
        
        
        #name_istate = ['-1_-1']
        #ref_state = ['+1_Plus']
        #name_istate = ['+1_PlusI','+1_+1','+1_-1','-1_Plus','-1_PlusI','-1_+1','-1_-1']
        name_istate = ['Plus_Plus','Plus_PlusI','Plus_+1','Plus_-1','PlusI_Plus','PlusI_PlusI','PlusI_+1','PlusI_-1','+1_Plus','+1_PlusI','+1_+1','+1_-1','-1_Plus','-1_PlusI','-1_+1','-1_-1']
        ref_state = ['Plus_Plus','Plus_PlusI','Plus_+1','PlusI_-1','PlusI_Plus','PlusI_PlusI','PlusI_+1','PlusI_-1','+1_Plus','+1_PlusI','+1_+1','+1_-1','-1_Plus','-1_PlusI','-1_+1','-1_-1']
        #name_istate  = ['+1_Plus','Plus_Plus','Plus_PlusI','-1_Plus','-1_Plus','+1_+1','+1_-1']

        for t in range(len(name_istate)):
        #for t in range(9):
            #self.prog.measurement.npulse = t
           
            seq_x = name_istate[t] + ' + 0'
            seq_y = name_istate[t] + ' + 90'
            self.prog.measurement.p_init_i = p[seq_x]
            self.prog.measurement.p_init_q = p[seq_y]
            
            #self.prog.measurement.istate_pulse = name_istate[t]
           
            ref_x = ref_state[0] + ' + 0'
            ref_y = ref_state[0] + ' + 90'
            self.prog.measurement.p_ref_i = p[ref_x]
            self.prog.measurement.p_ref_q = p[ref_y]
           
            self.prog.measurement.p_gate_i = cnot_gate_i
            self.prog.measurement.p_gate_q = cnot_gate_q
        
            self.prog.measurement.load()
            time.sleep(45.0)
           
            while self.prog.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
                 
            self.prog.measurement.submit()
           
            while self.prog.measurement.state != 'done':
                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                   break
                   
            if self.prog.measurement.state == 'done':          
            
               file_name = self.file_pos + '/loc1_cnot_' +  name_istate[t]
               self.prog.save_line_plot(file_name + '.png')
               self.prog.save(file_name + '.pyd')          
               time.sleep(10)
               
               self.prog.measurement.remove()
               self.prog.measurement.state = 'idle'   
               time.sleep(40)
               
    def _Do_tomo2(self):
        zero = Idle(1)
    
        from measurements.prog_gate_awg import EspinTomo_LoC2
        self.prog.measurement = EspinTomo_LoC2()
       
        self.prog.measurement.sweeps = 5.0e5
        self.prog.measurement.tau1 = 660
        
        istate_seq_file = 'D:/data/ProgGate/tomo/seq/diag'
        datfile = istate_seq_file + '/initial_seq.py'
        
        sampling = 1.2e9
        
        f1 = (self.prog.measurement.freq - self.prog.measurement.freq_center)/sampling
        f2 = (self.prog.measurement.freq_4 - self.prog.measurement.freq_center)/sampling
        f3= (self.prog.measurement.freq_2 - self.prog.measurement.freq_center)/sampling
        f4 = (self.prog.measurement.freq_3 - self.prog.measurement.freq_center)/sampling
        p = {}
        fileHandle = open (datfile) 
        #read the cotend of the file
        datfilelines=fileHandle.read()
        exec datfilelines 
        fileHandle.close()
        
        # gate
        datfile1 = istate_seq_file + '/cnot.py'
        fileHandle = open (datfile1) 
        #read the cotend of the file
        datfilelines=fileHandle.read()
        exec datfilelines 
        fileHandle.close()
        
        cnot_gate_i = va_vb_i + [Idle(12600 * 1.2)] + pi_i + [Idle(12600 * 1.2)] + ua_ub_i
        cnot_gate_q = va_vb_q + [Idle(12600 * 1.2)] + pi_q + [Idle(12600 * 1.2)] + ua_ub_q
        
        #cnot_gate_i = [Idle(1)]
        #cnot_gate_q = [Idle(1)]
        
        
        #name_istate = ['-1_-1']
        #ref_state = ['+1_Plus']
        #name_istate = ['+1_PlusI','+1_+1','+1_-1','-1_Plus','-1_PlusI','-1_+1','-1_-1']
        name_istate = ['Plus_Plus','Plus_PlusI','Plus_+1','Plus_-1','PlusI_Plus','PlusI_PlusI','PlusI_+1','PlusI_-1','+1_Plus','+1_PlusI','+1_+1','+1_-1','-1_Plus','-1_PlusI','-1_+1','-1_-1']
        ref_state = ['Plus_Plus','Plus_PlusI','Plus_+1','PlusI_-1','PlusI_Plus','PlusI_PlusI','PlusI_+1','PlusI_-1','+1_Plus','+1_PlusI','+1_+1','+1_-1','-1_Plus','-1_PlusI','-1_+1','-1_-1']
        #name_istate  = ['+1_Plus','Plus_Plus','Plus_PlusI','-1_Plus','-1_Plus','+1_+1','+1_-1']

        for t in range(len(name_istate)):
        #for t in range(9):
            #self.prog.measurement.npulse = t
           
            seq_x = name_istate[t] + ' + 0'
            seq_y = name_istate[t] + ' + 90'
            self.prog.measurement.p_init_i = p[seq_x]
            self.prog.measurement.p_init_q = p[seq_y]
            
            #self.prog.measurement.istate_pulse = name_istate[t]
           
            ref_x = ref_state[0] + ' + 0'
            ref_y = ref_state[0] + ' + 90'
            self.prog.measurement.p_ref_i = p[ref_x]
            self.prog.measurement.p_ref_q = p[ref_y]
           
            self.prog.measurement.p_gate_i = cnot_gate_i
            self.prog.measurement.p_gate_q = cnot_gate_q
        
            self.prog.measurement.load()
            time.sleep(45.0)
           
            while self.prog.measurement.reload == True:
                threading.currentThread().stop_request.wait(1.0)
                 
            self.prog.measurement.submit()
           
            while self.prog.measurement.state != 'done':
                threading.currentThread().stop_request.wait(1.0)
                if threading.currentThread().stop_request.isSet():
                   break
                   
            if self.prog.measurement.state == 'done':          
            
               file_name = self.file_pos + '/loc2_cnot_' +  name_istate[t]
               self.prog.save_line_plot(file_name + '.png')
               self.prog.save(file_name + '.pyd')          
               time.sleep(10)
               
               self.prog.measurement.remove()
               self.prog.measurement.state = 'idle'   
               time.sleep(40)           


    def _run(self):
        self._Do_tomo()
        self._Do_tomo2()
        
        