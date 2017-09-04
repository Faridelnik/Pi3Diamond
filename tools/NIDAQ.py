"""
This file is part of Diamond. Diamond is a confocal scanner written
in python / Qt4. It combines an intuitive gui with flexible
hardware abstraction classes.

Diamond is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Diamond is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with diamond. If not, see <http://www.gnu.org/licenses/>.

Copyright (C) 2009 Helmut Rathgen <helmut.rathgen@gmail.com>
"""

import ctypes

# lib = ctypes.CDLL('./libfoo.so')


import numpy
import time

dll = ctypes.windll.LoadLibrary('nicaiu.dll')

# dll.argtypes = []
# dll.restype = ctypes.c_void_p
#
# dll.argtypes = [ctypes.c_void_p]
# dll.restype = None

DAQmx_Val_Cfg_Default             = ctypes.c_int32(-1)
DAQmx_Val_RSE                     = ctypes.c_int32(10083)
DAQmx_Val_DoNotInvertPolarity     = ctypes.c_int32(0)
DAQmx_Val_GroupByChannel          = ctypes.c_int32(0)
DAQmx_Val_GroupByScanNumber       = ctypes.c_int32(1)
DAQmx_Val_ChanPerLine             = ctypes.c_int32(0)
DAQmx_Val_ChanForAllLines         = ctypes.c_int32(1)
DAQmx_Val_Acquired_Into_Buffer    = ctypes.c_int32(1)
DAQmx_Val_Ticks                   = ctypes.c_int32(10304)
DAQmx_Val_Rising                  = ctypes.c_int32(10280)
DAQmx_Val_Falling                 = ctypes.c_int32(10171)
DAQmx_Val_CountUp                 = ctypes.c_int32(10128)
DAQmx_Val_ContSamps               = ctypes.c_int32(10123)
DAQmx_Val_FiniteSamps             = ctypes.c_int32(10178)
DAQmx_Val_Hz                      = ctypes.c_int32(10373)
DAQmx_Val_Low                     = ctypes.c_int32(10214)
DAQmx_Val_High                    = ctypes.c_int32(10192)
DAQmx_Val_Volts                   = ctypes.c_int32(10348)
DAQmx_Val_MostRecentSamp          = ctypes.c_uint32(10428)
DAQmx_Val_OverwriteUnreadSamps    = ctypes.c_uint32(10252)
DAQmx_Val_HWTimedSinglePoint      = ctypes.c_int32(12522)
DAQmx_Val_SampClk                 = ctypes.c_int32(10388)
DAQmx_Val_OnDemand                = ctypes.c_int32(10390)
DAQmx_Val_ChangeDetection         = ctypes.c_int32(12504)
DAQmx_Val_CurrReadPos             = ctypes.c_int32(10425)
DAQmx_Val_MostRecentSamp          = ctypes.c_int32(10428)
DAQmx_Val_OverwriteUnreadSamps    = ctypes.c_int32(10252)
DAQmx_Val_DoNotOverwriteUnreadSamps  = ctypes.c_int32(10159)
DAQmx_Val_Task_Unreserve            = ctypes.c_int32(5)
DAQmx_Val_DigLvl                  = ctypes.c_int32(10152)

c_uint32_p = c_ulong_p = ctypes.POINTER(ctypes.c_uint32)
c_float64_p = c_double_p = ctypes.POINTER(ctypes.c_double)

def CHK(err):
    """a simple error checking routine"""
    if err < 0:
        buf_size = 1000
        buf = ctypes.create_string_buffer('\000' * buf_size)
        dll.DAQmxGetErrorString(err,ctypes.byref(buf),buf_size)
        raise RuntimeError('nidaq call failed with error %d: %s'%(err,repr(buf.value)))

# route signal
def Connect(source, destination):
    """Connect terminal 'source' to terminal 'destination'."""
    CHK( dll.DAQmxConnectTerms(source, destination, DAQmx_Val_DoNotInvertPolarity )  )

def Disconnect(source, destination):
    """Connect terminal 'source' to terminal 'destination'."""
    CHK( dll.DAQmxDisconnectTerms(source, destination)  )


class NIDAQ():
    """NIDAQ measurement card.
    2 counters are used for standard photon counting (time trace).
    The other 2 counters are used on demand for functions like scanning, odmr etc."""
    _CountAverageLength = 1
    _MaxCounts = 1e7
    _DefaultCountTime = 8e-3
    _DefaultSettlingTime = 2e-3
    _DefaultCountLength = 1000
    _RWTimeout = 60
    _DefaultAOLength = 1
    
    def __init__(self, photon_source, trace_counter_in, trace_counter_out, function_counter_in, function_counter_out, 
                 scanner_ao_channels, scanner_xrange, scanner_yrange, scanner_zrange, odmr_trig_channel=None, gate_in_channel=None,
                 aom_volt_ch=None, mirror_flip_ch=None, scanner_ai_channels=None, powermeter_ai_channel=None, 
                 aux1_ai_channel=None, aux2_ai_channel=None, rf_power_ai_channel=None, rf_power_trigger_channel=None):
        self._photon_source = photon_source                 # channel where apd is conntected
        self._odmr_trig_channel = odmr_trig_channel         # channel for mw trigger during odmr
        self._trace_counter_in = trace_counter_in           # counter for standard time trace (photon counting)
        self._trace_counter_out = trace_counter_out         # counter to generate pulse signal for standard photon counting
        self._function_counter_in = function_counter_in     # counter for seperate function / photon counting (eg scanning, odmr)
        self._function_counter_out = function_counter_out   # counter for pulse generation for seperate function
        self._scanner_xrange = scanner_xrange               # x range of scanner as (xmin, xmax)
        self._scanner_yrange = scanner_yrange
        self._scanner_zrange = scanner_zrange 
        self._scanner_xvolt_range = (10,-10)
        self._scanner_yvolt_range = (-10,10)
        self._scanner_zvolt_range = (-10,10)
        self.calibrated = False        
        self._scanner_ao_channels = scanner_ao_channels     # scanner ao channels for x,y,z control
        self._scanner_ai_channels = scanner_ai_channels     # scanner ai channels for x,y,z readout
        self._powermeter_ai_channel = powermeter_ai_channel
        self._aux1_ai_channel = aux1_ai_channel
        self._aux2_ai_channel = aux2_ai_channel
        self.gate_in_channel = gate_in_channel              # channel where external gate is connected for gated photon counting
        self.mirror_flip_ch = mirror_flip_ch
        self.aom_volt_ch = aom_volt_ch
        self._rf_power_ai_channel = rf_power_ai_channel
        self._rf_power_trigger_channel = rf_power_trigger_channel        

        self.function_state = 'idle'    # set to 'scan', 'odmr', during scanning, odmr. before starting function, check if state is idle

        self.scanner_x = 0.0    # current x, y, z values of scanner
        self.scanner_y = 0.0
        self.scanner_z = 0.0

        # init ao channels / task for scanner, should always be active
        self.scanner_ao_task = ctypes.c_uint32()
        CHK(  dll.DAQmxCreateTask('', ctypes.byref(self.scanner_ao_task))  )
        CHK(  dll.DAQmxCreateAOVoltageChan( self.scanner_ao_task,           # add to this task
                                            self._scanner_ao_channels, '',  # use sanncer ao_channels, name = ''
                                            ctypes.c_double(-10.),          # min voltage
                                            ctypes.c_double(10.),           # max voltage
                                            DAQmx_Val_Volts,'')    )        # units is Volt
        CHK( dll.DAQmxSetSampTimingType( self.scanner_ao_task, DAQmx_Val_OnDemand)  ) # set task timing to on demand, i.e. when demanded by software

        if self.mirror_flip_ch != None:
            self.mirror_flip_task = ctypes.c_uint32()
            CHK(  dll.DAQmxCreateTask('', ctypes.byref(self.mirror_flip_task))  )
            CHK(  dll.DAQmxCreateAOVoltageChan( self.mirror_flip_task,           # add to this task
                                                self.mirror_flip_ch, '',
                                                ctypes.c_double(-0.1),          # min voltage
                                                ctypes.c_double(5.1),           # max voltage
                                                DAQmx_Val_Volts,'')    )        # units is Volt
            CHK( dll.DAQmxSetSampTimingType( self.mirror_flip_task, DAQmx_Val_OnDemand)  ) # set task timing to on demand, i.e. when demanded by software

        if self.aom_volt_ch != None:
            self.aom_volt_task = ctypes.c_uint32()
            CHK(  dll.DAQmxCreateTask('', ctypes.byref(self.aom_volt_task))  )
            CHK(  dll.DAQmxCreateAOVoltageChan( self.aom_volt_task,           # add to this task
                                                self.aom_volt_ch, '',
                                                ctypes.c_double(-10.),          # min voltage
                                                ctypes.c_double(10.),           # max voltage
                                                DAQmx_Val_Volts,'')    )        # units is Volt
            CHK( dll.DAQmxSetSampTimingType( self.aom_volt_task, DAQmx_Val_OnDemand)  ) # set task timing to on demand, i.e. when demanded by software


        # init ai channels / task to read scanner position
        if self._scanner_ai_channels != None:
            self.scanner_ai_task = ctypes.c_uint32()
            CHK(  dll.DAQmxCreateTask('', ctypes.byref(self.scanner_ai_task))  )
            CHK(  dll.DAQmxCreateAIVoltageChan( self.scanner_ai_task,           # add to this task
                                                self._scanner_ai_channels,      # use scanner ao_channels
                                                '',                             # use scanner ao_channels name = ''
                                                DAQmx_Val_RSE,                  # measuring against ground? DAQmx_Val_Cfg_Default ?
                                                ctypes.c_double(-10.),          # min voltage
                                                ctypes.c_double(10.),           # max voltage
                                                DAQmx_Val_Volts,                # units is Volt
                                                '')    )                        # use no costum scaling
            CHK( dll.DAQmxSetSampTimingType( self.scanner_ai_task, DAQmx_Val_OnDemand)  ) # set task timing to on demand, i.e. when demanded by software



        # init ai channels / task to read powermeter
        if self._powermeter_ai_channel != None:
            self.powermeter_ai_task = ctypes.c_uint32()
            CHK(  dll.DAQmxCreateTask('', ctypes.byref(self.powermeter_ai_task))  )
            CHK(  dll.DAQmxCreateAIVoltageChan( self.powermeter_ai_task,           # add to this task
                                                self._powermeter_ai_channel,      # use scanner ao_channels
                                                '',                             # use scanner ao_channels name = ''
                                                DAQmx_Val_RSE,          # measuring against ground? DAQmx_Val_Cfg_Default ?
                                                ctypes.c_double(0.),          # min voltage
                                                ctypes.c_double(2.),           # max voltage
                                                DAQmx_Val_Volts,                # units is Volt
                                                '')    )                        # use no costum scaling
            CHK( dll.DAQmxSetSampTimingType( self.powermeter_ai_task, DAQmx_Val_OnDemand)  ) # set task timing to on demand, i.e. when demanded by software

        # init ai channels / task to read powermeter
        if self._powermeter_ai_channel != None:
            self.powermeter_ai_task = ctypes.c_uint32()
            CHK(  dll.DAQmxCreateTask('', ctypes.byref(self.powermeter_ai_task))  )
            CHK(  dll.DAQmxCreateAIVoltageChan( self.powermeter_ai_task,           # add to this task
                                                self._powermeter_ai_channel,      # use scanner ao_channels
                                                '',                             # use scanner ao_channels name = ''
                                                DAQmx_Val_RSE,          # measuring against ground? DAQmx_Val_Cfg_Default ?
                                                ctypes.c_double(0.),          # min voltage
                                                ctypes.c_double(2.),           # max voltage
                                                DAQmx_Val_Volts,                # units is Volt
                                                '')    )                        # use no costum scaling
            CHK( dll.DAQmxSetSampTimingType( self.powermeter_ai_task, DAQmx_Val_OnDemand)  ) # set task timing to on demand, i.e. when demanded by software

        # init ai channels / task to read auxillary 1
        if self._aux1_ai_channel != None:
            self.aux1_ai_task = ctypes.c_uint32()
            CHK(  dll.DAQmxCreateTask('', ctypes.byref(self.aux1_ai_task))  )
            CHK(  dll.DAQmxCreateAIVoltageChan( self.aux1_ai_task,           # add to this task
                                                self._aux1_ai_channel,      # use aux1 ai_channels
                                                '',                             # use aux1 ai_channels name = ''
                                                DAQmx_Val_RSE,          # measuring against ground? DAQmx_Val_Cfg_Default ?
                                                ctypes.c_double(-10.),          # min voltage
                                                ctypes.c_double(0.),           # max voltage
                                                DAQmx_Val_Volts,                # units is Volt
                                                '')    )                        # use no costum scaling
            CHK( dll.DAQmxSetSampTimingType( self.aux1_ai_task, DAQmx_Val_OnDemand)  ) # set task timing to on demand, i.e. when demanded by software

        # init ai channels / task to read auxillary 2
        if self._aux2_ai_channel != None:
            self.aux2_ai_task = ctypes.c_uint32()
            CHK(  dll.DAQmxCreateTask('', ctypes.byref(self.aux2_ai_task))  )
            CHK(  dll.DAQmxCreateAIVoltageChan( self.aux2_ai_task,           # add to this task
                                                self._aux2_ai_channel,      # use aux2 ai_channels
                                                '',                             # use aux2 ai_channels name = ''
                                                DAQmx_Val_RSE,          # measuring against ground? DAQmx_Val_Cfg_Default ?
                                                ctypes.c_double(0.),          # min voltage
                                                ctypes.c_double(1.),           # max voltage
                                                DAQmx_Val_Volts,                # units is Volt
                                                '')    )                        # use no costum scaling
            CHK( dll.DAQmxSetSampTimingType( self.aux2_ai_task, DAQmx_Val_OnDemand)  ) # set task timing to on demand, i.e. when demanded by software

        # init ai channels / task to read auxillary 2
        if self._rf_power_ai_channel != None:
            self.rf_power_ai_task = ctypes.c_uint32()
            CHK(  dll.DAQmxCreateTask('', ctypes.byref(self.rf_power_ai_task))  )
            CHK(  dll.DAQmxCreateAIVoltageChan( self.rf_power_ai_task,           # add to this task
                                                self._rf_power_ai_channel,      # use aux2 ai_channels
                                                '',                             # use aux2 ai_channels name = ''
                                                DAQmx_Val_RSE,          # measuring against ground? DAQmx_Val_Cfg_Default ?
                                                ctypes.c_double(-0.19),          # min voltage
                                                ctypes.c_double(0.1),           # max voltage
                                                DAQmx_Val_Volts,                # units is Volt
                                                '')    )                        # use no costum scaling
            CHK( dll.DAQmxCfgSampClkTiming(self.rf_power_ai_task,
                                           '',
                                           ctypes.c_double(250000),
                                           DAQmx_Val_Rising,
                                           DAQmx_Val_ContSamps,
                                           ctypes.c_uint64(10000)
                                            ) )
            CHK( dll.DAQmxSetPauseTrigType(self.rf_power_ai_task, DAQmx_Val_DigLvl) )
            CHK( dll.DAQmxSetDigLvlPauseTrigSrc(self.rf_power_ai_task, self._rf_power_trigger_channel) )
            CHK( dll.DAQmxSetDigLvlPauseTrigWhen(self.rf_power_ai_task, DAQmx_Val_Low)  )

    def start_rf_power_ai(self, buffer_length=4000):
        CHK( dll.DAQmxSetSampQuantSampPerChan(self.rf_power_ai_task, ctypes.c_uint64(buffer_length)) )
        CHK( dll.DAQmxSetReadRelativeTo(self.rf_power_ai_task, DAQmx_Val_MostRecentSamp) )
        CHK( dll.DAQmxSetReadOverWrite(self.rf_power_ai_task, DAQmx_Val_OverwriteUnreadSamps) )
        CHK( dll.DAQmxStartTask(self.rf_power_ai_task) )
    def stop_rf_power_ai(self):
        CHK( dll.DAQmxStopTask(self.rf_power_ai_task) )
    def read_rf_power_ai(self, number_samples):
        CHK( dll.DAQmxSetReadOffset(self.rf_power_ai_task, ctypes.c_int32(-number_samples)) )
        data = numpy.empty((1,number_samples), dtype=numpy.float64)
        n_read_samples = ctypes.c_int32()
        CHK(dll.DAQmxReadAnalogF64(self.rf_power_ai_task,
                                   ctypes.c_int32(-1),
                                   ctypes.c_double(self._RWTimeout),
                                   DAQmx_Val_GroupByChannel,
                                   data.ctypes.data_as(c_float64_p),
                                   ctypes.c_uint32(1*number_samples),
                                   ctypes.byref(n_read_samples),
                                   None
                                   )
            )
        return data[0]

    def mirror_flip(self):
        "Creates a pulse with 5V amplitude on mirror_flip_ch, length is undefined, probably some 10 ms"
        def set_voltage(volt):
            temp = ctypes.c_int32()
            CHK( dll.DAQmxWriteAnalogF64( self.mirror_flip_task,
                                          ctypes.c_int32(1), True,    # length of command, start task immediatly (True), or wait for software start (False)
                                          ctypes.c_double(self._RWTimeout),
                                          DAQmx_Val_GroupByChannel,
                                          volt.ctypes.data_as(c_float64_p),
                                          ctypes.byref(temp), None) )
        set_voltage(numpy.array(5.))
        time.sleep(0.1)
        set_voltage(numpy.array(0.))

    def set_aom_voltage(self, volt):
        "Set output voltage of aom_volt_ch"
        temp = ctypes.c_int32()
        volt = numpy.array((volt))
        CHK( dll.DAQmxWriteAnalogF64( self.aom_volt_task,
                                      ctypes.c_int32(1), True,    # length of command, start task immediatly (True), or wait for software start (False)
                                      ctypes.c_double(self._RWTimeout),
                                      DAQmx_Val_GroupByChannel,
                                      volt.ctypes.data_as(c_float64_p),
                                      ctypes.byref(temp), None) )

    #---------------- Counter ------------------------------------------
    def startCounter(self, count_interval = 0.01):
        "old confocal code uses this function instead of start_time_trace()"
        self.start_time_trace(count_interval)

    def start_time_trace(self, count_interval):
        """Initializes and starts tasks for standard photon counting.
        count_interval defines bin length for photon counting"""
        self.pulse_out_task = ctypes.c_uint64()  #create handle for task, this task will generate pulse signal for photon counting
        self.counter_in_task = ctypes.c_uint64() #this task will count photons with binning defined by pulse task
        CHK( dll.DAQmxCreateTask('', ctypes.byref(self.pulse_out_task)) )   #create task
        CHK( dll.DAQmxCreateTask('', ctypes.byref(self.counter_in_task)) )
        self.count_freq = 1./count_interval   #counting freqeuency
        #duty_cycle = 0.5       #width of pulse (relative to count_interval) during which photons are counted

        # now create pulse signal defined by f and duty_cycle
        CHK(  dll.DAQmxCreateCOPulseChanFreq( self.pulse_out_task,      #add to this task
                                              self._trace_counter_out,  #use this counter
                                              '',                       #name
                                              DAQmx_Val_Hz, DAQmx_Val_Low, ctypes.c_double(0), #units, idle state, inital delay
                                              ctypes.c_double(self.count_freq / 2.),   #pulse frequency, divide by 2 such that length of semi period = count_interval
                                              ctypes.c_double(0.5)) ) #duty cycle of pulses, 0.5 such that high and low duration are both = count_interval
        # set up semi period width measurement in photon ticks, i.e. the width of each pulse (high and low) generated by pulse_out_task is measured in photon ticks.
        CHK(  dll.DAQmxCreateCISemiPeriodChan( self.counter_in_task,    #add to this task
                                               self._trace_counter_in,  #use this counter
                                               '',  #name
                                               ctypes.c_double(0),  #expected minimum value
                                               ctypes.c_double(self._MaxCounts/2./self.count_freq),    #expected maximum value
                                               DAQmx_Val_Ticks, '')   )    #units of width measurement, here photon ticks
        # set the pulses to counter self._trace_counter_in
        CHK(  dll.DAQmxSetCISemiPeriodTerm( self.counter_in_task, self._trace_counter_in, self._trace_counter_out+'InternalOutput')  )
        # set the timebase for width measurement as self._photon_source
        CHK(  dll.DAQmxSetCICtrTimebaseSrc( self.counter_in_task, self._trace_counter_in, self._photon_source )  )

        # set timing to continous
        CHK(  dll.DAQmxCfgImplicitTiming( self.pulse_out_task,  #define task
                                          DAQmx_Val_ContSamps,  #continous running
                                          ctypes.c_ulonglong(10000))  ) #buffer length
        CHK(  dll.DAQmxCfgImplicitTiming( self.counter_in_task,
                                          DAQmx_Val_ContSamps,
                                          ctypes.c_ulonglong(10000))  )
        # read most recent samples
        CHK( dll.DAQmxSetReadRelativeTo(self.counter_in_task, DAQmx_Val_CurrReadPos) )
        CHK( dll.DAQmxSetReadOffset(self.counter_in_task, 0) )
        #unread data in buffer will be overwritten
        CHK( dll.DAQmxSetReadOverWrite(self.counter_in_task, DAQmx_Val_DoNotOverwriteUnreadSamps) )

        CHK( dll.DAQmxStartTask(self.pulse_out_task) )
        CHK( dll.DAQmxStartTask(self.counter_in_task) )
        time.sleep(5./self.count_freq) #wait until first data point is aquired

    def count_time_trace(self, samples=1):
        """Returns latest count sample (in sounts per second) aquired by nidaq for time trace."""
        #if samples != self._n_count_samples:
        #    CHK( dll.DAQmxSetReadOffset(self.counter_in_task, -samples) )
        #    self._n_count_samples = samples
        self._count_data = numpy.empty((samples,), dtype=numpy.uint32) # count data will be written here
        n_read_samples = ctypes.c_int32() #number of samples which were read will be stored here
        CHK( dll.DAQmxReadCounterU32(self.counter_in_task   #read from this task
                                     , ctypes.c_int32(samples)    #number of samples to read
                                     , ctypes.c_double(self._RWTimeout)
                                     , self._count_data.ctypes.data_as(c_uint32_p) #write into this array
                                     , ctypes.c_uint32(samples)   #length of array to write into
                                     , ctypes.byref(n_read_samples), None) ) #number of samples which were read (should be 1 obv.)
        return self._count_data * self.count_freq    #normalize to counts per second

    def Count(self):
        "Old confocal code calls this function instead of self.count_time_trace."
        return self.count_time_trace()

    def stop_time_trace(self):
        "Clear tasks, so that counters are not in use any more."
        CHK(  dll.DAQmxClearTask(self.pulse_out_task)  )
        CHK(  dll.DAQmxClearTask(self.counter_in_task)  )

    def read_powermeter_ai(self):
        line_points = 1
        data = numpy.empty((1,line_points), dtype=numpy.float64)
        n_read_samples = ctypes.c_int32()
        CHK(
            dll.DAQmxReadAnalogF64(
                                   self.powermeter_ai_task,
                                   ctypes.c_int32(line_points),
                                   ctypes.c_double(self._RWTimeout),
                                   DAQmx_Val_GroupByChannel,
                                   data.ctypes.data_as(c_float64_p),
                                   ctypes.c_uint32(1*line_points),
                                   ctypes.byref(n_read_samples),
                                   None
                                   )
            )
        return data

    def read_aux1_ai(self,line_points=1):
        #line_points = 1
        data = numpy.empty((1,line_points), dtype=numpy.float64)
        n_read_samples = ctypes.c_int32()
        CHK(
            dll.DAQmxReadAnalogF64(
                                   self.aux1_ai_task,
                                   ctypes.c_int32(line_points),
                                   ctypes.c_double(self._RWTimeout),
                                   DAQmx_Val_GroupByChannel,
                                   data.ctypes.data_as(c_float64_p),
                                   ctypes.c_uint32(1*line_points),
                                   ctypes.byref(n_read_samples),
                                   None
                                   )
            )
        return data

    def read_aux2_ai(self,line_points=1):
        #line_points = 1
        data = numpy.empty((1,line_points), dtype=numpy.float64)
        n_read_samples = ctypes.c_int32()
        CHK(
            dll.DAQmxReadAnalogF64(
                                   self.aux2_ai_task,
                                   ctypes.c_int32(line_points),
                                   ctypes.c_double(self._RWTimeout),
                                   DAQmx_Val_GroupByChannel,
                                   data.ctypes.data_as(c_float64_p),
                                   ctypes.c_uint32(1*line_points),
                                   ctypes.byref(n_read_samples),
                                   None
                                   )
            )
        return data


    #-------------- Scanner ----------------------------------
    def scanner_pos_to_volt(self, Pos):
        x = self._scanner_xrange
        y = self._scanner_yrange
        z = self._scanner_zrange
        return numpy.vstack(
            (
            -1. * (20.0 / (x[1]-x[0]) * (Pos[0]-x[0]) - 10),
            20.0 / (y[1]-y[0]) * (Pos[1]-y[0]) - 10,
            -1. * (20.0 / (z[1]-z[0]) * (Pos[2]-z[0]) - 10)
            )
        )

    def scanner_volt_to_pos(self, Volt):
        xpoint1 = (self._scanner_xvolt_range[0],self._scanner_xrange[0]) #(u1,x1)
        xpoint2 = (self._scanner_xvolt_range[1],self._scanner_xrange[1]) #(u2,x2)

        ypoint1 = (self._scanner_yvolt_range[0],self._scanner_yrange[0]) #(u1,z1)
        ypoint2 = (self._scanner_yvolt_range[1],self._scanner_yrange[1]) #(u2,z2)

        zpoint1 = (self._scanner_zvolt_range[0],self._scanner_zrange[0]) #(u1,z1)
        zpoint2 = (self._scanner_zvolt_range[1],self._scanner_zrange[1]) #(u2,z2)

        return numpy.vstack( ( xpoint1[1]  + ((Volt[0]-xpoint1[0]) * (xpoint2[1]-xpoint1[1])/(xpoint2[0]-xpoint1[0])),
                               ypoint1[1]  + ((Volt[1]-ypoint1[0]) * (ypoint2[1]-ypoint1[1])/(ypoint2[0]-ypoint1[0])),
                               zpoint1[1]  + ((Volt[2]-zpoint1[0]) * (zpoint2[1]-zpoint1[1])/(zpoint2[0]-zpoint1[0]))
                               ) )

        #x = self._scanner_xrange
        #y = self._scanner_yrange
        #z = self._scanner_zrange
        #return numpy.vstack( ( (Volt[0]/20.0+0.5) * (x[1]-x[0]),
        #                       (Volt[1]/20.0+0.5) * (y[1]-y[0]),
        #                       (Volt[2]/20.0+0.5) * (z[1]-z[0])  ) )

    def calibrate_scanner_remote(self):
        voltrange = []
        self.scanner_set_pos(self._scanner_xrange[0],self._scanner_yrange[0],self._scanner_zrange[0])
        time.sleep(0.5)

        voltrange.append(self.read_scanner_ai())

        self.scanner_set_pos(self._scanner_xrange[1],self._scanner_yrange[1],self._scanner_zrange[1])

        time.sleep(0.5)

        voltrange.append(self.read_scanner_ai())

        self._scanner_xvolt_range = []
        self._scanner_yvolt_range = []
        self._scanner_zvolt_range = []

        self._scanner_xvolt_range.extend([voltrange[0][0],voltrange[1][0]])
        self._scanner_yvolt_range.extend([voltrange[0][1],voltrange[1][1]])
        self._scanner_zvolt_range.extend([voltrange[0][2],voltrange[1][2]])
        self.calibrated = True

    def write_scanner_ao(self, data, length=1 ,start=False):
        self._AONwritten = ctypes.c_int32()
        CHK( dll.DAQmxWriteAnalogF64( self.scanner_ao_task,
                                      ctypes.c_int32(length),
                                      start,    # length of command, start task immediatly (True), or wait for software start (False)
                                      ctypes.c_double(self._RWTimeout),
                                      DAQmx_Val_GroupByChannel,
                                      data.ctypes.data_as(c_float64_p),
                                      None, #ctypes.byref(self._AONwritten),
                                      None ))
        #return self._AONwritten.value

    def read_scanner_ai(self):
        line_points = 1
        data = numpy.empty((3,line_points), dtype=numpy.float64)
        n_read_samples = ctypes.c_int32()
        CHK(
            dll.DAQmxReadAnalogF64(
                                   self.scanner_ai_task,
                                   ctypes.c_int32(line_points),
                                   ctypes.c_double(self._RWTimeout),
                                   DAQmx_Val_GroupByChannel,
                                   data.ctypes.data_as(c_float64_p),
                                   ctypes.c_uint32(3*line_points),
                                   ctypes.byref(n_read_samples),
                                   None
                                   )
            )
        return data

    def get_scanner_pos(self):
        if not self.calibrated:
            self.calibrate_scanner_remote()
        return self.scanner_volt_to_pos(self.read_scanner_ai())

    def move_x_to_0(self):
        self._reload_x = self.get_scanner_pos()[0]
        self.scanner_setx(0)

    def reload_x(self):
        self.scanner_setx(self._reload_x)

    def scanner_setx(self, x):
        """Move stage to x, y, z"""
        if self.function_state != 'scan':
            self.write_scanner_ao(self.scanner_pos_to_volt((x, self.scanner_y, self.scanner_z)), start=True)
        self.scanner_x = x

    def scanner_sety(self, y):
        """Move stage to x, y, z"""
        if self.function_state != 'scan':
            self.write_scanner_ao(self.scanner_pos_to_volt((self.scanner_x, y, self.scanner_z)), start=True)
        self.scanner_y = y

    def scanner_setz(self, z):
        """Move stage to x, y, z """
        if self.function_state != 'scan':
            self.write_scanner_ao(self.scanner_pos_to_volt((self.scanner_x, self.scanner_y, z)), start=True)
        self.scanner_z = z

    def scanner_set_pos(self, x, y, z):
        """Move stage to x, y, z"""
        if self.function_state != 'scan':
            self.write_scanner_ao(self.scanner_pos_to_volt((x, y, z)), start=True)
        self.scanner_x, self.scanner_y, self.scanner_z = x, y, z

    def init_scan(self, settling_time, count_time):
        """set up tasks for scanning."""
        #if self.function_state != 'idle':
        #    for i in range(10):
        #        time.sleep(0.1)
        #        if self.function_state == 'idle':
        #            break
        #    else:
        #        print 'error: nidaq state is not idle, counters are used by other function'
        #        return -1
        self.function_state = 'scan'
        self.scanner_pulse_task = ctypes.c_uint64()  #create handle for task, this task will generate pulse signal for photon counting
        self.scanner_count_task = ctypes.c_uint64() #this task will count photons with binning defined by pulse task
        CHK( dll.DAQmxCreateTask('', ctypes.byref(self.scanner_pulse_task)) )   #create task
        CHK( dll.DAQmxCreateTask('', ctypes.byref(self.scanner_count_task)) )
        # create pulses (clock) for scanning.
        self.scan_freq = 1./(count_time+settling_time)   #counting freqeuency
        duty_cycle = self.scan_freq * count_time     #width of pulse (relative to count_interval) during which photons are counted
        self.scan_duty_cycle = duty_cycle
        #now create pulse signal defined by f and duty_cycle
        CHK(  dll.DAQmxCreateCOPulseChanFreq( self.scanner_pulse_task,      #add to this task
                                              self._function_counter_out,   #use this counter
                                              '',                       #name
                                              DAQmx_Val_Hz, DAQmx_Val_Low, ctypes.c_double(0), #units, idle state, inital delay
                                              ctypes.c_double(self.scan_freq),   #pulse frequency
                                              ctypes.c_double(duty_cycle)) ) #duty cycle of pulses
        #set up pulse width measurement in photon ticks, i.e. the width of each pulse generated by pulse_out_task is measured in photon ticks.
        CHK(  dll.DAQmxCreateCIPulseWidthChan( self.scanner_count_task,    #add to this task
                                               self._function_counter_in,  #use this counter
                                               '',  #name
                                               ctypes.c_double(0),  #expected minimum value
                                               ctypes.c_double(self._MaxCounts*duty_cycle/self.scan_freq),    #expected maximum value
                                               DAQmx_Val_Ticks,     #units of width measurement, here photon ticks
                                               DAQmx_Val_Rising, '')   ) #start pulse width measurement on rising edge
        #set the pulses to counter self._trace_counter_in
        CHK(  dll.DAQmxSetCIPulseWidthTerm( self.scanner_count_task, self._function_counter_in, self._function_counter_out+'InternalOutput')  )
        #set the timebase for width measurement as self._photon_source
        CHK(  dll.DAQmxSetCICtrTimebaseSrc( self.scanner_count_task, self._function_counter_in, self._photon_source )  )
        self.line_points = -1   #reset variable, scan length is checked and adjusted for each line scan
        return 0

    def stop_scan(self):
        "Clear tasks, so that counters are not in use any more."
        self.function_state = 'idle'
        CHK( dll.DAQmxSetSampTimingType( self.scanner_ao_task, DAQmx_Val_OnDemand)  )
        CHK(  dll.DAQmxClearTask(self.scanner_pulse_task)  )
        CHK(  dll.DAQmxClearTask(self.scanner_count_task)  )

    def set_scan_length(self, N):
        """Set length for line scan, i.e. number of clock pulses, number of count samples etc."""
        CHK( dll.DAQmxSetSampTimingType( self.scanner_ao_task, DAQmx_Val_SampClk)  )  # set task timing to use a sampling clock
        if N < numpy.inf:
            CHK( dll.DAQmxCfgSampClkTiming( self.scanner_ao_task,   # set up sample clock for task timing
                                            self._function_counter_out+'InternalOutput',       # use these pulses as clock
                                            ctypes.c_double(self.scan_freq), # maximum expected clock frequency
                                            DAQmx_Val_Falling, DAQmx_Val_FiniteSamps, # genarate sample on falling edge, generate finite number of samples
                                            ctypes.c_ulonglong(N)) ) # samples to generate

        # set timing for scanner pulse and count task.
        # Because clock pulse starts high, but position is set on falling edge, first count will be ignored
        CHK(  dll.DAQmxCfgImplicitTiming( self.scanner_pulse_task, DAQmx_Val_FiniteSamps, ctypes.c_ulonglong(N+1))  )
        CHK(  dll.DAQmxCfgImplicitTiming( self.scanner_count_task, DAQmx_Val_FiniteSamps, ctypes.c_ulonglong(N+1))  )
        # read samples from beginning of acquisition, do not overwrite
        CHK( dll.DAQmxSetReadRelativeTo(self.scanner_count_task, DAQmx_Val_CurrReadPos) )
        CHK( dll.DAQmxSetReadOffset(self.scanner_count_task, 1) ) # do not read first sample
        CHK( dll.DAQmxSetReadOverWrite(self.scanner_count_task, DAQmx_Val_DoNotOverwriteUnreadSamps) )
        self._scan_count_timeout = 2. * (N+1) / self.scan_freq
        self.line_points = N

    def set_scan_read_length(self, N, trend = 'positive'):
        """Set length for line scan and position readout, i.e. number of clock pulses, number of count samples etc."""
        CHK( dll.DAQmxSetSampTimingType( self.scanner_ao_task, DAQmx_Val_SampClk)  )  # set task timing to use a sampling clock
        if N < numpy.inf:
            CHK( dll.DAQmxCfgSampClkTiming( self.scanner_ao_task,   # set up sample clock for task timing
                                            self._function_counter_out+'InternalOutput',       # use these pulses as clock
                                            ctypes.c_double(self.scan_freq), # maximum expected clock frequency
                                            DAQmx_Val_Falling, DAQmx_Val_FiniteSamps, # genarate sample on falling edge, generate finite number of samples
                                            ctypes.c_ulonglong(N)) ) # samples to generate

        # set timing for scanner pulse and count task.
        # Because clock pulse starts high, but position is set on falling edge, first count will be ignored
        CHK(  dll.DAQmxCfgImplicitTiming( self.scanner_pulse_task, DAQmx_Val_FiniteSamps, ctypes.c_ulonglong(N+1))  )
        CHK(  dll.DAQmxCfgImplicitTiming( self.scanner_count_task, DAQmx_Val_FiniteSamps, ctypes.c_ulonglong(N+1))  )
        # read samples from beginning of acquisition, do not overwrite
        CHK( dll.DAQmxSetReadRelativeTo(self.scanner_count_task, DAQmx_Val_CurrReadPos) )
        CHK( dll.DAQmxSetReadOffset(self.scanner_count_task, 1) ) # do not read first sample
        CHK( dll.DAQmxSetReadOverWrite(self.scanner_count_task, DAQmx_Val_DoNotOverwriteUnreadSamps) )
        self._scan_count_timeout = 2. * (N+1) / self.scan_freq
        self.line_points = N

    #def adjust_scan_timing(self, SettlingTime, CountTime):
    #    """Use to adjust scan timing during refocus when switching from xy scan to z scan."""
    #    if self.function_state != 'scan':
    #        print 'warning, cannot adjust scan timing if scan is not started'
    #        return
    #    self.scan_freq = 1. / ( CountTime + SettlingTime )
    #    self.scan_duty_cycle = CountTime * self.scan_freq
    #    CHK( dll.DAQmxSetCOPulseFreq( self.scanner_pulse_task, self._CODevice, ctypes.c_double(self.scan_freq)  )  )
    #    CHK( dll.DAQmxSetCOPulseDutyCyc( self.scanner_pulse_task, self._CODevice, ctypes.c_double(self.scan_duty_cycle)  )   )

    def config_readout_sampling(self, N, trend_is_positive):
        CHK( dll.DAQmxSetSampTimingType( self.scanner_ai_task, DAQmx_Val_SampClk)  )  # set task timing to use a sampling clock

        if trend_is_positive:
            active_edge = DAQmx_Val_Falling
        else:
            active_edge = DAQmx_Val_Rising

        CHK( dll.DAQmxCfgSampClkTiming( self.scanner_ai_task,   # set up sample clock for task timing
                                        self._function_counter_out+'InternalOutput',       # use these pulses as clock
                                        ctypes.c_double(self.scan_freq), # maximum expected clock frequency
                                        active_edge, DAQmx_Val_FiniteSamps, # genarate sample on falling edge, generate finite number of samples
                                        ctypes.c_ulonglong(N)) ) # samples to read

    def scan_line(self, Line):
        """Perform a line scan."""
        # check if length setup is correct, if not, adjust.
        line_points = Line.shape[1]
        if self.line_points != line_points:
            self.set_scan_length(line_points)

        # set up scanner ao channel
        self.write_scanner_ao( self.scanner_pos_to_volt(Line), length=self.line_points )
        #start tasks
        CHK( dll.DAQmxStartTask(self.scanner_ao_task) )
        CHK( dll.DAQmxStartTask(self.scanner_count_task) )
        CHK( dll.DAQmxStartTask(self.scanner_pulse_task) )

        CHK( dll.DAQmxWaitUntilTaskDone(self.scanner_count_task, ctypes.c_double(self._scan_count_timeout))  )

        self._scan_data = numpy.empty((self.line_points,), dtype=numpy.uint32) # count data will be written here
        n_read_samples = ctypes.c_int32() #number of samples which were read will be stored here
        CHK( dll.DAQmxReadCounterU32(self.scanner_count_task   #read from this task
                                     , ctypes.c_int32(self.line_points)    #read number of "line_points" samples
                                     , ctypes.c_double(self._RWTimeout)
                                     , self._scan_data.ctypes.data_as(c_uint32_p) #write into this array
                                     , ctypes.c_uint32(self.line_points)   #length of array to write into
                                     , ctypes.byref(n_read_samples), None) ) #number of samples which were read

        CHK( dll.DAQmxStopTask(self.scanner_count_task) )
        CHK( dll.DAQmxStopTask(self.scanner_pulse_task) )
        CHK( dll.DAQmxStopTask(self.scanner_ao_task) )

        return self._scan_data * self.scan_freq / self.scan_duty_cycle # normate to counts per second

    def scan_read_line(self, Line):
        """Perform a line scan and read."""
        # check if length setup is correct, if not, adjust.
        line_points = Line.shape[1]
        if Line[0][0] <= Line[0][-1]:
            trend_is_positive = True
        else:
            trend_is_positive = False

        if self.line_points != line_points:
            self.set_scan_length(line_points)

        self.config_readout_sampling(line_points,trend_is_positive)

        # set up scanner ao channel
        self.write_scanner_ao( self.scanner_pos_to_volt(Line), length=self.line_points )
        #start tasks
        CHK( dll.DAQmxStartTask(self.scanner_ao_task) )
        CHK( dll.DAQmxStartTask(self.scanner_ai_task) )
        CHK( dll.DAQmxStartTask(self.scanner_count_task) )
        CHK( dll.DAQmxStartTask(self.scanner_pulse_task) )

        CHK( dll.DAQmxWaitUntilTaskDone(self.scanner_count_task, ctypes.c_double(self._scan_count_timeout))  )

        self._scan_data = numpy.empty((self.line_points,), dtype=numpy.uint32) # count data will be written here
        self._read_pos_data = numpy.empty((3,self.line_points), dtype=numpy.float64) # read position data will be written here
        n_read_samples = ctypes.c_int32() #number of samples which were read will be stored here
        CHK( dll.DAQmxReadCounterU32(self.scanner_count_task   #read from this task
                                     , ctypes.c_int32(self.line_points)    #read number of "line_points" samples
                                     , ctypes.c_double(self._RWTimeout)
                                     , self._scan_data.ctypes.data_as(c_uint32_p) #write into this array
                                     , ctypes.c_uint32(self.line_points)   #length of array to write into
                                     , ctypes.byref(n_read_samples), None) ) #number of samples which were read
        # readout voltages positions here
        CHK(
            dll.DAQmxReadAnalogF64(
                                   self.scanner_ai_task,
                                   ctypes.c_int32(self.line_points),
                                   ctypes.c_double(self._RWTimeout),
                                   DAQmx_Val_GroupByChannel,
                                   self._read_pos_data.ctypes.data_as(c_float64_p),
                                   ctypes.c_uint32(3*self.line_points),
                                   ctypes.byref(n_read_samples),
                                   None
                                   )
            )

        CHK( dll.DAQmxStopTask(self.scanner_count_task) )
        CHK( dll.DAQmxStopTask(self.scanner_pulse_task) )
        CHK( dll.DAQmxStopTask(self.scanner_ao_task) )
        CHK( dll.DAQmxStopTask(self.scanner_ai_task) )

        return self.scanner_volt_to_pos(self._read_pos_data), self._scan_data * self.scan_freq / self.scan_duty_cycle  # normalize to counts per second, return also voltages/positions


    #------------------------- Gated photon counting -----------------------------------
    def start_gated_counting(self, buffer_length):
        """Initializes and starts task for external gated photon counting."""
        #if self.function_state != 'idle':
        #    for i in range(10):
        #        time.sleep(0.1)
        #        if self.function_state == 'idle':
        #            break
        #    else:
        #        print 'error: nidaq state is not idle, counters are used by other function'
        #        return -1
        self.function_state = 'gated'
        self.gated_count_task = ctypes.c_uint64() #this task will count photons with binning defined by pulse task
        CHK( dll.DAQmxCreateTask('', ctypes.byref(self.gated_count_task)) )   #create task
        #set up pulse width measurement in photon ticks, i.e. the width of each pulse generated by pulse_out_task is measured in photon ticks.
        CHK(  dll.DAQmxCreateCIPulseWidthChan( self.gated_count_task,    #add to this task
                                               self._function_counter_in,  #use this counter
                                               '',  #name
                                               ctypes.c_double(0),  #expected minimum value
                                               ctypes.c_double(self._MaxCounts),    #expected maximum value
                                               DAQmx_Val_Ticks,     #units of width measurement, here photon ticks
                                               DAQmx_Val_Rising, '')   ) #start pulse width measurement on rising edge
        #set the pulses to counter self._function_counter_in
        CHK(  dll.DAQmxSetCIPulseWidthTerm( self.gated_count_task, self._function_counter_in, self.gate_in_channel)  )
        #set the timebase for width measurement as self._photon_source
        CHK(  dll.DAQmxSetCICtrTimebaseSrc( self.gated_count_task, self._function_counter_in, self._photon_source )  )

        #set timing to continous
        CHK(  dll.DAQmxCfgImplicitTiming( self.gated_count_task,
                                          DAQmx_Val_ContSamps,
                                          ctypes.c_ulonglong(buffer_length))  )
        #read most recent samples
        CHK( dll.DAQmxSetReadRelativeTo(self.gated_count_task, DAQmx_Val_CurrReadPos) )
        CHK( dll.DAQmxSetReadOffset(self.gated_count_task, 0) )
        #unread data in buffer is not overwritten
        CHK( dll.DAQmxSetReadOverWrite(self.gated_count_task, DAQmx_Val_DoNotOverwriteUnreadSamps) )

        CHK( dll.DAQmxStartTask(self.gated_count_task) )
        return 0

    def read_gated_counts(self, samples, timeout=600):
        """Returns latest count samples aquired by gated photon counting."""
        self._gated_count_data = numpy.empty((samples,), dtype=numpy.uint32) # count data will be written here
        n_read_samples = ctypes.c_int32() #number of samples which were read will be stored here
        CHK( dll.DAQmxReadCounterU32(self.gated_count_task   #read from this task
                                     , ctypes.c_int32(samples)    #read number samples
                                     , ctypes.c_double(timeout)     # Timeout of read. Since nidaq waits for all samples to be aquired, make sure this is long enough.
                                     , self._gated_count_data.ctypes.data_as(c_uint32_p) #write into this array
                                     , ctypes.c_uint32(samples)   #length of array to write into
                                     , ctypes.byref(n_read_samples), None) ) #number of samples which were read
        return self._gated_count_data[:]

    def stop_gated_counting(self):
        "Clear tasks, so that counters are not in use any more."
        self.function_state = 'idle'
        CHK(  dll.DAQmxClearTask(self.gated_count_task)  )


    #--------------- ODMR -------------------------------------
    def init_odmr(self, settling_time, count_time, init_pulse=False):
        """Start task for odmr measurement.
        Use pulse width measurment with two counters: Counter 1 generates pulses for gate of counter two,
        which measures width of pulses in terms of photon ticks."""
        #if self.function_state != 'idle':
        #    for i in range(10):
        #        time.sleep(0.1)
        #        if self.function_state == 'idle':
        #            break
        #    else:
        #        print 'error: nidaq state is not idle, counters are used by other function'
        #        return -1
        self.function_state = 'odmr'
        self.odmr_pulse_task = ctypes.c_uint64()  #create handle for task, this task will generate pulse signal for photon counting
        self.odmr_count_task = ctypes.c_uint64() #this task will count photons with binning defined by pulse task
        CHK( dll.DAQmxCreateTask('', ctypes.byref(self.odmr_pulse_task)) )   #create task
        CHK( dll.DAQmxCreateTask('', ctypes.byref(self.odmr_count_task)) )
        # create pulses (clock) for odmr.
        self.odmr_freq = 1./(count_time+settling_time)   #counting freqeuency
        duty_cycle = self.odmr_freq * count_time     #width of pulse (relative to count_interval) during which photons are counted
        self.odmr_duty_cycle = duty_cycle
        #now create pulse signal defined by f and duty_cycle
        CHK(  dll.DAQmxCreateCOPulseChanFreq( self.odmr_pulse_task,      #add to this task
                                              self._function_counter_out,   #use this counter
                                              '',                       #name
                                              DAQmx_Val_Hz, DAQmx_Val_High, ctypes.c_double(0), #units, idle state, inital delay
                                              ctypes.c_double(self.odmr_freq),   #pulse frequency
                                              ctypes.c_double(1-duty_cycle)) ) #duty cycle of pulses
        if init_pulse == False:
            CHK( dll.DAQmxStartTask(self.odmr_pulse_task) ) #start and stop pulse task to correctly initiate idle state high voltage.
            CHK( dll.DAQmxStopTask(self.odmr_pulse_task) )  #otherwise, it will be low until task starts, and MW will receive wrong pulses.
        #set up pulse width measurement in photon ticks, i.e. the width of each pulse generated by pulse_out_task is measured in photon ticks.
        CHK(  dll.DAQmxCreateCIPulseWidthChan( self.odmr_count_task,    #add to this task
                                               self._function_counter_in,  #use this counter
                                               '',  #name
                                               ctypes.c_double(0),  #expected minimum value
                                               ctypes.c_double(self._MaxCounts*duty_cycle/self.odmr_freq),    #expected maximum value
                                               DAQmx_Val_Ticks,     #units of width measurement, here photon ticks
                                               DAQmx_Val_Falling, '')   ) #start pulse width measurement on falling edge
        #set the pulses to counter self._trace_counter_in
        CHK(  dll.DAQmxSetCIPulseWidthTerm( self.odmr_count_task, self._function_counter_in, self._function_counter_out+'InternalOutput')  )
        #set the timebase for width measurement as self._photon_source
        CHK(  dll.DAQmxSetCICtrTimebaseSrc( self.odmr_count_task, self._function_counter_in, self._photon_source )  )
        Connect(self._function_counter_out+'InternalOutput', self._odmr_trig_channel)
        self.odmr_points = -1   #reset variable, odmr length is checked and adjusted for each count operation
        return 0

    def stop_odmr(self):
        self.function_state = 'idle'
        Disconnect(self._function_counter_out+'InternalOutput', self._odmr_trig_channel)
        CHK(  dll.DAQmxClearTask(self.odmr_pulse_task)  )
        CHK(  dll.DAQmxClearTask(self.odmr_count_task)  )

    def set_odmr_length(self, N):
        """Set length for line scan, i.e. number of clock pulses, number of count samples etc."""
        # set timing for odmr pulse and count task.
        # Contrary to scanner, here first frequency points should be set without trigger, i.e. first count sample can be used. After last count sample pulse goes low, i.e. frequency generator will get trigger and return to list pos 1
        CHK(  dll.DAQmxCfgImplicitTiming( self.odmr_pulse_task, DAQmx_Val_FiniteSamps, ctypes.c_ulonglong(N))  )
        CHK(  dll.DAQmxCfgImplicitTiming( self.odmr_count_task, DAQmx_Val_FiniteSamps, ctypes.c_ulonglong(N))  )
        # read samples from beginning of acquisition, do not overwrite
        CHK( dll.DAQmxSetReadRelativeTo(self.odmr_count_task, DAQmx_Val_CurrReadPos) )
        CHK( dll.DAQmxSetReadOffset(self.odmr_count_task, 0) )
        CHK( dll.DAQmxSetReadOverWrite(self.odmr_count_task, DAQmx_Val_DoNotOverwriteUnreadSamps) )
        self._odmr_count_timeout = 2. * (N) / self.odmr_freq
        self.odmr_points = N

    def count_odmr(self, points):
        """Count one run of odmr."""
        # check if length setup is correct, if not, adjust.
        if self.odmr_points != points:
            self.set_odmr_length(points)

        #start tasks
        CHK( dll.DAQmxStartTask(self.odmr_count_task) )
        CHK( dll.DAQmxStartTask(self.odmr_pulse_task) )

        CHK( dll.DAQmxWaitUntilTaskDone(self.odmr_count_task, ctypes.c_double(self._odmr_count_timeout))  )

        self._odmr_data = numpy.empty((self.odmr_points,), dtype=numpy.uint32) # count data will be written here
        n_read_samples = ctypes.c_int32() #number of samples which were read will be stored here
        CHK( dll.DAQmxReadCounterU32(self.odmr_count_task   #read from this task
                                     , ctypes.c_int32(self.odmr_points)    #read number of "line_points" samples
                                     , ctypes.c_double(self._RWTimeout)
                                     , self._odmr_data.ctypes.data_as(c_uint32_p) #write into this array
                                     , ctypes.c_uint32(self.odmr_points)   #length of array to write into
                                     , ctypes.byref(n_read_samples), None) ) #number of samples which were read

        CHK( dll.DAQmxStopTask(self.odmr_count_task) )
        CHK( dll.DAQmxStopTask(self.odmr_pulse_task) )

        return self._odmr_data * self.odmr_freq / self.odmr_duty_cycle # normate to counts per second
    
    