import numpy
import ctypes
import time

dll = ctypes.windll.LoadLibrary('nicaiu.dll')

import logging


DAQmx_Val_ContSamps             = 10123
DAQmx_Val_FiniteSamps           = 10178
DAQmx_Val_Hz                    = 10373
DAQmx_Val_Low                   = 10214
DAQmx_Val_MostRecentSamp        = 10428
DAQmx_Val_OverwriteUnreadSamps  = 10252
DAQmx_Val_Rising                = 10280
DAQmx_Val_Ticks                 = 10304

def CHK(err):
    """a simple error checking routine"""
    if err < 0:
        buf_size = 1000
        buf = ctypes.create_string_buffer('\000' * buf_size)
        dll.DAQmxGetErrorString(err,ctypes.byref(buf),buf_size)
        raise RuntimeError('nidaq call failed with error %d: %s'%(err,repr(buf.value)))


class Counter(  ):
    
    def __init__(self, counter_out_device, counter_in_device, input_pad, bin_width, length):
        self.length = length
        self.bin_width = bin_width
        
        f = 1./bin_width
        buffer_size = max(1000, length)

        self.COTask = ctypes.c_ulong()
        self.CITask = ctypes.c_ulong()

        CHK(  dll.DAQmxCreateTask('', ctypes.byref(self.COTask))  )
        CHK(  dll.DAQmxCreateTask('', ctypes.byref(self.CITask))  )

        CHK(  dll.DAQmxCreateCOPulseChanFreq( self.COTask,
                                              counter_out_device, '',
                                              DAQmx_Val_Hz, DAQmx_Val_Low, ctypes.c_double(0),
                                              ctypes.c_double(f),
                                              ctypes.c_double(0.9) )
        )        

        CHK(  dll.DAQmxCreateCIPulseWidthChan( self.CITask,
                                               counter_in_device, '',
                                               ctypes.c_double(0),
                                               ctypes.c_double(1e7),
                                               DAQmx_Val_Ticks, DAQmx_Val_Rising, '')
        )

        CHK(  dll.DAQmxSetCIPulseWidthTerm( self.CITask, counter_in_device, counter_out_device+'InternalOutput' )  )
        CHK(  dll.DAQmxSetCICtrTimebaseSrc( self.CITask, counter_in_device, input_pad )  )

        CHK(  dll.DAQmxCfgImplicitTiming( self.COTask, DAQmx_Val_ContSamps, ctypes.c_ulonglong(buffer_size))  )
        CHK(  dll.DAQmxCfgImplicitTiming( self.CITask, DAQmx_Val_ContSamps, ctypes.c_ulonglong(buffer_size))  )

        # read most recent samples, overwrite buffer
        CHK( dll.DAQmxSetReadRelativeTo(self.CITask, DAQmx_Val_MostRecentSamp) )
        CHK( dll.DAQmxSetReadOffset(self.CITask, -length) )
        CHK( dll.DAQmxSetReadOverWrite(self.CITask, DAQmx_Val_OverwriteUnreadSamps) )
        
        CHK(  dll.DAQmxStartTask(self.COTask)  )
        CHK(  dll.DAQmxStartTask(self.CITask)  )
        
        self.n_read = ctypes.c_int32()
        self.data = numpy.empty((length,), dtype=numpy.uint32)
        self.timeout = 4*length*bin_width

    def getData(self):
        try:
            CHK(  dll.DAQmxReadCounterU32(self.CITask,
                                          ctypes.c_int32(self.length),
                                          ctypes.c_double(self.timeout),
                                          self.data.ctypes.data,
                                          ctypes.c_uint32(self.length),
                                          ctypes.byref(self.n_read), None)
            )
        except:
            time.sleep(self.timeout)
            CHK(  dll.DAQmxReadCounterU32(self.CITask,
                                          ctypes.c_int32(self.length),
                                          ctypes.c_double(self.timeout),
                                          self.data.ctypes.data,
                                          ctypes.c_uint32(self.length),
                                          ctypes.byref(self.n_read), None)
            )
        return self.data
    
    def __del__(self):
        try:
            CHK(  dll.DAQmxStopTask(self.COTask)  )
            CHK(  dll.DAQmxStopTask(self.CITask)  )
            CHK(  dll.DAQmxClearTask(self.COTask)  )
            CHK(  dll.DAQmxClearTask(self.CITask)  )
        except:
            pass

if __name__ == '__main__':
    c = Counter( counter_out_device='/Dev1/Ctr2', counter_in_device='/Dev1/Ctr3', input_pad='/Dev1/PFI3', bin_width=0.01, length=1000)
    
