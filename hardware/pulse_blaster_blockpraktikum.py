import ctypes
import numpy as np

dll = ctypes.cdll.LoadLibrary('spinapi.dll')

PULSE_PROGRAM  = 0
CONTINUE       = 0
STOP           = 1
LOOP           = 2
END_LOOP       = 3
LONG_DELAY     = 7
BRANCH         = 6
#ON             = 6<<21 # this doesn't work even though it is according to documentation
ON             = 0xE00000

def chk(err):
    """a simple error checking routine"""
    if err < 0:
        dll.pb_get_error.restype = ctypes.c_char_p
        err_str = dll.pb_get_error()
        raise RuntimeError('PulseBlaster error: %s' % err_str)
    return err

def factor(x):
    i = 256
    while i > 4:
        if x % i == 0:
            return i, x/i
        i -= 1
    return 1, x

def is_channel_list(item):
    """
    Returns True if the argument is a channel list.
    
    We assume that this is the case if the argument can be indexed but is empty
    or its first element is a string.
    """
    try:
        element = item[0]
        if type(element) == str:
            return True
        else:
            return False
    except IndexError: # empty list or empty tuple, so it must be a pulse
        return True
    
class PulseBlaster():

    """
    Basic control of a spin core pulse Blaster
    """
    
    def __init__(self,
                 clock=500.0,
                 channel_map = {'ch0':0,'ch1':1,'ch2':2,'ch3':3,'ch4':4,'ch5':5,'ch6':6,'ch7':7,'ch8':8,'ch9':9,'ch10':10,'ch11':11,},
                 ):
        self.channel_map = channel_map
        self.clock = clock
        self.dt = 1000./clock

    def High(self, channels):
        """Set specified channels to high, all others to low."""
        chk(dll.pb_init())
        chk(dll.pb_set_clock(ctypes.c_double(self.clock)))
        chk(dll.pb_start_programming(PULSE_PROGRAM))
        chk(dll.pb_inst_pbonly(ON|self.flags(channels), CONTINUE, None, ctypes.c_double(100)))
        chk(dll.pb_inst_pbonly(ON|self.flags(channels), STOP, None, ctypes.c_double(100)))
        chk(dll.pb_stop_programming())
        chk(dll.pb_start())
        chk(dll.pb_close())

    def Sequence(self, sequence, loop=np.inf):
        """Run sequence of instructions"""
        chk(dll.pb_init())
        chk(dll.pb_set_clock(ctypes.c_double(self.clock)))
        chk(dll.pb_start_programming(PULSE_PROGRAM))

        self.write_loop(sequence, loop)
        
        chk(dll.pb_stop_programming())
        chk(dll.pb_start())
        chk(dll.pb_close())

    def write_loop(self, sequence, loop=np.inf):
        """
        Write a loop (finite or infinite repetition). The first and last items inside the loop must be a pulse.
        All other items can be either a pulse or a sequence, that will be interpreted as sub loop.
        """
        item, value = sequence[0]
        if not is_channel_list(item):
            raise ValueError('First item inside a loop must be a pulse, not a sequence.')
        channel_bits = self.flags(item)
        # distinguish cases that we have a loop start or a normal pulse
        if loop < 1:
            raise ValueError('Loop count smaller than one.')
        elif loop==np.inf or loop==1: # first instruction is not a loop start but a normal command (last command will be stop or branch)
            start_label = chk(dll.pb_inst_pbonly( ON|channel_bits, CONTINUE, None, ctypes.c_double( value ) ))
        elif loop<=2**16: # finite repetitions, create a loop start
            start_label = chk(dll.pb_inst_pbonly( ON|channel_bits, LOOP, loop, ctypes.c_double( value ) ))
        else: # finite loop but loop count too large
            raise ValueError('Loop count too large.')
        # write the main body of the sequence
        for item, value in sequence[1:-1]:
            if not is_channel_list(item): # assume that this is a sub loop
                self.write_loop(item, value)
            else:
                channel_bits = self.flags(item)
                chk(dll.pb_inst_pbonly( ON|channel_bits, CONTINUE, None, ctypes.c_double( value ) ))
        item, value = sequence[-1]
        if not is_channel_list(item):
            raise ValueError('Last item inside a loop must be a pulse, not a sequence.')
        #write loop end instruction
        channel_bits = self.flags(item)
        if loop==np.inf: # branch to beginning of loop
            stop_label = chk(dll.pb_inst_pbonly( ON|channel_bits, BRANCH, start_label, ctypes.c_double( value ) ))
        elif loop>1:
            stop_label = chk(dll.pb_inst_pbonly( ON|channel_bits, END_LOOP, start_label, ctypes.c_double( value ) ))
        elif loop==1:
            chk(dll.pb_inst_pbonly( ON|channel_bits, CONTINUE, None, ctypes.c_double( 100 ) ))
            stop_label = chk(dll.pb_inst_pbonly( ON|channel_bits, STOP, None, ctypes.c_double( 100 ) ))        
        else:
            raise ValueError('Loop count too large or zero or negative.')
        if stop_label > 2**12 - 2:
            raise ValueError('Maximum number of commands %i exceeded.'%(2**12-2))
                    
    def flags(self,channels):
        """
        Converts the 'channels' argument into an integer. For each channel,
        the corresponding bit in this integer is high.
        
        Input:
            channels    can be an integer, a single string or a list of strings.
                        if it is an integer, this is directly used as the channel word.
                        if it is a single string, a channel word is returned that contains
                        zeros everywhere, except for the channel specified by the channel name.
                        It it is a list of strings, a channel word is returned that contains
                        a '1' at each specified channel.
        Output:
            bits        integer, the channel word
        """
        if type(channels) == int:
            return channels
        elif type(channels) == str:
            channels = [channels]
        bits = 0
        for channel in channels:
            bits = bits | 1<<self.channel_map[channel]
        return bits

    def Light(self):
        self.High(['laser'])

    def Night(self):
        self.High([])

    def Open(self):
        self.High(['laser','mw'])


if __name__ == '__main__':

    pulse_blaster=PulseBlaster()
    
    sub = [ (['ch11'], 10.), ([], 10.), ]
    
    sequence = [ (['ch0','ch11'], 10.), ([], 10.), (sub,6), (['ch11'], 1000.), ([], 1000.),]
    
    pulse_blaster.High(['ch11'])
    pulse_blaster.Sequence(sequence, loop=np.inf)
