#import time
from threading import Thread
import numpy as np
import visa
from socket import socket,SOL_SOCKET, SO_KEEPALIVE, AF_INET, SOCK_STREAM
from ftplib import FTP #The File Transfer Protocol, The FTP class implements the client side of the FTP protocol
from waveform import *

class AWG():
    """Controller for the Tektronix AWG5014C device.
    
    SCPI commands are issued via gpib.
    See device manual for command documentation.
    File management is done via FTP.
    
    """
    
    def __init__(self, address={'gpib':'GPIB0::20::INSTR',
                                'ftp':'129.69.46.166',
                                'socket':('129.69.46.166',4001)} ):
        self.address = address
        # setup ftp-connection
        self.ftp = FTP(self.address['ftp'])
        self.ftp.login('user', 'pass') 
        self.ftp.sock.setsockopt(SOL_SOCKET, SO_KEEPALIVE, 1)
        self.ftp.sock.settimeout(5.0)
        # setup gpib connection
        #self.gpib = visa.instrument(self.address['gpib'])
        self.gpib = visa.ResourceManager().get_instrument(self.address['gpib'])
        self.gpib.timeout = 10
        
    def __del__(self):
        self.gpib.close()
        self.ftp.close()
        #self.soc.close()

    
    # ____________
    # File Management
    
    PARALLEL = 0
    SERIAL = 1
    
    class UploadThread(Thread):
        """ Thread object, which uploads a single file to the AWG.
        """
        def __init__(self, address, file):
            self.ftp = FTP(address)
            self.res = self.ftp.login('user', 'pass') 
            self.file = file
            Thread.__init__(self)
            
        def run(self):
            try:
                if isinstance(self.file, (StringIO, Waveform)):
                    self.file.seek(0)
                    self.res = self.ftp.storbinary('STOR ' + self.file.name, self.file)
                elif isinstance(self.file, str):
                    f = open(self.file, 'rb')
                    self.res = self.ftp.storbinary('STOR ' + self.file, f)
                    f.close()
            finally:
                self.ftp.close()
             
    def upload(self, files, mode=PARALLEL):
        """Upload files to the AWG.
        
        """
        if not isinstance(files, (list, tuple)):
            files = [files]
        
        if mode == self.PARALLEL:
            threads = []
            for file in files:
                ut = self.UploadThread(self.address['ftp'], file)
                threads.append(ut)
                ut.start()
            for i,t in enumerate(threads):
                t.join()
            
        elif mode == self.SERIAL:
            for file in files:
                self.ftp = FTP(self.address['ftp'])
                self.ftp.login('user', 'pass') 
                if isinstance(file, (StringIO, Waveform)):
                    file.seek(0)
                    res = self.ftp.storbinary('STOR ' + file.name, file)
                elif isinstance(file, str):
                    with open(file, 'rb') as f:
                        res = self.ftp.storbinary('STOR ' + file.upper(), f)
                self.ftp.close()
        
    def delete(self, files):
        """Remove the specified files from the CWD.
        
        Delete a single file by providing its name as argument. Delete multiple
        files by passing a list of filenames as argument.
        
        files - string or list of strings
        
        """
        if not isinstance(files, (list, tuple)):
            files = [files]
        for filename in files:
            res = self.ftp.delete(filename)
        return res
    
    def delete_all(self):
        """Remove all files from the AWG's CWD.
        """
        res = 0
        filelist = self.ftp.nlst()
        try:
            filelist.remove('.')
            filelist.remove('..')
        except ValueError:
            pass
        for filename in filelist:
            res = self.ftp.delete(filename)
        return res
    
    # ____________
    # Operation Commands
    
    def tell(self, command):
        """Send a command string to the AWG."""
        self.gpib.write(command)
        
    def ask(self, query):
        """Send a query string to AWG and return the response."""
        self.gpib.write(query)
        try:
            res = self.gpib.read()
        except visa.VisaIOError as e:
            res = ''
            if 'Timeout' in e.message:
                print query, '--- no response from AWG'
            else:
                raise e
        return res
    
    def run(self):
        self.tell('AWGC:RUN')
    
    def stop(self):
        self.tell('AWGC:STOP')
        
    def force_trigger(self):
        self.tell('*TRG')
    
    def force_event(self, bitcode):
        self.tell('AWGC:EVEN:SOFT %i' %bitcode)
        
    def set_output(self, channel=0b1111):
        """Set the output state of specified channels.
        
        channels - int with states encoded on 4 LSB
                   e.g. bit=0b0000 closes all, bit=0b1111 opens all,
                        bit=0b1010 opens OUTP2 and 4 and closes OUTP1 and 3
        
        """
        for i in [0,1,2,3]:
            stat = channel >> i & 1 
            self.tell('OUTP%i %i' % ((i+1), stat) )
        
    def set_mode(self, mode):
        """Change the output mode.
        
        Options for mode (case-insensitive):
        continuous - 'C', 'CONT'
        triggered  - 'T', 'TRIG'
        gated      - 'G', 'GAT'
        sequence   - 'S', 'SEQ'
        
        """
        look_up = {'C' : 'CONT', 'CONT' : 'CONT',
                   'T' : 'TRIG', 'TRIG' : 'TRIG',
                   'G' : 'GAT' , 'GAT'  : 'GAT',
                   'S' : 'SEQ' , 'SEQ'  : 'SEQ'
                  }
        self.tell('AWGC:RMOD %s' % look_up[mode.upper()])
    
    def set_sample(self, frequency):
        """ Set the output sampling rate [GHz].
        
        """
        self.tell('SOUR:FREQ %.4GGHz' % frequency)
    
    def set_vpp(self, voltage, channel=0b1111):
        """ Set output peak-to-peak voltage of specified channel.
            
        """
        if channel & 1 == 1:
            self.tell('SOUR1:VOLT %.4GV' % voltage)
        if channel & 2 == 2:
            self.tell('SOUR2:VOLT %.4GV' % voltage)
        if channel & 4 == 4:
            self.tell('SOUR3:VOLT %.4GV' % voltage)
        if channel & 8 == 8:
            self.tell('SOUR4:VOLT %.4GV' % voltage)
    
    def load(self, filename, channel=1, cwd=None):
        """Load sequence or waveform file into RAM, preparing it for output.
        
        Waveforms and single channel sequences can be assigned to each or both
        channels. Double channel sequences must be assigned to channel 1.
        The AWG's file system is case-sensitive.
        
        """
        if cwd is None:
            cwd = '\waves' # default

        print 'SOUR%i:FUNC:USER "%s/%s"' % (channel, cwd, filename)
        self.tell('SOUR%i:FUNC:USER "%s/%s"' % (channel, cwd, filename))
        
    def reset(self):
        """ Reset the AWG settings. """
        self.tell('*RST')
        
    # TODO:
    # def play(self, *seq, **kw):
        # seq, waves = make(*seq, **kw)
'''        
class AWG():
    """Controller for the Tektronix AWG520 device.
    
    Commands are issued via Ethernet socket (default port: 4000).
    File management is done via FTP.
    
    """
    
    def __init__(self, address={'gpib':'GPIB0::20::INSTR',
                                'ftp':'192.168.0.44',
                                'socket':('192.168.0.44', 4001)} ):
        self.address = address
        # setup ftp-connection
        
        self.soc = socket(AF_INET, SOCK_STREAM)
        
        self.ftp = FTP(self.address['ftp'])
        self.ftp.login('user', 'pass') 
        self.ftp.sock.setsockopt(SOL_SOCKET, SO_KEEPALIVE, 1)
        self.ftp.sock.settimeout(5.0)
        
        self.input_buffer = 2 ** 11
        # setup gpib connection
        #self.gpib = visa.instrument(self.address['gpib'])
        #self.gpib = visa.ResourceManager().get_instrument(self.address['gpib'])
        #self.gpib.timeout = 5.0
    
    def __del__(self):
        self.soc.close()
        self.ftp.close()
    
    def _server_init(self):
        #self.soc = socket(AF_INET, SOCK_STREAM)
        self.soc.connect(('192.168.0.44',4000))
        print self.soc.getsockname()
        
    def _server_close(self):
        self.soc.close()
    # ____________
    # File Management
    
    PARALLEL = 0
    SERIAL = 1
    
    class UploadThread(Thread):
        """ Thread object, which uploads a single file to the AWG.
        """
        def __init__(self, address, file):
            self.ftp = FTP(address)
            self.res = self.ftp.login('user', 'pass') 
            self.file = file
            Thread.__init__(self)
            
        def run(self):
            try:
                if isinstance(self.file, (StringIO, Waveform)):
                    self.file.seek(0)
                    self.res = self.ftp.storbinary('STOR ' + self.file.name, self.file)
                elif isinstance(self.file, str):
                    f = open(self.file, 'rb')
                    self.res = self.ftp.storbinary('STOR ' + self.file, f)
                    f.close()
            finally:
                self.ftp.close()
                
    def upload(self, filelist):
        """Upload files to AWG's CWD.
        
        filelist may contain names of files in CWD or StrignIO objects.
        File names are converted to capital letters to be nonambiguous.
        
        """
        self.ftp.connect(self.address['ftp'])
        self.ftp.login('user', 'pass') 
        for file in filelist:
            if isinstance(file, StringIO):
                file.seek(0)
                res = self.ftp.storbinary('STOR ' + file.name, file)
            elif isinstance(file, str):
                with open(file, 'rb') as f:
                    res = self.ftp.storbinary('STOR ' + file.upper(), f)
            #else: raise AWGError
            #print 'uploading file %s %s' %(file, res) 
        self.ftp.quit()            
               
    def upload(self, files, mode=PARALLEL):
        """Upload files to the AWG.
        
        """
        if not isinstance(files, (list, tuple)):
            files = [files]
        
        if mode == self.PARALLEL:
            threads = []
            for file in files:
                ut = self.UploadThread(self.address['ftp'], file)
                threads.append(ut)
                ut.start()
            for i,t in enumerate(threads):
                t.join()
            
        elif mode == self.SERIAL:
            for file in files:
                self.ftp = FTP(self.address['ftp'])
                self.ftp.login('user', 'pass') 
                if isinstance(file, (StringIO, Waveform)):
                    file.seek(0)
                    res = self.ftp.storbinary('STOR ' + file.name, file)
                elif isinstance(file, str):
                    with open(file, 'rb') as f:
                        res = self.ftp.storbinary('STOR ' + file.upper(), f)
                self.ftp.close()
                
    def delete(self, files):
        """Remove the specified files from the CWD.
        
        Delete a single file by providing its name as argument. Delete multiple
        files by passing a list of filenames as argument.
        
        files - string or list of strings
        
        """
        if not isinstance(files, (list, tuple)):
            files = [files]
        for filename in files:
            res = self.ftp.delete(filename)
        return res
    
    def delete_all(self):
        """Remove all files from the AWG's CWD.
        """
        res = 0
        filelist = self.ftp.nlst()
        try:
            filelist.remove('.')
            filelist.remove('..')
        except ValueError:
            pass
        for filename in filelist:
            res = self.ftp.delete(filename)
        return res
    
    # ____________
    # Operation Commands
    
    def tell(self, command):
        """Send a command string to the AWG."""
        self.soc.send(command)
        
    
    def ask(self, command):
        """Send a command string to AWG and return the response.
        
        command - AWG command (see Programmer Manual)
        
        TODO: Debug
        
        """
        if not command.endswith('\n'): ## I always forget the line feed.
            command += '\n'
        self.soc.send(command)
        #time.sleep(0.000001)
        return self.soc.recvfrom(self.input_buffer) ## somewhat buggy
    
    def run(self):
        self.soc.send('AWGC:RUN\n')
    
    def stop(self):
        self.soc.send('AWGC:STOP\n')
        
    def force_trigger(self):
        self.soc.send('*TRG')
    
    def force_event(self, bitcode):
        self.soc.send('AWGC:EVEN:SOFT %i' %bitcode)
        

    def set_output(self, channel=0b1111):
        """Set the output state of specified channels.
        
        channels - int with states encoded on 4 LSB
                   e.g. bit=0b0000 closes all, bit=0b1111 opens all,
                        bit=0b1010 opens OUTP2 and 4 and closes OUTP1 and 3
        
        """
        for i in [0,1,2,3]:
            stat = channel >> i & 1 
            self.soc.send('OUTP%i %i' % ((i+1), stat) )
        
            
    def set_mode(self, mode):
        """Change the output mode.
        
        Options for mode (case-insensitive):
        continuous - 'C', 'CONT'
        triggered  - 'T', 'TRIG'
        gated      - 'G', 'GAT'
        sequence   - 'S', 'SEQ'
        
        """
        look_up = {'C' : 'CONT', 'CONT' : 'CONT',
                   'T' : 'TRIG', 'TRIG' : 'TRIG',
                   'G' : 'GAT' , 'GAT'  : 'GAT',
                   'S' : 'SEQ' , 'SEQ'  : 'SEQ'
                  }
        self.soc.send('AWGC:RMOD %s' % look_up[mode.upper()])
    
    def set_sample(self, frequency):
        """ Set the output sampling rate [GHz].
        
        """
        self.soc.send('SOUR:FREQ %.4GGHz' % frequency)
    
    def set_vpp(self, voltage, channel=0b1111):
        """ Set output peak-to-peak voltage of specified channel.
            
        """
        if channel & 1 == 1:
            self.soc.send('SOUR1:VOLT %.4GV' % voltage)
        if channel & 2 == 2:
            self.soc.send('SOUR2:VOLT %.4GV' % voltage)
        if channel & 4 == 4:
            self.soc.send('SOUR3:VOLT %.4GV' % voltage)
        if channel & 8 == 8:
            self.soc.send('SOUR4:VOLT %.4GV' % voltage)
    
    def load(self, filename, channel=1, cwd=None):
        """Load sequence or waveform file into RAM, preparing it for output.
        
        Waveforms and single channel sequences can be assigned to each or both
        channels. Double channel sequences must be assigned to channel 1.
        The AWG's file system is case-sensitive.
        
        """
        if cwd is None:
            cwd = '\waves' # default
        self.soc.send('SOUR%i:FUNC:USER "%s/%s"' % (channel, cwd, filename))
        
    def reset(self):
        """ Reset the AWG settings. """
        self.soc.send('*RST')
        '''
    
# _____________________________________________________________________________
# EXCEPTIONS:
    # TODO

# _____________________________________________________________________________
# DEBUG SCRIPT:






























