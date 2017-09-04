import time
from threading import Thread
import numpy as np
import visa
from socket import SOL_SOCKET, SO_KEEPALIVE
from ftplib import FTP, error_temp
#from waveform import *

# UI
from traits.api import HasTraits, Range, Float, Bool, Int, Str, Enum, Button, Property, Instance, on_trait_change
from traitsui.api import View, VGroup, HGroup, Item, TextEditor, EnumEditor


# TODO: File-transfer via GPIB as emergency

class AWG520( object ):
    """Controller for the Tektronix AWG520 device.
    
    SCPI commands are issued via gpib.
    See device manual for command documentation.
    File management is done via FTP.
    
    """
    
    def __init__(self, gpib='GPIB0::20::INSTR',
                       ftp='192.168.0.44', #129.69.46.221
                       socket=('192.168.0.44',4000) ):
        self.socket_addr = socket
        # set ftp parameters
        self.ftp_addr = ftp
        self.ftp_user = '\r'
        self.ftp_pw = '\r'
        self.ftp_cwd = '/main/waves'
        self.ftp_manager = FTPManager(self)
        self.todo = -1
        self.done = -1
        # setup gpib connection
        self.gpib_addr = gpib
        self.gpib = visa.instrument(self.gpib_addr)
        self.gpib.timeout = 5.0
        
    def __del__(self):
        self.gpib.close()
    
    # ____________
    # File Management
    
    def upload(self, files):
        # allow single files
        if not isinstance(files, (list, tuple)):
            files = [files]
    
        # opens up new ftp connections in separate threads
        self.todo = len(files)
        self.done = 0
        for file in files:
            self.ftp_manager.upload(file)
    
    def delete_all(self):
        """Remove all files from the AWG's CWD.
        """
        self.ftp_manager.delete_all()
        
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
                res = 'No response from AWG for: "' + query + '"'
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
        
    def set_output(self, channel=0b11):
        """Set the output state of specified channels.
        
        channels - int with states encoded on 2 LSB
                   e.g. bit=0b00 closes all, bit=0b11 opens all,
                        bit=0b10 opens OUTP2 and closes OUTP1
        
        """
        for i in [0,1]:
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
        look_up = {'C' : 'CONT', 'CON' : 'CONT', 'CONT' : 'CONT',
                   'T' : 'TRIG', 'TRI' : 'TRIG', 'TRIG' : 'TRIG',
                   'G' : 'GAT' , 'GAT' : 'GAT' , 'GATE' : 'GAT' ,
                   'E' : 'ENH' , 'ENH' : 'ENH' , 'ENHA' : 'ENH' ,
                  }
        self.tell('AWGC:RMOD %s' % look_up[mode.upper()])
    
    def set_sampling(self, frequency):
        """ Set the output sampling rate.
        
        """
        frequency *= 1e-9
        self.tell('SOUR:FREQ %.4GGHz' % frequency)
    
    def set_vpp(self, voltage, channel=0b11):
        """ Set output peak-to-peak voltage of specified channel.
            
        """
        if channel & 1 == 1:
            self.tell('SOUR1:VOLT %.4GV' % voltage)
        if channel & 2 == 2:
            self.tell('SOUR2:VOLT %.4GV' % voltage)
    
    def load(self, filename, channel=1, cwd='\waves', block=False):
        """Load sequence or waveform file into RAM, preparing it for output.
        
        Waveforms and single channel sequences can be assigned to each or both
        channels. Double channel sequences must be assigned to channel 1.
        The AWG's file system is case-sensitive.
        
        """
        self.tell('SOUR%i:FUNC:USER "%s/%s"' % (channel, cwd, filename))
        
        # block thread until the operation is complete
        while block:
            try:
                self.ask('*OPC?')
                self.tell('SYST:BEEP')
                block = False
            except visa.VisaIOError as e:
                if not 'Timeout' in e[0]: raise e
    
    def managed_load(self, filename, channel=1, cwd='\waves'):
        self.ftp_manager.load(filename, channel, cwd)
        
    def get_func(self, channel=1):
        res = self.ask('SOUR%i:FUNC:USER?' %channel)
        # res ~ '"/\\waves/0_MAIN.SEQ","MAIN"'
        return res.split(',')[0].split('/')[-1][:-1] # return ~ '0_MAIN.SEQ'
        
    def reset(self):
        """ Reset the AWG settings. """
        self.tell('*RST')
        

class FTPThread(Thread):
    """ Thread, which opens a new FTP connection.
    """
    def __init__(self, awg):
        # emulate state stuff
        class Foo(): pass
        self.file = Foo()
        self.file.state = 'compiling'
        
        self.awg = awg
        super(FTPThread, self).__init__()
        self.daemon = True
        self.file.state = 'ready'
        
    def setup_ftp(self):
        self.ftp = FTP(self.awg.ftp_addr, timeout=2.0)
        #self.ftp.set_pasv(False)
        self.ftp.login(self.awg.ftp_user, self.awg.ftp_pw)
        self.ftp.cwd(self.awg.ftp_cwd)
        
    def run(self):
        try:
            self.setup_ftp()
            self.task()
            self.ftp.close()
            self.file.state = 'finished'
        except Exception as e:
            try:
                self.ftp.close()
            except AttributeError as a:
                pass
            self.file.state = 'error'
            # dont raise error_temp
            # if not isinstance(e, error_temp):
            raise e
            
    def task(self): pass

class UploadThread(FTPThread):
    
    def __init__(self, awg, file):
        super(UploadThread, self).__init__(awg)
        self.file = file
    
    def task(self):
        self.file.seek(0)
        self.ftp.storbinary('STOR ' + self.file.name, self.file)

class DeleteAllThread(FTPThread):
    
    def task(self):
        filelist = self.ftp.nlst()
        try:
            filelist.remove('.')
            filelist.remove('..')
        except ValueError:
            pass
        self.awg.tell('MMEM:CDIR "%s"' % self.awg.ftp_cwd)
        for file in filelist:
            self.awg.tell('MMEM:DEL "%s"' % file)
            #self.ftp.delete(file)
        time.sleep(0.5)
            
class FTPManager(Thread):
    """ This Thread will prevent/workaround 421 session limit.
        
        It is only able to do to tasks, uploading files and deleting all files.
    """
    
    def __init__(self, awg):
        self.awg = awg
        self.threads = []
        self.clients = 0
        self.max_clients = 1
        self.awg.done = -1
        self.awg.todo = -1
        self.abort = False
        self.pause_set = False
        self.paused = False
        self.load_file = None
        super(FTPManager, self).__init__()
        self.daemon = True
        self.start()
        
    def upload(self, file):
        ut = UploadThread(self.awg, file)
        self.threads.append(ut)
             
    def delete_all(self):
        dt = DeleteAllThread(self.awg)
        self.threads.append(dt)
        
    def load(self, filename, channel=1, cwd='\waves'):
        self.load_file = { 'filename': filename,
                           'channel' : channel,
                           'cwd'     : cwd
                         }
        
    def reset(self):
        self.pause_set = True
        self.threads = []     # really bad practice!!! TODO: make stappable threads - stop and join them
        
        while not self.paused:
            time.sleep(0.1)
        self.clients = 0
        self.awg.done = -1
        self.awg.todo = -1
        self.pause_set = False
        
    def stop(self):
        self.abort = True
        
    def run(self):
        # Event loop 
        while True:
            # check list of threads repeatedly
            for thr in self.threads:
                if self.abort: return
                # ignore running threads
                if not thr.is_alive():
                    
                    # Case DeleteAllThread:
                    if isinstance(thr, DeleteAllThread):
                        # start a DeleteAllThread
                        if thr.file.state == 'ready' and self.clients == 0:
                            thr.start()
                            self.clients += self.max_clients
                            #time.sleep(0.001)
                        # remove finished DeleteAllThread
                        elif thr.file.state == 'finished':
                            self.clients = 0
                            self.threads.remove(thr)
                        # restart failed DeleteAllThread
                        elif thr.file.state == 'error':
                            self.clients = 0
                            self.threads.remove(thr)
                            self.delete_all()
                            
                    # Case UploadThread:
                    elif isinstance(thr, UploadThread):
                        # start new UploadThread
                        if thr.file.state == 'ready' and self.clients < self.max_clients:
                            thr.start()
                            self.clients += 1
                            #time.sleep(0.001)
                        # remove finished UploadThread
                        elif thr.file.state == 'finished':
                            self.clients -= 1
                            self.threads.remove(thr)
                            self.awg.done += 1
                        # restart failed UploadThread
                        elif thr.file.state == 'error':
                            self.clients -= 1
                            thr.file.seek(0)
                            thr.file.state = 'ready'
                            self.upload(thr.file)
                            self.threads.remove(thr)
                # stop threads if abort is set
                time.sleep(0.001)
            # check if there is something to load into RAM
            if len(self.threads) == 0 and self.awg.done != -1 and self.load_file is not None:
                f = self.load_file
                self.awg.load(f['filename'], f['channel'], f['cwd'], block=True)
                self.load_file = None
            if self.pause_set:
                self.paused = True
                while self.pause_set:
                    time.sleep(0.1)
                self.paused = False
                
                

class AWGHasTraits( HasTraits, AWG520 ):
    
    todo         = Int()
    done         = Int()
    progress     = Property(trait=Float, depends_on=['todo', 'done'],  format_str='%.1f')
    abort_button = Button(label='abort upload')
    
    en1  = Bool(False, label='CH 1', desc='enable CH1')
    en2  = Bool(False, label='CH 2', desc='enable CH2')
    
    current_wave = Str('', label='Wave', desc='last set waveform or sequence')
    
    mode = Enum(['ENH', 'CONT', 'TRIG', 'GATE'], desc='select run mode', editor=EnumEditor(values=['CONT', 'TRIG', 'GATE', 'ENH'], cols=4, format_str='%s'))
    
    run_button  = Button(label='Run', desc='Run')
    stop_button = Button(label='Stop', desc='Stop')
    trig_button = Button(label='Trigger', desc='Trigger')
    
    def __init__(self, gpib='GPIB0::14::INSTR',
                       ftp='192.168.1.2',
                       socket=('192.168.1.2',4000),
                       **kwargs
                ):
        AWG520.__init__(self, gpib, ftp, socket)
        HasTraits.__init__(self, **kwargs)
    
    def sync_upload_trait(self, client, trait_name):
        self.sync_trait('progress', client, trait_name, mutual=False)
    
    def _get_progress(self):
        return (100.0 * self.done) / self.todo
        
    def _abort_button_fired(self):
        self.ftp_manager.reset()
        
    def _run_button_fired(self):
        self.run()
        
    def _stop_button_fired(self):
        self.stop()
        
    def _trig_button_fired(self):
        self.force_trigger()
        
    @on_trait_change('en1', 'en2')
    def change_output(self):
        self.set_output((self.en2 << 1) + self.en1)
        
    @on_trait_change('mode')
    def change_mode(self):
        self.set_mode(self.mode)
    
    #override
    def load(self, filename, channel=1, cwd='\waves', block=False):
        self.current_wave = filename
        super(AWGHasTraits, self).load(filename, channel, cwd, block)
    
    view = View(VGroup(HGroup(Item('progress', width=40, style='readonly', format_str='%.1f'),
                              Item('todo', width=40, style='readonly'),
                              Item('abort_button', show_label=False),
                              label='FTP',
                              show_border=True,
                             ),
                       VGroup(HGroup(Item('en1'),
                                     Item('en2'),
                                     Item('run_button', show_label=False),
                                     Item('stop_button', show_label=False),
                                     Item('trig_button', show_label=False),
                                    ),
                              HGroup(Item('mode', style='custom', show_label=False, width=-250),
                                     Item('current_wave', style='readonly')
                                    ),
                              label='Output',
                              show_border=True,
                             ),
                      ),
                title='AWG520', width=550, buttons=[], resizable=True
               )
# _____________________________________________________________________________
# EXCEPTIONS:
    # TODO

# _____________________________________________________________________________
# DEBUG SCRIPT:

if __name__ == '__main__':
    pass























