
from tools.utility import singleton

@singleton
def Scanner():
    from nidaq import Scanner
    return Scanner( CounterIn='/Dev1/Ctr1',
                    CounterOut='/Dev1/Ctr0',
                    TickSource='/Dev1/PFI3',
                    AOChannels='/Dev1/ao0:2',
                    x_range=(0.0,100.0),
                    y_range=(0.0,100.0),
                    z_range=(-12.5,12.5),
                    v_range=(-10.,10.))

ScnnerA=Scanner

@singleton
def Counter():
    from nidaq import PulseTrainCounter
    return PulseTrainCounter( CounterIn='/Dev1/Ctr3',
                              CounterOut='/Dev1/Ctr2',
                              TickSource='/Dev1/PFI3' )
    
CounterA=Counter

"""
@singleton
def DOTask():
    from nidaq import DOTask
    return DOTask(DOChannels='/Dev2/port0/line0')
    # to turn on: DOTask().Write(np.array((1))), to turn off: DOTask().Write(np.array((0)))
"""
def CountTask(bin_width, length):
    return  TimeTagger.Counter(0,int(bin_width*1e12), length)
    """from ni import Counter as NICounter
    return NICounter(counter_out_device='/Dev1/Ctr2',
                     counter_in_device='/Dev1/Ctr3',
                     input_pad='/Dev1/PFI3',
                     bin_width=bin_width,
                     length=length )"""

"""
@singleton
def Laser():
    import laser
    return laser.Laser('/Dev1/ao3', voltage_range=(-10.,10.))
"""

"""
@singleton
def EOM():
    import laser
    return laser.Laser('/Dev2/ao1', voltage_range=(0.,10.))
"""
@singleton
def LaserRed():
    import laser_red
    return laser_red.LaserRed(visa_address='GPIB0::2', ao_chan='/Dev2/ao0', co_dev='/Dev2/Ctr1', ci_dev='/Dev2/Ctr2', ci_port='/Dev2/PFI0')
    """import laser_patrick
    return laser_patrick.LaserRed(visa_address='GPIB0::2', ao_chan='/Dev2/ao0', co_dev='/Dev2/Ctr1', ci_dev='/Dev2/Ctr2', ci_port='/Dev2/PFI0')
    """

@singleton
def DtgOptical():
    import dtg_optical
    return dtg_optical.DtgOptical(visa_address='GPIB0::1')
					 
@singleton
def Microwave():
    import microwave_sources
    return microwave_sources.SMIQ(visa_address='GPIB0::21')

MicrowaveA = Microwave

@singleton
def MicrowaveB():
    import microwave_sources
    return microwave_sources.SMIQ(visa_address='GPIB0::22')

@singleton
def MicrowaveC():
    import microwave_sources
    return microwave_sources.SMIQ(visa_address='GPIB0::23')

@singleton
def MicrowaveD():
    import microwave_sources
    return microwave_sources.SMIQ(visa_address='GPIB0::24')

@singleton
def MicrowaveE():
    import microwave_sources
    return microwave_sources.SMIQ(visa_address='GPIB0::25')

@singleton
def RFSource():
    import rf_source
    return rf_source.HP33120A(visa_address='GPIB0::11')
    # return rf_source.SMIQ_RF(visa_address='GPIB0::28')
    # return rf_source.Rigol1022('RigolB')

@singleton
def PulseGenerator():
    import pulse_generator
    #original version
    return pulse_generator.PulseGenerator(serial='11130001AO',channel_map={'red':0,'mw_o':1,'mw_d':1,'mw_b':1,'mw_y':1,'mw_r_an1':1,'mw_r_e':2,'laser':3,'mw':4,'mw_a':4,'mw_x':4,'rf':5,'mw_c':6,'mw_r_a':6,'sequence':7,'green':8,'aom':8,'ch10':9,'ch11':10,'ch12':11})
    #green out version
    #return pulse_generator.PulseGenerator(serial='11130001AO',channel_map={'red':0,'mw_o':1,'mw_d':1,'mw_b':1,'mw_y':1,'mw_r_an1':1,'mw_r_e':2,'sequence':3,'mw':4,'mw_a':4,'mw_x':4,'rf':5, 'mw_c':6,'mw_r_a':6,'green':7,'laser':7,'aom':7})
    #11130001AO

import TimeTagger
TimeTagger._Tagger.setSerial('12520004J2')
TimeTagger_OverFlow_Scanner = TimeTagger.OverFlow()

"""
from lake_shore import TCU

@singleton
def TempCtrl():
    return TCU() 
"""
