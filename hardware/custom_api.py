
from tools.utility import singleton

@singleton
def Scanner():
    from nidaq import Scanner
    return Scanner( CounterIn='/Dev1/Ctr1',
                    CounterOut='/Dev1/Ctr0',
                    TickSource='/Dev1/PFI3',
                    AOChannels='/Dev1/ao0:2',
                    x_range=(0.0,50.0),
                    y_range=(0.0,50.0),
                    z_range=(-25.0,25.0),
                    v_range=(0.,10.))

ScnnerA=Scanner

@singleton
def Counter():
    from nidaq import PulseTrainCounter
    return PulseTrainCounter( CounterIn='/Dev2/Ctr1',
                              CounterOut='/Dev2/Ctr0',
                              TickSource='/Dev1/PFI3' ) 
                              
@singleton
def Counter_SST():
    from nidaq import SST_PulseTrainCounter
    return SST_PulseTrainCounter( function_counter_in = '/Dev1/Ctr1',
                                  gate_in_channel = '/Dev1/PFI13',
                                  photon_source = '/Dev1/PFI3',
                                )
                              
                              
@singleton
def Counter_Ext():
    from nidaq import PulseTrainCounterExternal
    return PulseTrainCounterExternal( CounterIn='/Dev2/Ctr1',
                                      GateSignal='/Dev1/PFI4',
                                      TickSource='/Dev1/PFI3' )                              

@singleton
def Microwave():
    from microwave_sources import SMIQ
    return SMIQ(visa_address='GPIB0::29')

MicrowaveA = Microwave

@singleton
def Microwave_HMC():
    from microwave_sources_hmct import SMIQ_HMC
    return SMIQ_HMC(visa_address='GPIB1::30::INSTR')

@singleton
def PulseGenerator():
    import pulse_blaster
    return pulse_blaster

@singleton 
def AWG():
    from awg import AWG
    return AWG
    
@singleton
def FastComtec():
    import Pyro.core
    fast_counter = Pyro.core.getProxyForURI('PYROLOC://192.168.0.2:2000/FastComTec')
    return fast_counter
    
@singleton
def Coil():
    from coil import Coil
    return Coil    
    
@singleton
def Test():
    import test_hmc
    return  test_hmc

"""

@singleton
def RFSource():
    import rf_source
    return rf_source.HP33120A(visa_address='GPIB0::11')
    # return rf_source.SMIQ_RF(visa_address='GPIB0::28')
    # return rf_source.Rigol1022('RigolB')


import TimeTagger
TimeTagger._Tagger.setSerial('12520004J2')
TimeTagger_OverFlow_Scanner = TimeTagger.OverFlow()

from lake_shore import TCU

@singleton
def TempCtrl():
    return TCU() 
"""
