
from tools.utility import edit_singleton
from datetime import date
import os

import imp
#
def reimport_pulsed():
    import measurements.pulsed_awg
    imp.reload(measurements.pulsed_awg)

    rabi=measurements.pulsed_awg.Rabi()
    from analysis.pulsedawgan import PulsedAnalyzer
    p=PulsedAnalyzer()
    p.measurement=rabi
    p.edit_traits()

# start confocal including auto_focus tool and Toolbox
if __name__ == '__main__':

    # start confocal including auto_focus tool
    from hardware.api import Scanner
    scanner = Scanner()
    from measurements.confocal import Confocal
    confocal = Confocal(scanner)
    confocal.edit_traits()
    
    # some imports providing short names for the main modules
    import numpy as np
    import hardware.api as ha
    import measurements as me
    import analysis as an
    """
    laser = ha.Laser()
    laser.edit_traits()
    laser.voltage=0.0
    ha.PulseGenerator().Light()
    
    """
    
    from measurements.counter_trace import CounterTrace
    time_trace = CounterTrace()
    time_trace.edit_traits()
    
    
    from measurements.odmr import ODMR
    odmr=ODMR()
    odmr.edit_traits()
    
    from measurements.odmr_hmc import ODMR_HMC
    odmr_hmc=ODMR_HMC()
    odmr_hmc.edit_traits()
    
    
    from measurements.auto_focus import AutoFocus
    auto_focus = AutoFocus(confocal, odmr)
    auto_focus.edit_traits()
    
    try:
        auto_focus.load('defaults/auto_focus.pyd')
    except:
        pass
        
    from measurements.magnet_control import Magnet_alignment
    mag_al = Magnet_alignment(odmr, auto_focus)
    mag_al.edit_traits()
    
        
    #from field_align import Odmr_field
    #odmr_field = Odmr_field(odmr)
    
    from measurements.pulsed import Rabi
    Srabi=Rabi()
    from analysis.pulsedan import PulsedAnalyzer
    psrabi=PulsedAnalyzer()
    psrabi.measurement=Srabi
    psrabi.edit_traits()
    
    from measurements.pulsed import Hahn
    T2_pulse_blaster=Hahn()
    T2T1_pulse_blaster=an.pulsedan.DoubleSensingPulsedAnalyzer()
    T2T1_pulse_blaster.measurement=T2_pulse_blaster
    T2T1_pulse_blaster.edit_traits()
    
    # pulsed HMC without AWG to know the pi pulse for pulsed ODMR
    
    HMCRabi=me.pulsed_hmc.Rabi()
    hmcrabi=an.pulsedan.PulsedHMCAnalyzer()
    hmcrabi.measurements=HMCRabi
    hmcrabi.edit_traits()
    
    # from measurements.singleshot import SSTCounterTrace
    # trace = SSTCounterTrace()
    # from analysis.sstan import TracePulsedAnalyzer
    # sst_trace = TracePulsedAnalyzer()
    # sst_trace.measurements = trace
    # sst_trace.edit_traits()
    
    
    # import measurements.pulsed_awg
    # fid=measurements.pulsed_awg.Sing_FID()
    # from analysis.pulsedawgan import PulsedAnalyzer
    # psawg=PulsedAnalyzer()
    # psawg.measurement=fid
    # psawg.edit_traits()
    
    
    import measurements.pulsed_awg
    rabi=measurements.pulsed_awg.Rabi()
    from analysis.pulsedawgan import PulsedAnalyzer
    psawg2=PulsedAnalyzer()
    psawg2.measurement=rabi
    psawg2.edit_traits()
    
    # Rabi with AWG and HMC
    from analysis.pulsed_hmc_an import PulsedAnalyzer
    rabi2=me.pulsed_hmc.RabiAWG()
    rabi_hmc_awg=PulsedAnalyzer()
    rabi_hmc_awg.measurement=rabi2
    rabi_hmc_awg.edit_traits()
    
    # import measurements.prog_gate_awg
    # tomo= measurements.prog_gate_awg.EspinTomo_diag()
    # from analysis.pulsedawgan import ProgPulsedAnalyzer
    # prog=ProgPulsedAnalyzer()
    # prog.measurement=tomo
    # prog.edit_traits()
    
    from measurements.shallow_NV import Correlation_Spec_XY8_phase
    prabi= Correlation_Spec_XY8_phase()
    from analysis.pulsedawgan import SensingPulsedAnalyzer
    sensing=SensingPulsedAnalyzer()
    sensing.measurement=prabi
    sensing.edit_traits()
    
    
    import measurements.spotfinderv3
    sf=measurements.spotfinderv3.Spotfinder(confocal,auto_focus)
    sf.edit_traits()
    
    # import measurements.goodspot
    # goodspot=measurements.goodspot.Goodspot(confocal,auto_focus)
    # goodspot.edit_traits()
    
    #from measurements.Flopper import Flopper
    #flo = Flopper()
    #flo.edit_traits()
    
    T1 = measurements.shallow_NV.T1()
    from analysis.pulsedawgan import DoubleSensingPulsedAnalyzer
    pdawg=DoubleSensingPulsedAnalyzer()
    pdawg.measurement=T1
    pdawg.edit_traits()
    
    Hahn = measurements.shallow_NV.Hahn()
    from analysis.pulsedawgan import DoubleSensingPulsedAnalyzer
    pdawg2=DoubleSensingPulsedAnalyzer()
    pdawg2.measurement=Hahn
    pdawg2.edit_traits()
    
    XY8 = measurements.shallow_NV.XY8_Ref()
    from analysis.pulsedawgan import DoubleSensingPulsedAnalyzer
    pdawg3=DoubleSensingPulsedAnalyzer()
    pdawg3.measurement=XY8
    pdawg3.edit_traits()
    
    
    # T1 measurements with hmc and awg
    
    T1HMC = me.pulsed_hmc.T1()
    from analysis.pulsed_hmc_an import DoubleSensingPulsedAnalyzer
    pdawg4=DoubleSensingPulsedAnalyzer()
    pdawg4.measurement=T1HMC
    pdawg4.edit_traits()
    
    # DEER with two MW sources
    
    from measurements import DEER
    deer = DEER.Electron_Rabi()
    from analysis.pulsedDEERanalyzer import PulsedAnalyzer
    dr=PulsedAnalyzer()
    dr.measurement=deer
    dr.edit_traits()
    
    from measurements import deer_odmr
    deerodmr=deer_odmr.ODMR()
    deerodmr.edit_traits()
    
    # from analysis.dyn_dec_ssr_an import SSRAnalyzer
    # ddssr=SSRAnalyzer()
    # S=measurements.dyn_decoupl_with_ssr.XY8_with_SSR()
    # ddssr.measurement=S
    # ddssr.edit_traits()
    
    pg = ha.PulseGenerator()
    pg.High(['laser'])
    
    
    from tools.tau_calculation_XY8 import dip_position
    dip_pos = dip_position()
    dip_pos.edit_traits()
    
    # from analysis import distance_to_NV
    # dist=distance_to_NV.Distance_to_NV()
    # dist.edit_traits()
    
    from tools import equation18
    sn=equation18.Distance_to_NV()
    sn.edit_traits()
    
    
    # from hardware import temperature_sensor
    # TS = temperature_sensor.Temperature_Sensor()
    # TS.edit_traits()
    
    '''
    import measurements.pulsed_awg
    
    from measurements.nmr import NMR
    
    nmr=NMR()
    m_nmr=PulsedAnalyzer()
    m_nmr.measurement=nmr
    m_nmr.edit_traits()
    
    '''
    # from measurements.nuclear_rabi import NuclearRabi
    
    # nr=NuclearRabi()
    # m_nr=an.pulsed.PulsedAnalyzer()
    # m_nr.measurement=nr
    # m_nr.edit_traits()
    
    '''
    m_deer=PulsedAnalyzer()
    m_deer.edit_traits()
    
    from measurements.ple import PLE
    
    ple=PLE()
    ple.edit_traits()
    
   
    from measurements.auto_resonance import AutoResonance
    ar=AutoResonance()
    try:
        ar.load('defaults/ar.pyd')
    except:
        pass
    
    from measurements.linefinder import LineFinder
    lf=LineFinder()
    
   
    from measurements.qnd_red import QND_RED
    qnd_r=QND_RED(ar,auto_focus)
    from measurements.rabi_red import Rabi_RED
    rabi_r=Rabi_RED(ar,auto_focus)
    
    from measurements.hahn_red import Hahn_RED
    hr=Hahn_RED(ar,auto_focus)
    from measurements.nspin_red import Nspin_RED
    nsr=Nspin_RED(ar,auto_focus)
    
    
    from measurements.autocorrelation import Autocorrelation
    ac=Autocorrelation()
    ac.edit_traits()
   
    ha.PulseGenerator().Light()
    
    t = date.today()
    t = t.timetuple()
    if t.tm_mday < 10:
        d = '0' + str(t.tm_mday)
    else:
        d = str(t.tm_mday)
    
    if t.tm_mon < 10:
        m = '0' + str(t.tm_mon)
    else:
        m = str(t.tm_mon)
    y = str(t.tm_year)
    dirpath='D:/Data/' + y + '/' + y + '-' + m + '-' + d
    if not os.path.exists(dirpath):
        os.mkdir(dirpath) 
        '''