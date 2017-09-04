"""
Example file for custom hardware API hooks. To provide custom hardware api,
create a file 'custom_api.py', copy and paste desired elements from this
file and adapt them to your needs.
"""

from tools.utility import singleton

@singleton
def Scanner():
    from nidaq import Scanner
    return Scanner( CounterIn='/Dev1/Ctr3',
                    CounterOut='/Dev1/Ctr2',
                    TickSource='/Dev1/PFI0',
                    AOChannels='/Dev1/ao0:2',
                    TriggerChannels='/Dev1/port0/line0:1',
                    x_range=(0.0,75.0),
                    y_range=(0.0,75.0),
                    z_range=(-25.0,25.0),)

@singleton
def Counter():
    from nidaq import PulseTrainCounter
    return PulseTrainCounter( CounterIn='/Dev1/Ctr1',
                              CounterOut='/Dev1/Ctr0',
                              TickSource='/Dev1/PFI0' )

@singleton
def Microwave():
    import microwave_sources
    return microwave_sources.SMR20(visa_address='GPIB0::29')

@singleton
def PulseGenerator():
    import pulse_generator
    return pulse_generator.PulseGenerator(serial='HDbXzJrLZM',channel_map={'laser':0, 'mw':1, 'mw_a':1, 'mw_b':2, 'sequence':3})

import TimeTagger
TimeTagger._Tagger.setSerial('VJaAMgsRxh')

#ToDo: more examples

