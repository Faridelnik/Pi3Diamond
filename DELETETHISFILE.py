from measurements import shallow_NV
reload(shallow_NV)
prabi= shallow_NV.PulsePol()
from analysis import pulsedawgan
reload(pulsedawgan)
sensing=pulsedawgan.SensingPulsedAnalyzer()
sensing.measurement=prabi
sensing.edit_traits()
sensing.measurement.pi2_1 = 25.0
sensing.measurement.pi_1 = 50.
sensing.measurement.pulse_num = 237
sensing.measurement.tau_begin = 900.
sensing.measurement.tau_end = 930.
sensing.measurement.tau_delta  = 1.
sensing.measurement.load()
