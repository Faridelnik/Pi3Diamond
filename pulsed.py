import measurements.pulsed_awg
reload(measurements.pulsed_awg)
rabi=measurements.pulsed_awg.Rabi()
from analysis.pulsedawgan import PulsedAnalyzer
p=PulsedAnalyzer()
p.measurement=rabi
p.edit_traits()