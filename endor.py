from tools.utility import edit_singleton
from datetime import date
import os

if __name__ == '__main__':   
    
    from measurements.pulsed_awg import Double_RF_sweep
    endor1=Double_RF_sweep()
    from analysis.pulsedawgan import DoublePulsedAnalyzer
    q=DoublePulsedAnalyzer()
    q.measurement=endor1
    q.edit_traits()