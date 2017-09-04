###############
# stimulation
################
import time
from chaco import shell

from hardware.api import FastComtec
fc = FastComtec()

from hardware.api import Microwave
microwave = Microwave()

from hardware.microwave_sources import SMIQ
microwave_2 =  SMIQ(visa_address='GPIB0::28')

from hardware.api import PulseGenerator
pg = PulseGenerator()
pg.map['mw2'] = 5


### helper functions ###

from analysis.fitting import find_edge

def spin_state(c, dt, T, t0=0.0, t1=-1.):
   
    """
    Compute the spin state from a 2D array of count data.
   
    Parameters:
   
        c    = count data
        dt   = time step
        t0   = beginning of integration window relative to the edge
        t1   = None or beginning of integration window for normalization relative to edge
        T    = width of integration window
       
    Returns:
   
        y       = 1D array that contains the spin state
        profile = 1D array that contains the pulse profile
        edge    = position of the edge that was found from the pulse profile
       
    If t1<0, no normalization is performed. If t1>=0, each data point is divided by
    the value from the second integration window and multiplied with the mean of
    all normalization windows.
    """

    profile = c.sum(0)
    edge = find_edge(profile)
   
    I = int(round(T/float(dt)))
    i0 = edge + int(round(t0/float(dt)))
    y = np.empty((c.shape[0],))
    for i, slot in enumerate(c):
        y[i] = slot[i0:i0+I].sum()
    if t1 >= 0:
        i1 = edge + int(round(t1/float(dt)))    
        y1 = np.empty((c.shape[0],))
        for i, slot in enumerate(c):
            y1[i] = slot[i1:i1+I].sum()
        y = y/y1*y1.mean()
    return y, profile, edge
    
def find_trigger_pulses(sequence):
    n = 0
    prev = []
    for channels, t in sequence:
        if 'trigger' in channels and not 'trigger' in prev:
            n += 1
        prev = channels
    return n

def periodRound(period):
    from math import floor
    rounded = int(floor(period))
    control = rounded%2
    if control==1:
        return float(rounded +1) 
    else:
        return float(rounded)
    
    
##########  Rabi ramsy rabi pi messung ##################
############################################################

tau = np.arange(12., 20000., 50.) # ns


# alles in ns
rabiPerioden =  {"NV01":102.715968,"NV05":106.084182,"NV08":92.474017 ,"NV10":104.341812 ,"NV011":104.658512}
freqs1 =        {"NV01":2.56376135e+09,"NV05":2.56329954e+09,"NV08":2.56383882e+09 ,"NV10":2.56163565e+09 ,"NV011": 2.56325671e+09}
freqs2 =        {"NV01":3.18432006e+09,"NV05":3.18439341e+09,"NV08":3.18399623e+09 ,"NV10":3.18413424e+09 ,"NV011":3.18393480e+09}

actualNV = auto_focus.current_target

print "Die fokusierte NVStelle ist : %s"%actualNV


rabiPeriode = rabiPerioden[actualNV] ; 
piHalbePeriode =periodRound(rabiPeriode/4.);
dreiPiHalbePeriode=periodRound(3.*rabiPeriode/4.);

print "pi/2 period = %f"%piHalbePeriode
     
                
sequence = [(['laser'],3000)]
for t in tau:   
    # # # ramsey104.658512 mit pi halbe 
    sequence += [  ([],1000), (['mw'],piHalbePeriode), ([],t), (['mw'],piHalbePeriode),(['laser','trigger'],3000) ]
    
    # Ramsey mit 3 pi halbe 
    sequence += [  ([],1000), (['mw'],piHalbePeriode), ([],t), (['mw'],dreiPiHalbePeriode),(['laser','trigger'],3000) ]
sequence += [([],10000)]
    
power_a = -9 # dBm
power_b = -9 # dBm

frequency_a = freqs1[actualNV] # Hz
frequency_b = freqs2[actualNV] # Hz

microwave.setOutput(power_a, frequency_a)
microwave_2.setOutput(power_b, frequency_b)


###############
# acquisition
################

n_bins = 3000

binwidth = 1 # ns

n_detect = find_trigger_pulses(sequence) # number of laser pulses

#pulsed = time_tagger.Pulsed(n_bins, binwidth, n_detect, 1, 2, 3)



pg.High([])
fc.Configure(n_bins, binwidth, n_detect)
fc.SetCycles(np.inf)
fc.SetTime(np.inf)
fc.SetDelay(0)
fc.SetLevel(0.6, 0.6)
fc.Start()
time.sleep(2.0)
pg.Sequence( sequence )

#from analysis.pulsed import spin_state
    


def rabiFit(t,x):
    from analysis.fitting import Cosinus_phase, Cosinus_phaseEstimator, fit
    parameters = Cosinus_phaseEstimator(t,x)
    fitParameters = fit(t,x, Cosinus_phase, parameters)
    print "Period = %f  and frequency = %f "%(fitParameters[2], fitParameters[2]**(-1)*1e9)
    return Cosinus_phase(*fitParameters)
    
def fetch():
    mat = fc.GetData()
    y1, profile, edge = spin_state(mat[0::2], 1.0, 800.,t0=20)
    y2, profile, edge = spin_state(mat[1::2], 1.0, 800.,t0=20)
    return (y1,y2)

    
    
    
def plot(save=False):
    shell.close('all')
    x=fetch() #.reshape((len(tau),4))
    x1 = x[0]
    x2 = x[1]
    shell.figure()
    shell.plot(tau,x1)
    shell.hold(True)
    shell.plot(tau,x2,"b-")
    #fitPoints = rabiFit( tau,x)
    #shell.plot(tau,fitPoints(tau), "r-")
    shell.xtitle('tau [ns]')
    shell.ytitle('Fluorescence [a.u.]')
    
    shell.figure(title='ratio')
    shell.plot(tau,x[0]/x[1])
    shell.xtitle('tau [ns]')
    shell.ytitle('normalized fluorescence [a.u.]')

    shell.figure(title='difference')
    shell.plot(tau,x[0]-x[1])
    if save:
        np.savetxt("BP/Results/ramsey1.txt",x1);
        np.savetxt("BP/Results/ramsey2.txt",x2);
        np.savetxt("BP/Results/ramseyTau.txt",tau);

def stop():
    fc.Halt()
    pg.High(['laser'])
    microwave.Off()
    microwave_2.Off()
    
