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

##########  Rabi ramsy rabi pi messung ##################
############################################################

tau = np.arange(12., 270., 2.) # ns

sequence = [(['laser'],5000)]
for t in tau:
    
    # # # rabi
    sequence += [ (['mw'],t), (['laser', 'trigger'],3000), ([],1000) ]
    
power_a = -3 # dBm
power_b = -9 # dBm

frequency_a = 2.56329954e+09 # Hz
frequency_b = 2.56263901e+09 # Hz

microwave.setOutput(power_a, frequency_a)
microwave_2.setOutput(power_b, frequency_b)


###############
# acquisition
################

n_bins = 3000

binwidth = 1 # ns

n_detect = len(tau) # number of laser pulses

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
    y, profile, edge = spin_state(mat, 1.0, 300.)
    return y
    
def savePlot(filename):
    import matplotlib.pyplot as plt
    x=fetch() #.reshape((len(tau),4))
    plt.plot(tau,x)
    fitPoints = rabiFit( tau,x)
    plt.plot(tau,fitPoints(tau), "r-")
    plt.xlabel('tau [ns]')
    plt.ylabel('counts [a.u.]')
    plt.savefig("BP/Results/"+filename)
    plt.clf()

def plot(save = False):
    shell.close('all')
    x=fetch() #.reshape((len(tau),4))
    shell.figure()
    shell.plot(tau,x)
    shell.hold(True)
    fitPoints = rabiFit( tau,x)
    shell.plot(tau,fitPoints(tau), "r-")
    shell.xtitle('tau [ns]')
    shell.ytitle('counts [a.u.]')
    if save:
        np.savetxt("BP/Results/Rabi.txt",x);
        np.savetxt("BP/Results/RabiTau.txt",tau);
        np.savetxt("BP/Results/RabiFit.txt",fitPoints(tau));

def stop():
    fc.Halt()
    pg.High(['laser'])
    microwave.Off()
    microwave_2.Off()
    
