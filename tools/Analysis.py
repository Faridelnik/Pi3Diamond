import numpy
import numpy.fft
import pylab
import scipy.stats
import scipy.optimize
import cPickle
import Fit
#import elementtree.ElementTree as ET

#numpy.set_printoptions(linewidth=numpy.inf, threshold=numpy.inf)
pylab.ioff()

def fft(z, t):
    """Returns FFT and frequency mesh of vector 'z' and time mesh 't'."""
    Y = numpy.fft.fft(z)
    #Y[0] = 0.5*Y[0]
    N = len(Y)
    D = (t[-1] - t[0])  /  (len(t) - 1)
    return Y[:N/2+1]/float(N/2),  numpy.arange(N/2+1) / (N * D)

def autocorrelate(a):
    """Calculates autocorrelation of array 'a' with user supplied zero padding using FFT."""
    A = numpy.fft.fft(a)
    F = numpy.fft.ifft(A*A.conj()).real/len(a)*2
    N = len(F)
    return numpy.append(F[N/2:],F[0:N/2])

def correlate(a, b):
    """Calculates correlation between two arrays of same length, 'a' and 'b',
    with user supplied zero padding using FFT."""
    A = numpy.fft.fft(a)
    B = numpy.fft.fft(b)
    F = numpy.fft.ifft(A*B.conj()).real/len(a)*2
    N = len(F)
    return numpy.append(F[N/2:],F[0:N/2])


class Analysis(object):

    def Save(self, File):
        fil = open(File, 'wb')
        cPickle.dump(self, fil, 1)
        fil.close()

class AntiBunching(object):

    def __init__(self, y, AcquisitionBinwidth, Sweeps, RunTime):
        self.y = y
        self.AcquisitionBinwidth = AcquisitionBinwidth
        self.Sweeps = Sweeps
        self.RunTime = RunTime

    def Plot(self, Total, Background, file=None):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        p = 1 - Background / Total
        ax.plot(self.AcquisitionBinwidth*numpy.arange(len(self.y)), ( self.y / (0.25*Total**2 * (self.AcquisitionBinwidth*1e-9)**2 * self.Sweeps * len(self.y)) - (1-p**2) ) / p**2 )
        if file:
            pylab.savefig(file)
        else:
            fig.show()
        return fig

class ODMR(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def Frequency(self):
        if not hasattr(self, 'FitParameters'):
            self.Fit()
        return self.FitParameters[0]

    def Asymmetry(self, x0=2.87e9):
        x = self.x
        y = self.y
        i0 = x.searchsorted(x0)
        x1 = x[y[:i0].argmin()]
        x2 = x[i0+y[i0:].argmin()]
        return ( x0 - x1 ) / ( x2 - x0 )

    def Minimum(self):
        return self.x[self.y.argmin()]

    def Minima(self, fit=False, Bins=20):
        x = self.x
        y = self.y

        N, B = numpy.histogram(y, Bins)

        High = B[N.argmax()]
        Low = y.min()

        c = 0.5 * ( High + Low )

        dips = y < c
        edges = numpy.flatnonzero(dips[:-1] ^ dips[1:])
        edges = edges.reshape( (len(edges)/2, 2) )

        minpos = []
        minval = []
        for fall, rise in edges:
            i = fall+y[fall:rise].argmin()
            minpos.append( x[i] )
            minval.append( y[i] )

        if fit == True:
            p = Fit.Fit(self.x, self.y, Fit.TripleLorentzian, (minpos[0], minpos[1], minpos[2], 1e6, 1e6, 1e6, minval[0]-c, minval[1]-c, minval[2]-c, c))
            minpos = p[:3]

        return numpy.array(minpos)

    def EstimateMinima(self):
        x = self.x
        y = self.y

        c = scipy.stats.mode(y)[0][0]

        half = 0.5 * ( y.max() + y.min() )

        dips = y < half
        edges = numpy.flatnonzero(dips[:-1] ^ dips[1:])
        edges = edges.reshape( (len(edges)/2, 2) )

        minpos = []
        minval = []
        for fall, rise in edges:
            i = fall+y[fall:rise].argmin()
            minpos.append( x[i] )
            minval.append( y[i] )

        return (minpos[0], minpos[1], minpos[2], 2e5, 2e5, 2e5, minval[0]-c, minval[1]-c, minval[2]-c, c)

    def Fit(self):
        self.FitParameters, self.FitSuccess = self.FitLorentzian(self.y, self.x)
        return self.FitParameters, self.FitSuccess


    # TODO: fix this and move this to Fit
    def Lorentz2(self):
        self.FitParameters, self.FitSuccess = self.FitLorentzian2(self.y, self.x)
        return self.FitParameters, self.FitSuccess
    def FitLorentzian(self, y, x):
        """Returns (x0, gamma, amplitude, offset)
        the parameters of a Lorentzian fit to the ODMR measurement."""
        InitialParameters = self.EstimateFitParameters(y, x)
        errorfunction = lambda p: (self.Lorentzian(*p)(x) - y)
        return scipy.optimize.leastsq(errorfunction, InitialParameters)
    def FitLorentzian2(self, y, x):
        """Returns (x0, gamma, amplitude, offset)
        the parameters of a Lorentzian fit to the ODMR measurement."""
        InitialParameters = self.EstimateFitParameters2(y, x)
        errorfunction = lambda p: (self.Lorentzian2(*p)(x) - y)
        return scipy.optimize.leastsq(errorfunction, InitialParameters)
    def Lorentzian(self, x0, gamma, amplitude, offset):
        """Returns a Lorentzian with the given parameters"""
        return lambda x: offset + amplitude / numpy.pi  *   (  gamma / ( (x-x0)**2 + gamma**2 )  )
    def Lorentzian2(self, x0, x1, gamma0, gamma1, amplitude0, amplitude1, offset):
        """Returns a Lorentzian with the given parameters"""
        return lambda x: offset + amplitude0 / numpy.pi  *   (  gamma0 / ( (x-x0)**2 + gamma0**2 )  ) + amplitude1 / numpy.pi  *   (  gamma1 / ( (x-x1)**2 + gamma1**2 )  )
    def EstimateFitParameters(self, y, x):
        offset = scipy.stats.mode(y)[0][0]
        Y = numpy.sum(y-offset) / (x[-1] - x[0])
        y0 = y.min() - offset
        x0 = x[y.argmin()]
        #gamma = Y / (numpy.pi * y0)
        gamma = 5e6
        amplitude = y0 * numpy.pi * gamma
        return x0, gamma, amplitude, offset
    def EstimateFitParameters2(self, y, x):
        offset = scipy.stats.mode(y)[0][0]
        Y = numpy.sum(y-offset) / (x[-1] - x[0])
        y0 = y.min() - offset
        x0 = x[y.argmin()]
        x1 = x0
        #gamma = Y / (numpy.pi * y0)
        gamma0 = 5e6
        gamma1 = 5e6
        amplitude0 = y0 * numpy.pi * gamma0
        amplitude1 = amplitude0
        return x0, x1, gamma0 , gamma1,  amplitude0 , amplitude1,  offset

    def Plot(self, File=None, Minima=False):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.x, self.y)
        if hasattr(self, 'FitParameters'):
            ax.plot(self.x, self.Lorentzian(*self.FitParameters)(self.x))
        if Minima:
            for x in self.Minima(fit=True):
                ax.axvline(x)
                ax.text(x, self.y[self.x.searchsorted(x)], '%f'%(x*1e-9))
        if File:
            pylab.savefig(File)
        else:
            fig.show()
        return fig

    def Save(self, File):
        fil = open(File, 'wb')
        cPickle.dump(self, fil, 1)
        fil.close()


class Pulsed(object):

    def __init__(self, freq, power, sequence, y, Binwidth, Sweeps, RunTime):
        self.freq = freq
        self.power = power
        self.sequence = sequence
        self.y = y
        self.Binwidth = Binwidth
        self.Sweeps = Sweeps
        self.RunTime = RunTime

    def AnalyzeSequence(self, T, Delay=250, SearchWidth=50):
        #Delay          Offset [ns] between sequence time and timetag time
        #T              Window length [ns] (starting after Laser rise peak) during which counts are taken into account
        #SearchWidth    Window length [ns] around a laser pulse time offset (from sequence time) for finding the maximum count rate (in count data)
        self.z = [ ]
        self.zrise = [ ]
        self.zfall = [ ]
        self.trise = [ ]
        self.tfall = [ ]
        self.T = T
        t = Delay
        #pdb.set_trace()
        for step in self.sequence:
            if step[0] == ['LASER']:
                t1, t2, z1, z2 = self.AnalyzeLaserPulse(T, t, step[1], SearchWidth )
                self.trise.append( t1 )
                self.tfall.append( t2 )
                self.zrise.append( z1 )
                self.zfall.append( z2 )
                self.z.append( z1 / z2 )
            t += step[1]
        self.z = numpy.array(self.z)
        self.zrise = numpy.array(self.zrise)
        self.zfall = numpy.array(self.zfall)
        self.trise = numpy.array(self.trise)
        self.tfall = numpy.array(self.tfall)
        return self.z

    def AnalyzeLaserPulse(self, T, t0, PulseLength, SearchWidth):
        Di = int(T / self.Binwidth)
        i0 = int(t0 / self.Binwidth)
        I  = int(PulseLength  / self.Binwidth)
        di = int(SearchWidth / self.Binwidth)
        y = self.y
        i  = self.Trigger(y, i0, I, di)
        return i*self.Binwidth, (i+I)*self.Binwidth, y[i:i+Di].mean(), y[i+I-Di:i+I].mean()

    def Trigger(self, y, i0, I, di):
        #y          array of counts, index corresponds to timetagger bin number
        #i0         offset into y at which to start the search for maximum
        #I          length of laser pulse (converted to number of timetagger bins)
        #di         search window length (converted to number of timetagger bins)
        return i0 - di + numpy.argmax([ y[i:i+I].sum() for i in range(i0-di, i0+di) ])

    def PlotSequence(self, File=None):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.Binwidth * numpy.arange(self.y.shape[0]), self.y)
        ax.set_xlabel('t [ns]')
        ax.set_ylabel('counts')
        if hasattr(self, 'z'):
            for i in range(len(self.z)):
                ax.axvline(self.trise[i], color='red')
                ax.axvline(self.tfall[i], color='green')
                ax.plot((self.trise[i]+0.5*self.T,), (self.zrise[i],), 'r.', markersize=10)
                ax.plot((self.tfall[i]-0.5*self.T,), (self.zfall[i],), 'g.', markersize=10)
        if File is None:
            fig.show()
        else:
            fig.savefig(File)
        return fig

    def Dump(self, FileName):
        fil = open(FileName, 'wb')
        cPickle.dump(self, fil, 1)
        fil.close()


class Sequential(object):

    def __init__(self, freq, power, sequence, y, Binwidth, Cycles, RunTime):
        self.freq = freq
        self.power = power
        self.sequence = sequence
        self.y = y
        self.Binwidth = Binwidth
        self.Cycles = Cycles
        self.RunTime = RunTime

    def AnalyzeSequence(self, T=200, t0=0, t1=2600, SearchMax=600):
        self.T = T
        self.t0 = t0
        self.t1 = t1
        self.SearchMax = SearchMax
        z0 = [ ]
        z1 = [ ]
        self.yy = self.y.sum(0)
        flank = self.Trigger(self.yy, self.SearchMax)
        i0 = int(numpy.round((flank+t0)/self.Binwidth))
        i1 = int(numpy.round((flank+t1)/self.Binwidth))
        I = T / self.Binwidth
        for y in self.y:
            z0.append( y[i0:i0+I].mean() )
            z1.append( y[i1:i1+I].mean() )
        self.z0 = numpy.array(z0)
        self.z1 = numpy.array(z1)
        self.z  = self.z0 / self.z1
        return self.z

    def Trigger(self, y, SearchMax):
        high, low = self.HighLow(y)
        I = int(SearchMax/self.Binwidth)
        self.chisqr = [ self.ChiSquare(y, high, low, i, I) for i in range(I) ]
        self.delay = numpy.argmin(self.chisqr) * self.Binwidth
        return self.delay

    def HighLow(self, y):
        threshold = 0.5 * ( y.max() + y.min() )
        self.high = y[y>threshold].mean()
        self.low  = y.min()
        return self.high, self.low

    def ChiSquare(self, y, high, low, i, I):
        return numpy.sum((y[0:i].astype(float)-low)**2) + numpy.sum((y[i:I].astype(float)-high)**2)

    def PlotDelay(self, file=None):
        fig = pylab.figure()
        s1 = fig.add_subplot(121)
        s1.plot(self.Binwidth*numpy.arange(len(self.chisqr)), self.chisqr)
        s1.axvline(self.delay, color='red')
        s1.set_xlabel('t [ns]')
        s1.set_ylabel('chisqr [a.u.]')
        s2 = fig.add_subplot(122)
        s2.plot(self.Binwidth * numpy.arange(self.yy.shape[0]), self.yy)
        s2.axhline(self.high, color='red')
        s2.axhline(self.low, color='green')
        s2.axvline(self.delay, color='red')
        s2.set_xlabel('t [ns]')
        s2.set_ylabel('fluorescence [a.u.]')
        if file:
            pylab.savefig(file)
        else:
            fig.show()
        return fig

    def PlotSequence(self):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        di = numpy.max((int(10./self.Binwidth), 1))
        ax.plot(self.Binwidth * numpy.arange(numpy.prod(self.y.shape))[::di], self.y.flatten()[::di])
        ax.set_xlabel('t [ns]')
        ax.set_ylabel('counts')
        TT = self.y.shape[1] * self.Binwidth
        if hasattr(self, 'z'):
            for i in range(len(self.z)):
                ax.axvline(i*TT + self.delay, color='black')
                ax.axvline(i*TT + self.delay + self.t0, color='red')
                ax.axvline(i*TT + self.delay + self.t1 + self.T, color='green')
                ax.axvline((i+1)*TT, color='black')
                ax.plot((i*TT + self.delay + self.t0 + 0.5*self.T,), (self.z0[i],), 'r.', markersize=10)
                ax.plot((i*TT + self.delay + self.t1 + 0.5*self.T,), (self.z1[i],), 'g.', markersize=10)
        return fig

    #def PlotPower(self):
    #    fig = pylab.figure()
    #    ax = fig.add_subplot(111)
    #    ax.plot(self.TransmittedPower[:,0], self.TransmittedPower[:,1])
    #    ax.set_xlabel('t [s]')
    #    ax.set_ylabel('dP/P')
    #    return fig

    def Dump(self, FileName):
        fil = open(FileName, 'wb')
        cPickle.dump(self, fil, 1)
        fil.close()


class Rabi(Analysis):

    def __init__(self, t):
        self.t = t

    def Tpi(self):
        return self.t[self.z.argmin()]

    def Fit(self, periods=0):
        if(periods>0):
            p = self.FitCosinus(self.z, self.t-self.t[0], periods)
        else:
            score = []
            for per in range(10):
                p = self.FitCosinus(self.z, self.t-self.t[0], per)
                fitfunc = self.Cosinus(*p)
                success = sum((fitfunc(self.t - self.t[0]) - self.z)**2)
                score.append(success)
            i = numpy.argmin(score)
            p = self.FitCosinus(self.z, self.t-self.t[0], i)
        p[-1] = p[-1] + self.t[0]
        self.FitParameters = p
        self.RabiMean = p[0]
        self.RabiAmpl = p[1]
        self.RabiPeriod = p[2]
        self.Rabix0 = p[3]

    def Period(self):
        if not hasattr(self, 'FitParameters'):
            p = self.FitCosinus(self.z, self.t-self.t[0], periods)
            p[-1] = p[-1] + self.t[0]
            self.FitParameters = p
            self.RabiPeriod = p[2]
        self.RabiPeriod = self.FitParameters[2]
        return self.RabiPeriod

    def FitCosinus(self, y, x, periods=1, InitialParameters=None):
        """Returns (height, x, y, width)
        the gaussian parameters of a 2D distribution found by a fit"""
        if InitialParameters == None:
            InitialParameters = self.EstimateFitParameters(y, x, periods)
        errorfunction = lambda p: (self.Cosinus(*p)(x) - y)
        p, success = scipy.optimize.leastsq(errorfunction, InitialParameters)
        if p[1] < 0:
            p[1] = -p[1]
            p[-1] =  ( ( p[-1]/p[-2] + 0.5 ) % 1 ) * p[-2]
            p, success = scipy.optimize.leastsq(errorfunction, p)
        if p[-1] / p[-2] < 0:
            p[-1] = (p[-1] / p[-2] + 1) * p[-2]
            p, success = scipy.optimize.leastsq(errorfunction, p)
        return p

    def Cosinus(self, offset, amplitude, period, x0):
        """Returns a Cosinus function with the given parameters"""
        return lambda x: offset + amplitude*numpy.cos( 2*numpy.pi*(x-x0)/float(period) )

    def EstimateFitParameters(self, y, x, periods=1):
        offset = y.mean()
        amplitude = 2**0.5 * numpy.sqrt( ((y-offset)**2).sum() )
        step = (y-offset)>0

        # a fairly advanced algorithm to guess the Rabi frequency,
        # replaced by something simpler and hopefully more reliable by FR, 090620
        #trigger = step[0:-1] ^ step[1:]
        #i = 0
        #j = [ ]
        #for i in range(len(trigger)):
        #    if trigger[i]:
        #        j.append( x[i] )
        #period = 2 * (j[-1]-j[0]) / (len(j)-1)

        #simpler version, preferable for less than one full cycle of oscillations
        period = self.t[-1]/periods
        return offset, amplitude, period, 0.

    def Plot(self, file=None):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.t, self.z)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Flourescence [a.u]')
        if hasattr(self, 'FitParameters'):
            ax.plot(self.t, self.Cosinus(*self.FitParameters)(self.t))
            ax.set_title('rabi period %.2f ns'%self.FitParameters[2])
        if file:
            pylab.savefig(file)
        else:
            fig.show()
        return fig

    def FFT(self):
        Y = scipy.fft(self.z)
        Y[0] = 0.5*Y[0]
        N = len(Y)
        D = (self.t[-1] - self.t[0])  /  (len(self.t) - 1)
        return numpy.arange(N/2+1) / (N * D),  Y[:N/2+1]/float(N/2)

class Hahn(object):

    def __init__(self, Wait, Tpi2, Tpi, T3pi2):
        #Pulsed.__init__(self, freq, power, sequence, y, Binwidth, Sweeps, RunTime)
        self.t = Wait
        self.Tpi2 = Tpi2
        self.Tpi = Tpi
        self.T3pi2 = T3pi2

    def T2(self):
        p = self.FitExp(self.z, self.t)
        self.FitParameters = p
        return self.FitParameters[2]

    def FitExp(self, y, x):
        """Returns (T2, amplitude, offset)
        the parameters of an exponential decay of the echo signal"""
        InitialParameters = self.EstimateFitParameters(y, x)
        errorfunction = lambda p: (self.Exp(*p)(x) - y)
        p, success = scipy.optimize.leastsq(errorfunction, InitialParameters)
        return p

    def Exp(self, offset, amplitude, T2):
        """Returns an exponential decay function with the given parameters"""
        return lambda x: offset + amplitude*numpy.exp( -x/T2 )

    def EstimateFitParameters(self, y, x):
        offset = float(y.min())
        amplitude = float(y.max() - y.min())
        T2 = float(.5*abs(x[y.argmin()] - x[y.argmax()]))
        return offset, amplitude, T2

    def Plot(self, file=None):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.t, self.z[1:len(self.t)+1], label='pi2 - pi - 3pi2')
        ax.plot(self.t, self.z[len(self.t)+1:2*len(self.t)+1], label='pi2 - pi - pi2')
        #ax.plot(self.t, self.z[2*len(self.t)+1:3*len(self.t)+1], label='pi2 - pi - pi2')
        ax.legend()
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Flourescence [a.u]')
        if hasattr(self, 'FitParameters'):
            ax.plot(self.t, self.Exp(*self.FitParameters)(self.t))
        if file:
            pylab.savefig(file)
        else:
            fig.show()
        return fig

class HahnLongPuls(object):

    def __init__(self, Wait, Tpi2, Tpi, T3pi2):
        #Pulsed.__init__(self, freq, power, sequence, y, Binwidth, Sweeps, RunTime)
        self.t = Wait
        self.Tpi2 = Tpi2
        self.Tpi = Tpi
        self.T3pi2 = T3pi2

    def T2(self):
        p = self.FitExp(self.z, self.t)
        self.FitParameters = p
        return self.FitParameters[2]

    def FitExp(self, y, x):
        """Returns (T2, amplitude, offset)
        the parameters of an exponential decay of the echo signal"""
        InitialParameters = self.EstimateFitParameters(y, x)
        errorfunction = lambda p: (self.Exp(*p)(x) - y)
        p, success = scipy.optimize.leastsq(errorfunction, InitialParameters)
        return p

    def Exp(self, offset, amplitude, T2):
        """Returns an exponential decay function with the given parameters"""
        return lambda x: offset + amplitude*numpy.exp( -x/T2 )

    def EstimateFitParameters(self, y, x):
        offset = float(y.min())
        amplitude = float(y.max() - y.min())
        T2 = float(.5*abs(x[y.argmin()] - x[y.argmax()]))
        return offset, amplitude, T2

    def Plot(self, file=None):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.t, self.z[1:len(self.t)+1], label='pi2 - pi - pi2')
        #ax.plot(self.t, self.z[len(self.t)+1:2*len(self.t)+1], label='pi2 - pi - pi2')
        #ax.plot(self.t, self.z[2*len(self.t)+1:3*len(self.t)+1], label='pi2 - pi - pi2')
        ax.legend()
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Flourescence [a.u]')
        if hasattr(self, 'FitParameters'):
            ax.plot(self.t, self.Exp(*self.FitParameters)(self.t))
        if file:
            pylab.savefig(file)
        else:
            fig.show()
        return fig


class HahnEcho(object):

    def __init__(self, Wait, Tpi2, Tpi, T3pi2):
        #Pulsed.__init__(self, freq, power, sequence, y, Binwidth, Sweeps, RunTime)
        self.t = Wait
        self.Tpi2 = Tpi2
        self.Tpi = Tpi
        self.T3pi2 = T3pi2

    def T2(self):
        p = self.FitExp(self.z, self.t)
        self.FitParameters = p
        return self.FitParameters[2]

    def FitExp(self, y, x):
        """Returns (T2, amplitude, offset)
        the parameters of an exponential decay of the echo signal"""
        InitialParameters = self.EstimateFitParameters(y, x)
        errorfunction = lambda p: (self.Exp(*p)(x) - y)
        p, success = scipy.optimize.leastsq(errorfunction, InitialParameters)
        return p

    def Exp(self, offset, amplitude, T2):
        """Returns an exponential decay function with the given parameters"""
        return lambda x: offset + amplitude*numpy.exp( -x/T2 )

    def EstimateFitParameters(self, y, x):
        offset = float(y.min())
        amplitude = float(y.max() - y.min())
        T2 = float(.5*abs(x[y.argmin()] - x[y.argmax()]))
        return offset, amplitude, T2

    def Plot(self, file=None):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.t, self.z[1:len(self.t)+1], label='pi2 - pi - pi2')
        #ax.plot(self.t, self.z[len(self.t)+1:2*len(self.t)+1], label='pi2 - pi - pi2')
        #ax.plot(self.t, self.z[2*len(self.t)+1:3*len(self.t)+1], label='pi2 - pi - pi2')
        ax.legend()
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Flourescence [a.u]')
        if hasattr(self, 'FitParameters'):
            ax.plot(self.t, self.Exp(*self.FitParameters)(self.t))
        if file:
            pylab.savefig(file)
        else:
            fig.show()
        return fig

class PulsedRabi(Pulsed, Rabi):

    def __init__(self, freq, power, sequence, y, Binwidth, Microwave, Sweeps, RunTime):
        Pulsed.__init__(self, freq, power, sequence, y, Binwidth, Sweeps, RunTime)
        Rabi.__init__(self, Microwave)

class SequentialRabi(Sequential, Rabi):

    def __init__(self, freq, power, sequence, y, Binwidth, Microwave, Cycles, RunTime):
        Sequential.__init__(self, freq, power, sequence, y, Binwidth, Cycles, RunTime)
        Rabi.__init__(self, Microwave)

class PulsedHahn(Pulsed, Hahn):

    def __init__(self, freq, power, sequence, y, Binwidth, Wait, Sweeps, RunTime, Tpi2, Tpi, T3pi2):
        Pulsed.__init__(self, freq, power, sequence, y, Binwidth, Sweeps, RunTime)
        Hahn.__init__(self, Wait, Tpi2, Tpi, T3pi2)


class PulsedHahnEfield(Pulsed, Hahn):

    def __init__(self, freq, power, sequence, y, Binwidth, Wait, Sweeps, RunTime, Tpi2, Tpi, T3pi2):
        Pulsed.__init__(self, freq, power, sequence, y, Binwidth, Sweeps, RunTime)
        Hahn.__init__(self, Wait, Tpi2, Tpi, T3pi2)


class SequentialHahn(Sequential, Hahn):

    def __init__(self, freq, power, sequence, y, Binwidth, Wait, Cycles, RunTime, Tpi2, Tpi, T3pi2):
        Sequential.__init__(self, freq, power, sequence, y, Binwidth, Cycles, RunTime)
        Hahn.__init__(self, Wait, Tpi2, Tpi, T3pi2)


class SequentialHahnLongPuls(Sequential, HahnLongPuls):

    def __init__(self, freq, power, sequence, y, Binwidth, Wait, Cycles, RunTime, Tpi2, Tpi, T3pi2):
        Sequential.__init__(self, freq, power, sequence, y, Binwidth, Cycles, RunTime)
        HahnLongPuls.__init__(self, Wait, Tpi2, Tpi, T3pi2)

class SequentialHahnEcho(Sequential, HahnEcho):

    def __init__(self, freq, power, sequence, y, Binwidth, Wait, Cycles, RunTime, Tpi2, Tpi, T3pi2):
        Sequential.__init__(self, freq, power, sequence, y, Binwidth, [[0,0],[0,0]], Cycles, RunTime)
        HahnEcho.__init__(self, Wait, Tpi2, Tpi, T3pi2)


class T1(Sequential):

    def __init__(self, freq, power, sequence, y, Binwidth, Tau, Tpi, Cycles, RunTime):
        Sequential.__init__(self, freq, power, sequence, y, Binwidth, Cycles, RunTime)
        self.Tau = Tau
        self.Tpi = Tpi

    def T1(self):
        self.FitParameters = self.FitExp(self.z, self.Tau)
        return self.FitParameters[2]

    def FitExp(self, y, x):
        """Returns (T, amplitude, offset)
        the parameters of an exponential decay of the echo signal"""
        InitialParameters = self.EstimateFitParameters(y, x)
        errorfunction = lambda p: (self.Exp(*p)(x) - y)
        p, success = scipy.optimize.leastsq(errorfunction, InitialParameters)
        return p

    def Exp(self, offset, amplitude, T):
        """Returns an exponential decay function with the given parameters"""
        return lambda x: offset + amplitude*numpy.exp( -x/T )

    def EstimateFitParameters(self, y, x):
        offset = float(y.min())
        amplitude = float(y.max() - y.min())
        T = float(.5*abs(x[y.argmin()] - x[y.argmax()]))
        return offset, amplitude, T

    def Plot(self, file=None):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.Tau, self.z)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Flourescence [a.u]')
        if hasattr(self, 'FitParameters'):
            ax.plot(self.t, self.Exp(*self.FitParameters)(self.Tau))
        if file:
            pylab.savefig(file)
        else:
            fig.show()
        return fig



class Spectrum(object):

    def __init__(self, Y, DT):
        self.Y = Y
        self.DT = DT
        self.DF = 1./(2*len(Y)*DT)

    def Plot(self, file=None):
        fig = pylab.figure()
        ax  = fig.add_subplot(111)
        Y = self.Y[:len(self.Y)/2+1]
        ax.plot(self.DF*numpy.arange(len(Y)), Y)
        ax.set_xlabel('frequency [GHz]')
        ax.set_ylabel('power [a.u.]')
        if file:
            pylab.savefig(file)
        else:
            fig.show()
        return fig

class RabiSpectrum(Spectrum):

    def __init__(self, freq, power, Y, Binwidth):
        Spectrum.__init__(self, Y, Binwidth)
        self.freq = freq
        self.power = power


class PowerDrift(object):

    def __init__(self, freq, power, sequence, Binwidth, Y, Time, Sweeps, Power, TrackEvents):
        self.freq = freq
        self.power = power
        self.sequence = sequence
        self.Binwidth = Binwidth
        self.Y = Y
        self.Time = Time
        self.Sweeps = Sweeps
        self.Power = Power
        self.TrackEvents = TrackEvents

    def PlotPulse(self, File=None):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        y = numpy.array(self.Y).sum(0)
        ax.plot(self.Binwidth * numpy.arange(len(y)), y)
        if File:
            pylab.savefig(File)
        else:
            fig.show()
        return fig

    def Analyze(self, T, t0, t1, delay=500):
        i0 = int( (delay+t0) / self.Binwidth)
        i1 = int( (delay+t1) / self.Binwidth)
        Di = int(T / self.Binwidth)
        z = []
        for y in self.Y:
            z.append( y[i0:i0+Di].mean() / float(y[i1:i1+Di].mean()) )
        self.z = numpy.array(z)

    def Plot(self, TrackEvents=False, File=None):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.Time, self.z)
        if TrackEvents:
            for t in self.TrackEvents:
                ax.axvline(t, color='red')
        if File:
            pylab.savefig(File)
        else:
            fig.show()
        return fig

    def Save(self, File):
        fil = open(File, 'wb')
        cPickle.dump(self, fil, 1)
        fil.close()


class MovingPoint(object):

    def __init__(self, freq, power, Binwidth, Microwave, Y, Time, Sweeps, Power, TrackEvents):
        self.freq = freq
        self.power = power
        self.Binwidth = Binwidth
        self.Microwave = Microwave
        self.Y = Y
        self.Time = Time
        self.Sweeps = Sweeps
        self.Power = Power
        self.TrackEvents = TrackEvents

    def PlotPulse(self, File=None):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        y = numpy.array(self.Y).sum(0)
        ax.plot(self.Binwidth * numpy.arange(len(y)), y)
        if File:
            pylab.savefig(File)
        else:
            fig.show()
        return fig

    def Analyze(self, T, t0, t1, delay=500):
        i0 = int( (delay+t0) / self.Binwidth)
        i1 = int( (delay+t1) / self.Binwidth)
        Di = int(T / self.Binwidth)
        z = []
        for y in self.Y:
            z.append( y[i0:i0+Di].mean() / float(y[i1:i1+Di].mean()) )
        self.z = numpy.array(z)

    def Plot(self, TrackEvents=False, File=None):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.Time, self.z)
        if TrackEvents:
            for t in self.TrackEvents:
                ax.axvline(t, color='red')
        if File:
            pylab.savefig(File)
        else:
            fig.show()
        return fig

    def PlotRabi(self, File=None):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.Microwave, self.z)
        if File:
            pylab.savefig(File)
        else:
            fig.show()
        return fig

    def Save(self, File):
        fil = open(File, 'wb')
        cPickle.dump(self, fil, 1)
        fil.close()


class Trace(object):

    def __init__(self, Trace, Threshold, Pulse):
        self._Trace = Trace
        self._FilteredTrace = self.Filter(Trace, Threshold)
        self._BinaryTrace = Trace>=Threshold
        self._BinaryFilteredTrace = self._BinaryTrace[numpy.where(self._BinaryTrace[:-1] == 0)[0]+1]
        self._Threshold = Threshold
        self._Pulse = Pulse

    @staticmethod
    def Filter(Trace, Threshold):
        return Trace[numpy.where(Trace[:-1] < Threshold)[0]+1]

    @staticmethod
    def Hist(Y):
        Bins = numpy.arange(Y.min(), Y.max()+1, dtype=numpy.int32)
        N = numpy.zeros(Bins.shape)
        y0 = Bins[0]
        m = len(N)
        for y in Y:
            i = int(y-y0)
            if 0 <= i < m:
                N[i] += 1
        return N, Bins

    def Normal(self):
        return self._Trace

    def Filtered(self, Threshold=None):
        if Threshold is None:
            Threshold = self._Threshold
        return self._Trace[numpy.where(self._Trace[:-1] < Threshold)[0]+1]

    def Binary(self, Threshold=None):
        if Threshold is None:
            Threshold = self._Threshold
        return self._Trace >= Threshold

    def BinaryFiltered(self, Threshold=None):
        if Threshold is None:
            Threshold = self._Threshold
        BinaryTrace = self.Binary(Threshold)
        return BinaryTrace[numpy.where(BinaryTrace[:-1] == 0)[0]+1]

    def Mean(self, Filter=False, Binary=False):
        if Filter:
            if Binary:
                return self.BinaryFiltered().mean()
            else:
                return self.Filtered().mean()
        else:
            if Binary:
                return self.Binary().mean()
            else:
                return self.Normal().mean()

    def PlotTrace(self, File=None):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(self._Y)
        if File is None:
            fig.show()
        else:
            fig.savefig(File)
        return fig

    def Plot(self, File=None, xlim=None, fit=False):
        fig = pylab.figure(figsize=(10,6))
        N, Bins = self.Hist(self._Trace)
        ax = fig.add_subplot(121)
        ax.bar(Bins, N, align='center')
        if fit:
            FitParameters = Fit.Fit(Bins, N, Fit.DoubleGaussian, Fit.DoubleGaussianEstimator)
            ax.plot(Bins, Fit.DoubleGaussian(*FitParameters)(Bins))
            ax.axvline(self._Threshold, color='red', linestyle='-')
            ax.axvline(0.5*(FitParameters[2]+FitParameters[3]), color='green', linestyle='--')
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_title('unfiltered')
        ax = fig.add_subplot(122)
        N, Bins = self.Hist(self._FilteredTrace)
        ax.bar(Bins, N, align='center')
        if fit:
            FitParameters = Fit.Fit(Bins, N, Fit.DoubleGaussian, Fit.DoubleGaussianEstimator)
            ax.plot(Bins, Fit.DoubleGaussian(*FitParameters)(Bins))
            ax.axvline(self._Threshold, color='red', linestyle='-')
            ax.axvline(0.5*(FitParameters[2]+FitParameters[3]), color='green', linestyle='--')
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_title('conditional')
        if File is None:
            fig.show()
        else:
            fig.savefig(File)
        return fig

    def Save(self, File):
        if File[-4:].upper() == '.XML':
            ET.ElementTree(self.xml()).write(File)
        else:
            fil = open(File, 'wb')
            cPickle.dump(self, fil, 1)
            fil.close()

    def xml(self):
        Trace = ET.Element("Trace")
        s = ''
        for x in self._Trace:
            s += '%i '%x
        Trace.text = s
        Trace.attrib['Threshold'] = '%i'% self._Threshold
        Trace.attrib['Pulse'] = '%i'% self._Pulse
        return Trace


class NuclearRabi(object):

    def __init__(self, PROBE, ProbeRepetition, RFFreq, RFPower, RFTau, Traces, Threshold):

        self.RFFreq = RFFreq
        self.RFPower = RFPower
        self.PROBE = PROBE
        self.ProbeRepetition = ProbeRepetition
        self.RFTau = RFTau
        self.Traces = Traces
        self.Threshold = Threshold

    def Analyze(self, Threshold=None):
        if Threshold is None:
            Threshold = self.Threshold
        self.Normal = []
        self.Filtered = []
        self.Binary = []
        self.BinaryFiltered = []
        for i, data in enumerate(self.Traces):
            trace = Trace(data, self.Threshold, 1)
            self.Normal.append( trace.Normal() )
            self.Filtered.append( trace.Mean(Filter=True) )
            self.Binary.append( trace.Mean(Binary=True) )
            self.BinaryFiltered.append( trace.Mean(Filter=True, Binary=True) )
        self.Normal = numpy.array( self.Normal )
        self.Filtered = numpy.array( self.Filtered )
        self.Binary = numpy.array( self.Binary )
        self.BinaryFiltered = numpy.array( self.BinaryFiltered )

    def Fit(self):
        if not hasattr(self, 'Normal'):
            self.Analyze()
        p = Fit.Fit(self.RFTau, self.BinaryFiltered, Fit.Cosinus, Fit.CosinusEstimator)
        # if p[1] < 0:
            # p[1] = -p[1]
            # p[-1] =  ( ( p[-1]/p[-2] + 0.5 ) % 1 ) * p[-2]
            # p = Fit.Fit(self.RFTau, self.BinaryFiltered, Fit.Cosinus, p)
        # if p[-1] / p[-2] < 0:
            # p[-1] = (p[-1] / p[-2] + 1) * p[-2]
            # p = Fit.Fit(self.RFTau, self.BinaryFiltered, Fit.Cosinus, p)
        self.FitParameters = p
        self.RabiMean = p[0]
        self.RabiAmpl = p[1]
        self.RabiPeriod = p[2]
        self.Rabix0 = p[3]

    def Period(self):
        if not hasattr(self, 'FitParameters'):
            self.Fit()
        return self.RabiPeriod

    def Plot(self, File=None):
        if not hasattr(self, 'Normal'):
            self.Analyze()
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.RFTau, self.BinaryFiltered)
        ax.set_xlabel(r'$\tau [ns]$')
        ax.set_ylabel('spin flip probability')
        if hasattr(self, 'FitParameters'):
            ax.plot(self.RFTau, Fit.Cosinus(*self.FitParameters)(self.RFTau))
            ax.set_title('Rabi period %.2f ns'%self.FitParameters[2])
        if File is None:
            fig.show()
        else:
            fig.savefig(File)
        return fig

    def PlotHistogram(self, tau):
        i = self.RFTau.searchsorted( tau )
        return Trace(self.Traces[i], self.Threshold, 1).Plot()

    def Save(self, File):
        #if File[-4:].upper() == '.XML':
        #    ET.ElementTree(self.xml()).write(File)

        #if File[-4:].upper() == '.DAT':
        # self.Analyze()
        # X=self.RFTau
        # Y=self.BinaryFiltered
        # fil = open( File[-4:]+'.DAT','w')
        # for i in range(len(X)):
            # fil.write('%f   %f\n'%(X[i],Y[i]) )
        # fil.close()
        #else:
        fil = open(File, 'wb')
        cPickle.dump(self, fil, 1)
        fil.close()

class NuclearSweep(object):

    def __init__(self, PROBE, ProbeRepetition, RFFreq, RFPower, RFTau, Traces, Threshold):

        self.RFFreq = RFFreq
        self.RFPower = RFPower
        self.PROBE = PROBE
        self.ProbeRepetition = ProbeRepetition
        self.RFTau = RFTau
        self.Traces = Traces
        self.Threshold = Threshold

    def Analyze(self, Threshold=None):
        if Threshold is None:
            Threshold = self.Threshold
        self.Normal = []
        self.Filtered = []
        self.Binary = []
        self.BinaryFiltered = []
        for i, data in enumerate(self.Traces):
            trace = Trace(data, self.Threshold, 1)
            self.Normal.append( trace.Normal() )
            self.Filtered.append( trace.Mean(Filter=True) )
            self.Binary.append( trace.Mean(Binary=True) )
            self.BinaryFiltered.append( trace.Mean(Filter=True, Binary=True) )
        self.Normal = numpy.array( self.Normal )
        self.Filtered = numpy.array( self.Filtered )
        self.Binary = numpy.array( self.Binary )
        self.BinaryFiltered = numpy.array( self.BinaryFiltered )

    def PlotHistogram(self, f):
        i = self.RFFreq.searchsorted( f )
        return Trace(self.Traces[i], self.Threshold, 1).Plot()

    def Plot(self, File=None):
        if not hasattr(self, 'Normal'):
            self.Analyze()
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.RFFreq, self.BinaryFiltered)
        if File is None:
            fig.show()
        else:
            fig.savefig(File)
        return fig

    def Save(self, File):
        if File[-4:].upper() == '.XML':
            ET.ElementTree(self.xml()).write(File)
        else:
            fil = open(File, 'wb')
            cPickle.dump(self, fil, 1)
            fil.close()
    # def xml(self):
        # if not hasattr(self, 'Normal'):
            # self.Analyze()
        # xmlRoot = ET.Element("NuclearRabi")
        # xmlRoot.text = numpy.array_str(self.)[1:-1]
        # xmlRoot.attrib['Threshold'] = '%i'% self._Threshold
        # xmlRoot.attrib['Pulse'] = '%i'% self._Pulse
        # return Trace

    # fig = pylab.figure()
    # ax = fig.add_subplot(121)
    # ax.hist(trace, bins=50)
    # ax.set_xlim(50,250)
    # MeanNormal.append( trace.mean() )
    # FilteredTrace = trace[numpy.where(trace[:-1] < threshold)[0]+1]
    # ax = fig.add_subplot(122)
    # ax.hist(FilteredTrace, bins=50)
    # ax.set_xlim(50,250)
    # fig.savefig(path+'%.f.png'%F[i])
    # MeanFiltered.append( FilteredTrace.mean() )
    # BinaryTrace = trace>=threshold
    # BinaryNormal.append( BinaryTrace.mean() )
    # BinaryFilteredTrace = BinaryTrace[numpy.where(BinaryTrace[:-1] == 0)[0]+1]
    # BinaryFiltered.append( BinaryFilteredTrace.mean() )
# fig = pylab.figure(figsize=(20,20))
# ax = fig.add_subplot(221)
# ax.plot(F, MeanNormal)
# ax.set_title('Mean')
# ax = fig.add_subplot(222)
# ax.plot(F, MeanFiltered)
# ax.set_title('MeanFiltered')
# ax = fig.add_subplot(223)
# ax.plot(F, BinaryNormal)
# ax.set_title('Binary')
# ax = fig.add_subplot(224)
# ax.plot(F, BinaryFiltered)
# ax.set_title('BinaryFiltered')
# fig.savefig(path+'sweep.png')













#~ class Trace(object):

    #~ def __init__(self, y, AcquisitionBinwidth, TransmittedPower, Sweeps, RunTime):
        #~ self.y = y
        #~ self.AcquisitionBinwidth = AcquisitionBinwidth
        #~ self.TransmittedPower = TransmittedPower
        #~ self.Sweeps = Sweeps
        #~ self.RunTime = RunTime

    #~ def autocorrelate(self):
        #~ return fft.autocorrelate(numpy.append(self.y, numpy.zeros(self.y.shape)))

#~ class RabiTrace(Trace):

    #~ def __init__(self, freq, power, y, Binwidth, TransmittedPower, Sweeps, RunTime):
        #~ Trace.__init__(self, y, Binwidth, TransmittedPower, Sweeps, RunTime)
        #~ self.freq = freq
        #~ self.power = power


#~ class ConstantPowerRabi(Pulsed):

    #~ def __init__(self, freq, power, sequence, y, Binwidth, MicrowaveOn, LaserDutyCycle, MicrowaveDutyCycle, Sweeps, RunTime):
        #~ Pulsed.__init__(self, freq, power, sequence, y, Binwidth, Sweeps, RunTime)
        #~ self.t = MicrowaveOn
        #~ self.LaserDutyCycle = LaserDutyCycle
        #~ self.MicrowaveDutyCycle = MicrowaveDutyCycle

    #~ def Period(self):
        #~ p = self.FitCosinus(self.z, self.t)
        #~ self.FitParameters = p
        #~ self.RabiPeriod = p[2]
        #~ return self.RabiPeriod

    #~ def FitCosinus(self, y, x):
        #~ """Returns (height, x, y, width)
        #~ the gaussian parameters of a 2D distribution found by a fit"""
        #~ InitialParameters = self.EstimateFitParameters(y, x)
        #~ errorfunction = lambda p: (self.Cosinus(*p)(x) - y)
        #~ p, success = scipy.optimize.leastsq(errorfunction, InitialParameters)
        #~ return p

    #~ def Cosinus(self, offset, amplitude, period, x0):
        #~ """Returns a Cosinus function with the given parameters"""
        #~ return lambda x: offset + amplitude*numpy.cos( 2*numpy.pi*(x-x0)/float(period) )

    #~ def EstimateFitParameters(self, y, x):
        #~ offset = y.mean()
        #~ amplitude = 2**0.5 * numpy.sqrt( ((y-offset)**2).sum() )
        #~ step = (y-offset)>0
        #~ trigger = step[1:] ^ step[0:-1]
        #~ i = 0
        #~ j = [ ]
        #~ for i in range(len(trigger)):
            #~ if trigger[i] == 1:
                #~ j.append( x[i] )
        #~ period = j[2]-j[0]
        #~ return offset, amplitude, period, 0.

    #~ def Plot(self):
        #~ fig = pylab.figure()
        #~ ax = fig.add_subplot(111)
        #~ ax.plot(self.t, self.z)
        #~ ax.set_xlabel('Time [ns]')
        #~ ax.set_ylabel('Flourescence [a.u]')
        #~ if hasattr(self, 'FitParameters'):
            #~ ax.plot(self.t, self.Cosinus(*self.FitParameters)(self.t))
        #~ return fig






    #~ def Period(self):
        #~ p = self.FitCosinus(self.z, self.t)
        #~ self.FitParameters = p
        #~ self.RabiPeriod = p[2]
        #~ return self.RabiPeriod

    #~ def FitCosinus(self, y, x):
        #~ """Returns (height, x, y, width)
        #~ the gaussian parameters of a 2D distribution found by a fit"""
        #~ InitialParameters = self.EstimateFitParameters(y, x)
        #~ errorfunction = lambda p: (self.Cosinus(*p)(x) - y)
        #~ p, success = scipy.optimize.leastsq(errorfunction, InitialParameters)
        #~ return p

    #~ def Cosinus(self, offset, amplitude, period, x0):
        #~ """Returns a Cosinus function with the given parameters"""
        #~ return lambda x: offset + amplitude*numpy.cos( 2*numpy.pi*(x-x0)/float(period) )

    #~ def EstimateFitParameters(self, y, x):
        #~ offset = y.mean()
        #~ amplitude = 2**0.5 * numpy.sqrt( ((y-offset)**2).sum() )
        #~ step = (y-offset)>0
        #~ trigger = step[1:] ^ step[0:-1]
        #~ i = 0
        #~ j = [ ]
        #~ for i in range(len(trigger)):
            #~ if trigger[i] == 1:
                #~ j.append( x[i] )
        #~ period = j[2]-j[0]
        #~ return offset, amplitude, period, 0.

    #~ def Plot(self):
        #~ fig = pylab.figure()
        #~ ax = fig.add_subplot(111)
        #~ ax.plot(self.t, self.z)
        #~ ax.set_xlabel('Time [ns]')
        #~ ax.set_ylabel('Flourescence [a.u]')
        #~ if hasattr(self, 'FitParameters'):
            #~ ax.plot(self.t, self.Cosinus(*self.FitParameters)(self.t))
        #~ return fig
