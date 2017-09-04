import numpy

class Pulsed( ):
    
    def __init__(self, n_bins, bin_width, n_slots, channel, shot_trigger, sequence_trigger=None):
        self.n_bins = n_bins
        self.n_slots = n_slots
        data = numpy.zeros((n_slots,n_bins))
        m0 = int(n_bins/5)
        m = float(n_bins-m0)
        M = numpy.arange(m, dtype=float)
        n = float(n_slots)
        k = n_slots/2
        for i in range(n_slots):
            """Rabi Data"""
            data[i,m0:] = 30*numpy.cos(3*2*numpy.pi*i/n)*numpy.exp(-5*M/m)+100
            """Hahn Data        
            data[i,m0:] = 30*numpy.exp(-9*i**2/n**2)*numpy.exp(-5*M/m)+100
            """
            """Hahn 3pi2 Data
            if i < k:
                data[i,m0:] = 30*numpy.exp(-9*i**2/float(k**2))*numpy.exp(-5*M/m)+100
            else:
                data[i,m0:] = -30*numpy.exp(-9*(i-k)**2/float(k**2))*numpy.exp(-5*M/m)+100
            """
            """T1 Data
            data[i,m0:] = 30*numpy.exp(-3*i/n)*numpy.exp(-5*M/m)+100
            """
        self.data = data
        self.counter = 1

    def getData(self):
        self.counter += 1
        return numpy.random.poisson(self.counter*self.data)

    def getCounts(self):
        return self.counter
    
    def start(self):
        pass

    def stop(self):
        pass

class Countrate( ):
    
    def __init__(self, channel):
        self.rate = 0.
        
    def getData(self):
        self.rate += 1.
        return 1e5/(1+20./self.rate)
    
    def clear(self):
        pass

class Counter( ):
    
    def __init__(self, channel, bins_per_point, length):
        self.channel = channel
        self.seconds_per_point = float(bins_per_point)/800000000
        self.length = length
        
    def getData(self):
        return numpy.random.random_integers(100000,120000, self.length)*self.seconds_per_point
    