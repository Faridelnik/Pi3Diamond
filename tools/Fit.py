"""
This file is part of Diamond. Diamond is a confocal scanner written
in python / Qt4. It combines an intuitive gui with flexible
hardware abstraction classes.

Diamond is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Diamond is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with diamond. If not, see <http://www.gnu.org/licenses/>.

Copyright (C) 2009 Helmut Rathgen <helmut.rathgen@gmail.com>
"""

import numpy
import scipy.optimize, scipy.stats

def Cosinus(a, T, x0, c):
    """Returns a Cosinus function with the given parameters"""
    return lambda x: a*numpy.cos( 2*numpy.pi*(x-x0)/float(T) ) + c
setattr(Cosinus, 'Formula', r'$cos(c,a,T,x0;x)=a\cos(2\pi(x-x0)/T)+c$')

def CosinusEstimator(x, y):
    c = y.mean()
    a = 2**0.5 * numpy.sqrt( ((y-c)**2).sum() )
    # better to do estimation of period from
    Y = numpy.fft.fft(y)
    N = len(Y)
    D = float(x[1] - x[0])
    i = abs(Y[1:N/2+1]).argmax()+1
    T = (N * D) / i
    x0 = 0
    return a, T, x0, c

def CosinusNoOffset(a, T, x0):
    """Returns a Cosinus function with the given parameters"""
    return lambda x: a*numpy.cos( 2*numpy.pi*(x-x0)/float(T) )
setattr(Cosinus, 'Formula', r'$cos(a,T,x0;x)=a\cos(2\pi(x-x0)/T)$')

def CosinusNoOffsetEstimator(x, y):
    a = 2**0.5 * numpy.sqrt( (y**2).sum() )
    # better to do estimation of period from
    Y = numpy.fft.fft(y)
    N = len(Y)
    D = float(x[1] - x[0])
    i = abs(Y[1:N/2+1]).argmax()+1
    T = (N * D) / i
    x0 = 0
    return a, T, x0

def brot_transitions_upper(B, D, E, phase):
    return lambda theta: 3./2. * B**2/D * numpy.sin(theta + phase)**2 + ( B**2 * numpy.cos(theta + phase)**2 + (E + B**2/(2*D) * numpy.sin(theta+phase)**2)**2)**0.5 + D
    
def brot_transitions_lower(B, D, E, phase):
    return lambda theta: 3./2. * B**2/D * numpy.sin(theta + phase)**2 - ( B**2 * numpy.cos(theta + phase)**2 + (E + B**2/(2*D) * numpy.sin(theta+phase)**2)**2)**0.5 + D

def FCSTranslationRotation(alpha, tau_r, tau_t, N):
    """Fluorescence Correlation Spectroscopy. g(2) accounting for translational and rotational diffusion."""
    return lambda t: (1 + alpha*numpy.exp(-t/tau_r) ) / (N * (1 + t/tau_t) )
setattr(FCSTranslationRotation, 'Formula', r'$g(\alpha,\tau_R,\tau_T,N;t)=\frac{1 + \alpha \exp(-t/\tau_R)}{N (1 + t/\tau_T)}$')

def FCSTranslation(tau, N):
    """Fluorescence Correlation Spectroscopy. g(2) accounting for translational diffusion."""
    return lambda t: 1. / (N * (1 + t/tau) )
setattr(FCSTranslation, 'Formula', r'$g(\tau,N;t)=\frac{1}{N (1 + t/\tau)}$')

def Antibunching(alpha, c, tau, t0):
    """Antibunching. g(2) accounting for Poissonian background."""
    return lambda t: c*(1-alpha*numpy.exp(-(t-t0)/tau))
setattr(Antibunching, 'Formula', r'$g(\alpha,c,\tau,t_0;t)=c(1 - \alpha \exp(-(t-t_0)/\tau))$')

def Gaussian(c, a, x0, w):
    """Gaussian function with offset."""
    return lambda x: c + a*numpy.exp( -0.5*((x-x0)/w)**2   )
setattr(Gaussian, 'Formula', r'$f(c,a,x0,w;x)=c+a\exp(-0.5((x-x_0)/w)^2)$')

def DoubleGaussian(a1, a2, x01, x02, w1, w2):
    """Gaussian function with offset."""
    return lambda x: a1*numpy.exp( -0.5*((x-x01)/w1)**2   ) + a2*numpy.exp( -0.5*((x-x02)/w2)**2   )
setattr(Gaussian, 'Formula', r'$f(c,a1, a2,x01, x02,w1,w2;x)=a_1\exp(-0.5((x-x_{01})/w_1)^2)+a_2\exp(-0.5((x-x_{02})/w_2)^2)$')

def DoubleGaussianEstimator(x, y):
	center = (x*y).sum() / y.sum()
	ylow = y[x < center]
	yhigh = y[x > center]
	x01 = x[ylow.argmax()]
	x02 = x[len(ylow)+yhigh.argmax()]
	a1 = ylow.max()
	a2 = yhigh.max()
	w1 = w2 = center**0.5
	return a1, a2, x01, x02, w1, w2

# important note: lorentzian can also be parametrized with an a' instead of a,
# such that a' is directly related to the amplitude (a'=f(x=x0)). In this case a'=a/(pi*g)
# and f = a * g**2 / ( (x-x0)**2 + g**2 ) + c.
# However, this results in much poorer fitting success. Probably the g**2 in the numerator
# causes problems in Levenberg-Marquardt algorithm when derivatives
# w.r.t the parameters are evaluated. Therefore it is strongly recommended
# to stick to the parametrization given below.
def Lorentzian(x0, g, a, c):
    """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
    return lambda x: a / numpy.pi * (  g / ( (x-x0)**2 + g**2 )  ) + c
setattr(Lorentzian, 'Formula', r'$f(x0,g,a,c;x)=a/\pi (g/((x-x_0)^2+g^2)) + c$')

def LorentzianEstimator(x, y):
    c = scipy.stats.mode(y)[0][0]
    yp = y - c
    Y = numpy.sum(yp) * (x[-1] - x[0]) / len(x)
    ymin = yp.min()
    ymax = yp.max()
    if ymax > abs(ymin):
        y0 = ymax
    else:
        y0 = ymin
    x0 = x[y.argmin()]
    g = Y / (numpy.pi * y0)
    a = y0 * numpy.pi * g
    return x0, g, a, c

def DoubleLorentzian(x0, x1, g0, g1, a0, a1, c):
    """2 Lorentzian centered at x0/x1, with amplitude a0/a1, offset y0 and HWHM g0/g1."""
    return lambda x: a0 / numpy.pi * (  g0 / ( (x-x0)**2 + g0**2 )  )+a1 / numpy.pi * (  g1 / ( (x-x1)**2 + g1**2 )  ) + c
setattr(Lorentzian, 'Formula', r'$f(x0,g,a,c;x)=a/\pi (g/((x-x_0)^2+g^2)) + c$')

def DoubleLorentzianEstimator(x, y):
    c = scipy.stats.mode(y)[0][0]
    yp = y - c
    Y = numpy.sum(yp) * (x[-1] - x[0]) / len(x)
    ymin = yp.min()
    ymax = yp.max()
    if ymax > abs(ymin):
        y0 = ymax
    else:
        y0 = ymin
    if y.argmin()<2.87e9:
        x0 = x[y.argmin()]
        x1 = 2*2.87e9-x0
    else:
        x1 = x[y.argmin()]
        x0 = 2*2.87e9-x1
    g0 = Y / (numpy.pi * y0)
    g1=g0
    a0 = y0 * numpy.pi * g0
    a1=a0
    return x0, x1, g0, g1, a0, a1, c

def TripleLorentzian(x1, x2, x3, g1, g2, g3, a1, a2, a3, c):
    """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
    return lambda x: a1 / numpy.pi * (  g1**2 / ( (x-x1)**2 + g1**2 )  ) + a2 / numpy.pi * (  g2**2 / ( (x-x2)**2 + g2**2 )  ) + a3 / numpy.pi * (  g3**2 / ( (x-x3)**2 + g3**2 )  ) + c
setattr(Lorentzian, 'Formula', r'$f(x0,g,a,c;x)=a/\pi (g/((x-x_0)^2+g^2)) + c$')

def SumOverFunctions( functions ):
    """Creates a factory that returns a function representing the sum over 'functions'.
    'functions' is a list of functions. 
    The resulting factory takes as arguments the parameters to all functions,
    flattened and in the same order as in 'functions'."""
    def function_factory(*args):
        def f(x):
            y = numpy.zeros(x.shape)
            i = 0
            for func in functions:
                n = func.func_code.co_argcount
                y += func(*args[i,i+n])(x)
                i += n
        return f
    return function_factory


def Fit(x, y, Model, Estimator):
    """Perform least-squares fit of two dimensional data (x,y) to model 'Model' using Levenberg-Marquardt algorithm.\n
    'Model' is a callable that takes as an argument the model parameters and returns a function representing the model.\n
    'Estimator' can either be an N-tuple containing a starting guess of the fit parameters, or a callable that returns a respective N-tuple for given x and y."""
    if callable(Estimator):
        return scipy.optimize.leastsq(lambda pp: Model(*pp)(x) - y, Estimator(x,y))[0]
    else:
        return scipy.optimize.leastsq(lambda pp: Model(*pp)(x) - y, Estimator)[0]

class Gaussfit2D(object):

    def __init__(self, data):      
        self.data=data

    def gauss(self, A0, A, x0, y0, wx, wy, theta):
        wx = numpy.float(wx)
        wy = numpy.float(wy)
        #def f(x,y):
        #    x = (x-x0)*numpy.cos(theta) + (y-y0)*numpy.sin(theta)
        #    y = (x-x0)*numpy.sin(theta) + (y-y0)*numpy.cos(theta)
        #    return A0**2+A*A*numpy.exp(-((x/wx)**2+(y/wy)**2)/2)
        #return f
        return lambda x,y: A0**2+A*A*numpy.exp(-(((x0-x)/wx)**2+((y0-y)/wy)**2)/2)
       
    def moments(self, data):
        total = data.sum()
        X, Y = numpy.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        wx = numpy.sqrt(abs((numpy.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        wy = numpy.sqrt(abs((numpy.arange(row.size)-x)**2*row).sum()/row.sum())
        A = numpy.sqrt(data.max())
        A0 = numpy.sqrt(data.min())
        theta = 0
        return A0 , A , x , y , wx , wy, theta
       
    def fitgaussian(self, data):
        params = self.moments(data)
        errorfunction = lambda p: numpy.ravel(self.gauss(*p)(*numpy.indices(data.shape))-data)
        p, success = scipy.optimize.leastsq(errorfunction, params)
        return p
       
    def Execute(self, data=None):
        if data is None:
            data = self.data
        params = self.fitgaussian(data)      
        (A0, A , y , x , wx , wy, theta) = params
        #a  = A**2       
        #return x , y , a
        return A0**2, A**2, y, x, wx, wy, theta