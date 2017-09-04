def cosinusExp(a, x0, T, c,T1):
    """Returns a Cosinus function.
    
        f = a\cos(2\pi(x-x0)/T)+c
    
    Parameter:
    
    a    = amplitude
    T    = period
    x0   = shift position
    c    = offset in y-direction
    """
    return lambda x: np.exp(-x/T1)*a*np.cos( 2*np.pi*(x-x0)/float(T)) + c
    
    
def Cosinus_phaseEstimator(x, y,T1=3000):
    c = y.mean()
    a = 2**0.5 * np.sqrt( ((y-c)**2).sum() )
    # better to do estimation of period from
    Y = np.fft.fft(y)
    N = len(Y)
    D = float(x[1] - x[0])
    i = abs(Y[1:N/2+1]).argmax()+1
    T = (N * D) / i
    x0 = 0.0
    return (a, x0, T, c, T1)