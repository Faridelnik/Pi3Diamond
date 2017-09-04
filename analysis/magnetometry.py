"""
Determination of magnetic field from the Zeeman shift of several NVs.

Problem statement

Suppose we have a single crystalline bulk diamond. Suppose the crystal
is placed in a homogeneous magnetic field B. NV centers can have 4 possible
orientations with respect to the crystal lattice. We consider N single NVs
who's orientations are known. If the orientations are unknown, we
can assume specific orientations. Thereby we fix how the the lab coordinate
system is related to the cartesian crystal coordinate. Using ODMR measurements
we can determine the ESR transitions of each NV. 

    v1_i=e1_i-e0_i
    v2_i=e2_i-e0_i

From the spin Hamiltonian model we can compute the energies e0_i, e1_i, e2_i.
We assume the energies are ordered with increasing energy. Then we can directly relate
measured energies to calculated energies.

If E<<B<<D, (i.e. strain is small compared to the Zeeman shift and
non-axial Zeeman shifts are negligible) we can use the following
linear approxmation to estimate the magnetic field vector.

For every NV, we can estimate the absolute value of
the projection of the magnetic field b_i onto the NV axis from the
Zeeman splitting s_i through

    |b_i| = s_i / ( 2 * 2.8 MHz / Gauss)

Below we will evaluate how we can estimate the field B from the |b_i|
(up to certain ambiguity of the solution coming from the |  | in the equation).


Unit vectors along the four possible NV axis.
 
We consider a cube with a tetrahedron inside. The center
of the tetrahedron coincides with the center of the cube.

The four possible NV directions point along diagonals of the cube.

A possible choice is

    v_1 = (1,1,1)
    v_2 = (-1,-1,1)
    v_3 = (1,-1,-1)
    v_4 = (-1,1,-1)

In the following we will use the normalized vectors

    n_i = v_i / sqrt(3)

Finding the magnetic field from given splittings.

Let B be the magnetic field. Let there be N NVs with orientations
n_i. Let b_i be the magnetic field components parallel to the NVs.

We have

   n_1 * B = b_1
       .
       .
       .
   n_N * B = b_N

We write this in matrix form

    A B = b

Where A is a Nx3 matrix, B is a 3-vector, b is an N-vector. If we use
4 different NVs, this is an over determined system. We can find
the best solution in the least square sense by singular value decomposition.

Singular value decomposition decomposes matrix A into three matrices

    A = U S V

Where U and V are unitary and S is diagonal. U is Nx3, S is 3x3 and V is 3x3.

So, we have

    U S V B = b
   
Thus

    B =  V^T S^-1 U^T b

We can use this to find the field B if we know the signs of each equation,
such that we can remove the | |. However, in reality, the signs are unknown,
so we have to try all possible combinations of signs.

For every set of signs we compute the chisqr as

    A = { n_i }

    A = U S V

    B' =  V^T S^-1 U^T b
    
    b' = A B'

    chisqr = (b - b')**2

Chisqr is minimal when the set of signs is the correct one.

Helmut Fedder <helmut.fedder@gmail.com>

first version: 31 May 2012
"""

import numpy as np

import scipy.optimize
import scipy.special

import cPickle
import sys

# transformations to NV coordinate systems
def axis_angle(n0,n1):
    """Compute the rotation axis and angle to transform n0 to n1.
    n0 and n1 are of length 1."""
    n = np.cross(n0,n1)
    norm_n = np.linalg.norm(n)
    if norm_n == 0.:
        n = np.array((0.,0.,1.))
        phi = 0.0
    else:
        n=n/norm_n
        phi = np.arccos(np.dot(n0,n1))
    return n, phi

def rotation_matrix(n,phi):
    """Compute the rotation axis from an axis and angle."""
    x,y,z=n
    c=np.cos(phi)
    s=np.sin(phi)
    t=1-c
    R=np.array((( t*x*x + c  , t*x*y - z*s, t*x*z + y*s),
                ( t*y*x + z*s, t*y*y + c  , t*y*z - x*s),
                ( t*z*x - y*s, t*z*y + x*s, t*z*z + c  )))
    return R

class TripletSpin():
    """Triplet state Hamiltonian with D, E and Zeeman shift."""
    
    def __init__(self,D=2870.6,E=0.0,g=2.0028*9.27400915/6.626068):
        self.D=D
        self.E=E
        self.g=g
    
    def energy_levels(self,B):
        """Calculate the energy levels for given magnetic field."""
        D=self.D
        E=self.E
        g=self.g
        H = np.array(((           0.,  1j*g*B[1],  -1j*g*B[0]),
                      (   -1j*g*B[1],        D-E,   1j*g*B[2]),
                      (    1j*g*B[0], -1j*g*B[2],         D+E)))
        e=np.linalg.eigvalsh(H)
        e.sort()
        return e 

    def transitions(self,B):
        """
        Calculate the ESR transitions for given external field.
        Note that this method of sorting the energies fails above the LAC.
        """
        e=self.energy_levels(B)
        return np.array((e[1]-e[0],e[2]-e[0])) 


class Magnetometer():

    axis = np.array(((1,1,1), (-1,-1,1), (1,-1,-1), (-1,1,-1))) / 3**0.5

    rotations=np.empty((4,3,3))

    for i,n in enumerate(axis):
        nr,phi = axis_angle(n,np.array((0,0,1)))
        rotations[i] = rotation_matrix(nr,phi)
    
    def __init__(self, hamiltonians):
                
        self.hamiltonians = hamiltonians

    def get_field(self, transitions, full_output=False):
        """
        Calculate the magnetic field from the +1 and -1
        energy levels of three or four NVs.
        
        Input:
            transitions    energies of the +1 and -1 level [MHz]
            [full_output]  defaults to 'False'. If 'True', return
                           also the estimated starting guesses
                           for the non-linear fitting and all
                           corresponding minimized fields and
                           chisqrs.
                           
        Returns:
            b                3-vector containing the magnetic field
                             vector
            [b_estimates]    4x3 or 8x3 array containing the magnetic
                             field vectors evaluated from linear
                             approximation of the hamiltonian that
                             are used as starting guesses for
                             non-linear minimization
            [b_candidates]   4x3 or 8x3 array containing all magnetic
                             field vectors obtained from non-linear
                             minimization
            [chisqr]         4 or 8 element array containing all
                             chisqrs obtained from non-linear
                             minimization
                           
        """
        # The strategy is to perform a non-linear fit
        # using the spin Hamiltonians.
        #
        # The nonlinear fit converges quickly
        # to the global minimum, provided that the starting
        # guess is chosen properly.
        #
        # To find a starting guess of the field, a
        # suitable linear approximation of the hamiltonian
        # can be used.
        # 
        # Currently only an estimate in the range
        # E<<B<<D is implemented. In this case,
        # the magnetic field can be estimated from
        # the energy difference between the +1 and -1
        # levels. However, this estimate is ambiguous
        # and 4 or 8 possible magnetic fields are
        # obtained.
        #
        # The global minimum can be found by trying
        # all different estimates as starting guess
        # and afterwards choosing the one that has the
        # smallest chisqr.
        
        # calculate parallel magnetic field components from transitions
        # and individual g_factors of the spin hamiltonians
        # (usually the g_factors will be all the same).
        g_factors = [ham.g for ham in self.hamiltonians]
        if len(transitions) == 3:
            g_factors = g_factors[:3]
        b_parallel = 0.5*(transitions[:,1]-transitions[:,0])/g_factors

        # get all possible (ambiguous) estimates of the magnetic
        # field
        b_estimates = self.intermediate_field_estimate(b_parallel)

        # construct the chi
        theo = np.empty_like(transitions)
        rotations = self.rotations
        hamiltonians = self.hamiltonians
        def chi(b):
            for i in range(len(transitions)):
                theo[i] = hamiltonians[i].transitions(np.dot(rotations[i],b))
            return (theo-transitions).flatten()

        # perform non-linear fit with all possible starting guesses and
        # evaluate the chisqr for each of them
        b_candidates = np.empty_like(b_estimates)
        chisqr = np.empty((len(b_estimates)))
        for i,b0 in enumerate(b_estimates):
            b = scipy.optimize.leastsq(chi, b0, full_output=True)[0]
            b_candidates[i] = b
            chisqr[i] = np.sum(chi(b)**2)
            
        b = b_candidates[chisqr.argmin()]
        
        if full_output:
            return b, b_estimates, b_candidates, chisqr 
        else:
            return b
    
    def intermediate_field_estimate(self, b_parallel):
        """
        Linear estimate of the field vector from the energy
        difference between +1 and -1 level. Valid as long as
        E<<B<<D. The solution is ambiguous due to the
        unknown sign in each of the linear equations.
        
        This function returns all possible solutions.
        
        Input:
            b_parallel   parallel magnetic field component
                         in Gauss for three or four NVs.
            
        Returns:
            b            (Nx3) array containing the ambiguous solutions for b.
        """
        b = np.empty((4,3))
        n_nvs = len(b_parallel)
        
        
        if n_nvs == 4: # use nv_s with largest splittings
            mask = np.arange(4)!=b_parallel.argmin()
            b_parallel = b_parallel[mask]
        elif n_nvs == 3:
            mask = np.array((True, True, True))
        else:
            raise ValueError('Array of NV-axis must be of length 3 or 4.')
        
        signs = np.array( ((1, 1, 1),
                           (1, 1,-1),
                           (1,-1, 1),
                           (1,-1,-1)
                           )
                         )
        
        n_signs=4
        
        b = np.empty((n_signs,3))
        
        for i,sign in enumerate(signs):
            A = (self.axis[mask].transpose()*sign).transpose()
            U, s, V = np.linalg.svd(A,full_matrices=False)
            Ainv = np.dot(V.transpose(),np.dot(np.diag(1./s),U.transpose()))
            b[i] = np.dot(Ainv,b_parallel)
            
        return b
    
if __name__=='__main__':

    """
    Simulate the fitting process.
    """

    # use NV Hamiltonians with default values for D,E, and g
    hams = (TripletSpin(),TripletSpin(),TripletSpin(),TripletSpin())

    magnetometer = Magnetometer(hams)
    
    # chose an applied magnetic field
    #b_input=np.array((0.5,10.,20.1))
    b_input=np.array((0.0,10.0,20.1)) # when one of the field components is 0, there are two possible solutions
    b_input=np.array((0.0,10.0,20.1)) # when two of the field components are 0, there are three possible solutions
    # ToDo: check for this case in the method and issue a warning

    b_parallel_input = np.array([ np.dot(b_input,axis) for axis in magnetometer.axis ])
    
    # calculate the resulting ESR transitions [MHz]
    transitions = np.array([ham.transitions(np.dot(magnetometer.rotations[i],b_input)) for i,ham in enumerate(hams[:-1])]).astype(float)
    
    g_factors = [ham.g for ham in magnetometer.hamiltonians]
    if len(transitions) == 3:
        g_factors = g_factors[:3]
    b_parallel = 0.5*(transitions[:,1]-transitions[:,0])/g_factors

    # put an error on the transitions, such as a global shift of the resonances (due to temperature drift --> D changes)
    # or random error due to measurement uncertainity
    #transitions += .1*np.random.random((4,2))
    #transitions += .1*np.ones((4,2))
    
    b_output,b_est,b_cand,chisqr = magnetometer.get_field(transitions,full_output=True)
    
    print b_input
    print b_output
    print np.linalg.norm(b_input)
    print [np.linalg.norm(b_i) for b_i in b_est]
    print [np.linalg.norm(b_i) for b_i in b_cand]
    print chisqr
    print b_est
    print b_cand

