"""
Determination of magnetic field from the Zeeman shift of several NVs

Problem statement

Suppose we have a single crystalline bulk diamond. Suppose the crystal
is placed in a homogeneous magnetic field B. NV centers can have 4 possible
orientations with respect to the crystal lattice. We consider N single NVs
who's orientations are unknown. Using ODMR measurements we can determine
the ESR transitions of each NV. 

    v1_i=e1_i-e0_i
    v2_i=e2_i-e0_i

From the spin Hamiltonian model we can compute the energies e0_i, e1_i, e2_i.
We assume the energies are ordered with increasing energy. Then we can directly relate
measured energies to calculated energies.

We will use a brute force method to find the magnetic field. To this end, we compute
the transition frequencies v1 and v2 for a predetermined
set of magnetic fields. For numerical efficiency, we will pre-compute the ESR transitions
and store the data on the disc.

The search will typically be performed in three steps:

1. coarse brute force search using pre-computed ESR data.
2. fine brute force search using on-demand calculation on a fine mesh
3. non-linear least-square fitting

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

def v1v2_table(nvs,rotations,bx,by,bz):
    table=np.empty((len(bx),len(by),len(bz),len(nvs),2))
    for i,bxi in enumerate(bx):
        sys.stdout.write('\r%i%%'%int(round(i/float(len(bx))*100)))
        sys.stdout.flush()
        for j,byj in enumerate(by):
            for k,bzk in enumerate(bz):
                b = np.array((bxi,byj,bzk))
                for m in range(4):
                    table[i,j,k,m] = nvs[m].transitions(np.dot(rotations[m],b))
        sys.stdout.write('\r')
        sys.stdout.flush()
    return table

def landscape_transitions(orientations,transitions,table):
    return np.sum(np.sum((table[:,:,:,orientations,:]-transitions)**2,axis=-1),axis=-1)

def landscape_splittings(orientations,splittings,table):
    return np.sum((table[:,:,:,orientations,1]-table[:,:,:,orientations,0]-splittings)**2,axis=-1)

def index_minimum(landscape):
    i_flat=landscape.argmin()
    return np.unravel_index(i_flat,landscape.shape)

def chi_transitions(nvs,rotations,orientations,transitions,s_transitions=None):
    """Chi for transition frequencies"""
    if s_transitions is None:
        def chi(b):
            theo = np.empty_like(transitions)
            for i in range(len(transitions)):
                theo[i] = nvs[i].transitions(np.dot(rotations[orientations[i]],b))
            return (theo-transitions).flatten()
    else:
        def chi(b):
            theo = np.empty_like(transitions)
            for i in range(len(transitions)):
                theo[i] = nvs[i].transitions(np.dot(rotations[orientations[i]],b))
            return ((theo-transitions)/s_transitions).flatten()
    return chi

def chi_splittings(nvs,rotations,orientations,splittings,s_splittings=None):
    """Chi for splittings."""
    if s_splittings is None:
        def chi(b):
            theo = np.empty((len(splittings),2))
            for i in range(len(splittings)):
                theo[i] = nvs[i].transitions(np.dot(rotations[orientations[i]],b))
            return theo[:,1]-theo[:,0]-splittings
    else:
        def chi(b):
            theo = np.empty((len(splittings),2))
            for i in range(len(splittings)):
                theo[i] = nvs[i].transitions(np.dot(rotations[orientations[i]],b))
            return (theo[:,1]-theo[:,0]-splittings)/s_splittings
    return chi

def generate_table(spin_models, rotations, b_range, b_delta, table_file):
    # magnetic field range
    b = np.arange(-b_range,b_range,b_delta)
    table = v1v2_table(spin_models,rotations,b,b,b)
    fil=open(table_file,'wb')
    cPickle.dump((spin_models,rotations,b,b,b,table),fil,1)
    fil.close()

def load_table(filename):
    fil=open(table_file)
    spin_models,rotations,bx,by,bz,table=cPickle.load(fil)
    fil.close()
    return spin_models,rotations,bx,by,bz,table

def find_field(spin_models, rotations, orientations, bx, by, bz, table, transitions, s_transitions=None, method='transitions'):
    """
    Try to find the magnetic field from given ESR frequencies or alternatively ESR splittings
    of 3 or four different NVs.
    
    Input:
        spin_models:    3- or 4-tuple of TripletSpin. Here you can pass
                        D,E, and g-factor for the individual NVs
        orientations:   list of orientations of the NVs ( 0-->(1,1,1), 1-->(-1,-1,1), 2-->(1,-1,-1), 3-->(-1,1,-1) )
        bx, by, bz:     The x,y,z magnetic field mesh over which the table was computed
        table:          table of pre-computed ESR transitions corresponding to the spi-models
                        and magnetic field meshes. 
        transitions:    (3,2) or (4,2) array containing the ESR transition frequencies
                        of three or four different NVs.
        s_transitions:  standard deviations of the transitions frequencies or None
        method:         Determines the parameter used for the fitting. Can be either 'transitions'
                        or 'splittings'. In the prior case, the transitions are fitted
                        in the latter case, the splittings, i.e. the difference between
                        the transitions are fitted.
                        
    Returns:
        b :             3-tuple of vectors: b-field vectors after the three stages
                            stage one: brute force using lookup table
                            stage two: refined brute force
                            stage three: nonlinear least-square minimization
        s_b:            standard deviation of b-field from least-square minimization (stage three)
        chisqr:         3-tuple: chisqr deviations after the three stages
        q:              quality of fit from least-square minimization (only meaningful if
                        standard deviations of the transition frequencies were provided)
    """

    if method=='transitions':
        # stage one: lookup
        i,j,k = index_minimum(landscape_transitions(orientations,transitions,table))
        b1=np.array((bx[i],by[j],bz[k]))
        
        # stage two: refined brute force
        bxf=np.linspace(bx[i-1],bx[i+1],11)
        byf=np.linspace(by[j-1],by[j+1],11)
        bzf=np.linspace(bz[k-1],bz[k+1],11)
    
        fine_table=v1v2_table(spin_models,rotations,bxf,byf,bzf)
        i,j,k = index_minimum(landscape_transitions(orientations,transitions,fine_table))        
        b2=np.array((bxf[i],byf[j],bzf[k]))
        
        # stage three: nonlinear least square minimization
        result = scipy.optimize.leastsq(chi_transitions(spin_models,rotations,orientations,transitions,s_transitions), b2, full_output=True)
    elif method=='splittings':
        splittings = transitions[:,1]-transitions[:,0]
        if s_transitions is not None:
            s_splittings = (s_transitions[:,1]**2+s_transitions[:,0]**2)**0.5
        else:
            s_splittings = None

        # stage one: lookup
        i,j,k = index_minimum(landscape_splittings(orientations,splittings,table))
        b1=np.array((bx[i],by[j],bz[k]))

        # stage two: refined brute force
        bxf=np.linspace(bx[i-1],bx[i+1],11)
        byf=np.linspace(by[j-1],by[j+1],11)
        bzf=np.linspace(bz[k-1],bz[k+1],11)
    
        fine_table=v1v2_table(spin_models,rotations,bxf,byf,bzf)
        i,j,k = index_minimum(landscape_splittings(orientations,splittings,fine_table))
        b2=np.array((bxf[i],byf[j],bzf[k]))
        
        # stage three: nonlinear least square minimization
        result = scipy.optimize.leastsq(chi_splittings(spin_models,rotations,orientations,splittings), b2, full_output=True)

    # resulting b
    b3 = result[0]
    
    # error of b
    s_b3 = np.diag(result[1])**0.5
    
    # goodness of fit analysis
    chi0 = result[2]['fvec']
    chisqr0 = np.sum(chi0**2)
    nu = 2*len(transitions) - 3
    
    # quality of the fit
    q = scipy.special.gammaincc(0.5*nu,0.5*chisqr0)
    
    if method=='transitions':
        chisqr1=np.sum(chi_transitions(spin_models,rotations,orientations,transitions)(b1)**2)
        chisqr2=np.sum(chi_transitions(spin_models,rotations,orientations,transitions)(b2)**2)
        chisqr3=np.sum(chi_transitions(spin_models,rotations,orientations,transitions)(b3)**2)
    elif method=='splittings':
        chisqr1=np.sum(chi_splittings(spin_models,rotations,orientations,splittings)(b1)**2)
        chisqr2=np.sum(chi_splittings(spin_models,rotations,orientations,splittings)(b2)**2)
        chisqr3=np.sum(chi_splittings(spin_models,rotations,orientations,splittings)(b3)**2)

    return (b1, b2, b3), s_b3, (chisqr1, chisqr2, chisqr3), q
    
if __name__=='__main__':

    """
    # generate table_file

    # use NV's with standard parameters D=2870.6 MHz, E=0.0 MHz, g=2.0028
    # here you can pass individual parameters for each of the NVs
    nvs=(TripletSpin(),TripletSpin(),TripletSpin(),TripletSpin())

    # NV orientations
    nv_axis = np.array(((1,1,1), (-1,-1,1), (1,-1,-1), (-1,1,-1))) / 3**0.5 

    n0=np.array((0,0,1))

    rotations=np.empty((4,3,3))

    for i,n in enumerate(nv_axis):
        nr,phi = axis_angle(n,n0)
        rotations[i] = rotation_matrix(nr,phi)

    # generate table with magnetic field in the range -200 to 200 Gauss for bx, by, bz 
    generate_table(nvs,rotations,200.,2.,'table_200.pys')
    """

    # load pre-computed data
    if not 'table' in dir():
        print 'loading table file...'
        fil=open('table_200.pys')
        spin_models,rotations,bx,by,bz,table=cPickle.load(fil)
        fil.close()

    # normally we have as experimental data the ESR resonance lines.
    # in this case, we have only the splittings and compute 'fake'
    # transition frequencies
    splittings = np.array((31.99, 90.93, 103.5, 162.6))
    
    transitions = np.vstack((np.zeros_like(splittings), splittings)).transpose()

    orientations = [0,1,2,3]

    b, s_b, chisqr, q = find_field(spin_models,orientations,bx,by,bz,table,transitions,method='splittings')

    print np.linalg.norm(b[2])
    
    