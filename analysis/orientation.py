    
"""
Determination of magnetic field from the Zeeman shift of several NVs

Problem statement

Suppose we have a single crystalline bulk diamond. Suppose the crystal
is placed in a homogeneous magnetic field B. NV centers can have 4 possible
orientations with respect to the crystal lattice. We consider N single NVs
whos orientations are unknown. Using ODMR measurements we can determine
the Zeeman shift of each NV. Assume that strain is small compared to
the Zeeman shift and non-axial Zeeman shifts are negligible.
Then, for every NV, we can estimate the absolute value of
the projection of the magnetic field b_i onto the NV axis from the
Zeeman splitting s_i through

    |b_i| = s_i / ( 2 * 2.8 MHz / Gauss)

Below we will evaluate we can determine the field B and the orientations
o_i of all NVs from the |b_i|.


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

We enumerate the vectors in the following way

    0 --> n_1
    1 --> n_2
    2 --> n_3
    3 --> n_4


Finding NV orientations and magnetic field from given splittings.

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

Where A is a Nx3 matrix, B is a 3-vector, b is an N-vector. In general
this is an over determined system. We can find the best solution
in the least square sense by singular value decomposition.

Singular value decomposition decomposes matrix A into three matrices

    A = U S V

Where U and V are unitary and S is diagonal. U is Nx3, S is 3x3 and V is 3x3.

So, we have

    U S V B = b
   
Thus

    B =  V^T S^-1 U^T b

We can use this to find the field B for a given set of orientations n_i.

Now assume that the NV orientations are not known. To find the right
NV orientations, we try all possible sets of orientations. For every
set of NV orientations we compute the chisqr as

    A = { n_i }

    A = U S V

    B' =  V^T S^-1 U^T b
    
    b' = A B'

    chisqr = (b - b')**2

Chisqr is minimal when the set of NV orientations is the correct one.

Now assume we do not know b (the field components parallel to the NV axis),
but rather their absolute values |b|.

To find the solution, we also have to try the two possible signs -b and b.
"""

import numpy as np

nv_axis = np.array(((1,1,1), (-1,-1,1), (1,-1,-1), (-1,1,-1))) / 3**0.5 

def solve(o, b):

    A = nv_axis[o]
    
    U, s, V = np.linalg.svd(A,full_matrices=False)

    Ainv = np.dot(V.transpose(),np.dot(np.diag(1./s),U.transpose()))
        
    x = np.dot(Ainv,b)

    return x


def deviation(o,b):
    B = solve(o,b)
    return np.sum((np.dot(nv_axis[o],B) - b)**2, axis=0)


def chisqr(os, bs):
    
    result = np.empty((len(os), bs.shape[1]))
    
    for i,oi in enumerate(os):
        result[i] = deviation(oi,bs)
    
    return result


def random_samples(N,M):
    return np.random.randint(0, 4, (N,M))

def random_signs(N,M):
    (-1)**np.random.randint(0,2,(N,M))
    
def systematic_samples(N):
    
    os = []

    for m in range(4**N):
        l = []
        for n in range(N-1,-1,-1):
            k = 4**n
            l.append(m/k)
            m = m%k
        os.append(l)

    return os


def systematic_signs(N):
    
    signs = []

    for m in range(2**N):
        l = []
        for n in range(N-1,-1,-1):
            k = 2**n
            l.append(m/k)
            m = m%k
        signs.append(l)

    return (-1)**np.array(signs)


def find_field(b):
    """
    Compute the magnetic field (least square optimum).
    
    Input:
       b = measured B0 fields (from 3 or more differently oriented NVs)
       
    Output:
       o = orientations of the NV's (up to 180 deg. ambiguity) described by
           an integer between 0 and 3, representing one of the four
           possible NV axis defined above. 
       B = magnetic field vector in the cartesian coordinate system
           (same coordinate system where the NV axis vectors are defined).
       landscape = least square landscape that was evaluated during computation 
    """

    N = len(b)
    
    M = 2**N
    
    orientations = systematic_samples(N)
    signs = systematic_signs(N)
    
    landscape = chisqr(orientations,(signs*b).transpose())
    
    ii = landscape.argmin()
    
    i = ii / M
    j = ii % M
    
    o = orientations[i]
    s = signs[j]
    
    B = solve(o, s*b)
    
    return o, B, landscape

def angles(o, B):
    return np.arccos(np.dot(nv_axis[o],B)/np.linalg.norm(B))*180/np.pi
    
def angles_mod90(o, B):
    """
    Convenience function to compute the angle between the NV axis
    and the magnetic field for a set of NV orientations 'o'. 
    """
    phi = np.arccos(np.dot(nv_axis[o],B)/np.linalg.norm(B))*180/np.pi
    return np.array([ p if p <= 90 else 180-p for p in phi ])
    
def components(o, B):
    """
    Convenience function to compute the parallel and perpendicular
    magnetic field components for a set of NV orientations 'o'. 
    """
    b_para = abs(np.dot(nv_axis[o],B))
    b_perp = np.array([ np.linalg.norm(v) for v in np.cross(nv_axis[o],B) ])
    return np.array((b_para, b_perp)).transpose()


if __name__ == '__main__':
    
    # magnetic field in Gauss
    B0 = np.array((10., 11., 12.))
        
    # directions of NVs
    directions = [0,1,2,3]
    
    # splitting in MHz (two times the zeeman shift)
    splittings = 2 * 2.8 * abs(np.dot(nv_axis[directions],B0))
    
    b0 = splittings / (2*2.8)

    o, B, landscape = find_field(b0)

    print 'B0: ', B0
    print 'B:  ', B
    
    print 'phi0: ', angles(directions, B0)
    print 'phi:  ', angles(o, B)

    print 'compo0: ', components(directions, B0)
    print 'compo:  ', components(o, B)

    directions = [0,1,2]
    
    # NV 19:10.6, 20:0.9, 23:1.1
    splittings = 2 * 2.8 * abs(np.array((10.6, 1.1, 0.9)))

    b0 = splittings / (2*2.8)

    o, B, landscape = find_field(b0)

    print 'B:  ', B
    print 'phi:  ', angles(o, B)
    print 'compo:  ', components(o, B)
