import numpy as np
from scipy.integrate import solve_ivp

def integrate_geodesic(x0, v0, range, dim=4):
    z0 = np.append(v0.x, x0.x)
    def geodeseq(t, z):
        a = np.zeros(2*dim)
        a[:dim] = -np.einsum('abc,bc->a', v0.coordinate_type.christoffel(z[dim:]), np.outer(z[:dim], z[:dim])).reshape(dim)
        a[dim:] = z[:dim].reshape(dim)
        #TODO: Add corrections to 4-velocity?
        return a

    return solve_ivp(geodeseq, range, z0, vectorized=True)