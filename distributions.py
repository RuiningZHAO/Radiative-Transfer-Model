#! -*- coding: UTF-8 -*-
from numba import jit

@jit( nopython = True )
def modified_power_law( r_scaled, params ):

    '''
    Modified power law (A. Li type) for radial density distribution:
    n(r) ~ ( 1 - 1 / r_scaled )**beta * ( 1 / r_scaled )**gamma,
    where r_scaled = r / r_min

    Parameters
    ----------
    r_scaled : array_like
        Scaled shell radius i.e. r / r_min
    params : array_like
        Parameter list containing beta and gamma.
        beta  : scalar
            Larger beta means larger r_peak and weaker 9.7 micron feature,
            typically, 2 - 20. 
        gamma : scalar
            Larger gamma means smaller r_peak, stronger 9.7um feature and
            weaker 18um feature, typically, 1 - 4.

    Returns
    -------
    n_r : array_like
        n(r).
    '''

    # beta = params[0]; gamma = params[1]
    # r_peak = ( beta + gamma ) / gamma #?
    # dn_dr_max = ( 1 - 1 / r_peak )**beta * ( 1 / r_peak )**gamma
    # dn_dr = ( 1 - 1 / r_scaled )**beta * ( 1 / r_scaled )**gamma / dn_dr_max
    # dn_dr *= (r_scaled >= 1)

    beta = params[0]; gamma = params[1]
    n_r = ( 1 - 1 / r_scaled )**beta * ( 1 / r_scaled )**gamma
    n_r *= (r_scaled >= 1)

    return n_r