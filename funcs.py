#! -*- coding: UTF-8 -*-
import numpy as np
from numba import jit

def opt_filename( dust, type ):

    if dust == 'sil':

        if type == 'DL84':
            # Draine-Lee silicate (Draine & Lee 1984)
            filename = 'dlsi_opct.dat'
        elif type == 'LD01':
            # crystalline silicate (Li & Draine 2001)
            filename = 'crstsi_opct.dat'
        # elif type == 'LG97':
        #     # Li-Greenberg silicate (Greenberg & Li 1996; Li & Greenberg 1997)
        #     filename = 'lgsi_opct.dat'
        else:
            raise ValueError( 'There is no such type of silicate: {}'.format(type) )

    elif dust == 'car':

        if type == 'amca':
            # amorphous carbon (Rouleau & Martin 1991)
            filename = 'amca_opct.dat'
        elif type == 'corf':
            # carbonaceous organic refractory dust (Li & Greenberg 1997)
            filename = 'corf_opct.dat'
        else:
            raise ValueError( 'There is no such type of carbon: {}'.format(type) )

    elif dust == 'ice':

        if type == 'LG98':
            # H2O ice (Li & Greenberg 1998)
            filename = 'ice_opct.dat'
        else:
            raise ValueError( 'There is no such type of ice: {}'.format(type) )

    return filename

def column_density_grid( r_grid, theta_grid, radial_dist, normalization, dist_params, delta = 0.1 ):

    '''
    Subroutine to obtain optical depth that outgoing light ray
    experiences from each radial layer.

    Parameters
    ----------
    r_grid : array_like
        Shell radius in cm.
    theta_grid : array_like
        Angle.
    radial_dist : function
        Radial number density distribution n(r).
    normalization : Scalar
        Normalization factor for radial_dist which guarantees 
        ∫ n(r) * dr = 1.
    dist_params : array_like
        Distribution parameters passed into radial_dist.
    delta : Scalar
        Differential element along |r - r'| in au. Default 0.1.

    Returns
    -------
    N_grid_normed : 3-D array
        N_grid_normed[i, j, k] stores unit column density along |r - r'| where
        r and r' are vectors of length r_grid[i] and r_grid[j], respectively.
        The angle between two vectors is theta_grid[j].
    r_to_r : 3-D array
        |r - r'|.
    '''

    # Convert delta into CGS unit
    au = 1.49597871e+13 #
    delta *= au

    # Scaling
    scaling_factor = r_grid.min()
    r_grid_scaled = r_grid / scaling_factor
    delta_scaled  = delta  / scaling_factor

    # Initialize tau grid
    N_grid_scaled = np.zeros( (r_grid.shape[0], theta_grid.shape[0], r_grid.shape[0]) )
    # r to r'
    r_to_r = np.zeros( (r_grid.shape[0], theta_grid.shape[0], r_grid.shape[0]) )

    # Cos(theta)
    cos_theta_grid = np.cos( theta_grid )

    for i, r_strt in enumerate(r_grid_scaled):
        for j, cos_theta in enumerate(cos_theta_grid):
            for k, r_end in enumerate(r_grid_scaled):
                if (r_strt == r_end) & (cos_theta == 1.0):
                    N_grid_scaled[i, j, k] = 0.0
                    r_to_r[i, j ,k] = 0.0
                else:
                    r_to_r[i, j ,k] = np.sqrt( r_strt**2 + r_end**2 - 2 * r_strt * r_end * cos_theta )
                    cos_phi = ( r_strt - r_end * cos_theta ) / r_to_r[i, j ,k]
                    d = np.arange( delta_scaled, r_to_r[i, j ,k], delta_scaled )
                    r2 = r_strt**2 + ( r_to_r[i, j ,k] - d )**2 - 2 * r_strt * ( r_to_r[i, j ,k] - d ) * cos_phi; r2 *= (r2 > 0)
                    r = np.sqrt( r2 )
                    N_grid_scaled[i, j, k] = np.trapz( radial_dist( r, dist_params ), d )

    return N_grid_scaled * scaling_factor * normalization, r_to_r * scaling_factor

# @jit( nopython = True )
def column_density_outgoing( N_grid, r_grid, theta_grid, theta_thres ):

    '''
    Subroutine to obtain optical depth that outgoing light ray
    experiences from each radial layer.

    Parameters
    ----------
    N_grid : 3-D array
        N_grid[i, j, k] means column density that light ray 
        experiences from the i-th layer to k-th layer, the angle
        between which is the j-th theta.
    r_grid : array_like
        Shell radius in cm.
    theta_grid : array_like
        Angle.
    theta_thres : scalar
        Block angle.

    Returns
    -------
    N_out : 2-D array
        N_out[i, j] means column density that light ray experiences
        from i-th layer at j-th theta to outside along line of sight
        (LoS).
    '''

    # Initialize N_out
    N_out = np.zeros( (r_grid.shape[0], theta_grid.shape[0]) )
    r_to_r_max = np.zeros( (r_grid.shape[0], theta_grid.shape[0]) )

    sin_theta_grid = np.sin( theta_grid )
    cos_theta_grid = np.cos( theta_grid )
    for i, r in enumerate(r_grid):
        for j, theta in enumerate(theta_grid):
            # Distance from grid r to r_max along LOS
            if (r == r_grid[-1]) & (theta <= 0.5 * np.pi):
                r_to_r_max[i, j] = 0
            else:
                r_to_r_max[i, j] = np.sqrt( r_grid[-1]**2 - ( r * sin_theta_grid[j] )**2 ) - r * cos_theta_grid[j]
            # Angle between grid (r_max, phi) and grid (r, theta)
            cos_theta_minus_phi = ( r_grid[-1]**2 + r**2 - r_to_r_max[i, j]**2 ) / ( 2 * r * r_grid[-1] )
            if cos_theta_minus_phi >= 1.0:
                theta_minus_phi = 0
            elif cos_theta_minus_phi <= -1.0:
                theta_minus_phi = np.pi
            else:
                theta_minus_phi = np.arccos( cos_theta_minus_phi )
            # Get rid of the grids blocked by central star        
            if theta <= theta_thres:
                idx = np.argmin( np.abs( theta_minus_phi - theta_grid ) )
                N_out[i, j] = N_grid[ -1, idx, i ]

    return N_out, r_to_r_max

def cross_sections( a, lbda, filename ):

    '''
    Subroutine to calculate cross sections.

    Parameters
    ----------
    a : array_like
        Dust size (radius) in cm.
    lbda : array_like
        Wavelength in cm.
    filename : str
        Name of the file containing optical constants.

    Returns
    -------
    c_abs : 2D array
        Absorption cross section (cm^2).
    c_sca : 2D array
        Scattering cross section (cm^2).
    c_ext : 2D array
        Extinction cross section (cm^2).
    c_rp : 2D array
        Radiative pressure cross section (cm^2).
    g : 2D array
        g.
    '''
    import mie_cabs_ali

    # Load optical files
    with open( './lib/opt/{}'.format( filename ), 'r' ) as f:
        lbda_opt, n, k = np.loadtxt(f).T
        n = np.interp( lbda, lbda_opt * 1e-4, n )
        k = np.interp( lbda, lbda_opt * 1e-4, k )

    c_abs = np.zeros( (a.shape[0], lbda.shape[0]) )
    c_sca = np.zeros( (a.shape[0], lbda.shape[0]) )
    c_ext = np.zeros( (a.shape[0], lbda.shape[0]) )
    c_rp  = np.zeros( (a.shape[0], lbda.shape[0]) )
    g     = np.zeros( (a.shape[0], lbda.shape[0]) )
    for i in range( a.shape[0] ):
        for j in range( lbda.shape[0] ):
            c_abs[i, j], c_sca[i, j], c_ext[i, j], c_rp[i, j], g[i, j] = mie_cabs_ali.mie_cabs_ali( a[i], lbda[j], n[j], k[j] )

    return c_abs, c_sca, c_ext, c_rp, g

@jit( nopython = True )
def blackbody( lbda, T ):

    '''
    Blackbody Radiation.

    Parameters
    ----------
    lbda : array_like
        Wavelength (cm).
    T : scalar
        Temperature (K).

    Returns
    -------
    I_nu : array
        Intensity (erg/s/cm^3/sr) under T(K).
    '''

    # Math constants
    e  = 2.718281828459045
    # Physical constants in cgs
    h = 6.62607004e-27 # erg * s
    c = 2.99792458e+10 #  cm / s
    k = 1.38064852e-16 # erg / K

    I_nu = 2 * h * c**2 / ( lbda**5 ) / ( e**( h * c / (lbda * k * T) ) - 1 )

    return I_nu

@jit( nopython = True )
def delta_rate(T, lbda, heating_rate, c_abs):

    '''
    Function caculating the difference between heating rate and cooling rate.

    Parameters
    ----------
    T : scalar
        Temperature (K).
    lbda : array_like
        Wavelength (cm).
    heating_rate : scalar
        Heating rate (erg/s) which cooling rate (erg/s) needs to be equal to.
    c_abs : array_like
        Wavelength depentent absorption cross section (cm^2).

    Returns
    -------
    delta : scalar
        Heating rate minus calculated cooling rate under T (K).
    '''

    # Math constants
    e  = 2.718281828459045
    pi = 3.141592653589793
    # Physical constants in cgs
    h = 6.62607004e-27 # erg * s
    c = 2.99792458e+10 #  cm / s
    k = 1.38064852e-16 # erg / K

    F_nu = pi * ( 2 * h * c**2 / ( lbda**5 ) / ( e**( h * c / (lbda * k * T) ) - 1 ) )
    cooling_rate_nu = c_abs * F_nu
    dlbda = lbda[1:] - lbda[:-1]
    cooling_rate = np.sum( ( cooling_rate_nu[:-1] + cooling_rate_nu[1:] ) * dlbda / 2 )
    delta = heating_rate - cooling_rate

    return delta

def Equilibrium_Temperature(c_abs, lbda, F_nu_r, F_dust_emi, F_dust_sca, scatter = False):

    '''
    Subroutine to obtain equilibrium temperature as a function of dust 
    size and shell radius. Dust emission is also returned.

    Parameters
    ----------
    c_abs : 2-D array
        c_abs[i, j] means absorption cross section of i-th size at j-th 
        wavelength. (unit: cm^2)
    lbda : array_like
        Wavelength in cm.
    F_nu_r : 2-D array
        F_nu_r[i, j] means central star flux at j-th wavelength received 
        by grains in i-th layer. (unit: erg/s/cm^3)
    F_dust_emi, F_dust_sca : 2-D array
        Dust emission and scattering flux. Both have the same dimensions
        as F_nu_r. (unit: erg/s/cm^3)
    scatter: bool
        If True, F_dust_sca will be added into F_dust_r. Default `False`.

    Returns
    -------
    T_dust : 2-D array
        Equilibrium temperature of grains of j-th size in i-th layer
        is stored in T_dust[i, j]. (unit: K)
    F_nu_r_a_dust : 3-D array
        Wavelength dependent flux from grains of j-th size in i-th layer.
        Wavelength is the third axis. (unit: erg/s/cm^3)
    '''
    from scipy.optimize import brentq

    F_nu_r_tot = F_nu_r + F_dust_emi

    if scatter:
        F_nu_r_tot += F_dust_sca

    heating_rate_nu = np.einsum('ij, kj -> kij', c_abs, F_nu_r_tot) # erg/s @ different wavelengths
    # Intergrate wavelengths
    heating_rate = np.trapz( heating_rate_nu, lbda, axis = 2 ) # erg/s

    # Initialization
    T_dust = np.zeros( [heating_rate.shape[0], heating_rate.shape[1]] )
    F_nu_r_a_dust = np.zeros( [heating_rate.shape[0], heating_rate.shape[1], lbda.shape[0] ] )
    for i in range( heating_rate.shape[0] ):         # Iter over shell radius
        for j in range( heating_rate.shape[1] ):     # Iter over dust size
            # Look for zero point
            T_dust[i, j] = brentq( delta_rate, args = (lbda, heating_rate[i, j], c_abs[j, :]), a = 10, b = 4000, xtol = 1e-8, maxiter = 100, disp = True )
            F_nu_r_a_dust[i, j, :] = np.pi * blackbody( lbda, T_dust[i, j] )

    return T_dust, F_nu_r_a_dust

def debug():
    pass

if __name__ == '__main__':
    debug()  