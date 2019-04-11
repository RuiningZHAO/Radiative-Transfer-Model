import time
import numpy as np
from funcs import *
import astropy.units as u
from distributions import *
from tabulate import tabulate
import astropy.constants as const
import matplotlib.pyplot as plt

class shell(object):

    def __init__(self):
        
        # STELLAR PARAMETERS
        self.stellar_params = dict()
        self.stellar_params['M'] =  15
        self.stellar_params['R'] = 900
        self.stellar_params['Teff'] = 3300
        self.stellar_params['logg'] = 0.00

        # GRIDS
        self.grid_params = dict()
        # =====================================================================================|
        # WAVELENGTH GRID                                                                      |
        # -------------------------------------------------------------------------------------+
        self.grid_params['lbda_min']  = 0.0912 # Minimum Wavelength                   | um     |
        self.grid_params['lbda_max']  = 10000  # Maximum Wavelength                   | um     |
        self.grid_params['nlbda']     = 1000   # Number of wavelengths                |        |
        # =====================================================================================|
        # SHELL GRID                                                                           |
        # -------------------------------------------------------------------------------------+
        self.grid_params['r_min']     = 18     # Inner edge                           | AU     |
        self.grid_params['r_max']     = 1000   # Outer edge                           | AU     |
        self.grid_params['nr']        = 52     # Number of layers                     |        |
        # -------------------------------------------------------------------------------------+
        self.grid_params['ntheta']    = 25     # Number of theta grid                 |        |
        # =====================================================================================|
        # DUST SIZE GRID                                                                       |
        # -------------------------------------------------------------------------------------+
        self.grid_params['a_min']     = 0.01   # Minimum dust radius                  | um     |
        self.grid_params['a_max']     = 1000   # Maximum dust radius                  | um     |
        self.grid_params['na']        =  102   # Number of radius                     |        |
        # -------------------------------------------------------------------------------------+

        # DUST PARAMETERS
        self.dust_params = dict()
        # =====================================================================================|
        # DUST COMPONENTS                                                                      |
        # -------------------------------------------------------------------------------------+
        self.dust_params['siltype']   = 'DL84' # Draine-Lee silicate (Draine & Lee 1984)       |
        self.dust_params['cartype']   = 'amca' # amorphous carbon (Rouleau & Martin 1991)      |
        self.dust_params['icetype']   = 'LG98' # H2O ice (Li & Greenberg 1998)                 |
        # -------------------------------------------------------------------------------------+
        # Note: f_sil + f_car + f_ice = 1                                                      |
        # -------------------------------------------------------------------------------------+
        self.dust_params['f_sil']     =   0    # Volume fraction of silicate dust     |        |
        self.dust_params['f_car']     =   1    # Volume fraction of carbonaceous dust |        |
        self.dust_params['f_ice']     =   0    # Volume fraction of ice               |        |
        self.dust_params['rho_sil']   = 3.5    # Mass density of silicate dust        | g/cm^3 |
        self.dust_params['rho_car']   = 3.3    # Mass density of carbonaceous dust    | g/cm^3 |
        self.dust_params['rho_ice']   = 1.2    # Mass density of ice                  | g/cm^3 |
        # -------------------------------------------------------------------------------------+
        # RADIAL DISTRIBUTION
        self.dust_params['radial_distribution'] = modified_power_law
        self.dust_params['distribution_parameters'] = [ 2.0, 4.0 ]
        # -------------------------------------------------------------------------------------+

        return None

    def setup(self, stellar_params, grid_params, dust_params, show = 0):

        '''
        Subroutine to 
            1) build up grids based on the input hyper parameters (stellar_params, 
               grid_params & dust_params)
            3) convert all physical quantities into CGS units
            4) obtain quantities remaining unchanged, e.g. cross sections, optical
               depth along LoS, etc.

        Parameters
        ----------
        stellar_params : dict
            Dictionary containing stellar parameters.
        grid_params : dict
            Dictionary containing grid parameters.
        dust_params : dict
            Dictionary containing dust parameters.
        show : logic
            Print out parameters or not.

        Returns
        -------
            None
        '''

        # SETLLAR PARAMETERS
        self.R = (stellar_params['R'] * const.R_sun).to( u.cm ).value
        self.M = (stellar_params['M'] * const.M_sun).to( u.g  ).value
        self.Teff = stellar_params['Teff']
        self.logg = stellar_params['logg']

        # GRIDS
        self.lbda = ( np.logspace( np.log10(grid_params['lbda_min']), 
                                   np.log10(grid_params['lbda_max']), 
                                   grid_params['nlbda'] ) * u.um ).to( u.cm ).value

        self.r = ( np.logspace( np.log10(grid_params['r_min']), 
                                np.log10(grid_params['r_max']), 
                                grid_params['nr'] ) * u.au ).to( u.cm ).value
        
        self.a = ( np.logspace( np.log10(grid_params['a_min']), 
                                np.log10(grid_params['a_max']), 
                                grid_params['na'] ) * u.um ).to( u.cm ).value

        self.theta = np.linspace( 0, np.pi, grid_params['ntheta'] )

        # DUST COMPONENTS
        self.f_sil = dust_params['f_sil']
        self.f_car = dust_params['f_car']
        self.rho_sil = dust_params['rho_sil']
        self.rho_car = dust_params['rho_car']

        # CALCULATE CROSS SECTIONS
        self.c_abs_sil, _, self.c_ext_sil, _, _ = \
            cross_sections( self.a, self.lbda, opt_filename( 'sil', dust_params['siltype'] ) )
        self.c_abs_car, _, self.c_ext_car, _, _ = \
            cross_sections( self.a, self.lbda, opt_filename( 'car', dust_params['cartype'] ) )

        self.c_abs = self.f_sil * self.c_abs_sil + self.f_car * self.rho_sil / self.rho_car * self.c_abs_car
        self.c_ext = self.f_sil * self.c_ext_sil + self.f_car * self.rho_sil / self.rho_car * self.c_ext_car

        # RADIAL DISTRIBUTION
        self.radial_distribution = dust_params['radial_distribution']
        self.distribution_parameters = dust_params['distribution_parameters']
        self.n_r = self.radial_distribution( self.r/self.r.min(), self.distribution_parameters )
        self.dr = self.r * np.log(10) * np.log10( self.r.max() / self.r.min() ) / ( self.r.shape[0] - 1 )
        self.normfac_r = 1 / ( 0.5 * np.sum( self.n_r[[0, -1]] * self.dr[[0, -1]] ) + np.sum( self.n_r[ 1: -1 ] * self.dr[ 1: -1 ] ) )
        self.n_r *= self.normfac_r

        # COLUMN DENSITY GRID
        self.N_grid_normed, self.r_to_r = column_density_grid( r_grid = self.r, 
                                                               theta_grid = self.theta, 
                                                               delta = 0.1,
                                                               radial_dist = self.radial_distribution, 
                                                               normalization = self.normfac_r,
                                                               dist_params = self.distribution_parameters )
        # BLOCK ANGLE
        self.block_angle = np.arccos( self.R / self.r.min() ) + np.pi / 2
        # COLUMN DENSITY along LoS
        self.N_out_normed, self.r_to_r_max = column_density_outgoing( self.N_grid_normed, 
                                                                      r_grid = self.r, 
                                                                      theta_grid = self.theta, 
                                                                      theta_thres = self.block_angle )

        if show:
            # STELLAR PARAMETERS
            print( 'Stellar parameters:')
            print( '    R    = {:7.1f} R_sun'.format( self.R / const.R_sun.cgs.value ) )
            print( '    M    = {:7.1f} M_sun'.format( self.M / const.M_sun.cgs.value ) )
            print( '    Teff = {:7.1f} K'.format( self.Teff ) )
            print( '    logg = {:7.1f}'.format( self.logg ) )
            # GRIDS
            print( '\nGrids:' )
            print( '    Wavelength   (#{:4d}): {:10.4f} - {:10.4f} micron'.format( self.lbda.shape[0], self.lbda[0] * 1e4, self.lbda[-1] * 1e4 ) )
            print( '    Shell Radius (#{:4d}): {:10.1f} - {:10.1f} au'.format( self.r.shape[0], 
                                                                        self.r[0] / (1 * u.au).to( u.cm ).value, 
                                                                        self.r[-1] / (1 * u.au).to( u.cm ).value ) )
            print( '    Dust Size    (#{:4d}): {:10.2f} - {:10.2f} micron'.format( self.a.shape[0], self.a[0] * 1e4, self.a[-1] * 1e4 ) )
            print( '    Theta        (#{:4d}): {:>10} - {:>10}'.format( self.theta.shape[0], '0', 'pi' ) )

            # DUST
            print( '\nDust properties:' )
            print( ' '*4 + '-'*59 )
            print( '    Dust Component  Volume Fraction  Mass Density (g/cm3)  Type' )
            print( ' '*4 + '-'*59 )
            print( '      Silicate  {:13.0%}{:18.2f}{:>16}'.format( self.f_sil, self.rho_sil, dust_params['siltype'] ) )
            print( '      Carbon    {:13.0%}{:18.2f}{:>16}'.format( self.f_car, self.rho_car, dust_params['cartype'] ) )
            print( ' '*4 + '-'*59 )
            print( '    Radial Density Distribution: {}'.format( self.radial_distribution.__name__ ) )
            print( '        Distribution Parameters: {}'.format( self.distribution_parameters ) )

        return None

    def initialize( self, F_nu_R, show = 1 ):

        '''
        Subroutine to initialize the model. Under this subroutine, all the quantities 
        that are not necessary to be obtained in the main loop are derived, especially
        the intial equilibrium temperature.

        Parameters
        ----------
        F_nu_R : array_like
            Outgoing spectrum (SED) from the surface of the central star with unit 
            erg/s/cm3.
        show : logic
            Print out parameters or not.

        Returns
        -------
            None
        '''

        self.F_nu_R = F_nu_R

        # Dilution factor
        W = ( 1 - np.sqrt( 1 - ( self.R / self.r )**2 ) ) / 2
        self.F_nu_r = np.einsum('i, j -> ij', W, F_nu_R )

        # Equilibrium Temperature
        self.dust_emi = np.zeros( self.lbda.shape[0] )
        self.dust_sca = np.zeros( self.lbda.shape[0] )
        self.L_nu_r_a_sil = np.zeros( (self.r.shape[0], self.a.shape[0], self.lbda.shape[0]) )
        self.L_nu_r_a_car = np.zeros( (self.r.shape[0], self.a.shape[0], self.lbda.shape[0]) )
        if self.f_sil > 0:
            self.T_eq_sil, self.F_nu_r_a_sil = Equilibrium_Temperature( self.c_abs_sil, self.lbda, self.F_nu_r, self.dust_emi, self.dust_sca )
            self.L_nu_r_a_sil = np.einsum('jk, ijk -> ijk', self.c_abs_sil, self.F_nu_r_a_sil )
        if self.f_car > 0:
            self.T_eq_car, self.F_nu_r_a_car = Equilibrium_Temperature( self.c_abs_car, self.lbda, self.F_nu_r, self.dust_emi, self.dust_sca )
            self.L_nu_r_a_car = np.einsum('jk, ijk -> ijk', self.c_abs_car, self.F_nu_r_a_car )
        self.L_nu_r_a_dust = self.f_sil * self.L_nu_r_a_sil + self.rho_sil / self.rho_car * self.f_car * self.L_nu_r_a_car

        # Print
        if show:
            print( '\nInitial Equilibrium Temperature: \n' )
            first_row = ['Radius  (au)']; first_row.extend( np.round( self.r[::10] / const.au.cgs.value, 2 ).tolist() )
            rows = [ first_row ]
            if self.f_sil > 0:
                row = ['silicate (K)']; row.extend( np.round( self.T_eq_sil[::10, 0], 2 ).tolist() ); rows.append( row )
            if self.f_car > 0:
                row = ['carbon   (K)']; row.extend( np.round( self.T_eq_car[::10, 0], 2 ).tolist() ); rows.append( row )
            print( tabulate(rows, headers = 'firstrow' ) )

        return None

    def run( self, alphas, optical_depths, lbda_tau = 5500 ):

        '''
        Subroutine to calculate the outgoing SED.

        Parameters
        ----------
        alphas : scalar or array_like
            MRN power law index a^(-alpha)
        optical_depths : scalar or array_like
            Optical depth

        Returns
        -------
        '''

        # Check alphas is a number or an array-like variable
        if np.size(alphas) == 1:
            alphas = [alphas]
        else:
            pass
        # Check optical_depths is a number or an array-like variable
        if np.size(optical_depths) == 1:
           optical_depths = [optical_depths]
        else:
            pass

        lbda_tau = (lbda_tau * u.AA).to( u.cm ).value
        da = self.a * np.log(10) * np.log10( self.a.max() / self.a.min() ) / ( self.a.shape[0] - 1 )

        # Loop over alpha
        for alpha in alphas:
            # n(a)
            if alpha == 1:
                normfac_a = 1 / np.log( self.a.max() / self.a.min() )
            else:
                normfac_a = ( 1 - alpha ) / ( self.a.max()**( 1 - alpha ) - self.a.min()**( 1 - alpha ) )
            dn = normfac_a * self.a**( -alpha ) * da

            # Integrate dust emission over the size distribution
            L_nu_r_dust = 0.5 * np.sum( np.einsum('ijk, j -> ijk', self.L_nu_r_a_dust[:, [0, -1], :], dn[[0, -1]]), axis = 1 ) +\
                                np.sum( np.einsum('ijk, j -> ijk', self.L_nu_r_a_dust[:,  1: -1 , :], dn[ 1: -1 ]), axis = 1 )
            # Dust size averaged extinction cross sections
            c_ext_nu = 0.5 * np.sum( np.einsum('ij, i -> ij', self.c_ext[[0, -1], :], dn[[0, -1]]), axis = 0 ) +\
                             np.sum( np.einsum('ij, i -> ij', self.c_ext[ 1: -1 , :], dn[ 1: -1 ]), axis = 0 )
            # Interpolate extinction cross section at lbda_tau
            c_ext_5500 = np.interp(lbda_tau, self.lbda, c_ext_nu)
            # т(r, θ, r', λ)
            tau_grid_normed = np.einsum('ijk, l -> ijkl', self.N_grid_normed, c_ext_nu)
            # т along LoS
            tau_out_normed  = np.einsum('ij, k -> ijk', self.N_out_normed, c_ext_nu)
            tau_r_normed    = tau_grid_normed[:, 0, 0, :]

            # Loop over optical depth
            for tau_5500 in optical_depths:
                # --- Column density ---
                # From definition of optical depth,
                #      т(5500A) = ∫ c_ext(5500A) * n(r) * dr.
                # Since c_ext(5500A) is radius independent,
                #      т(5500A) = c_ext(5500A) * ∫ n(r) * dr
                #                 = c_ext(5500A) * N.
                # factor, thus the column density can be derived by
                #      N = т(5500A) / c_ext(5500A)
                N = tau_5500 / c_ext_5500
                # Recall that ∫n(r)dr is set to be 1 by a normalization
                tau_grid = N * tau_grid_normed
                # Outgoing Star Spectrum
                F_nu_r_attenuated = (self.F_nu_R * self.R**2 / self.r[-1]**2) * np.exp( -N * tau_r_normed[-1, :] )

                # --- Attenuation factor ---
                # Get rid of warnings of `divided by 0`
                self.r_to_r[ np.where( self.r_to_r == 0 ) ] = np.inf
                # Assignment
                attenuation_factor = np.einsum('ijkl, ijk -> ijkl', np.exp(-tau_grid), 1.0 / (4 * np.pi * self.r_to_r**2) )
                # Get rid of block angle
                attenuation_factor[:, np.where( self.theta > self.block_angle )[0], :, :] = 0

                # --- Total dust emission @ r ---
                # dust_emi(r, λ) = ∫ dφ ∫ ( ∫ F(r, θ, r', λ) * sinθ * dθ ) * r'^2 * n(r') * dr'
                # where F(r, θ, r', λ) = L0(r, λ) * exp( -т(|r - r'|) ) / (4 * pi * |r - r'|^2)
                # step1: attenuation_factor_theta_inted = ∫ exp( -т(|r - r'|) ) / (4 * pi * |r - r'|^2) * sinθ * dθ
                attenuation_factor_theta_inted = np.trapz( y = np.einsum( 'j, ijkl -> ijkl', np.sin( self.theta ), attenuation_factor ), 
                                                           x = self.theta, 
                                                           axis = 1 )

                niter = 0; iter_flag = 1
                while iter_flag:
                    niter += 1; print( 'niter = {}'.format( niter ) )
                    # step2: dust_emi(r, r', λ) = L0(r, λ) * (∫ exp( -т(|r - r'|) ) / (4 * pi * |r - r'|^2) * sinθ * dθ) * r'^2 * n(r')
                    dust_emi_r = N * np.einsum( 'jk, ijk, j, j -> ijk', L_nu_r_dust, attenuation_factor_theta_inted, self.r**2, self.n_r )
                    # step3: dust_emi(r, λ) = ∫ dφ ∫ dust_emi(r, r', λ) * dr'
                    self.dust_emi = 2 * np.pi * ( 0.5 * np.sum( np.einsum( 'ijk, j -> ijk', dust_emi_r[:, [0, -1], :], self.dr[[0, -1]]), axis = 1 ) +\
                                                        np.sum( np.einsum( 'ijk, j -> ijk', dust_emi_r[:,  1: -1 , :], self.dr[ 1: -1 ]), axis = 1 ) )

                    if self.f_sil > 0:
                        T_eq_sil_0 = self.T_eq_sil + 0.0
                        self.T_eq_sil, self.F_nu_r_a_sil = Equilibrium_Temperature( self.c_abs_sil, self.lbda, self.F_nu_r, self.dust_emi, self.dust_sca )
                        self.L_nu_r_a_sil = np.einsum('jk, ijk -> ijk', self.c_abs_sil, self.F_nu_r_a_sil )
                        iter_flag = np.sum( np.abs( T_eq_sil_0 - self.T_eq_sil ) > 0.1 )
                    if self.f_car > 0:
                        T_eq_car_0 = self.T_eq_car + 0.0
                        self.T_eq_car, self.F_nu_r_a_car = Equilibrium_Temperature( self.c_abs_car, self.lbda, self.F_nu_r, self.dust_emi, self.dust_sca )
                        self.L_nu_r_a_car = np.einsum('jk, ijk -> ijk', self.c_abs_car, self.F_nu_r_a_car )
                        iter_flag += np.sum( np.abs( T_eq_car_0 - self.T_eq_car ) > 0.1 )
                    self.L_nu_r_a_dust = self.f_sil * self.L_nu_r_a_sil + self.rho_sil / self.rho_car * self.f_car * self.L_nu_r_a_car

                    # Integrate dust emission over the size distribution
                    L_nu_r_dust = 0.5 * np.sum( np.einsum('ijk, j -> ijk', self.L_nu_r_a_dust[:, [0, -1], :], dn[[0, -1]]), axis = 1 ) +\
                                        np.sum( np.einsum('ijk, j -> ijk', self.L_nu_r_a_dust[:,  1: -1 , :], dn[ 1: -1 ]), axis = 1 )

                    # Print
                    print( '\nEquilibrium Temperature: \n' )
                    first_row = ['Radius  (au)']; first_row.extend( np.round( self.r[::10] / const.au.cgs.value, 2 ).tolist() )
                    rows = [ first_row ]
                    if self.f_sil > 0:
                        row = ['silicate (K)']; row.extend( np.round( self.T_eq_sil[::10, 0], 2 ).tolist() ); rows.append( row )
                    if self.f_car > 0:
                        row = ['carbon   (K)']; row.extend( np.round( self.T_eq_car[::10, 0], 2 ).tolist() ); rows.append( row )
                    print( tabulate(rows, headers = 'firstrow' ) )

                    if niter == 10:
                        iter_flag = 0
        
                self.r_to_r_max[ np.where( self.r_to_r_max == 0 ) ] = np.inf
                tau_out = N * tau_out_normed
                attenuation_factor_out = np.einsum('ijk, ij -> ijk', np.exp(-tau_out), 1.0 / (4 * np.pi * self.r_to_r_max**2) )
                attenuation_factor_out[:, np.where( self.theta > self.block_angle )[0], :] = 0
                attenuation_factor_out_theta_inted = np.trapz( y = np.einsum('j, ijk -> ijk', np.sin( self.theta ), attenuation_factor_out ), 
                                                               x = self.theta, 
                                                               axis = 1 )
                dust_emi_r_out = N * np.einsum('ij, ij, i, i -> ij', L_nu_r_dust, attenuation_factor_out_theta_inted, self.r**2, self.n_r)
                dust_emi_out = 2 * np.pi * ( 0.5 * np.sum( np.einsum('ij, i -> ij', dust_emi_r_out[[0, -1], :], self.dr[[0, -1]]), axis = 0 ) +\
                                                   np.sum( np.einsum('ij, i -> ij', dust_emi_r_out[ 1: -1 , :], self.dr[ 1: -1 ]), axis = 0 ) )

                N_dust = np.trapz( N * self.n_r * 4 * np.pi * self.r**2, self.r )
                V_dust = 4 * np.pi / 3 * N_dust * ( 0.5 * np.sum( self.a[[0, -1]]**3 * dn[[0, -1]], axis = 0 ) +\
                                                          np.sum( self.a[ 1: -1 ]**3 * dn[ 1: -1 ], axis = 0 ) )
                M_sil = self.f_sil * V_dust * self.rho_sil
                M_car = self.f_car * V_dust * self.rho_car

                print( M_sil, M_car )

        return F_nu_r_attenuated, dust_emi_out
