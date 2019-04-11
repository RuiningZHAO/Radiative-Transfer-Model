import numpy as np
import matplotlib.pyplot as plt
import RT
import astropy.units as u
from astropy.modeling.blackbody import blackbody_lambda

d = (6.72e6 * u.pc).to( u.cm ) # cm

# Load Incident Spectrum
Teff = 3300 # 4500
logg = -0.5 # 0.00

lbda, F_nu = np.loadtxt( './Data/Teff_{}_logg_{}.dat'.format( str( '%04d' %Teff ), 
                                                              str( '%.2f' %logg ) ), skiprows = 2 ).T

F_bb = (blackbody_lambda( lbda*u.cm, Teff*u.K ) * np.pi * u.sr).to( u.erg / u.s / u.cm**3 ).value

# tau = np.loadtxt( 'tmp.dat' )
# fig, ax = plt.subplots(1, 1, figsize = (10, 6))
# ax.plot( lbda * 1e4, tau[:1000], 'y-' )
# ax.plot( lbda * 1e4, tau[1000:], 'k:' )
# ax.set_xscale('log'); ax.set_yscale('log')
# ax.set_xlim(0.1, 1000)
# plt.show()

# Define a new shell model
RT_Model = RT.shell()

# Adjust parameters
RT_Model.stellar_params['M'] = 15.0
RT_Model.stellar_params['Teff'] = Teff
RT_Model.stellar_params['logg'] = logg
RT_Model.dust_params['f_sil'] = 0.75
RT_Model.dust_params['f_car'] = 0.25
RT_Model.dust_params['distribution_parameters'] = [2.0, 4.0]

# Use setup() method to set up the model with new dictionaries and
# print out updated parameters by setting show = 1.
RT_Model.setup( stellar_params = RT_Model.stellar_params, 
                grid_params    = RT_Model.grid_params, 
                dust_params    = RT_Model.dust_params, 
                show           = 1 )

# Initialize
RT_Model.initialize( F_bb )

F_star_nu, dust_emi_nu = RT_Model.run( alphas = 4.0, optical_depths = 1.2 )

fig, ax = plt.subplots(1, 1, figsize = (6, 6))
ax.plot( lbda * 1e4, lbda * F_bb * (RT_Model.R/d)**2, 'k-' )
ax.plot( lbda * 1e4, lbda * F_star_nu * (RT_Model.r[-1]/d)**2, 'b--' )
ax.plot( lbda * 1e4, lbda * dust_emi_nu * (RT_Model.r[-1]/d)**2, 'y--' )
ax.plot( lbda * 1e4, lbda * (F_star_nu + dust_emi_nu) * (RT_Model.r[-1]/d)**2, 'r-' )
ax.set_xlabel('Wavelength($\mathrm{\mu m}$)', fontsize = 16)
ax.set_ylabel('$\lambda F_{\lambda}$($\mathrm{erg/s/cm^2}$)', fontsize = 16)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.2, 100)
ax.set_ylim(5e-18, 5e-12)
ax.tick_params(which = 'major', direction = 'in', top = True, right = True, length = 6, width = 1.2)
ax.tick_params(which = 'minor', direction = 'in', top = True, right = True, length = 4, width = 1.2)
plt.show()