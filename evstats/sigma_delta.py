import numpy as np
import scipy as sp
from astropy.cosmology import Planck18_arXiv_v2

cosmo = Planck18_arXiv_v2

all_sky = 41252.96125
f = np.pi/180

def sigma_delta_evs(z_r, z_OD, dDEC, dRA, z_min_survey, z_max_survey, dRA_survey, dDEC_survey, max_sig = 10, n_sig = int(1e7)):
    ''' All DEC/RA's should be given in degrees'''
    d1 = dDEC*f*cosmo.comoving_distance(z_OD)
    d2 = dRA*f*cosmo.comoving_transverse_distance(z_OD)
    dz = cosmo.comoving_distance(z_OD+z_r)-cosmo.comoving_distance(z_OD-z_r)

    V_OD = 4/3*np.pi*d1*d2*dz/8

    z_av = (z_min_survey+z_max_survey)/2
    d1 = dDEC_survey*f*cosmo.comoving_distance(z_av)
    d2 = dRA_survey*f*cosmo.comoving_transverse_distance(z_av)
    dz = cosmo.comoving_distance(z_max_survey)-cosmo.comoving_distance(z_min_survey)

    # V_survey = dDEC_survey*dRA_survey/all_sky*(cosmo.comoving_volume(z_max_survey) - cosmo.comoving_volume(z_min_survey))
    V_survey = d1*d2*dz
    N = V_survey/V_OD
    print(N)
    x = np.linspace(-max_sig, max_sig, n_sig)
    evs = N*sp.stats.norm.pdf(x)*pow(sp.stats.norm.cdf(x), N - 1.)
    return evs, x

z_r = 0.003
z_OD = 4.622
dDEC, dRA = 49, 191
dRA_survey, dDEC_survey = np.sqrt(400/3600), np.sqrt(400/3600)
z_min_survey, z_max_survey = 3, 5

print(sigma_delta_evs(z_r, z_OD, dDEC/3600, dRA/3600, z_min_survey, z_max_survey, dRA_survey, dDEC_survey))