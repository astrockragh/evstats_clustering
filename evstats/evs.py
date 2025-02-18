import numpy as np
from scipy import integrate
from scipy.stats import norm
from scipy.special import gammaincinv
import hmf, astropy, sys, tqdm, time
import pandas as pd
from astropy.cosmology import Planck18_arXiv_v2

little_h = 0.68
baryon_frac = 0.188

mmin = 3 # Minimum halo mass for HMF
mmax = 15 # Maximum halo mass for HMF

maxMass = 12.5
dM = 0.5
dlog10m = 0.01 #numerical HMF resolution, should be <<mass resolution
Ndm = int(0.5//dlog10m) #number of steps in a 0.5 dex mass interval
Nm = 100 # set to 100
mbins = np.arange(3., maxMass+dM/Nm-1e-10, dM/Nm)
mbins=np.round(mbins, 3)

# Method to take one trial of a gamma distribution with a given variance and mean
def trial_minsig(sig_v, mean, Ntrials=10000, minsig = -7):
    Ntrials = int(Ntrials)
    var = sig_v**2*mean**2+mean #cv + poisson
    k = mean**2/var
    t = var/mean
    y = np.linspace(norm.cdf(minsig),1-1e-8, Ntrials)
    rand = np.rint(t*gammaincinv(k, y, out = None)) # this is only accurate when sigma_CV>0.05
    return rand

def evs_delta_sig(cv_df_loc = 'calculate_sigma_cv/dfs/scaled_4.623_0.003.csv', mf = hmf.MassFunction(), vol = 10, z = 4, Ntrials = int(1e4)):
    """

    Parameters
    ----------

    Returns
    -------
    phi:

    """

    cv_df = pd.read_csv(cv_df_loc)
    mf.update(z = z, Mmin = mmin, Mmax = mmax, dlog10m = dlog10m)
    dndm = mf.dndlog10m/little_h**4*vol
    mass = mf.m*little_h
    sbf = 0.02856609803835385 + 0.012162496006188494 * (z - 4) # could possibly include some uncertainty here
    stellar_mass = mass * sbf * baryon_frac 

    N_trapz = []
    for i in range(len(mass)-Ndm):
        inte = np.trapz(dndm[i:i+Ndm], mass[i:i+Ndm]) #integrate over bin
        N_trapz.append(inte)
    N_trapz = np.array(N_trapz)   

    stellar_mass = stellar_mass[int(Ndm)//2:-int(Ndm)//2] #redefine to fit integration range
    smfs = []
    for m in mbins[:-1]:
        arg = np.argmin(np.abs(np.log10(stellar_mass)-m))
        N = N_trapz[arg-1]
        cv = cv_df.iloc[np.argmin(np.abs(cv_df['z']-z))][str(m)]
        smfs.append(trial_minsig(float(cv), N, Ntrials = Ntrials, minsig = -10))
    
    smfs = np.vstack(smfs)
    Ns = np.sum(smfs, axis = 1).reshape(-1, 1)
    fs = smfs/Ns
    Fs = np.cumsum(smfs, axis = 1)/Ns
    phi_maxs = Ns*fs*pow(Fs, Ns-1)
    return phi_maxs


def evs_hypersurface_pdf(mf = hmf.MassFunction(), V = 33510.321):
    """
    Calculate extreme value probability density function for the dark matter
    halo population on a spatial hypersurface (fixed redshift).

    Parameters
    ----------
    mf : mass function, from `hmf` package
    V : volume (default: a sphere with radius 20 Mpc)

    Returns
    -------
    phi:

    """

    n_tot = integrate.trapz(mf.dndlog10m, np.log10(mf.m))
    f = mf.dndlog10m[:-1] / n_tot
    F = integrate.cumtrapz(mf.dndlog10m, np.log10(mf.m)) / n_tot
    N = V*n_tot
    phi_max = N*f*(F**(N-1))
    return phi_max



def evs_bin_pdf(mf = hmf.MassFunction(), zmin=0., zmax=0.1, dz=0.01, mmin=12, mmax=18, dm = 0.01, fsky=1.):
    """
    Calculate EVS in redshift and mass bin

    Parameters
    ----------
    zmin : z minimum
    zmax : z maximum
    dz: delta z
    mmin: mass minimum (log10 (h^{-1} M_{\sol}) )
    mmax: mass maximum (log10 (h^{-1} M_{\sol}) )
    dm: delta m (log10 (h^{-1} M_{\sol}) )
    fsky: fraction of sky

    Returns
    -------
    phi: probability density function
    ln10m_range: corresponding mass values for PDF (log10 (h^{-1} M_{\sol}) )
    """

    N, f, F, ln10m_range = _evs_bin(mf=mf, zmin=zmin, zmax=zmax, dz=dz, mmin=mmin, mmax=mmax, dm=dm)

    phi =_apply_fsky(N, f, F, fsky)

    return phi, ln10m_range


def _apply_fsky(N, f, F, fsky):     
    N_sky = N * fsky
    f_sky = f / N_sky
    f_sky *= fsky
    
    F_sky = F / N_sky
    F_sky *= fsky

    return N_sky * f_sky * pow(F_sky, N_sky - 1.)
    
    
def _evs_bin(mf = hmf.MassFunction(), zmin=0., zmax=0.1, dz=0.01, mmin=12, mmax=18, dm = 0.01):
    """
    Calculate EVS (ignoring fsky dependence). Worker function for `evs_bin_pdf`

    Parameters
    ----------
    zmin : z minimum
    zmax : z maximum
    dz: delta z
    mmin: mass minimum (log10 (h^{-1} M_{\sol}) )
    mmax: mass maximum (log10 (h^{-1} M_{\sol}) )
    dm: delta m (log10 (h^{-1} M_{\sol}) )

    Returns
    -------
    N (float)
    f (array)
    F (array)
    ln10m_range (array)
    
    """

    mf.update(Mmin=mmin, Mmax=mmax, dlog10m=dm)

    N = _computeNinbin(mf=mf, zmin=zmin, zmax=zmax, dz=dz)

    # need to set lower limit slightly higher otherwise hmf complains.
    # should have no impact on F if set sufficiently low.
    ln10m_range = np.log10(mf.m[np.log10(mf.m) >= mmin+1])

    F = np.array([_computeNinbin(mf=mf, zmin=zmin, zmax=zmax, lnmax=lnmax, dz=dz) \
     for lnmax in ln10m_range])

    f = np.gradient(F, mf.dlog10m)
    
    return N, f, F, ln10m_range


def _computeNinbin(mf, zmin, zmax, lnmax=False, dz=0.01):

    if lnmax: mf.update(Mmax=lnmax)

    zees = np.arange(zmin, zmax, dz, dtype='longdouble')  # z range

    # calculate dvdz in advance to take advantage of vectorization
    dvdz = mf.cosmo.differential_comoving_volume(zees).value.astype('longdouble') * (4. * np.pi)

    dndmdz = [_dNdlnmdz(z=z, mf=mf, dvdz=dv) for z, dv in zip(zees,dvdz)]
    
    # integrate over z
    return integrate.trapz(dndmdz, zees)


def _dNdlnmdz(z, mf, dvdz):
    mf.update(z=z)
    return integrate.trapz(mf.dndlnm.astype('longdouble') * dvdz, np.log(mf.m))


if __name__ == '__main__':

    # initialise mass function
    mass_function = hmf.MassFunction()

    # set cosmology using astropy
    mass_function.cosmo_model = Planck18_arXiv_v2

    # set redshift
    mass_function.z = 0.0

    # set mass range and resolution
    mass_function.dlog10m = 0.1
    mass_function.Mmin = 12
    mass_function.Mmax = 18

    phi_max = evs_hypersurface_pdf(mf = mass_function)
