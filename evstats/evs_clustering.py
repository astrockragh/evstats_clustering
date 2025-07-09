import numpy as np
from scipy import integrate
from scipy.stats import norm, uniform
from scipy.special import gammaincinv
import hmf, h5py, astropy, sys, tqdm, time, os, warnings
import pandas as pd
from astropy.cosmology import Planck18_arXiv_v2

pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=RuntimeWarning)

out_str = 'demo2'
z_r = 0.003
z_OD = 4.622
dDEC_OD, dRA_OD = 49/3600, 191/3600 # in degrees
dRA_survey, dDEC_survey = np.sqrt(400/3600), np.sqrt(400/3600) # in degrees
z_min_survey, z_max_survey = 3, 5
z_min_survey = float(z_min_survey)
z_max_survey = float(z_max_survey)

# calculate cosmic variance
recalculate_cv = False
df_loc_precomputed = f'dfs/scaled_{z_OD}_{z_r}.csv'

# computational parameters
low_z = z_min_survey 
max_z = 15
n_z = 2
Ntrials = int(1e6) # number of overdensities to sample, make sure that this is larger than the number of overdensities that fit in your survey volume

# define basic parameters
cosmo = Planck18_arXiv_v2
all_sky = 41252.96125 #number of square degrees in the sky
f = np.pi/180 # change between degrees and radians
little_h = 0.68 
baryon_frac = 0.16

mmin = 3 # Minimum halo mass for HMF
mmax = 15 # Maximum halo mass for HMF

maxMass = 12.5
dM = 0.5
dlog10m = 0.01 #numerical HMF resolution, should be <<mass resolution
Ndm = int(0.5//dlog10m) #number of steps in a 0.5 dex mass interval
Nm = 50 #mass evaluation grid resolution
mbins = np.arange(7., maxMass+dM/Nm-1e-10, dM/Nm)
mbins = np.round(mbins, 3)
# print(mbins)

# Method to take trials from the inverse CDF
# of a gamma distribution with a given variance and mean
def trial_minsig(sig_v, mean, Ntrials=10000):
    Ntrials = int(Ntrials)
    var = sig_v**2*mean**2+mean #cv + poisson
    k = mean**2/var
    t = var/mean
    x = np.linspace(1/Ntrials, 1-1/Ntrials, Ntrials)
    rand = t*gammaincinv(k, x, out = None) # this is only accurate when sigma_CV>0.05
    return rand

def sigma_delta_evs(z_r, z_OD, dDEC_OD, dRA_OD, z_min_survey, z_max_survey, dRA_survey, dDEC_survey, n_samp = int(1e5)):
    ''' All DEC/RA's should be given in degrees'''
    d1 = dDEC_OD*f*cosmo.comoving_distance(z_OD)
    d2 = dRA_OD*f*cosmo.comoving_transverse_distance(z_OD)
    dz = cosmo.comoving_distance(z_OD+z_r)-cosmo.comoving_distance(z_OD-z_r)

    V_OD = 4/3*np.pi*d1*d2*dz/8/(1+z_OD)**3

    z_av = (z_min_survey+z_max_survey)/2
    d1 = dDEC_survey*f*cosmo.comoving_distance(z_av)
    d2 = dRA_survey*f*cosmo.comoving_transverse_distance(z_av)
    dz = cosmo.comoving_distance(z_max_survey)-cosmo.comoving_distance(z_min_survey)

    V_survey = ( cosmo.comoving_volume(z_max_survey) - cosmo.comoving_volume(z_min_survey) )*(dRA_survey*dDEC_survey/all_sky)/(1+z_av)**3

    # V_survey = d1*d2*dz
    N = V_survey/V_OD
    x = np.linspace(1/n_samp, 1-1/n_samp, n_samp)
    evs = N*uniform.pdf(x)*pow(uniform.cdf(x), N - 1.)
    return evs, x, V_OD, V_survey

def evs_clustering(cv_df, mf = hmf.MassFunction(), V = 1, z = 4, Ntrials = int(1e5)):
    """

    Parameters
    ----------

    Returns
    -------
    phi:

    """
    
    mf.update(z = z, Mmin = mmin, Mmax = mmax, dlog10m = dlog10m)
    dndm = mf.dndlog10m/little_h**4*V.value
    mass = mf.m*little_h
    sbf = 0.03856609803835385 + 0.012162496006188494 * (z - 4) # could possibly include some uncertainty here
    stellar_mass = mass * sbf * baryon_frac 

    N_trapz = []
    for i in range(len(mass)-Ndm):
        inte = np.trapz(dndm[i:i+Ndm], np.log10(mass[i:i+Ndm]) ) #integrate over bin
        N_trapz.append(inte)
        
    N_trapz = np.array(N_trapz)   

    ## correction for bin size, applying later
    n_tot = integrate.trapz(dndm, np.log10(mass))
    f = n_tot/np.sum(N_trapz)
    stellar_mass = stellar_mass[int(Ndm)//2:-int(Ndm)//2] #redefine to fit integration range
    smfs = []
    cv_df_z  = cv_df.iloc[ np.argmin(np.abs(cv_df['z']-z)) ]
    cols = []
    for m in mbins[:-1]:
        cols.append(str(m) )

    ## make sure that the cv increases monotonically
    cv_df_z[cols] = np.maximum.accumulate(cv_df_z[cols])

    for m in mbins[:-1]:
        arg = np.argmin(np.abs(np.log10(stellar_mass)-m))
        N = N_trapz[arg-1]
        cv = cv_df_z[str(m)]
        smfs.append(trial_minsig(float(cv), N, Ntrials = Ntrials))
    
    smfs = np.vstack(smfs)*f
    Ns = np.sum(smfs, axis = 0)
    fs = smfs/Ns
    fs = fs.T
    Fs = np.cumsum(smfs, axis = 0)
    Fs = Fs.T/Ns.reshape(-1, 1)
    phi_maxs = Ns.reshape(-1, 1)*fs*pow(Fs, Ns.reshape(-1, 1)-1)

    return phi_maxs.T, smfs, mbins, N_trapz, f

# # Get the absolute path of the parent directory
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# # Add the parent directory to sys.path
# sys.path.append(parent_dir)

try:
    from make_scaled_cv import scale_cv
except:
    from evstats.make_scaled_cv import scale_cv

if __name__ == '__main__':
    
    if recalculate_cv:
        start = time.time()

        cv_df = scale_cv( dDEC_OD, dRA_OD, z_OD, z_r, low_z = z_min_survey, max_z = max_z, n_z = n_z)

        stop = time.time()

        print(f'Finished calculating cosmic variance in {(stop-start)/60:.2f} minutes')
        df_loc =  f'dfs/scaled_{z_OD}_{z_r}.csv'
        cv_df = pd.read_csv(df_loc)
    else:
        df_loc =  df_loc_precomputed   
        cv_df = pd.read_csv(df_loc)
        cv_df = cv_df.iloc[np.arange(len(cv_df))[::len(cv_df)//n_z]]

    zs = np.array(cv_df['z'])
    evs_OD, x_OD, V_OD, V_survey = sigma_delta_evs(z_r, z_OD, dDEC_OD, dRA_OD, \
                                                z_min_survey, z_max_survey, dRA_survey, dDEC_survey, n_samp = Ntrials)

    phi_max_convolved_z = []
    phi_maxs_z = []

    smfs_z = []
    mbins_z = []
    N_trapz_z = []
    f_z = []
    for z in tqdm.tqdm(zs, total = len(zs)):
        pm, smfs, mbins, N_trapz, f = evs_clustering(cv_df = cv_df, V = V_OD, z = z, Ntrials = Ntrials)
        mask = ~np.any(np.isnan(pm), axis = 0)
        evs_ODm = evs_OD[mask]
        pdf_norm = np.sum(pm[:,mask]*evs_ODm, axis = 1)/np.sum(evs_ODm)
        phi_max_convolved_z.append(pdf_norm)
        phi_maxs_z.append(pm)
        smfs_z.append(smfs)
        mbins_z.append(mbins)
        N_trapz_z.append(N_trapz)
        f_z.append(f)

    if not os.path.exists('data'):
        os.mkdir('data')

    with h5py.File('data/'+out_str+'.h5', 'w') as hf:
        hf.create_dataset('log10m', data = mbins)
        hf.create_dataset('z', data = zs)
        # hf.create_dataset('smf', data = np.array(smfs_z, dtype = np.float32))
        hf.create_dataset('phi_max_conv', data = phi_max_convolved_z)
        hf.create_dataset('phi_maxs', data = np.array(phi_maxs_z, dtype = np.float32))
        hf.create_dataset('evs_OD', data = evs_OD)
        hf.create_dataset('f', data = f)
        hf.create_dataset('N_trapz', data = N_trapz_z)

