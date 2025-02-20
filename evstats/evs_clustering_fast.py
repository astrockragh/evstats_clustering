import numpy as np
from scipy import integrate
from scipy.stats import norm, uniform, poisson
from scipy.special import gammaincinv
import hmf, h5py, astropy, sys, tqdm, time, os, warnings
import pandas as pd
from astropy.cosmology import Planck18_arXiv_v2

pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=RuntimeWarning)

add_sig = 0
od = 'big'
out_str = f'{od}_OD_fast_{add_sig}'
recalculate_cv = False
n_z = 15

if od == 'small':
    z_r = 0.003
    z_OD = 4.622
    dDEC_OD, dRA_OD = 49/3600, 191/3600 # in degrees
    Ntrials = int(5e6)

if od == 'big':
    z_r = 0.07
    z_OD = 3.21
    dDEC_OD, dRA_OD = 437/3600, 609/3600 # in degrees
    Ntrials = int(1e4)

if od == 'demo':
    z_r = 0.003
    z_OD = 4.622
    dDEC_OD, dRA_OD = 49/3600, 191/3600 # in degrees
    Ntrials = int(1e3)

dRA_survey, dDEC_survey = np.sqrt(160/3600), np.sqrt(160/3600) # in degrees
z_min_survey, z_max_survey = 3, 5
z_min_survey = float(z_min_survey)
z_max_survey = float(z_max_survey)

# calculate cosmic variance
df_loc_precomputed = f'dfs/scaled_{z_OD}_{z_r}.csv'

# computational parameters
low_z = z_min_survey 
max_z = 18

# define basic parameters
cosmo = Planck18_arXiv_v2
all_sky = 41252.96125 #number of square degrees in the sky
f = np.pi/180 # change between degrees and radians
baryon_frac = 0.16

mmin = 3 # Minimum halo mass for HMF
mmax = 15 # Maximum halo mass for HMF

maxMass = 12.5
dM = 0.5 #clustering scale
dlog10m = 0.01 #numerical HMF resolution, should be <<mass resolution
Ndm = int(dM//dlog10m) #number of steps in a dM dex mass interval
Nm = 100 #mass evaluation grid resolution, set so that we get dM/Nm = 0.005
mbins = np.arange(7., maxMass+dM/Nm-1e-10, dM/Nm)
mbins = np.round(mbins, 3)

# Method to take trials from the inverse CDF
# of a gamma distribution with a given variance and mean
def trial_OD(x, sig_v, mean):
    var = sig_v**2*mean**2+mean #cv + poisson
    k = mean**2/var
    t = var/mean
    rand = t*gammaincinv(k, x, out = None) # this is only accurate when sigma_CV*N>0.05
    return rand

def sigma_delta_evs(z_r, z_OD, dDEC_OD, dRA_OD, z_min_survey, z_max_survey, dRA_survey, dDEC_survey, n_samp = int(1e5), add_sig = 0.0):
    ''' All DEC/RA's should be given in degrees'''
    d1 = dDEC_OD*f*cosmo.comoving_distance(z_OD)
    d2 = dRA_OD*f*cosmo.comoving_transverse_distance(z_OD)
    dz = cosmo.comoving_distance(z_OD+z_r)-cosmo.comoving_distance(z_OD-z_r)

    V_OD = 4/3*np.pi*d1*d2*dz/8/(1+z_OD)**3

    z_av = (z_min_survey+z_max_survey)/2
    d1 = dDEC_survey*f*cosmo.comoving_distance(z_av)
    d2 = dRA_survey*f*cosmo.comoving_transverse_distance(z_av)
    dz = cosmo.comoving_distance(z_max_survey)-cosmo.comoving_distance(z_min_survey)

    V_survey = d1*d2*dz/(1+z_min_survey)**3
    N = V_survey/V_OD
    sig = norm.isf(1/N)
    if add_sig:
        N = 1/(1-norm.cdf(sig+add_sig))
        sig = norm.isf(1/N)
    
    print(N, V_OD, V_survey, sig)
    x = np.linspace(1/n_samp, 1-1/n_samp, n_samp)
    evs = N*uniform.pdf(x)*pow(uniform.cdf(x), N - 1.)
    return evs, x, V_OD*(1+z_OD)**3, V_survey*(1+z_min_survey)**3

def evs_clustering(cv_df, x, mf = hmf.MassFunction(hmf_model="Behroozi"), V = 1, z = 4):
    """

    Parameters
    ----------

    Returns
    -------
    phi:

    """
    
    mf.update(z = z, Mmin = mmin, Mmax = mmax, dlog10m = dlog10m)
    mf.cosmo_model = Planck18_arXiv_v2

    dndm = mf.dndlog10m*V.value
    mass = mf.m
    sbf = 0.1 + 0.02 * (z - 4) # could possibly include some uncertainty here
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
        cv = cv_df_z[str(m)]/3
        smfs.append(trial_OD(x, float(cv), N))
    
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
        z_min_survey, z_max_survey, dRA_survey, dDEC_survey, n_samp = Ntrials, add_sig = add_sig)

    phi_max_convolved_z = []
    phi_maxs_z = []

    smfs_z = []
    mbins_z = []
    N_trapz_z = []
    mask0 = evs_OD>1e-4
    evs_OD = evs_OD[mask0]
    x_OD = x_OD[mask0]
    print(zs)
    f_z = []
    for z in tqdm.tqdm(zs, total = len(zs)):
        pm, smfs, mbins, N_trapz, f = evs_clustering(cv_df, x_OD, V = V_OD, z = z)
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
        hf.create_dataset('smf', data = np.array(smfs_z, dtype = np.float32))
        hf.create_dataset('phi_max_conv', data = phi_max_convolved_z)
        hf.create_dataset('phi_maxs', data = np.array(phi_maxs_z, dtype = np.float32))
        hf.create_dataset('evs_OD', data = evs_OD)
        hf.create_dataset('f', data = [f])
        hf.create_dataset('N_trapz', data = N_trapz_z)
