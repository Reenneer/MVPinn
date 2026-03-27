import numpy as np
from scipy.special import voigt_profile
from dataclasses import dataclass
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Optional, Tuple
import astropy.io.fits as fits
from lmfit import Model, Parameters, Minimizer
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Union
import time
from tqdm import tqdm  # For progress bars
import sys
from functools import partial
from skimage.transform import rotate
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool, cpu_count, shared_memory

def get_noise(data1d):
    """Calculate noise estimate from 1D data"""
    kernel = np.array([-1., 2., -1.])
    a = np.convolve(data1d, kernel, mode='same')
    return np.std(a[1:-1]) / np.sqrt(6.)

def init_par(x: np.ndarray, data: np.ndarray, cont_ind: np.ndarray, 
             display: bool = False) -> np.ndarray:
    """
    Initialize parameters for ME inversion
    
    Parameters:
        x (np.ndarray): wavelength array
        data (np.ndarray): Stokes profiles data
        cont_ind (np.ndarray): indices of continuum points
        display (bool): whether to display diagnostic plots
        
    Returns:
        np.ndarray: array of initial parameters
    """
    dwl_core = 0.15  # width of the line core in A
    half_width = 0.6  # full half width in A (from line center to cont)
    bt_coeff = 5000.  # magnetic field strength coefficient
    
    n = len(x) // 4
    
    # Extract Stokes profiles
    wlength = x[:n]
    prof_i = data[:,0]
    prof_q = data[:,1]
    prof_u = data[:,2]
    prof_v = data[:,3]
    
    # Calculate continuum level
    continuum = np.median(prof_i[cont_ind])
    
    # Get line indices
    line_ind = np.where(np.abs(wlength) <= half_width)[0]
    core_ind = np.where(np.abs(wlength) <= dwl_core)[0]
    
    # Calculate B_los using Center of Gravity method
    blos, _ = niris_cogmag(wlength[line_ind], 
                          prof_i[line_ind]/continuum,
                          prof_v[line_ind]/continuum)
   
    # Get core profiles
    wcore = wlength[core_ind]
    icore = prof_i[core_ind]
    qicore = prof_q[core_ind] / np.mean(icore)
    uicore = prof_u[core_ind] / np.mean(icore)
    
    # Calculate B transverse
    bt = bt_coeff * np.sqrt(np.sqrt(np.mean(qicore**2 + uicore**2)))

    # Calculate total magnetic field
    bfield = np.sqrt(blos[0]**2 + bt**2)
    if abs(bfield) > 5000:
        bfield = np.sign(bfield) * 5000.
    
    # Calculate field inclination and azimuth
    theta = np.arctan2(bt, blos[0])
    chi = np.arctan2(-uicore, -qicore)
    chi = 0.5 * np.median(chi + 2*np.pi*(chi < 0))
    
    # Calculate other parameters
    eta0 = continuum / np.median(icore)
    dlambda_d = 0.15  # Doppler width in angstrom
    adamp = 1.0  # damping parameter
    lambda0 = 0.0  # line center in A
    
    # Source function parameters
    b1 = 2 * (continuum - np.min(icore))
    b0 = continuum - b1
    
    return np.array([bfield, theta, chi, eta0, dlambda_d, adamp, lambda0, b0, b1])

def niris_cogmag(wv, idata, vdata):
    """
    Determine the line-of-sight field from Stokes I and V profiles
    using the center of gravity method
    
    Parameters:
        wv (array): wavelength in angstrom
        idata (array): Stokes I data normalized by continuum intensity
        vdata (array): Stokes V data normalized by continuum intensity
        
    Returns:
        tuple: (B, wlc) - magnetic field strength and wavelength center
    """
    geff = 3.0
    lambda_rest = 15648.5
    
    npoint = idata.shape[1] if len(idata.shape) > 1 else 1
    
    wv1 = np.zeros(npoint)
    wv2 = np.zeros(npoint)
    
    for k in range(npoint):
        weight1 = 1 - (idata + vdata) if npoint == 1 else 1 - (idata[:, k] + vdata[:, k])
        weight2 = 1 - (idata - vdata) if npoint == 1 else 1 - (idata[:, k] - vdata[:, k])
        
        wv2[k] = np.sum(weight1 * wv) / np.sum(weight1)
        wv1[k] = np.sum(weight2 * wv) / np.sum(weight2)
    
    coeff = 4.67e-13 * lambda_rest**2 * geff
    B = (wv2 - wv1) / (2 * coeff)
    wlc = 0.5 * (wv2 + wv1)
    
    return B, wlc

def niris_MEsinglet(x: np.ndarray, *params) -> np.ndarray:
    """
    ME inversion core function
    
    Parameters:
        x (np.ndarray): wavelength array
        *params: unpacked parameters [B, theta, chi, eta0, dlambdaD, a, lambda0, B0, B1]
        
    Returns:
        np.ndarray: array of calculated Stokes parameters [I, Q, U, V]
    """
    # Convert params tuple to array
    params = np.array(params)
    
    # Constants
    geff = 3.0
    lambda_rest = 15648.5
    
    # Unpack parameters
    B, theta, chi, eta0, dlambdaD, a, lambda0, B0, B1 = params
    
    # Calculate wavelength-related parameters
    n = len(x) // 4
    lambda_arr = x[:n]
    v = (lambda_arr - lambda0) / dlambdaD
    vb = geff * (4.67e-13 * lambda_rest**2 * B) / dlambdaD

    # Calculate Voigt profiles
    phib, psib = ch_voigt(a, v + vb)
    phip, psip = ch_voigt(a, v)
    phir, psir = ch_voigt(a, v - vb)
    
    # Normalize
    factor = 1.0 / np.sqrt(np.pi)
    phib *= factor
    psib *= factor
    phip *= factor
    psip *= factor
    phir *= factor
    psir *= factor
    
    # Calculate trigonometric terms
    st = np.sin(theta)
    st2 = st**2
    ct = np.cos(theta)
    
    # Calculate absorption and dispersion profiles
    etaI = 1 + 0.5 * eta0 * (phip * st2 + 0.5 * (phib + phir) * (1 + ct**2))
    etaQ = eta0 * 0.5 * (phip - 0.5 * (phib + phir)) * st2 * np.cos(2 * chi)
    etaU = eta0 * 0.5 * (phip - 0.5 * (phib + phir)) * st2 * np.sin(2 * chi)
    etaV = eta0 * 0.5 * (phir - phib) * ct
    
    rhoQ = eta0 * 0.5 * (psip - 0.5 * (psib + psir)) * st2 * np.cos(2 * chi)
    rhoU = eta0 * 0.5 * (psip - 0.5 * (psib + psir)) * st2 * np.sin(2 * chi)
    rhoV = eta0 * 0.5 * (psir - psib) * ct
    
    # Calculate determinant
    Delta = (etaI**2 * (etaI**2 - etaQ**2 - etaU**2 - etaV**2 + rhoQ**2 + rhoU**2 + rhoV**2) - 
            (etaQ * rhoQ + etaU * rhoU + etaV * rhoV)**2)
    
    # Calculate Stokes parameters
    I = B0 + B1 * etaI * (etaI**2 + rhoQ**2 + rhoU**2 + rhoV**2) / Delta
    Q = -B1 * (etaI**2 * etaQ + etaI * (etaV * rhoU - etaU * rhoV) + 
               rhoQ * (etaQ * rhoQ + etaU * rhoU + etaV * rhoV)) / Delta
    U = -B1 * (etaI**2 * etaU + etaI * (etaQ * rhoV - etaV * rhoQ) + 
               rhoU * (etaQ * rhoQ + etaU * rhoU + etaV * rhoV)) / Delta
    V = -B1 * (etaI**2 * etaV + etaI * (etaU * rhoQ - etaQ * rhoU) + 
               rhoV * (etaQ * rhoQ + etaU * rhoU + etaV * rhoV)) / Delta
    
    return np.concatenate([I, Q, U, V])

def ch_voigt(a, u, calc_derivatives=False):
    """
    Calculate the values of the Voigt function and its associated dispersion function,
    and their partial derivatives
    
    Parameters:
        a (float/array): damping parameter(s)
        u (float/array): dimensionless wavelength offset(s)
        calc_derivatives (bool): whether to calculate derivatives
        
    Returns:
        tuple: (vgt, dis, vgtda, vgtdu, disda, disdu) - last 4 items only if calc_derivatives=True
    """
    # Convert inputs to complex numbers
    z = a + 1j * (-u)
    
    # Constants for the approximation
    a_coeffs = np.array([
        122.607931777104326,
        214.382388694706425,
        181.928533092181549,
        93.155580458138441,
        30.180142196210589,
        5.912626209773153,
        0.564189583562615
    ])
    
    b_coeffs = np.array([
        122.60793177387535,
        352.730625110963558,
        457.334478783897737,
        348.703917719495792,
        170.354001821091472,
        53.992906912940207,
        10.479857114260399
    ])
    
    # Calculate numerator and denominator polynomials
    num = np.polyval(a_coeffs[::-1], z)
    den = np.polyval(np.append(b_coeffs[::-1], 1), z)
    
    fz = num / den
    
    vgt = np.real(fz)
    dis = np.imag(fz)
    
    if not calc_derivatives:
        return vgt, dis
    
    # Calculate derivatives if requested
    numdz = np.polyval(np.polyder(a_coeffs[::-1]), z)
    dendz = np.polyval(np.polyder(np.append(b_coeffs[::-1], 1)), z)
    
    fzdz = numdz/den - num*dendz/den**2
    
    vgtda = np.real(fzdz)
    vgtdu = np.imag(fzdz)
    disda = vgtdu
    disdu = -vgtda
    
    return vgt, dis, vgtda, vgtdu, disda, disdu

def niris_mefit(x: np.ndarray, data: np.ndarray, initial_params: np.ndarray, 
                display: bool = False) -> Tuple[np.ndarray, float]:
    """
    Perform ME fitting using scipy's curve_fit
    
    Parameters:
        x (np.ndarray): wavelength array
        data (np.ndarray): observed Stokes profiles
        initial_params (np.ndarray): initial parameter estimates
        display (bool): whether to display fit results
        
    Returns:
        Tuple[np.ndarray, float]: (best fit parameters, chi-square value)
    """
    n = len(x) // 4
    
    # Calculate weights based on noise
    weights = np.array([get_noise(data[:,0])] * n)
    for k in range(1, 4):
        weights = np.concatenate([weights, [get_noise(data[:,k])] * n])
    weights = 1.0 / weights**2
    
    # Define parameter bounds
    bounds = (
        [0, 0, 0, 0.5, 0.12, 0, -0.25, -np.inf, -np.inf],  # lower bounds
        [5000, np.pi, np.pi, 100, 0.17, 10, 0.25, np.inf, np.inf]  # upper bounds
    )
    
    # Perform the fit
    popt, pcov = curve_fit(niris_MEsinglet, x, np.swapaxes(data, 0, 1).flatten(), 
                      p0=initial_params,
                      sigma=weights, 
                      bounds=bounds, 
                      maxfev=10,
                      method='lm')
    # Calculate chi-square
    fitted = niris_MEsinglet(x, *popt)
    chisq = np.sum(weights * (np.swapaxes(data, 0, 1).flatten() - fitted)**2) / (4 * n)
    
    if display:
        plot_fit_results(x, np.swapaxes(data, 0, 1).flatten(), fitted)
            
    return popt, chisq

def niris_mefit_lmfit(x: np.ndarray, data: np.ndarray, initial_params: np.ndarray, 
                display: bool = False, max_nfev: int = 50) -> Tuple[np.ndarray, float]:
    """
    Perform ME fitting using lmfit
    
    Parameters:
        x (np.ndarray): wavelength array
        data (np.ndarray): observed Stokes profiles
        initial_params (np.ndarray): initial parameter estimates
        display (bool): whether to display fit results
        max_nfev (int): maximum number of function evaluations
        
    Returns:
        Tuple[np.ndarray, float]: (best fit parameters, chi-square value)
    """
    n = len(x) // 4
    
    # Calculate weights based on noise
    weights = np.array([get_noise(data[:,0])] * n)
    for k in range(1, 4):
        weights = np.concatenate([weights, [get_noise(data[:,k])] * n])
    weights = 1.0 / weights**2
    
    # Create lmfit Parameters object
    params = Parameters()
    param_names = ['B', 'theta', 'chi', 'eta0', 'dlambdaD', 'a', 'lambda0', 'B0', 'B1']
    bounds = {
        'B': (0, 5000),
        'theta': (0, np.pi),
        'chi': (0, np.pi), 
        'eta0': (0.5, 20),
        'dlambdaD': (0.12, 0.25),
        'a': (0, 10),
        'lambda0': (-0.25, 0.25),
        'B0': (-np.inf, np.inf),
        'B1': (-np.inf, np.inf)
    }
    
    # Add parameters with bounds
    for name, value, (min_val, max_val) in zip(param_names, initial_params, bounds.values()):
        params.add(name, value=value, min=min_val, max=max_val, vary=True)
    
    # Create model
    def model_func(x, B, theta, chi, eta0, dlambdaD, a, lambda0, B0, B1):
        return niris_MEsinglet(x, B, theta, chi, eta0, dlambdaD, a, lambda0, B0, B1)
    
    model = Model(model_func)
    
    # Set up fit options for numerical derivatives
    # For leastsq method, epsfcn controls the step size for numerical derivatives
    fit_kws = {
        'ftol': 1e-4,  # Function tolerance
        'gtol': 1e-4,  # Gradient tolerance
        'xtol': 1e-4,  # Parameter tolerance
        'epsfcn': 1e-7   # Step size for numerical derivatives
    }
    
    data_flat = np.swapaxes(data, 0, 1).flatten()
    
    # Use numerical derivatives by setting calc_covar=False
    result = model.fit(data_flat, 
                      params, 
                      x=x, 
                      weights=weights,
                      method='leastsq',  # Levenberg-Marquardt
                      max_nfev=max_nfev,
                      calc_covar=False,  # Don't calculate covariance matrix analytically
                      fit_kws=fit_kws)
    
    # Get best fit parameters
    popt = np.array([result.params[name].value for name in param_names])
    
    # Calculate chi-square
    fitted = niris_MEsinglet(x, *popt)
    chisq = np.sum(weights * (data_flat - fitted)**2) / (4 * n)
    # figs, axs = plt.subplots(1, 2, figsize=(12, 4))
    # axs[0].plot(data_flat)
    # axs[0].plot(fitted)
    # axs[1].plot(weights * (data_flat - fitted)**2)
    # plt.show()
    if display:
        plot_fit_results(x, data_flat, fitted)
        
        # Additional fit diagnostics from lmfit
        print("\nFit Report:")
        print(result.fit_report())
            
    return popt, chisq

def plot_fit_results(x: np.ndarray, data: np.ndarray, fitted: np.ndarray):
    """Plot the fitting results"""
    n = len(x) // 4
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    titles = ['I/Ic', 'Q/Ic', 'U/Ic', 'V/Ic']
    
    for i, ax in enumerate(axs.flat):
        y_range = [-0.1, 0.15] if i > 0 else [0.5, 1.2]
        ax.plot(x[:n], data[i*n:(i+1)*n], 'k.', label='Observed')
        ax.plot(x[:n], fitted[i*n:(i+1)*n], 'r-', label='Fitted')
        ax.set_ylim(y_range)
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel(titles[i])
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def save_results(res_dir: Path, data_file: Path, parmap: np.ndarray, 
                 parmap_wfa: np.ndarray, original: np.ndarray, 
                 fit_array: np.ndarray, chisqa: np.ndarray, 
                 hdr: fits.Header, wfa_only=False, fits_only=False,
                 npz_only=False):
    """Save inversion results to files"""
    # Create output arrays
    nx, ny = parmap.shape[:2]
    
    # 9 params + continuum = 10
    data = np.zeros((nx, ny, 10))
    data_wfa = np.zeros((nx, ny, 10))
    cont = np.mean(original[..., :16, 0], axis=2)
     
    data[..., :9] = parmap[..., :9]  
    data[..., 9] = cont              
    
    data_wfa[..., :9] = parmap_wfa[..., :9]
    data_wfa[..., 9] = cont
    
    # Update FITS header
    hdr['COMMENT'] = 'B total; Incl; Azim; opacity rate; Dopp width; Damping; Dopp Shift; B0; B1; continuum'
    
    # Generate output filenames
    tag = 'wfa' if wfa_only else 'inv'
    base_name = f'gst_niris_{tag}_20{data_file.stem[5:]}'
    
    # Save files
    if npz_only:
        np.savez(res_dir / f'{base_name}.npz',
                 data=data,
                 chisqa=chisqa if not wfa_only else None,
                 header=dict(hdr))
    
    if fits_only:
        fits.writeto(res_dir / f'{base_name}.fits',
                     data,
                     header=hdr,
                     overwrite=True)

    if wfa_only:
        fits.writeto(res_dir / f'{base_name}_wfa.fits',
                     data_wfa,
                     header=hdr,
                     overwrite=True)
    
def process_ME_inversion(data_file: Path) -> None:
    """
    Process ME inversion for a given data file
    
    Parameters:
        data_file (Path): Path to the input data file
        display (bool): Whether to display diagnostic plots
    """
    # Setup paths
    res_dir = Path(data_file.parent)
    print(f"Processing {data_file}")
    print(f"Data will be saved in {res_dir}")

    # Load data
    with fits.open(data_file) as hdul:
        b = hdul[0].data
        hdr = hdul[0].header
    
    # swap b four axes in reverse order
    b = np.swapaxes(b, 0, 3)
    b = np.swapaxes(b, 1, 2)

    # Get wavelength array
    lambda_rest = hdr['REFWV']

    # Get array dimensions
    nx, ny, nw = b.shape[0:3]

    # generate lambda array using start and end scan range
    l_step = (hdr['ENDWV'] - hdr['STARTWV']) / (nw - 1)
    dlambda = np.arange(nw) * l_step + hdr['STARTWV']

    # Find line center position
    aver_prof0 = np.mean(b[..., 0].reshape(-1, nw), axis=0)
    x = np.arange(nw)
    coeffs = np.polyfit(x, aver_prof0, 2)
    yfit = np.polyval(coeffs, x)
    aver_prof = aver_prof0 - yfit + np.mean(yfit)
    ind = np.argmin(aver_prof)
        
    if abs(dlambda[ind]) > l_step:
        lambda_shift = dlambda[ind]
    else:
        lambda_shift = 0
        
    if abs(lambda_shift) > 1.:
        raise ValueError('Lambda Shift is too large')

    # shift lambda scale
    dlambda = dlambda - lambda_shift
    
    # Select wavelength range
    s = np.where(abs(dlambda) <= 2.5)[0]
    
    # Initialize arrays
    factor = np.median(b[..., 40 if 40 < nw else nw-1, 0])
    parmap = np.zeros((nx, ny, 9))
    parmap_wfa = np.zeros((nx, ny, 9))
    chisqa = np.zeros((nx, ny))
    fit_array = np.zeros((nx, ny, len(s), 4))
    original = np.zeros((nx, ny, len(s), 4))
    
    # Start timer
    t1 = time.time()
    
    # Main inversion loop
    for xpos in tqdm(range(nx)):
        for ypos in range(ny):
            # define x (4 x lambda) array for fitting
            x = np.tile(dlambda[s], 4)
            
            # populate y (Stokes profiles for fitting)
            b1 = b[xpos, ypos, s, :]
            
            # get continuum indexes at |lambda| gt 1.5A 
            cont_ind = np.where(abs(x[:len(s)]) >= 1.5)[0]
            if len(cont_ind) == 0:
                cont_ind = np.where(abs(x[:len(s)]) >= 1.0)[0]

            # normalize data
            data = b1 / factor
            
            # adjustment of non-zero bias in Q, U and V
            for k in range(1, 4):
                data[:, k] -= np.median(data[:, k][cont_ind])
            
            # prepare profiles for fitting/inversions
            original[xpos, ypos] = data.reshape(4, -1).T
            
            try:
                # Initialize parameters using weak-field approximation
                par = init_par(x, data, cont_ind)
                
                # Perform ME fitting
                result_popt, chisq = niris_mefit_lmfit(x, data, par, display=False)
                yfit = niris_MEsinglet(x, *result_popt)
                fit_array[xpos, ypos] = yfit.reshape(4, -1).T
                chisqa[xpos, ypos] = chisq / (len(s) * 4)
                parmap[xpos, ypos] = result_popt
                parmap_wfa[xpos, ypos] = par
                
            except Exception as e:
                print(f"Error at position ({xpos}, {ypos}): {str(e)}")
                # Use initial parameters if fit fails
                yfit = niris_MEsinglet(x, *par)
                fit_array[xpos, ypos] = yfit.reshape(4, -1).T
                chisqa[xpos, ypos] = np.nan
                parmap[xpos, ypos] = par
                parmap_wfa[xpos, ypos] = par

    t2 = time.time()
    print(f'Computing time: {(t2-t1)/60:.2f} min')
    
    # Save results
    save_results(res_dir, data_file, parmap, original, fit_array, chisqa, hdr, wfa_only=True)

def process_single_row(xpos: int, params: dict, max_nfev: int = 50) -> tuple:
    """
    Process a single row of the data
    
    Parameters:
        xpos (int): x position to process
        params (dict): dictionary containing all necessary parameters
        
    Returns:
        tuple: (xpos, results for this row)
    """
    ny = params['ny']
    dlambda = params['dlambda']
    s = params['s']
    b = params['b']
    factor = params['factor']
    
    # Initialize arrays for this row
    row_parmap = np.zeros((ny, 9))
    row_parmap_wfa = np.zeros((ny, 9))
    row_chisqa = np.zeros(ny)
    row_fit_array = np.zeros((ny, len(s), 4))
    row_original = np.zeros((ny, len(s), 4))
    
    for ypos in range(ny):
        try:
            # define x (4 x lambda) array for fitting
            x = np.tile(dlambda[s], 4)
            
            # populate y (Stokes profiles for fitting)
            b1 = b[xpos, ypos, s, :]
            
            # get continuum indexes at |lambda| gt 1.5A 
            cont_ind = np.where(abs(x[:len(s)]) >= 1.5)[0]
            if len(cont_ind) == 0:
                cont_ind = np.where(abs(x[:len(s)]) >= 1.0)[0]

            # normalize data
            data = b1 / factor
            
            # adjustment of non-zero bias in Q, U and V
            for k in range(1, 4):
                data[:, k] -= np.median(data[:, k][cont_ind])
            
            # prepare profiles for fitting/inversions
            row_original[ypos] = data.reshape(4, -1).T
            
            # Initialize parameters
            par = init_par(x, data, cont_ind)
            
            # Perform ME fitting
            result_popt, chisq= niris_mefit_lmfit(x, data, par, display=False, max_nfev=max_nfev)
            yfit = niris_MEsinglet(x, *result_popt)
            row_fit_array[ypos] = yfit.reshape(4, -1).T
            row_chisqa[ypos] = chisq / (len(s) * 4)
            row_parmap[ypos] = result_popt
            row_parmap_wfa[ypos] = par
            
        except Exception as e:
            print(f"Error at position ({xpos}, {ypos}): {str(e)}")
            # Use initial parameters if fit fails
            yfit = niris_MEsinglet(x, *par)
            row_fit_array[ypos] = yfit.reshape(4, -1).T
            row_chisqa[ypos] = np.nan
            row_parmap[ypos] = par
            row_parmap_wfa[ypos] = par
    
    return xpos, (row_parmap, row_parmap_wfa, row_chisqa, row_fit_array, row_original)

def plot_results(res_dir: Path, data_file: Path) -> None:

    #Read the output
    res = np.load(Path(res_dir, f'gst_niris_inv_20{data_file.stem[5:]}.npz'))
    # print the keys
    print(res.files)
    # read the data
    data = res['data']
    # read each data [Bfield, theta, chi, eta0, dlambdaD,adamp, lambda0, B0, B1]
    Bfield = data[..., 0]
    theta = data[..., 1]
    chi = data[..., 2]
    eta0 = data[..., 3]
    dlambdaD = data[..., 4]
    adamp = data[..., 5]
    lambda0 = data[..., 6]
    B0 = data[..., 7]
    B1 = data[..., 8]

    # plot Bfield, inclination, azimuth
    figs, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(Bfield, origin='lower', cmap='gray', vmin=0, vmax=3000)
    axs[0].set_title('Bfield')
    axs[1].imshow(theta, origin='lower', cmap='gray', vmin=0, vmax=np.pi)
    axs[1].set_title('Inclination')
    axs[2].imshow(chi, origin='lower', cmap='gray', vmin=0, vmax=np.pi)
    axs[2].set_title('Azimuth')
    plt.show()

    # plot Bx, By, Bz
    Bx = Bfield * np.cos(chi) * np.sin(theta)
    By = Bfield * np.sin(chi) * np.sin(theta)
    Bz = Bfield * np.cos(theta)

    figs, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(Bx, origin='lower', cmap='gray', vmin=-1500, vmax=1500)
    axs[0].set_title('Bx')
    axs[1].imshow(By, origin='lower', cmap='gray', vmin=-1500, vmax=1500)
    axs[1].set_title('By')
    axs[2].imshow(Bz, origin='lower', cmap='gray', vmin=-1500, vmax=1500)
    axs[2].set_title('Bz')
    plt.savefig('Bx_By_Bz_python.png', dpi=300)
    plt.show()

def process_ME_inversion_parallel(data_file, res_dir=None, n_cpu: Optional[int] = None, max_nfev: int = 50, display: bool = False) -> None:
    """
    Process ME inversion for a given data file
    
    Parameters:
        data_file (Path): Path to the input data file
        display (bool): Whether to display diagnostic plots
        n_cpu (int, optional): Number of CPUs to use. If None, uses all available CPUs - 1
    """
    # Setup paths and load data
    if res_dir is None:
        res_dir = Path(data_file.parent)
    print(f"Processing {data_file}")
    print(f"Data will be saved in {res_dir}")

    with fits.open(data_file) as hdul:
        b = hdul[0].data
        hdr = hdul[0].header
    
    # swap b four axes in reverse order
    b = np.swapaxes(b, 0, 3)
    b = np.swapaxes(b, 1, 2)

    # Get dimensions and calculate wavelength array
    nx, ny, nw = b.shape[0:3]
    l_step = (hdr['ENDWV'] - hdr['STARTWV']) / (nw - 1)
    dlambda = np.arange(nw) * l_step + hdr['STARTWV']

    # Find line center position
    aver_prof0 = np.mean(b[..., 0].reshape(-1, nw), axis=0)
    x = np.arange(nw)
    coeffs = np.polyfit(x, aver_prof0, 2)
    yfit = np.polyval(coeffs, x)
    aver_prof = aver_prof0 - yfit + np.mean(yfit)
    ind = np.argmin(aver_prof)
        
    if abs(dlambda[ind]) > l_step:
        lambda_shift = dlambda[ind]
    else:
        lambda_shift = 0
        
    if abs(lambda_shift) > 1.:
        raise ValueError('Lambda Shift is too large')

    # shift lambda scale
    dlambda = dlambda - lambda_shift
    
    # Select wavelength range
    s = np.where(abs(dlambda) <= 2.5)[0]
    
    # Initialize arrays
    factor = np.median(b[..., 40 if 40 < nw else nw-1, 0])
    parmap = np.zeros((nx, ny, 9))
    parmap_wfa = np.zeros((nx, ny, 9))
    chisqa = np.zeros((nx, ny))
    fit_array = np.zeros((nx, ny, len(s), 4))
    original = np.zeros((nx, ny, len(s), 4))
    
    # Prepare parameters dictionary for parallel processing
    params = {
        'ny': ny,
        'dlambda': dlambda,
        's': s,
        'b': b,
        'factor': factor
    }
    
    # Start timer
    t1 = time.time()
    
    # Set number of CPUs
    if n_cpu is None:
        n_cpu = max(1, cpu_count() - 1)  # Leave one CPU free
    
    print(f"Using {n_cpu} CPUs")
    
    # Create pool and run parallel processing
    with Pool(n_cpu) as pool:
        process_func = partial(process_single_row, params=params, max_nfev=max_nfev)
        results = list(tqdm(pool.imap(process_func, range(nx)), total=nx))
    
    # Collect results
    for xpos, (row_parmap, row_parmap_wfa, row_chisqa, row_fit_array, row_original) in results:
        parmap[xpos] = row_parmap
        parmap_wfa[xpos] = row_parmap_wfa
        chisqa[xpos] = row_chisqa
        fit_array[xpos] = row_fit_array
        original[xpos] = row_original
    
    t2 = time.time()
    print(f'Computing time: {(t2-t1)/60:.2f} min')
    
    # Save results
    save_results(res_dir, data_file, parmap, parmap_wfa, original, fit_array, chisqa, hdr, npz_only=True)
    
    if display:
        plot_results(res_dir, data_file)

def initialize_shared_memory(shm_name, shape, dtype):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    global b_shared
    b_shared = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

def process_chunk_shared(chunk_info, dlambda, s, factor, ny, max_nfev, display):
    global b_shared
    start_row, end_row = chunk_info
    num_rows = end_row - start_row
    parmap = np.zeros((num_rows, ny, 9))
    parmap_wfa = np.zeros((num_rows, ny, 9))
    chisqa = np.zeros((num_rows, ny))
    fit_array = np.zeros((num_rows, ny, len(s), 4))
    original = np.zeros((num_rows, ny, len(s), 4))
    
    x = np.tile(dlambda[s], 4)
    
    for i, xpos in enumerate(range(start_row, end_row)):
        for ypos in range(ny):
            try:
                b1 = b_shared[xpos, ypos, s, :]
                cont_ind = np.where(abs(x[:len(s)]) >= 1.5)[0]
                if len(cont_ind) == 0:
                    cont_ind = np.where(abs(x[:len(s)]) >= 1.0)[0]

                data = b1 / factor
                for k in range(1, 4):
                    data[:, k] -= np.median(data[:, k][cont_ind])

                par = init_par(x, data, cont_ind)
                result_popt, chisq_val = niris_mefit_lmfit(x, data, par, display=False, max_nfev=max_nfev)
                yfit = niris_MEsinglet(x, *result_popt)

                parmap[i, ypos] = result_popt
                parmap_wfa[i, ypos] = par
                chisqa[i, ypos] = chisq_val
                fit_array[i, ypos] = yfit.reshape(4, -1).T
                original[i, ypos] = data

            except Exception as e:
                print(f"Error at position ({xpos}, {ypos}): {str(e)}")
                parmap[i, ypos] = par
                parmap_wfa[i, ypos] = par
                chisqa[i, ypos] = np.inf
                fit_array[i, ypos] = niris_MEsinglet(x, *par).reshape(4, -1).T
                original[i, ypos] = data
    
    return (start_row, parmap, parmap_wfa, chisqa, fit_array, original)

def process_ME_inversion_parallel_shared_memory(data_file: Path, n_cpu: int = 32, display: bool = False) -> None:
    """Optimized parallel processing for ME inversion with shared memory."""
    # Load data
    with fits.open(data_file) as hdul:
        b = hdul[0].data
        hdr = hdul[0].header

    # Swap axes as required
    b = np.swapaxes(b, 0, 3)
    b = np.swapaxes(b, 1, 2)

    # Get wavelength array
    nw = b.shape[2]
    l_step = (hdr['ENDWV'] - hdr['STARTWV']) / (nw - 1)
    dlambda = np.arange(nw) * l_step + hdr['STARTWV']

    # Find line center position
    aver_prof0 = np.mean(b[..., 0].reshape(-1, nw), axis=0)
    x_fit = np.arange(nw)
    coeffs = np.polyfit(x_fit, aver_prof0, 2)
    yfit_poly = np.polyval(coeffs, x_fit)
    aver_prof = aver_prof0 - yfit_poly + np.mean(yfit_poly)
    ind = np.argmin(aver_prof)

    lambda_shift = dlambda[ind] if abs(dlambda[ind]) > l_step else 0
    if abs(lambda_shift) > 1.:
        raise ValueError('Lambda Shift is too large')

    dlambda = dlambda - lambda_shift
    s = np.where(abs(dlambda) <= 2.5)[0]

    # Initialize parameters
    nx, ny = b.shape[0:2]
    factor = np.median(b[..., 40 if 40 < nw else nw-1, 0])

    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=b.nbytes)
    b_shared = np.ndarray(b.shape, dtype=b.dtype, buffer=shm.buf)
    b_shared[:] = b[:]

    # Prepare chunks
    chunk_size = max(20, nx // (n_cpu * 2))
    chunks = [(i, min(i + chunk_size, nx)) for i in range(0, nx, chunk_size)]
    print(f"Processing {len(chunks)} chunks of {chunk_size} rows each using {n_cpu} CPUs")

    # Create partial function with fixed arguments
    process_func = partial(process_chunk_shared, dlambda=dlambda, s=s, factor=factor, ny=ny, max_nfev=250, display=False)

    # Initialize Pool with initializer to attach shared memory
    with Pool(n_cpu, initializer=initialize_shared_memory, initargs=(shm.name, b.shape, b.dtype)) as pool:
        results = list(tqdm(pool.imap(process_func, chunks), total=len(chunks)))

    # Clean up shared memory
    shm.close()
    shm.unlink()

    # Initialize result arrays
    parmap = np.zeros((nx, ny, 9))
    parmap_wfa = np.zeros((nx, ny, 9))
    chisqa = np.zeros((nx, ny))
    fit_array = np.zeros((nx, ny, len(s), 4))
    original = np.zeros((nx, ny, len(s), 4))

    # Aggregate results
    for start_row, chunk_p, chunk_pwfa, chunk_c, chunk_f, chunk_o in results:
        end_row = start_row + chunk_p.shape[0]
        parmap[start_row:end_row] = chunk_p
        parmap_wfa[start_row:end_row] = chunk_pwfa
        chisqa[start_row:end_row] = chunk_c
        fit_array[start_row:end_row] = chunk_f
        original[start_row:end_row] = chunk_o

    # Save results
    save_results(res_dir=data_file.parent, data_file=data_file, parmap=parmap, parmap_wfa=parmap_wfa,
                original=original, fit_array=fit_array, chisqa=chisqa, hdr=hdr, wfa_only=False)

    if display:
        plot_results(res_dir=data_file.parent, data_file=data_file)

def process_ME_inversion_single_point(data_file: Path, x_pos: int, y_pos: int, display: bool = True, max_nfev: int = 50) -> dict:
    """
    Process ME inversion for a single point in the data
    
    Parameters:
        data_file (Path): Path to the input data file
        x_pos (int): X coordinate of the point to analyze
        y_pos (int): Y coordinate of the point to analyze
        display (bool): Whether to display diagnostic plots
        
    Returns:
        dict: Dictionary containing the fitting results and parameters
    """
    # Load data
    with fits.open(data_file) as hdul:
        b = hdul[0].data
        hdr = hdul[0].header
    
    # swap b four axes in reverse order
    b = np.swapaxes(b, 0, 3)
    b = np.swapaxes(b, 1, 2)

    # Get wavelength array
    l_step = (hdr['ENDWV'] - hdr['STARTWV']) / (b.shape[2] - 1)
    dlambda = np.arange(b.shape[2]) * l_step + hdr['STARTWV']

    # Find line center position and shift wavelength scale
    aver_prof0 = np.mean(b[..., 0].reshape(-1, b.shape[2]), axis=0)
    x = np.arange(b.shape[2])
    coeffs = np.polyfit(x, aver_prof0, 2)
    yfit = np.polyval(coeffs, x)
    aver_prof = aver_prof0 - yfit + np.mean(yfit)
    ind = np.argmin(aver_prof)
    
    lambda_shift = dlambda[ind] if abs(dlambda[ind])-0.00005 > l_step else 0
    if abs(lambda_shift) > 1.:
        raise ValueError('Lambda Shift is too large')
    

    dlambda = dlambda - lambda_shift
 
    # Select wavelength range
    s = np.where(abs(dlambda) <= 2.0)[0]
    
    # Get normalization factor
    factor = np.median(b[..., 40 if 40 < b.shape[2] else b.shape[2]-1, 0])
    
    # Extract data for the selected point
    x = np.tile(dlambda[s], 4)
    b1 = b[x_pos, y_pos, s, :]
    
    # Get continuum indices
    cont_ind = np.where(abs(x[:len(s)]) >= 1.5)[0]
    if len(cont_ind) == 0:
        cont_ind = np.where(abs(x[:len(s)]) >= 1.0)[0]

    # Normalize data
    data = b1 / factor
   
    # Adjust non-zero bias in Q, U and V
    for k in range(1, 4):
        data[:, k] -= np.median(data[:, k][cont_ind])
  

    # Initialize parameters
    par = init_par(x, data, cont_ind)

    # Perform ME fitting
    result_popt, chisq = niris_mefit_lmfit(x, data, par, display=False, max_nfev=max_nfev)
    yfit = niris_MEsinglet(x, *result_popt)
    initial_fit = niris_MEsinglet(x, *par)
    # Calculate magnetic field components
    B, theta, chi = result_popt[:3]
    Bx = B * np.cos(chi) * np.sin(theta)
    By = B * np.sin(chi) * np.sin(theta)
    Bz = B * np.cos(theta)
    
    # Prepare results dictionary
    results = {
        'wavelength': x[:len(s)],
        'observed_profiles': data,
        'fitted_profiles': yfit.reshape(4, -1).T,
        'initial_parameters': par,
        'initial_fit': initial_fit.reshape(4, -1).T,
        'final_parameters': result_popt,
        'chi_square': chisq,
        'magnetic_field': {
            'B_total': B,
            'B_x': Bx,
            'B_y': By,
            'B_z': Bz,
            'inclination': theta * 180/np.pi,  # Convert to degrees
            'azimuth': chi * 180/np.pi        # Convert to degrees
        }
    }
    
    if display:
        # Plot the results
        print(par)
        print(result_popt)
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        titles = ['I/Ic', 'Q/Ic', 'U/Ic', 'V/Ic']
        y_ranges = [(0.5, 1.2), (-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)]
        
        for i, (ax, title, y_range) in enumerate(zip(axs.flat, titles, y_ranges)):
            ax.plot(x[:len(s)], data[:, i], 'k.', label='Observed')
            ax.plot(x[:len(s)], initial_fit.reshape(4, -1).T[:, i], 'g-', label='Initial')
            ax.plot(x[:len(s)], yfit.reshape(4, -1).T[:, i], 'r-', label='Fitted')
            ax.set_ylim(y_range)
            ax.set_xlabel('Wavelength (Å)')
            ax.set_ylabel(title)
            ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print magnetic field information
        print("\nMagnetic Field Results:")
        print(f"Total Field Strength: {B:.1f} G")
        print(f"Components: Bx = {Bx:.1f} G, By = {By:.1f} G, Bz = {Bz:.1f} G")
        print(f"Inclination: {theta * 180/np.pi:.1f}°")
        print(f"Azimuth: {chi * 180/np.pi:.1f}°")
        print(f"Chi-square: {chisq:.3f}")
    np.save('/project/bs644/ql47/ME/PINN4ME/wavelengths.npy', x)
    np.save('/project/bs644/ql47/ME/PINN4ME/data.npy', data)
    return results
    
# Example usage:
if __name__ == "__main__":
    data_file = Path('/project/bs644/ql47/ME/cal_240725_181800.fts')
    process_ME_inversion_parallel(data_file, n_cpu=None, max_nfev=50, display=False)  # Uses all CPUs - 1