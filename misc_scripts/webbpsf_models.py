import numpy as np
from astropy.io import fits
import webbpsf
import tqdm
import glob
import photutils
import sys
import os


def calculate_mrs_psf(ifu_cube, wave=None, oversample=4, broadening='both'): 
    """
    Calculate a model of the MIRI/MRS Point-Spread Function (PSF) using the webbpsf package.
    N.B. this routine relies on a version of the webbpsf package that is currently in-development that supports
    the MRS IFU mode; this function will not work on the current release version of webbpsf!
        - See the pull request for the modified version here: https://github.com/spacetelescope/webbpsf/pull/691
    
    :param ifu_cube: HDUList
        A FITS object containing observational data from the channel/band that one wants to create a PSF model for.
        This is only necessary to obtain information on the wavelength and spatial sampling scales.
    :param wave: ndarray, optional
        An optional argument specifying which monochromatic wavelengths to generate PSFs for. If left empty, then PSFs
        will be generated for every wavelength slice in the channel/band of the `ifu_cube` parameter.
    :param oversample: integer
        The oversampling factor when calculating the PSF models. Default = 4.
    :param broadening: str
        Whether or not to broaden the PSF to match the observed MRS PSF FWHM. May be 'both' for both alpha and beta axes,
        None for neither axis, or anything else to broaden just the beta axis.
    """

    # Get channel/band information
    channel = ifu_cube[0].header['CHANNEL']
    band = ifu_cube[0].header['BAND']
    detector = ifu_cube[0].header['DETECTOR']
    assert channel in ('1', '2', '3', '4'), f"Unrecognized channel: {channel}, please only input single-channel cubes"
    assert band in ('SHORT', 'MEDIUM', 'LONG'), f"Unrecognized band: {band}, please only input single-band cubes"
    assert detector in ('MIRIFUSHORT', 'MIRIFULONG'), f"Unrecognized detector: {detector}, please only input single-detector cubes"

    # Get wavelength array
    hdr = ifu_cube[1].header
    wave_full = hdr["CRVAL3"] + hdr["CDELT3"] * (hdr["CRPIX3"] + np.arange(hdr["NAXIS3"]) - 1)
    if wave is None:
        wave = wave_full
    if type(wave) in (float, int):
        wave = [wave]

    # Get the field of view in arcseconds
    pix_as = np.sqrt(hdr["PIXAR_A2"])
    # pix_as = {'1': 0.13, '2': 0.17, '3': 0.2, '4': 0.34}[channel]
    fov_as = np.array(ifu_cube[1].data.shape[1:]) * pix_as

    print("Setting up MIRI detector object")
    # Set up MIRI object with the IFU configurations
    miri = webbpsf.MIRI()
    # Get OPD inormation at the time of the observation
    miri.load_wss_opd_by_date(ifu_cube[0].header['DATE-BEG'])

    # Configure for the proper channel/band setup
    assert (f'MIRI-IFU_{channel}' in miri.image_mask_list) and (f'D{band}' in miri.filter_list), "Channel/Band setup not found in webbpsf " + \
         "MIRI options. Please make sure you are using a version of webbpsf that supports MIRI MRS mode."
    miri.image_mask = f'MIRI-IFU_{channel}'
    miri.filter = f'D{band}'
    miri.deector = detector
    miri.pixelscale = pix_as

    # Calculate offsets based on centroiding
    data2d = np.nansum(ifu_cube['SCI'].data, axis=0)
    err2d = np.sqrt(np.nansum(ifu_cube['ERR'].data**2, axis=0))
    mx = np.unravel_index(np.nanargmax(data2d), data2d.shape)
    centroid_x, centroid_y = photutils.centroids.centroid_2dg(data2d[mx[0]-5:mx[0]+5, mx[1]-5:mx[1]+5], 
                                                              error=err2d[mx[0]-5:mx[0]+5, mx[1]-5:mx[1]+5]) + (mx[1], mx[0]) - 5 + 1
    center_y, center_x = np.array(ifu_cube[1].data.shape[1:]) / 2 + 0.5
    offset_x_as = (centroid_x - center_x) * pix_as
    offset_y_as = (centroid_y - center_y) * pix_as
    miri.options['source_offset_x'] = offset_x_as
    miri.options['source_offset_y'] = offset_y_as

    # Prepare output array
    psf_cube = np.zeros((len(wave), *ifu_cube[1].data.shape[1:]))

    for i in tqdm.trange(len(wave)):
        # Calculate the PSF
        try:
            psf = miri.calc_psf(monochromatic=wave[i], broadening=broadening, add_distortion=False, 
                                oversample=oversample, display=False, fov_arcsec=fov_as, crop_psf=True)
        except:
            # This is dumb but it works for some reason
            miri.pixelscale = pix_as
            psf = miri.calc_psf(monochromatic=wave[i], broadening=broadening, add_distortion=False, 
                                oversample=oversample, display=False, fov_arcsec=fov_as, crop_psf=True)
        # Add to output array
        psf_cube[i, :, :] = psf[3].data
    
    if psf_cube.shape[0] == 1:
        psf_cube = psf_cube[0, :, :]
    
    # Save
    psfhdu = fits.HDUList([fits.PrimaryHDU(header=ifu_cube[0].header), fits.ImageHDU(psf_cube, header=ifu_cube[1].header)])
    psfhdu.writeto(f'../src/templates/webbpsf/webbpsf_model_ch{channel}{band}_s3d.fits', overwrite=True)

    # Return the PSF cube object 
    return psf_cube

# Run this as a script
if __name__ == '__main__':

    path = sys.argv[1]
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*.fits'))
    else:
        files = [path]
    
    for f in files:
        # Load
        hdu = fits.open(f)
        # Calculate PSF
        calculate_mrs_psf(hdu)

