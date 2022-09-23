# Module for JWST Data IO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.visualization import quantity_support

# Useful constants
C_KMS = 299792.458  # speed of light in km/s

# MATPLOTLIB SETTINGS TO MAKE PLOTS LOOK PRETTY :)
SMALL = 12
MED = 14
BIG = 16

plt.rc('font', size=MED)          # controls default text sizes
plt.rc('axes', titlesize=MED)     # fontsize of the axes title
plt.rc('axes', labelsize=MED)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL)  # fontsize of the tick labels
plt.rc('legend', fontsize=MED)    # legend fontsize
plt.rc('figure', titlesize=BIG)   # fontsize of the figure title
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{amsmath}\usepackage{physics}")
plt.rc('font', family="Times New Roman")
quantity_support()


class CubeData:

    __slots__ = ['_wave', '_intensity', '_error', '_mask', 'z', 'wcs', '_omega', 'name', '_ra', '_dec', 
        'instrument', 'detector', 'channel', 'band']

    @u.quantity_input(wave="AA", intensity="erg / (s cm2 AA sr)", error="erg / (s cm2 AA sr)", omega="sr/pix", ra="deg", dec="deg",
        equivalencies=u.spectral_density(None))
    def __init__(self, wave, intensity, error, mask=None, z=None, wcs=None, omega=None, name='Generic Object', ra=None, dec=None,
        instrument=None, detector=None, channel=None, band=None):
        """
        An object for holding 3D IFU spectroscopy data.

        :param wave: array
            1D array of wavelengths, in units convertible to Angstroms
        :param intensity: array
            3D array of intensity, in units convertible to erg s^-1 cm^-2 Angstrom^-1 sr^-1
        :param error: array
            3D array of uncertainties, in units convertible to erg s^-1 cm^-2 Angstrom^-1 sr^-1
        :param mask: array, optional
            3D array of booleans acting as a mask for the flux/error data
        :param z: float, optional
            the redshift of the source
        :param wcs: WCS, optional
            an astropy World Coordinate System conversion object
        :param omega: float, optional
            the solid angle subtended by each spaxel, in units convertible to sr/spax
        :param name: str, optional
            a label for the source / data
        :param ra: float, optional
            the right ascension of the source, in units convertible to decimal degrees
        :param dec: float, optional
            the declination of the source, in units convertible to decimal degrees
        :param instrument: str, optional
            the instrument name, i.e. 'MIRI'
        :param detector: str, optional
            the detector name, i.e. 'MIRIFULONG'
        :param channel: int, optional
            the channel, i.e. 4
        :param band: str, optional
            the band, i.e. 'MULTIPLE'
        """
        # Check wavelength dimensions and units
        assert wave.ndim == 1, "wave array must be 1D"

        # Check dimensions
        assert (intensity.ndim == 3) and (intensity.shape[0] == wave.shape[0]), "intensity array's first axis must match wavelength"
        assert (error.ndim == 3) and (error.shape[0] == wave.shape[0]), "err array's first axis must match wavelength"
        assert omega.ndim == 0, "omega must be a scalar"

        # Populate instance attributes in the correct units
        self._wave = wave.to("AA").value
        _wv_extend = np.broadcast_to(self.wave[:, np.newaxis, np.newaxis], intensity.shape) << u.AA
        # Converting from MJy to CGS flux units:
        # 1 MJy = 10^6 Jy
        # 1 Jy = 10^-23 erg s^-1 cm^-2 Hz^-1
        # Fλdλ = Fνdν  --->  Fλ = Fν|dν/dλ| = Fν(c/λ^2)
        self._intensity = intensity.to("erg/(s cm2 AA sr)", equivalencies=u.spectral_density(_wv_extend)).value
        self._error = error.to("erg/(s cm2 AA sr)", equivalencies=u.spectral_density(_wv_extend)).value
        self._omega = omega.to("sr/pix").value
        self._ra = ra.to("deg").value
        self._dec = dec.to("deg").value
        if mask is None:
            mask = np.array([False] * self.intensity.size).reshape(self.intensity.shape)
        self._mask = mask

        # Additional non-unit data
        self.name = name
        self.wcs = wcs
        self.instrument = instrument
        self.detector = detector
        self.channel = channel
        self.band = band
        self.z = z

        # Apply mask
        self.intensity[self.mask] = np.nan
        self.error[self.mask] = np.nan
    
    def __repr__(self):
        """
        A pretty representation of the object
        """
        s =   '########################################################################################\n'
        s += f'############################       {self.name} IFU CUBE       #############################\n'
        s +=  '########################################################################################\n'
        s += f'INSTRUMENT:   {self.instrument}\n'
        s += f'DETECTOR:     {self.detector}\n'
        s += f'CHANNEL:      {self.channel}\n'
        s += f'BAND:         {self.band}\n'
        s += f'COORDINATES:  ({self.ra:.2f},{self.dec:.2f})\n'
        s += f'REDSHIFT:     {self.z:.4f}\n'
        s += f'WAVELENGTH:   ({np.nanmin(self.wave)/1e4:.2f} - {np.nanmax(self.wave)/1e4:.2f}) um\n'
        s += '########################################################################################\n'
        return s

    def plot_2d(self, fname, intensity="sr", error="sr", log_i=10, log_e=None, colormap=cm.cubehelix, use_wcs=True,
        space='wave'):
        """
        A plotting utility function for 2D maps of the intensity / error

        :param fname: str
            The file name of the plot to be saved.
        :param intensity: bool, str
            If 'sr', plot the intensity in flux sr^-1 units. If 'spax' or 'pix', plot in flux spax^-1 units. If False,
            do not plot the intensity.
        :param err: bool
            If 'sr', plot the error in flux sr^-1 units. If 'spax' or 'pix', plot in flux spax^-1 units. If False,
            do not plot the error.
        :param log_i: float, optional
            The base of the logarithm to take for intensity data. Set to None to not take the logarithm.
        :param log_e: float, optional
            The base of the logarithm to take for the error data. Set to None to not take the logarithm.
        :param colormap: Colormap, str
            Matplotlib colormap for the data.
        :param use_wcs: bool
            Whether or not to project the spatial axes into RA/Dec or to keep them in spaxels.
        :param space: str
            Specifies if in wavelength or frequency space, accepts 'wave', 'wavelength', 'freq', 'frequency'
        
        :return None:
        """
        if (intensity is not False) and (error is not False) and (intensity != error):
            raise ValueError("intensity and error should be plotted in the same units!")

        i_plot = self.intensity.copy()
        e_plot = self.error.copy()

        # nu or lambda subscript
        if space in ('freq', 'frequency'):
            sub = r'\nu'
            # Fλdλ = Fνdν  --->  Fν = Fλ|dλ/dν| = Fν(λ^2/c)
            i_plot *= self._wv_extend**2 / (C_KMS * 1e13)
            e_plot *= self._wv_extend**2 / (C_KMS * 1e13)
            # i_plot = i_plot.to("erg / (s cm2 Hz sr)", equivalencies=u.spectral_density(self._wv_extend))
            # e_plot = e_plot.to("erg / (s cm2 Hz sr)", equivalencies=u.spectral_density(self._wv_extend))
        elif space in ('wave', 'wavelength'):
            sub = r'\lambda'
        else:
            raise ValueError("unrecognized space argument: accepts 'wave', 'wavelength', 'freq', 'frequency'")

        # Sum data along the wavelength axis
        i_plot = np.nansum(i_plot, axis=0)
        e_plot = np.sqrt(np.nansum(e_plot**2, axis=0))
        # Reapply masks
        i_plot[i_plot == 0.] = np.nan
        e_plot[e_plot == 0.] = np.nan

        # Convert to pixel units if plotting w/r/t spaxels
        if intensity in ("spax", "pix"):
            i_plot *= self.omega
            e_plot *= self.omega

        # Take the logarithm if specified
        if log_e:
            e_plot = e_plot / np.abs(np.log(log_e) * i_plot)
        if log_i:
            i_plot = np.log(i_plot) / np.log(log_i)

        unit_str = r"erg$\,$s$^{-1}\,$cm$^{-2}\,$" + \
            (r"${\rm \AA}^{-1}\,$" if space in ("wave", "wavelength") else r"Hz$^{-1}\,$") + \
            (r"sr$^{-1}$" if intensity == "sr" or error == "sr" else r"spax$^{-1}$")
        ax1 = ax2 = None
        fig = plt.figure(figsize=(12,6) if (intensity and error) else (12,12))
        if intensity:
            # Plot intensity using WCS projection if specified
            ax1 = fig.add_subplot(121, projection=self.wcs if use_wcs else None)
            ax1.set_title(self.name)
            cdata = ax1.imshow(i_plot, origin='lower', cmap=colormap, 
                vmin=np.nanpercentile(i_plot, 1), vmax=np.nanpercentile(i_plot, 99))
            fig.colorbar(cdata, ax=ax1, fraction=0.046, pad=0.04,
                label=('' if log_i is None else r'$\log_{%s}$(' % str(log_i)) + r'$I_{%s}\,/\,$' % sub + unit_str + (r')' if log_i else ''))
            ax1.set_xlabel(r'$\alpha$' if use_wcs else r'$x$ (spaxels)')
            ax1.set_ylabel(r'$\delta$' if use_wcs else r'$y$ (spaxels)')
            ax1.tick_params(direction='in')

        if error:
            # Plot err using WCS projection if specified
            ax2 = fig.add_subplot(122, projection=self.wcs if use_wcs else None, sharey=ax1)
            ax2.set_title(self.name)
            cdata = ax2.imshow(e_plot, origin='lower', cmap=colormap,
                vmin=np.nanpercentile(e_plot, 1), vmax=np.nanpercentile(e_plot, 99))
            fig.colorbar(cdata, ax=ax2, fraction=0.046, pad=0.04,
                label=r'$\sigma_{I_{%s}}\,/\,$' % sub + unit_str if log_e is None else r'$\sigma_{{\rm log_{%s}} I_{%s}}$' % (str(log_e), sub))
            ax2.set_xlabel(r'$\alpha$' if use_wcs else r'$x$ (spaxels)')
            ax2.set_ylabel(r'$\delta$' if use_wcs else r'$y$ (spaxels)')
            # Turn off the left ticks if plotting both
            if intensity:
                ax2.set_ylabel(r' ')
                ax2.tick_params(axis='y', which='both', labelleft=False)
            ax2.tick_params(direction='in')

        # Save and close plot
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_1d(self, fname, intensity="sr", error="sr", log=False, space='wave', spaxel=None, linestyle='-'):
        """
        A plotting utility function for 1D spectra of individual spaxels or the full cube.
        
        :param fname: str
            The file name of the plot to be saved.
        :param intensity: bool, str
            If 'sr', plot the intensity in flux sr^-1 units. If 'spax' or 'pix', plot in flux spax^-1 units. If False,
            do not plot the intensity.
        :param err: bool
            If 'sr', plot the error in flux sr^-1 units. If 'spax' or 'pix', plot in flux spax^-1 units. If False,
            do not plot the error.
        :param log: float, optional
            The base of the logarithm to take for the flux/error data. Set to None to not take the logarithm.
        :param space: str
            Specifies if in wavelength or frequency space, accepts 'wave', 'wavelength', 'freq', 'frequency'
        :param spaxel: tuple, optional
            Tuple of (y,x) spaxel coordinates to plot the 1D spectrum for; otherwise sum over all spaxels.
        :param linestyle: str
            The matplotlib linestyle argument.
        
        :return None:
        """
        if (intensity is not False) and (error is not False) and (intensity != error):
            raise ValueError("intensity and error should be plotted in the same units!")

        i_plot = self.intensity.copy()
        e_plot = self.error.copy()

        # Conversion to frequency space
        if space in ('wave', 'wavelength'):
            sub = r'\lambda'
            # Convert to um
            xval = self.wave / 1e4
        elif space in ('freq', 'frequency'):
            sub = r'\nu'
            # c = λν ---> ν = c/λ
            # factor of 1e13 for AA/km but factor of 1/1e12 for THz
            xval = (C_KMS * 10) / self.wave
            # xval = self.wave.to(u.THz, equivalencies=u.spectral())
            # Fλdλ = Fνdν  --->  Fν = Fλ|dλ/dν| = Fν(λ^2/c)
            i_plot *= self._wv_extend**2 / (C_KMS * 1e13)
            e_plot *= self._wv_extend**2 / (C_KMS * 1e13)
            # i_plot = i_plot.to("erg / (s cm2 Hz sr)", equivalencies=u.spectral_density(self._wv_extend))
            # e_plot = e_plot.to("erg / (s cm2 Hz sr)", equivalencies=u.spectral_density(self._wv_extend))

        # Convert to per pixel units
        if intensity in ("pix", "spax") or error in ("pix", "spax"):
            i_plot *= self.omega
            e_plot *= self.omega

        if spaxel is None:
            # Sum data along the spatial axes
            i_plot = np.nansum(i_plot, axis=(1, 2))
            e_plot = np.sqrt(np.nansum(e_plot**2, axis=(1, 2)))
            # Reapply masks
            i_plot[i_plot == 0.] = np.nan
            e_plot[e_plot == 0.] = np.nan

            # If flux was not per spaxel then we picked up a spaxel/sr term that must be canceled
            if intensity == "sr":
                # Count up the number of GOOD pixels used in the summation for each wavelength bin
                i_plot /= np.nansum(~self.mask, axis=(1, 2))
            if error == "sr":
                e_plot /= np.nansum(~self.mask, axis=(1, 2))

        else:
            # Get data for the single spaxel to plot
            i_plot = i_plot[:, spaxel[0], spaxel[1]]
            e_plot = e_plot[:, spaxel[0], spaxel[1]]

        # Take the logarithm if specified
        if space in ('wave', 'wavelength') and log:
            e_plot = e_plot / np.abs(i_plot * np.log(log))
            i_plot = np.log(self.wave * i_plot) / np.log(log)
        elif space in ('freq', 'frequency') and log:
            e_plot = e_plot / np.abs(i_plot * np.log(log))
            freq_Hz = (C_KMS * 1e13) / self.wave
            i_plot = np.log(freq_Hz * i_plot) / np.log(log)
        
        xunit = r"${\rm \mu m}$" if space in ('wave', 'wavelength') else r"THz"
        yunit = r"erg$\,$s$^{-1}\,$cm$^{-2}\,$" + \
            (r"${\rm \AA}^{-1}\,$" if space in ('wave', 'wavelength') and log is False else r"Hz$^{-1}\,$" if space in ('freq', 'frequency') and log is False else r"") + \
            (r"sr$^{-1}\,$" if (intensity == "sr" or error == "sr") else r"spax$^{-1}$" if (intensity in ("spax", "pix") or error in ("spax", "pix")) and spaxel is not None else r"")
        yunittype = "F" if (intensity in ("spax", "pix") or error in ("spax", "pix")) and (spaxel is None) else "I"

        # Plot formatting
        fig, ax = plt.subplots(figsize=(10,5))
        if intensity:
            ax.plot(xval, i_plot, 'k', linestyle=linestyle, label='Data')
        if error and not intensity:
            ax.plot(xval, e_plot, 'k', linestyle=linestyle, label='$1\\sigma$ Error')
        if error and intensity:
            ax.fill_between(xval, i_plot-e_plot, i_plot+e_plot, color='k', alpha=0.5, label='$1\\sigma$ Error')
        ax.set_xlabel(r'$%s$' % sub + ' (' + xunit + ')')
        if not log:
            ax.set_ylabel(r'$%s_{%s}$' % (yunittype, sub) + ' (' + yunit + ')')
        else:
            ax.set_ylabel(r'$\log_{%s}{%s}{%s}_{%s}$' % (log, sub, yunittype, sub) + ' (' + yunit + ')')
        ax.legend(loc='upper right', frameon=False)
        ax.set_xlim(np.nanmin(xval), np.nanmax(xval))
        ax.set_title(self.name + (f'' if spaxel is None else f' Spaxel({spaxel[1]},{spaxel[0]})'))
        ax.tick_params(direction='in')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
    
    @classmethod
    def from_fits(cls, filepath, z=0):
        """
        A method for reading IFU data from JWST 's3d' fits files

        :param filepath: str
            the path to the 's3d' JWST FITS file
        :param z: float, optional
            the redshift of the source, if known
        
        :return cls:
            the object
        """
        # Load in the fits file
        hdu = fits.open(filepath)
        assert hdu[0].header['DATAMODL'] == 'IFUCubeModel', "The FITS file must contain IFU cube data!"

        # Wavelength dimension
        nz = hdu['SCI'].header['NAXIS3']
        # Solid angle of each spaxel
        omega = hdu['SCI'].header['PIXAR_SR'] << u.sr / u.pix
        # Construct the (linearly binned) wavelength array
        wave = (hdu['SCI'].header['CRVAL3'] + hdu['SCI'].header['CDELT3'] * np.arange(nz)) << u.Unit(hdu['SCI'].header['CUNIT3'])
        # Construct intensity and error array with units
        intensity = hdu['SCI'].data << u.Unit(hdu['SCI'].header['BUNIT'])
        error = hdu['ERR'].data << u.Unit(hdu['SCI'].header['BUNIT'])
        # Construct world coordinate system with only the first two axes
        wcs = WCS(hdu['SCI'].header, naxis=2)
        # Data quality map into mask
        dq = hdu['DQ'].data
        mask = (dq != 0) | ~np.isfinite(intensity) | ~np.isfinite(error)

        # Target info from the header
        name = hdu[0].header['TARGNAME']
        ra = hdu[0].header['TARG_RA'] << u.deg
        dec = hdu[0].header['TARG_DEC'] << u.deg
        inst = hdu[0].header['INSTRUME']
        detector = hdu[0].header['DETECTOR']
        channel = int(hdu[0].header['CHANNEL'])
        band = hdu[0].header['BAND']

        return cls(wave, intensity, error, mask, z, wcs, omega, name, ra, dec, inst, detector, channel, band)
    
    @property
    def _wv_extend(self):
        """
        Convenience function for broadcasting the wavelength vector to the shape of the intensity/error arrays
        """
        return np.broadcast_to(self.wave[:, np.newaxis, np.newaxis], self.intensity.shape)
    
    @_wv_extend.setter
    def _wv_extend(self, val):
        raise ValueError("the _wv_extend property cannot be set.")
    
    @_wv_extend.deleter
    def _wv_extend(self):
        raise ValueError("the _wv_extend property cannot be deleted.")
    
    # Basic data should be read-only
    @property
    def wave(self):
        return self._wave
    
    @wave.setter
    def wave(self, val):
        raise ValueError("this data is read-only; if you need different data, make a new CubeData object")
    
    @wave.deleter
    def wave(self):
        raise ValueError("this data is read-only; if you need different data, make a new CubeData object")

    @property
    def intensity(self):
        return self._intensity
    
    @intensity.setter
    def intensity(self, val):
        raise ValueError("this data is read-only; if you need different data, make a new CubeData object")
    
    @intensity.deleter
    def intensity(self):
        raise ValueError("this data is read-only; if you need different data, make a new CubeData object")

    @property
    def error(self):
        return self._error
    
    @error.setter
    def error(self, val):
        raise ValueError("this data is read-only; if you need different data, make a new CubeData object")
    
    @error.deleter
    def error(self):
        raise ValueError("this data is read-only; if you need different data, make a new CubeData object")
    
    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self, val):
        raise ValueError("this data is read-only; if you need different data, make a new CubeData object")
    
    @mask.deleter
    def mask(self):
        raise ValueError("this data is read-only; if you need different data, make a new CubeData object")
    
    @property
    def omega(self):
        return self._omega
    
    @omega.setter
    def omega(self, val):
        raise ValueError("this data is read-only; if you need different data, make a new CubeData object")
    
    @omega.deleter
    def omega(self):
        raise ValueError("this data is read-only; if you need different data, make a new CubeData object")
    
    @property
    def ra(self):
        return self._ra
    
    @ra.setter
    def ra(self, val):
        raise ValueError("this data is read-only; if you need different data, make a new CubeData object")
    
    @ra.deleter
    def ra(self):
        raise ValueError("this data is read-only; if you need different data, make a new CubeData object")
    
    @property
    def dec(self):
        return self._dec
    
    @dec.setter
    def dec(self, val):
        raise ValueError("this data is read-only; if you need different data, make a new CubeData object")
    
    @dec.deleter
    def dec(self):
        raise ValueError("this data is read-only; if you need different data, make a new CubeData object")
    