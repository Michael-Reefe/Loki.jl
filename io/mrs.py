# Module for JWST Data IO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.visualization import quantity_support

# MATPLOTLIB SETTINGS
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

    __slots__ = ['wave', 'intensity', 'err', 'mask', 'z', 'wcs', 'omega', 'name', 'ra', 'dec', 
        'instrument', 'detector', 'channel', 'band', 'nx', 'ny', 'nz']

    def __init__(self, wave, intensity, err, mask=None, z=None, wcs=None, omega=None, name='Generic Object', ra=None, dec=None,
        instrument=None, detector=None, channel=None, band=None):
        """
        An object for holding 3D IFU spectroscopy data.

        :param wave: array
            1D array of wavelengths, with units
        :param intensity: array
            3D array of intensity, with units
        :param err: array
            3D array of uncertainties, with units
        :param mask: array, optional
            3D array of booleans acting as a mask for the flux/error data
        :param z: float, optional
            the redshift of the source
        :param wcs: WCS, optional
            an astropy World Coordinate System conversion object
        :param omega: float, optional
            the solid angle subtended by each spaxel, with units
        :param name: str, optional
            a label for the source / data
        :param ra: float, optional
            the right ascension of the source, with units
        :param dec: float, optional
            the declination of the source, with units
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
        assert u.get_physical_type(wave) == 'length', "wave must have units of length"
        # Check dimensions
        assert (intensity.ndim == 3) and (intensity.shape[0] == wave.shape[0]), "intensity array's first axis must match wavelength"
        assert (err.ndim == 3) and (err.shape[0] == wave.shape[0]), "err array's first axis must match wavelength"
        assert omega.ndim == 0, "omega must be a scalar"
        # Store dimensions of the cube as parameters
        self.nz, self.ny, self.nx = intensity.shape

        @u.quantity_input(_wave=u.AA, _intensity=u.erg/u.s/u.cm**2/u.AA/u.sr, _err=u.erg/u.s/u.cm**2/u.AA/u.sr, _omega=u.sr/u.pix,
            _ra=u.deg, _dec=u.deg, equivalencies=u.spectral_density(None))
        def __assign_data(self, _wave, _intensity, _err, _omega, _ra=None, _dec=None):
            """
            A sub-function for ensuring that all inputs quantities are units with the right dimensions.
            The spectral_density equivalency has an empty wavelength input since nothing is *actually* being converted (yet),
            astropy just needs to know that wavelength and frequency units *can* be interchanged here.
            """
            # Populate instance attributes
            self.wave = _wave
            self.intensity = _intensity
            self.err = _err
            self.omega = _omega
            self.ra = _ra
            self.dec = _dec

        # Call the nested function to assign all data
        __assign_data(self, wave, intensity, err, omega, ra, dec)

        # Additional non-unit data
        self.name = name
        self.wcs = wcs
        if mask is None:
            mask = np.array([False] * self.intensity.size).reshape(self.intensity.shape)
        self.mask = mask
        self.instrument = instrument
        self.detector = detector
        self.channel = channel
        self.band = band
        self.z = z

        # Apply mask
        self.intensity[mask] = np.nan
        self.err[mask] = np.nan
    
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
        s += f'COORDINATES: ({self.ra.value:.2f},{self.dec.value:.2f})\n'
        s += f'REDSHIFT:     {self.z:.4f}\n'
        s += f'WAVELENGTH:  ({np.nanmin(self.wave.value):.2f} - {np.nanmax(self.wave.value):.2f}) {self.wave.unit}\n'
        s += '########################################################################################\n'
        return s
    
    def convert(self, wave_unit=None, new_flux_unit=None, to_spaxels=False, to_angles=False, inplace=True):
        """
        Convert the units of the wavelength, intensity, and error arrays into a new set, which must be
        physically consistent with the old set of units.

        :param wave_unit: Unit, str
            The new wavelength unit, must be a unit of length
        :param new_flux_unit: Unit, str, optional
            The new FLUX & error unit, must be a unit of specific FLUX (i.e. erg/s/cm2/AA), not accounting
            for the solid angle^-1 or pixel^-1 dependence of the intensity.
        :param to_spaxels: bool
            If true, convert intensity from sr^-1 to pix^-1 by multiplying by the plate scale (sr/pix)
        :param to_angles: str, unit
            If a string or unit, convert intensity from pix^-1 to sr^-1 by dividing by the plate scale (pix/sr); 
            convert the plate scale to the new unit of solid angle, i.e. pix/arcsec2.
        :param inplace: bool
            If true, set the new values to the instance attributes of the object, otherwise just return them.
        
        :return w_out: array
            The converted wavelength quantity
        :return i_out: array
            The converted intensity quantity
        :return e_out: array
            The converted error quantity
        """
        # Convert wavelengths
        w_out = self.wave
        if wave_unit is not None:
            w_out = self.wave.to(wave_unit)
        i_out = self.intensity
        e_out = self.err

        # If converting to solid angles
        if to_angles:
            if "sr" not in str(self.intensity.unit) and "deg2" not in str(self.intensity.unit) \
                and "arcsec2" not in str(self.intensity.unit) and "arcmin2" not in str(self.intensity.unit):
                i_out /= self.omega.to(to_angles / u.pix)
                e_out /= self.omega.to(to_angles / u.pix)

        # Convert the rest of the units with astropy
        if new_flux_unit is not None:
            i_out = i_out.to(new_flux_unit / (u.sr if not to_angles else u.Unit(to_angles)), equivalencies=u.spectral_density(self._wv_extend))
            e_out = e_out.to(new_flux_unit / (u.sr if not to_angles else u.Unit(to_angles)), equivalencies=u.spectral_density(self._wv_extend))

        # If converting to pixels (via plate scales)
        if to_spaxels:
            assert "pix" not in str(self.intensity.unit), "intensity is already per pixel!"
            i_out *= self.omega
            e_out *= self.omega
        
        # Only set new values if inplace argument is true
        if inplace:
            self.wave = w_out
            self.intensity = i_out
            self.err = e_out
        
        return w_out, i_out, e_out

    def plot_2d(self, fname, intensity=True, err=True, log_i=10, log_e=None, colormap=cm.cubehelix, use_wcs=True,
        space='wave'):
        """
        A plotting utility function for 2D maps of the intensity / error

        :param fname: str
            The file name of the plot to be saved.
        :param intensity: bool
            True to plot the intensity.
        :param err: bool
            True to plot the error.
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
        # Sum data along the wavelength axis
        i_plot = np.nansum(self.intensity, axis=0)
        e_plot = np.sqrt(np.nansum(self.err**2, axis=0))
        # Reapply masks
        i_plot[i_plot == 0.] = np.nan
        e_plot[e_plot == 0.] = np.nan
        # Convert to pixel units if plotting w/r/t spaxels
        if not use_wcs and 'pix' not in str(i_plot.unit):
            d = u.sr if 'sr' in str(i_plot.unit) else u.deg**2 if 'deg2' in str(i_plot.unit) else u.arcsec**2 \
                if 'arcsec2' in str(i_plot.unit) else u.arcmin**2 if 'arcmin2' in str(i_plot.unit) else u.sr
            i_plot *= self.omega.to(d / u.pix)
            e_plot *= self.omega.to(d / u.pix)
        if use_wcs and 'pix' in str(i_plot.unit):
            i_plot /= self.omega
            e_plot /= self.omega

        units = i_plot.unit
        # Take the logarithm if specified
        if log_e:
            e_plot = e_plot / np.abs(np.log(log_e) * i_plot)
        if log_i:
            i_plot = np.log(i_plot / i_plot.unit) / np.log(log_i)
        # nu or lambda subscript
        if space in ('freq', 'frequency'):
            sub = r'\nu'
        elif space in ('wave', 'wavelength'):
            sub = r'\lambda'
        else:
            raise ValueError("unrecognized space argument: accepts 'wave', 'wavelength', 'freq', 'frequency'")

        ax1 = ax2 = None
        fig = plt.figure(figsize=(12,6) if (intensity and err) else (12,12))
        if intensity:
            # Plot intensity using WCS projection if specified
            ax1 = fig.add_subplot(121, projection=self.wcs if use_wcs else None)
            ax1.set_title(self.name)
            cdata = ax1.imshow(i_plot, origin='lower', cmap=colormap, 
                vmin=np.nanpercentile(i_plot, 1).value, vmax=np.nanpercentile(i_plot, 99).value)
            fig.colorbar(cdata, ax=ax1, fraction=0.046, pad=0.04,
                label=('' if log_i is None else r'$\log_{%s}$(' % str(log_i)) + r'$I_{%s}\,/\,$' % sub + u.format.LatexInline.to_string(units) + (r')' if log_i else ''))
            ax1.set_xlabel(r'$\alpha$' if use_wcs else r'$x$ (spaxels)')
            ax1.set_ylabel(r'$\delta$' if use_wcs else r'$y$ (spaxels)')
            ax1.tick_params(direction='in')
        if err:
            # Plot err using WCS projection if specified
            ax2 = fig.add_subplot(122, projection=self.wcs if use_wcs else None, sharey=ax1)
            ax2.set_title(self.name)
            cdata = ax2.imshow(e_plot, origin='lower', cmap=colormap,
                vmin=np.nanpercentile(e_plot, 1).value, vmax=np.nanpercentile(e_plot, 99).value)
            fig.colorbar(cdata, ax=ax2, fraction=0.046, pad=0.04,
                label=r'$\sigma_{I_{%s}}\,/\,$' % sub + u.format.LatexInline.to_string(units) if log_e is None else r'$\sigma_{{\rm log_{%s}} I_{%s}}$' % (str(log_e), sub))
            ax2.set_xlabel(r'$\alpha$' if use_wcs else r'$x$ (spaxels)')
            ax2.set_ylabel(r'$\delta$' if use_wcs else r'$y$ (spaxels)')
            # Turn off the left ticks if plotting both
            if intensity:
                ax2.set_ylabel(r' ')
                ax2.tick_params(axis='y', which='both', labelleft=False)
            ax2.tick_params(direction='in')
        if intensity and err:
            fig.subplots_adjust(wspace=0.25)
        
        # Save and close plot
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_1d(self, fname, intensity=True, err=True, log=False, space='wave', spaxel=None, linestyle='-'):
        """
        A plotting utility function for 1D spectra of individual spaxels or the full cube.
        
        :param fname: str
            The file name of the plot to be saved.
        :param intensity: bool
            True to plot the intensity.
        :param err: bool
            True to plot the error.
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
        # Check if in spaxel units before summing
        pix = True
        if 'pix' not in str(self.intensity.unit):
            pix = False

        if spaxel is None:
            # Sum data along the spatial axes
            i_plot = np.nansum(self.intensity, axis=(1, 2))
            e_plot = np.sqrt(np.nansum(self.err**2, axis=(1, 2)))
            # Reapply masks
            i_plot[i_plot == 0.] = np.nan
            e_plot[e_plot == 0.] = np.nan
            # If flux was not per spaxel then we picked up a spaxel/sr term that must be canceled
            if not pix:
                # Count up the number of GOOD pixels used in the summation for each wavelength bin
                i_plot /= np.nansum(~self.mask, axis=(1, 2))
                e_plot /= np.nansum(~self.mask, axis=(1, 2))
                units = i_plot.unit
            else:
                units = i_plot.unit * u.pix
        else:
            # Get data for the single spaxel to plot
            i_plot = self.intensity[:, spaxel[0], spaxel[1]]
            e_plot = self.err[:, spaxel[0], spaxel[1]]
            units = i_plot.unit

        # Take the logarithm if specified
        if space in ('wave', 'wavelength'):
            sub = r'\lambda'
            xval = self.wave
            if log:
                units = units * self.wave.unit
                e_plot = e_plot / np.abs(i_plot * np.log(log))
                i_plot = np.log(self.wave / self.wave.unit * i_plot / i_plot.unit) / np.log(log)
        elif space in ('freq', 'frequency'):
            sub = r'\nu'
            xval = self.wave.to(u.GHz, equivalencies=u.spectral())
            if log:
                units = units * u.Hz
                e_plot = e_plot / np.abs(i_plot * np.log(log))
                i_plot = np.log(self.wave.to('Hz', equivalencies=u.spectral()) / u.Hz * i_plot / i_plot.unit) / np.log(log)
        
        # Plot formatting
        fig, ax = plt.subplots(figsize=(10,5))
        if intensity:
            ax.plot(xval, i_plot, 'k', linestyle=linestyle, label='Data')
        if err and not intensity:
            ax.plot(xval, e_plot, 'k', linestyle=linestyle, label='$1\\sigma$ Error')
        if err and intensity:
            ax.fill_between(xval, i_plot-e_plot, i_plot+e_plot, color='k', alpha=0.5, label='$1\\sigma$ Error')
        ax.set_xlabel(r'$%s$' % sub + ' (' + u.format.LatexInline.to_string(xval.unit) + ')')
        if not log:
            ax.set_ylabel(r'$F_{%s}$' % sub + ' (' + u.format.LatexInline.to_string(units) + ')')
        else:
            ax.set_ylabel(r'$\log_{%s}{%s}F_{%s}$' % (log, sub, sub) + ' (' + u.format.LatexInline.to_string(units) + ')')
        ax.legend(loc='upper right', frameon=False)
        ax.set_xlim(np.nanmin(xval), np.nanmax(xval))
        ax.set_title(self.name + (f'' if spaxel is None else f' Spaxel({spaxel[1]},{spaxel[0]})'))
        ax.tick_params(direction='in')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    
    # Framework for extending the wavelength array into 3D for usage with astropy unit conversions
    @property
    def _wv_extend(self):
        return np.dstack([np.vstack([self.wave] * self.ny).T] * self.nx)
    
    @_wv_extend.setter
    def _wv_extend(self, value):
        raise ValueError("The _wv_extend property may not be set.")

    @_wv_extend.deleter
    def _wv_extend(self):
        raise ValueError("The _wv_extend property may not be deleted.")
    
    @classmethod
    def from_fits(cls, filepath, z=0):
        """
        A method for reading IFU data from JWST 's3d' fits files

        :param filepath: str
            the path to the 's3d' JWST FITS file
        :param z: float, optional
            the redshift of the source
        
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
        err = hdu['ERR'].data << u.Unit(hdu['SCI'].header['BUNIT'])
        # Construct world coordinate system with only the first two axes
        wcs = WCS(hdu['SCI'].header, naxis=2)
        # Data quality map into mask
        dq = hdu['DQ'].data
        mask = (dq != 0) | ~np.isfinite(intensity) | ~np.isfinite(err)

        # Target info from the header
        name = hdu[0].header['TARGNAME']
        ra = hdu[0].header['TARG_RA'] << u.deg
        dec = hdu[0].header['TARG_DEC'] << u.deg
        inst = hdu[0].header['INSTRUME']
        detector = hdu[0].header['DETECTOR']
        channel = int(hdu[0].header['CHANNEL'])
        band = hdu[0].header['BAND']

        return cls(wave, intensity, err, mask, z, wcs, omega, name, ra, dec, inst, detector, channel, band)

