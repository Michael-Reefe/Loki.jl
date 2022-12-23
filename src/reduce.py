"""
This python script has been adapted from a Jupyter Notebook provided by the JWST 
team for reducing uncalibrated MIRI MRS data.  Original documentation is pasted below:
--------------------------------------------------------------------------------------------------------------

MIRI MRS Batch Processing Notebook
Author: David Law, AURA Associate Astronomer, MIRI branch


**Last Updated**: October 14, 2022
**Pipeline Version**: 1.8.0+
The purpose of this notebook is to provide a framework for batch processing of MIRI MRS data 
through all three pipeline stages. Data is assumed to be located in two Observation folders 
(science and background) according to paths set up below.

This example is set up to use observations from APT program 2732 (Stephan's Quintet ERO observations) 
using a standard 4-pt dither with dedicated background in all three grating settings. Input data for 
this notebook can be obtained by downloading the 'uncal' files from MAST and placing them in directories 
set up below.

Changes:
Sep 1 2022: Add some commentary and example on how to use multicore processing in Detector 1
Sep 12 2022: Disable unnecessary cube/1d spectra production for individual science exposures in Spec 2
Oct 14 2022: Include residual fringe correction in spec2 (note that this will CRASH earlier pipeline versions!

--------------------------------------------------------------------------------------------------------------
"""
# Basic system utilities for interacting with files
import sys
import os
import pdb
import glob
import time
import shutil
import warnings
import zipfile
import urllib.request

# Astropy utilities for opening FITS and ASCII files
from astropy.io import fits
from astropy.io import ascii
from astropy.utils.data import download_file
# Astropy utilities for making plots
from astropy.visualization import (LinearStretch, LogStretch, ImageNormalize, ZScaleInterval)

# Numpy for doing calculations
import numpy as np

# Matplotlib for making plots
import matplotlib.pyplot as plt
from matplotlib import rc

# Import the base JWST package
import jwst

# JWST pipelines (encompassing many steps)
from jwst.pipeline import Detector1Pipeline
from jwst.pipeline import Spec2Pipeline
from jwst.pipeline import Spec3Pipeline

# JWST pipeline utilities
from jwst import datamodels # JWST datamodels
from jwst.associations import asn_from_list as afl # Tools for creating association files
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase # Definition of a Lvl2 association file
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base # Definition of a Lvl3 association file

from stcal import dqflags # Utilities for working with the data quality (DQ) arrays


# Configure the script to take command-line inputs for the directories that will be reduced

# CRDS context (if overriding)
#%env CRDS_CONTEXT jwst_0771.pmap

# Point to where the uncalibrated FITS files are from the science observation
input_dir = sys.argv[1]
# Point to where you want the output science results to go
output_dir = sys.argv[2]
# Point to where the uncalibrated FITS files are from the background observation
# If no background observation, leave this blank
input_bgdir = sys.argv[3]
# Point to where the output background observations should go
# If no background observation, leave this blank
output_bgdir = sys.argv[4]

# Whether or not to run a given pipeline stage
# Science and background are processed independently through det1+spec2, and jointly in spec3

# Science processing
dodet1 = True
dospec2 = True
dospec3 = True

# Background processing
dodet1bg = True
dospec2bg = True

# If there is no background folder, ensure we don't try to process it
if (input_bgdir == ''):
    dodet1bg = False
    dospec2bg = False

# Output subdirectories to keep science data products organized
# Note that the pipeline might complain about this as it is intended to work with everything in a single
# directory, but it nonetheless works fine for the examples given here.
det1_dir = os.path.join(output_dir, 'stage1/') # Detector1 pipeline outputs will go here
spec2_dir = os.path.join(output_dir, 'stage2/') # Spec2 pipeline outputs will go here
spec3_dir = os.path.join(output_dir, 'stage3/') # Spec3 pipeline outputs will go here

# We need to check that the desired output directories exist, and if not create them
if not os.path.exists(det1_dir):
    os.makedirs(det1_dir)
if not os.path.exists(spec2_dir):
    os.makedirs(spec2_dir)
if not os.path.exists(spec3_dir):
    os.makedirs(spec3_dir)

# Output subdirectories to keep background data products organized
det1_bgdir = os.path.join(output_bgdir, 'stage1/') # Detector1 pipeline outputs will go here
spec2_bgdir = os.path.join(output_bgdir, 'stage2/') # Spec2 pipeline outputs will go here

# We need to check that the desired output directories exist, and if not create them
if (output_bgdir != ''):
    if not os.path.exists(det1_bgdir):
        os.makedirs(det1_bgdir)
    if not os.path.exists(spec2_bgdir):
        os.makedirs(spec2_bgdir)

# Start a timer to keep track of runtime
time0 = time.perf_counter()


# First we'll define a function that will call the detector1 pipeline with our desired set of parameters
# We won't enumerate the individual steps
def rundet1(filename, outdir):
    print(filename)
    det1 = Detector1Pipeline() # Instantiate the pipeline
    det1.output_dir = outdir # Specify where the output should go
    
    # Overrides for whether or not certain steps should be skipped
    #det1.dq_init.skip = False
    #det1.saturation.skip = False
    #det1.firstframe.skip = False
    #det1.lastframe.skip = False
    #det1.reset.skip = False
    #det1.linearity.skip = False
    #det1.rscd.skip = False
    #det1.dark_current.skip = False
    #det1.refpix.skip = False
    #det1.jump.skip = False
    #det1.ramp_fit.skip = False
    #det1.gain_scale.skip = False
    
    # The jump and ramp fitting steps can benefit from multi-core processing, but this is off by default
    # Turn them on here if desired by choosing how many cores to use (quarter, half, or all)
    # det1.jump.maximum_cores='half'
    # det1.ramp_fit.maximum_cores='half'
    
    # Bad pixel mask overrides
    #det1.dq_init.override_mask = 'myfile.fits'

    # Saturation overrides
    #et1.saturation.override_saturation = 'myfile.fits'
    
    # Reset overrides
    #det1.reset.override_reset = 'myfile.fits'
        
    # Linearity overrides
    #det1.linearity.override_linearity = 'myfile.fits'

    # RSCD overrides
    #det1.rscd.override_rscd = 'myfile.fits'
        
    # DARK overrides
    #det1.dark_current.override_dark = 'myfile.fits'
        
    # GAIN overrides
    #det1.jump.override_gain = 'myfile.fits'
    #det1.ramp_fit.override_gain = 'myfile.fits'
                
    # READNOISE overrides
    #det1.jump.override_readnoise = 'myfile.fits'
    #det1.ramp_fit.override_readnoise = 'myfile.fits'
        
    det1.save_results = True # Save the final resulting _rate.fits files
    det1(filename) # Run the pipeline on an input list of files


# Now let's look for input files of the form *uncal.fits from the science observation
sstring = input_dir + 'jw*mirifu*uncal.fits'
lvl1b_files = sorted(glob.glob(sstring))
print('Found ' + str(len(lvl1b_files)) + ' science input files to process')

# Run the pipeline on these input files by a simple loop over our pipeline function
if dodet1:
    for file in lvl1b_files:
        rundet1(file, det1_dir)
else:
    print('Skipping Detector1 processing')

# Now let's look for input files of the form *uncal.fits from the background observation
sstring = input_bgdir + 'jw*mirifu*uncal.fits'
lvl1b_files = sorted(glob.glob(sstring))
print('Found ' + str(len(lvl1b_files)) + ' background input files to process')

# Run the pipeline on these input files by a simple loop over our pipeline function
if dodet1bg:
    for file in lvl1b_files:
        rundet1(file, det1_bgdir)
else:
    print('Skipping Detector1 processing')

# Print out the time benchmark
time1 = time.perf_counter()
print(f"Runtime so far: {time1 - time0:0.4f} seconds")


# Define a function that will call the spec2 pipeline with our desired set of parameters
# We'll list the individual steps just to make it clear what's running
def runspec2(filename, outdir, nocubes=False):
    spec2 = Spec2Pipeline()
    spec2.output_dir = outdir

    # Assign_wcs overrides
    #spec2.assign_wcs.override_distortion = 'myfile.asdf'
    #spec2.assign_wcs.override_regions = 'myfile.asdf'
    #spec2.assign_wcs.override_specwcs = 'myfile.asdf'
    #spec2.assign_wcs.override_wavelengthrange = 'myfile.asdf'

    # Flatfield overrides
    #spec2.flat.override_flat = 'myfile.fits'
        
    # Straylight overrides
    #spec2.straylight.override_mrsxartcorr = 'myfile.fits'
        
    # Fringe overrides
    #spec2.fringe.override_fringe = 'myfile.fits'
    
    # Photom overrides
    #spec2.photom.override_photom = 'myfile.fits'

    # Cubepar overrides
    #spec2.cube_build.override_cubepar = 'myfile.fits'
        
    # Extract1D overrides
    #spec2.extract1d.override_extract1d = 'myfile.asdf'
    #spec2.extract1d.override_apcorr = 'myfile.asdf'
        
    # Overrides for whether or not certain steps should be skipped
    #spec2.assign_wcs.skip = False
    #spec2.bkg_subtract.skip = True
    #spec2.flat_field.skip = False
    #spec2.srctype.skip = False
    #spec2.straylight.skip = False
    #spec2.fringe.skip = False
    #spec2.photom.skip = False
    spec2.residual_fringe.skip = False
    #spec2.cube_build.skip = False
    #spec2.extract_1d.skip = False
    
    # This nocubes option allows us to skip the cube building and 1d spectral extraction for individual
    # science data frames, but run it for the background data (as the 1d spectra are needed later
    # for the master background step in Spec3)
    if (nocubes):
        spec2.cube_build.skip = True
        spec2.extract_1d.skip = True
    
    # Some cube building options
    #spec2.cube_build.weighting='drizzle'
    #spec2.cube_build.coord_system='ifualign' # If aligning cubes with IFU axes instead of sky
      
    spec2.save_results = True
    spec2(filename)


# Look for uncalibrated science slope files from the Detector1 pipeline
sstring = det1_dir + 'jw*mirifu*rate.fits'
ratefiles = sorted(glob.glob(sstring))
ratefiles=np.array(ratefiles)
print('Found ' + str(len(ratefiles)) + ' input files to process')

if dospec2:
    for file in ratefiles:
        runspec2(file, spec2_dir, nocubes=True)
else:
    print('Skipping Spec2 processing')

# Look for uncalibrated background slope files from the Detector1 pipeline
sstring = det1_bgdir + 'jw*mirifu*rate.fits'
ratefiles = sorted(glob.glob(sstring))
ratefiles=np.array(ratefiles)
print('Found ' + str(len(ratefiles)) + ' input files to process')

if dospec2bg:
    for file in ratefiles:
        runspec2(file, spec2_bgdir)
else:
    print('Skipping Spec2 processing')

# Print out the time benchmark
time1 = time.perf_counter()
print(f"Runtime so far: {time1 - time0:0.4f} seconds")


# Define a useful function to write out a Lvl3 association file from an input list
# Note that any background exposures have to be of type x1d.
def writel3asn(scifiles, bgfiles, asnfile, prodname):
    # Define the basic association of science files
    asn = afl.asn_from_list(scifiles, rule=DMS_Level3_Base, product_name=prodname)
        
    # Add background files to the association
    nbg=len(bgfiles)
    for ii in range(0,nbg):
        asn['products'][0]['members'].append({'expname': bgfiles[ii], 'exptype': 'background'})
        
    # Write the association to a json file
    _, serialized = asn.dump()
    with open(asnfile, 'w') as outfile:
        outfile.write(serialized)


# Find and sort all of the input files

# Science Files need the cal.fits files
sstring = spec2_dir + 'jw*mirifu*cal.fits'
calfiles = np.array(sorted(glob.glob(sstring)))

# Background Files need the x1d.fits files
sstring = spec2_bgdir + 'jw*mirifu*x1d.fits'
bgfiles = np.array(sorted(glob.glob(sstring)))

print('Found ' + str(len(calfiles)) + ' science files to process')
print('Found ' + str(len(bgfiles)) + ' background files to process')

# Make an association file that includes all of the different exposures
asnfile=os.path.join(output_dir, 'l3asn.json')
if dospec3:
    writel3asn(calfiles, bgfiles, asnfile, 'Level3')


# Define a function that will call the spec3 pipeline with our desired set of parameters
# This is designed to run on an association file
def runspec3(filename):
    # This initial setup is just to make sure that we get the latest parameter reference files
    # pulled in for our files.  This is a temporary workaround to get around an issue with
    # how this pipeline calling method works.
    crds_config = Spec3Pipeline.get_config_from_reference(filename)
    spec3 = Spec3Pipeline.from_config_section(crds_config)
    
    spec3.output_dir = spec3_dir
    spec3.save_results = True
    
    # Cube building configuration options
    # spec3.cube_build.output_file = 'bandcube' # Custom output name
    # spec3.cube_build.output_type = 'band' # 'band', 'channel', or 'multi' type cube output
    # spec3.cube_build.channel = '1' # Build everything from just channel 1 into a single cube (we could also choose '2','3','4', or 'ALL')
    # spec3.cube_build.weighting = 'drizzle' # 'emsm' or 'drizzle'
    # spec3.cube_build.coord_system = 'ifualign' # 'ifualign', 'skyalign', or 'internal_cal'
    # spec3.cube_build.scale1 = 0.5 # Output cube spaxel scale (arcsec) in dimension 1 if setting it by hand
    # spec3.cube_build.scale2 = 0.5 # Output cube spaxel scale (arcsec) in dimension 2 if setting it by hand
    # spec3.cube_build.scalew = 0.002 # Output cube spaxel size (microns) in dimension 3 if setting it by hand
    
    # Overrides for whether or not certain steps should be skipped
    #spec3.assign_mtwcs.skip = False
    #spec3.master_background.skip = True
    #spec3.outlier_detection.skip = True
    #spec3.mrs_imatch.skip = True
    #spec3.cube_build.skip = False
    #spec3.extract_1d.skip = False
    
    # Cubepar overrides
    #spec3.cube_build.override_cubepar = 'myfile.fits'
        
    # Extract1D overrides
    #spec3.extract1d.override_extract1d = 'myfile.asdf'
    #spec3.extract1d.override_apcorr = 'myfile.asdf'

    spec3(filename)


if dospec3:
    runspec3(asnfile)
else:
    print('Skipping Spec3 processing')


# Print out the time benchmark
time1 = time.perf_counter()
print(f"Total runtime: {time1 - time0:0.4f} seconds")
