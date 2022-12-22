#= 
THE REDUCE MODULE
-----------------

This lightweight / supplementary module is separated from the rest of the code and
is essentially just a wrapper around the Python JWST pipeline for reducing raw data,
with the addition of the residual fringe step.  It is included purely for convenience
to allow running all steps of data analysis in one clean pipeline.
=#

module Reduce

export run_jwst_pipeline

"""
    run_jwst_pipeline(data_input, data_output, bkg_input, bkg_output)

Run the JWST python reduction pipeline on a given set of uncalibrated input data and background files.

# Arguments
- `data_input::String`: The input folder containing the uncalibrated science data
- `data_output::String`: The destination folder that will contain the calibrated results
- `bkg_input::String`: The input folder containing the uncalibrated background data
- `bkg_output::String`: The destination folder that will contain the calibrated background data 
"""
function run_jwst_pipeline(data_input::String, data_output::String, bkg_input::String, bkg_output::String)
    cmd = `python $(@__DIR__)/reduce.py $data_input $data_output $bkg_input $bkg_output`
    run(cmd)
end

end