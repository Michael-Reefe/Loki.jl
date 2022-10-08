module CubeFit

export pahfit_spaxel

# Import packages
using PyCall

astropy_units = pyimport("astropy.units")
helpers = pyimport("pahfit.helpers")

include("utils.jl")
using .Utils

function pahfit_spaxel(λ::Vector{Float64}, F::Vector{Float64}, σ::Vector{Float64}, label::String; nostart::Bool=false,
    maxiter=10_000)

    # Add astropy units
    λ_q = λ << astropy_units.Angstrom
    F_q = F << astropy_units.Unit("erg / (s cm2 AA)")
    σ_q = σ << astropy_units.Unit("erg / (s cm2 AA)")

    # Create the data object
    pah_obs = Dict(
        "x" => λ_q.to("um") << astropy_units.um,
        "y" => F_q.to("Jy", equivalencies=astropy_units.spectral_density(λ_q)) << astropy_units.Jy,
        "unc" => σ_q.to("Jy", equivalencies=astropy_units.spectral_density(λ_q)) << astropy_units.Jy
        )
    
    # Create the model
    pah_model = helpers.initialize_model("scipack_ExGal_SpitzerIRSSLLL.ipac", pah_obs, !nostart)

    # Fit the spectrum with LevMarLSQFitter
    pah_fit = helpers.fit_spectrum(pah_obs, pah_model, maxiter=maxiter)

    pah_model.save(pah_fit, label, "ipac")
    pah_model = helpers.initialize_model("$(label)_output.ipac", pah_obs, false)

    compounds = helpers.calculate_compounds(pah_obs, pah_model)
    cont_flux_Jy = (compounds["tot_cont"] .+ compounds["dust_features"]) .* compounds["extinction_model"]

    F_cont = cont_flux_Jy .* 1e-23 .* (C_MS .* 1e10) ./ (pah_obs["x"].value .* 1e4).^2

    return pah_model, F_cont
end

end