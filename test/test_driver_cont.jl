using DataStructures
using TOML

options = OrderedDict()

options["stellar_continuum_temp"] = OrderedDict("val" => 5000., "prior" => "Uniform", "pval" => [0.,1.], "locked" => true)
options["dust_continuum_temps"] = [
    OrderedDict("val" => 300., "prior" => "Uniform", "pval" => [0.,1.], "locked" => true), 
    OrderedDict("val" => 200., "prior" => "Uniform", "pval" => [0.,1.], "locked" => true), 
    OrderedDict("val" => 135., "prior" => "Uniform", "pval" => [0.,1.], "locked" => true), 
    OrderedDict("val" => 90., "prior" => "Uniform", "pval" => [0.,1.], "locked" => true), 
    OrderedDict("val" => 65., "prior" => "Uniform", "pval" => [0.,1.], "locked" => true), 
    OrderedDict("val" => 50., "prior" => "Uniform", "pval" => [0.,1.], "locked" => true), 
    OrderedDict("val" => 40., "prior" => "Uniform", "pval" => [0.,1.], "locked" => true), 
    OrderedDict("val" => 35., "prior" => "Uniform", "pval" => [0.,1.], "locked" => true)
]

dust_features = OrderedDict(
    "PAH_5.24"  => OrderedDict(
        "wave" => OrderedDict("val" => 5.24, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.058, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_5.27"  => OrderedDict(
        "wave" => OrderedDict("val" => 5.27, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.179, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_5.70"  => OrderedDict(
        "wave" => OrderedDict("val" => 5.70, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.200, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_5.87"  => OrderedDict(
        "wave" => OrderedDict("val" => 5.87, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.200, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_6.00"  => OrderedDict(
        "wave" => OrderedDict("val" => 6.00, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.200, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_6.18"  => OrderedDict(
        "wave" => OrderedDict("val" => 6.18, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.100, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_6.30"  => OrderedDict(
        "wave" => OrderedDict("val" => 6.30, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.187, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_6.69"  => OrderedDict(
        "wave" => OrderedDict("val" => 6.69, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.468, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_7.42"  => OrderedDict(
        "wave" => OrderedDict("val" => 7.42, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.935, "prior" => "Uniform", "pval" => [0.4, 1.1], "locked" => false)
    ),
    "PAH_7.52"  => OrderedDict(
        "wave" => OrderedDict("val" => 7.52, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.240, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_7.62"  => OrderedDict(
        "wave" => OrderedDict("val" => 7.62, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.240, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_7.85"  => OrderedDict(
        "wave" => OrderedDict("val" => 7.85, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.416, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_8.33"  => OrderedDict(
        "wave" => OrderedDict("val" => 8.33, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.417, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_8.61"  => OrderedDict(
        "wave" => OrderedDict("val" => 8.61, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.336, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_10.68"  => OrderedDict(
        "wave" => OrderedDict("val" => 10.68, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.214, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_11.00"  => OrderedDict(
        "wave" => OrderedDict("val" => 11.00, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.100, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_11.15"  => OrderedDict(
        "wave" => OrderedDict("val" => 11.15, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.030, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_11.20"  => OrderedDict(
        "wave" => OrderedDict("val" => 11.20, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.030, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_11.22"  => OrderedDict(
        "wave" => OrderedDict("val" => 11.22, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.100, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_11.25"  => OrderedDict(
        "wave" => OrderedDict("val" => 11.25, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.135, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_11.33"  => OrderedDict(
        "wave" => OrderedDict("val" => 11.33, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.363, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_11.99"  => OrderedDict(
        "wave" => OrderedDict("val" => 11.99, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.540, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_12.62"  => OrderedDict(
        "wave" => OrderedDict("val" => 12.62, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.530, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_12.69"  => OrderedDict(
        "wave" => OrderedDict("val" => 12.69, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.120, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_13.48"  => OrderedDict(
        "wave" => OrderedDict("val" => 13.48, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.539, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_14.04"  => OrderedDict(
        "wave" => OrderedDict("val" => 14.04, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.225, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_14.19"  => OrderedDict(
        "wave" => OrderedDict("val" => 14.19, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.355, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_14.65"  => OrderedDict(
        "wave" => OrderedDict("val" => 14.65, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.500, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_15.90"  => OrderedDict(
        "wave" => OrderedDict("val" => 15.90, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.318, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_16.45"  => OrderedDict(
        "wave" => OrderedDict("val" => 16.45, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.230, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_17.04"  => OrderedDict(
        "wave" => OrderedDict("val" => 17.04, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 1.108, "prior" => "Uniform", "pval" => [0.4, 1.1], "locked" => false)
    ),
    "PAH_17.375"  => OrderedDict(
        "wave" => OrderedDict("val" => 17.375, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.209, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_17.87"  => OrderedDict(
        "wave" => OrderedDict("val" => 17.87, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.286, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_18.92"  => OrderedDict(
        "wave" => OrderedDict("val" => 18.92, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 0.359, "prior" => "Uniform", "pval" => [0.4, 1.4], "locked" => false)
    ),
    "PAH_33.10"  => OrderedDict(
        "wave" => OrderedDict("val" => 33.10, "prior" => "Uniform", "pval" => [-0.05,0.05], "locked" => false), 
        "fwhm" => OrderedDict("val" => 1.655, "prior" => "Uniform", "pval" => [0.4, 1.1], "locked" => false)
    )
)

options["dust_features"] = dust_features

extinction = OrderedDict(
    "tau_9_7" => OrderedDict("val" => 0.5, "prior" => "Uniform", "pval" => [0., 2.], "locked" => false),
    "beta" => OrderedDict("val" => 0.1, "prior" => "Uniform", "pval" => [0., 1.], "locked" => true)
)
options["extinction"] = extinction

open("options.toml", "w") do f
    TOML.print(f, options)
end

lines = OrderedDict(
    "SI_69865" => OrderedDict(
        "wave" => 6.9865, 
        "profile" => "Gaussian", 
        "voff" => OrderedDict("val" => 0., "prior" => "Uniform", "pval" => [-500., 500.], "locked" => false),
        "fwhm" => OrderedDict("val" => 100., "prior" => "Uniform", "pval" => [10., 1000.], "locked" => false)
        )
    )

open("lines.toml", "w") do f
    TOML.print(f, lines)
end
