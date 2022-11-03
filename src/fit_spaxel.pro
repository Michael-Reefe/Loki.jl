name = COMMAND_LINE_ARGS()

data = READ_CSV(name)
dust_features=[{WAVELENGTH: 5.24D,  FRAC_FWHM: 0.011D}, $
                    {WAVELENGTH: 5.27D,  FRAC_FWHM: 0.034D}, $
                    {WAVELENGTH: 5.70D,  FRAC_FWHM: 0.035D}, $
                    {WAVELENGTH: 5.87D,  FRAC_FWHM: 0.034D}, $
                    {WAVELENGTH: 6.00D,  FRAC_FWHM: 0.033D}, $
                    {WAVELENGTH: 6.18D,  FRAC_FWHM: 0.016D}, $
;;                  {WAVELENGTH: 6.22D,  FRAC_FWHM: 0.030D}, $
                    {WAVELENGTH: 6.30D,  FRAC_FWHM: 0.030D}, $
                    {WAVELENGTH: 6.69D,  FRAC_FWHM: 0.07D},  $
                    {WAVELENGTH: 7.42D,  FRAC_FWHM: 0.126D}, $
;;                  {WAVELENGTH: 7.60D,  FRAC_FWHM: 0.044D}, $
                    {WAVELENGTH: 7.52D,  FRAC_FWHM: 0.030D}, $  ;; MY ADDITION -> TURN THE 7.60um FEATURE INTO TWO DRUDE PROFILES
                    {WAVELENGTH: 7.62D,  FRAC_FWHM: 0.020D}, $  ;;                AT 7.52 AND 7.62 um!!
                    {WAVELENGTH: 7.85D,  FRAC_FWHM: 0.053D}, $
                    {WAVELENGTH: 8.33D,  FRAC_FWHM: 0.05D},  $
                    {WAVELENGTH: 8.61D,  FRAC_FWHM: 0.039D}, $
                    {WAVELENGTH: 10.68D, FRAC_FWHM: 0.02D},  $
                    {WAVELENGTH: 11.00D, FRAC_FWHM: 0.009D}, $
                    {WAVELENGTH: 11.15D, FRAC_FWHM: 0.003D}, $
                    {WAVELENGTH: 11.20D, FRAC_FWHM: 0.003D}, $
                    {WAVELENGTH: 11.22D, FRAC_FWHM: 0.009D}, $
;;                  {WAVELENGTH: 11.23D, FRAC_FWHM: 0.012D}, $
                    {WAVELENGTH: 11.25D, FRAC_FWHM: 0.012D}, $
                    {WAVELENGTH: 11.33D, FRAC_FWHM: 0.032D}, $
                    {WAVELENGTH: 11.99D, FRAC_FWHM: 0.045D}, $
                    {WAVELENGTH: 12.62D, FRAC_FWHM: 0.042D}, $
                    {WAVELENGTH: 12.69D, FRAC_FWHM: 0.013D}, $
                    {WAVELENGTH: 13.48D, FRAC_FWHM: 0.04D},  $
                    {WAVELENGTH: 14.04D, FRAC_FWHM: 0.016D}, $
                    {WAVELENGTH: 14.19D, FRAC_FWHM: 0.025D}, $
                    {WAVELENGTH: 14.65D, FRAC_FWHM: 0.034D}, $
                    {WAVELENGTH: 15.9D,  FRAC_FWHM: 0.02D},  $
                    {WAVELENGTH: 16.45D, FRAC_FWHM: 0.014D}, $
                    {WAVELENGTH: 17.04D, FRAC_FWHM: 0.065D}, $
                    {WAVELENGTH: 17.375D,FRAC_FWHM: 0.012D}, $
                    {WAVELENGTH: 17.87D, FRAC_FWHM: 0.016D}, $
                    {WAVELENGTH: 18.92D, FRAC_FWHM: 0.019D}, $
                   {WAVELENGTH: 33.1D,  FRAC_FWHM: 0.05D}]

n = size(dust_features, /n_elements)
df_out = []
for i=0,n-1 do if (dust_features[i].WAVELENGTH ge min(data.field1)-.5) && (dust_features[i].WAVELENGTH le max(data.field1)+.5) then df_out = [df_out, dust_features[i]]
lines = [{WAVELENGTH: 5.0D, NAME: "1"}, {WAVELENGTH: 5.1D, NAME: "2"}]

fit = pahfit(data.field1, data.field2, data.field3, REDSHIFT=0., /REPORT, /PLOT_PROGRESS, XSIZE=1000, YSIZE=600, DUST_FEATURES=df_out)

t_stellar = fit.STARLIGHT.TEMPERATURE[0] 
t_stellar_unc = fit.STARLIGHT.TEMPERATURE_UNC[0]
a_stellar = fit.STARLIGHT.TAU[0]
a_stellar_unc = fit.STARLIGHT.TAU_UNC[0]

t_dc = fit.DUST_CONTINUUM.TEMPERATURE[0]
t_dc_unc = fit.DUST_CONTINUUM.TEMPERATURE_UNC[0]
a_dc = fit.DUST_CONTINUUM.TAU[0]
a_dc_unc = fit.DUST_CONTINUUM.TAU_UNC[0]

amp_df = fit.DUST_FEATURES.CENTRAL_INTEN[0]
amp_df_unc = fit.DUST_FEATURES.CENTRAL_INTEN_UNC[0]
mean_df = fit.DUST_FEATURES.WAVELENGTH[0]
mean_df_unc = fit.DUST_FEATURES.WAVELENGTH_UNC[0]
fwhm_df = fit.DUST_FEATURES.FWHM[0] * fit.DUST_FEATURES.WAVELENGTH[0]
fwhm_df_unc = sqrt(fit.DUST_FEATURES.FWHM_UNC[0]^2 + fit.DUST_FEATURES.WAVELENGTH_UNC[0]^2)

tau_97 = fit.EXTINCTION.TAU_9_7[0]
tau_97_unc = fit.EXTINCTION.TAU_9_7_UNC[0]
beta = fit.EXTINCTION.BETA[0]
beta_unc = fit.EXTINCTION.BETA_UNC[0]

params = [a_stellar, t_stellar]
uncs = [a_stellar_unc, t_stellar_unc]
n_dc = size(a_dc, /n_elements)
n_df = size(amp_df, /n_elements)

for i=0,n_dc-1 do params = [params, a_dc[i], t_dc[i]]
for i=0,n_dc-1 do uncs = [uncs, a_dc_unc[i], t_dc_unc[i]]

for j=0,n_df-1 do params = [params, amp_df[j], mean_df[j], fwhm_df[j]]
for j=0,n_df-1 do uncs = [uncs, amp_df_unc[j], mean_df_unc[j], fwhm_df_unc[j]]

params = [params, tau_97, beta]
uncs = [uncs, tau_97_unc, beta_unc]

out = REPSTR(name, ".csv", "_params.csv")
WRITE_CSV, out, params, uncs, header=["params", "uncs"]

I_cont = fit.FINAL_FIT

out2 = REPSTR(name, ".csv", "_fit.csv")
WRITE_CSV, out2, data.field1, I_cont, header=["wave", "intensity"]




