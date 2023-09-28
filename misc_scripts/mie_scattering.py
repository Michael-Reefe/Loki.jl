import numpy as np
import os, sys
import miepython

def read_dorschner_nk():
    data = np.loadtxt(os.path.join('..', 'src', 'templates', 'dorschner_olivine_pyroxene_nk.txt'), delimiter=' ')
    wave = data[:,0]
    pyr_dict = {}
    pyr_dict[100] = data[:,1] - 1j*data[:,2]
    pyr_dict[95] = data[:,3] - 1j*data[:,4]
    pyr_dict[80] = data[:,5] - 1j*data[:,6]
    pyr_dict[70] = data[:,7] - 1j*data[:,8]
    pyr_dict[60] = data[:,9] - 1j*data[:,10]
    pyr_dict[50] = data[:,11] - 1j*data[:,12]
    pyr_dict[40] = data[:,13] - 1j*data[:,14]
    oli_dict = {}
    oli_dict[50] = data[:,15] - 1j*data[:,16]
    oli_dict[40] = data[:,17] - 1j*data[:,18]
    return wave, pyr_dict, oli_dict

def calculate_miecoeff(wave, n, a=None):
    # Particle radius array
    if a is None:
        a = np.geomspace(1e-3, 10, 81)
    if type(a) in (int, float):
        a = np.array([a])
    qabs = np.zeros((len(wave), len(a)))
    qsca = np.zeros((len(wave), len(a)))
    qback = np.zeros((len(wave), len(a)))
    g = np.zeros((len(wave), len(a)))
    for (i,ai) in enumerate(a):
        # Size parameter
        x = 2*np.pi*ai / wave
        # Extinction and scattering efficiencies
        qext_i, qsca_i, qback_i, g_i = miepython.mie(n, x)
        qsca[:,i] = qsca_i
        qback[:,i] = qback_i
        g[:,i] = g_i
        # Absorption efficiency
        qabs[:,i] = qext_i - qsca_i

    return qabs, qsca, qback, g


if __name__ == '__main__':

    _all = sys.argv[1]

    if _all == 'all':
        wave, pyr_dict, oli_dict = read_dorschner_nk()
        a = np.geomspace(1e-3, 10, 81)
        pyr_qabs = np.zeros((len(wave), len(a), len(pyr_dict.keys())))
        pyr_qsca = np.copy(pyr_qabs)
        oli_qabs = np.zeros((len(wave), len(a), len(oli_dict.keys())))
        oli_qsca = np.copy(oli_qabs)

        for (i, key) in enumerate(pyr_dict.keys()):
            pyr_qabs[:, :, i], pyr_qsca[:, :, i], _, _ = calculate_miecoeff(wave, pyr_dict[key])
        for (j, key) in enumerate(oli_dict.keys()):
            oli_qabs[:, :, j], oli_qsca[:, :, j], _, _ = calculate_miecoeff(wave, oli_dict[key])

        x = np.array(list(pyr_dict.keys()))
        y = np.array(list(oli_dict.keys()))

        np.savetxt(os.path.join('..', 'src', 'templates', 'dorschner_wave.txt'), wave)
        np.savetxt(os.path.join('..', 'src', 'templates', 'dorschner_x.txt'), x)
        np.savetxt(os.path.join('..', 'src', 'templates', 'dorschner_y.txt'), y)
        np.savetxt(os.path.join('..', 'src', 'templates', 'dorschner_qabs_pyr_all.txt'), pyr_qabs.reshape(len(wave)*len(a), len(x)))
        np.savetxt(os.path.join('..', 'src', 'templates', 'dorschner_qabs_oli_all.txt'), oli_qabs.reshape(len(wave)*len(a), len(y)))
        np.savetxt(os.path.join('..', 'src', 'templates', 'dorschner_qsca_pyr_all.txt'), pyr_qsca.reshape(len(wave)*len(a), len(x)))
        np.savetxt(os.path.join('..', 'src', 'templates', 'dorschner_qsca_oli_all.txt'), oli_qsca.reshape(len(wave)*len(a), len(y)))

    else:
        # Specify x and y for pyroxene and olivine, as well as particle size a in microns
        x = int(float(sys.argv[1]) * 100)  # must be '0.4', '0.5', '0.6', '0.7', '0.8', '0.95', or '1.0'
        y = int(float(sys.argv[2]) * 100)  # must be '0.4' or '0.5'
        a = float(sys.argv[3])

        wave, pyr_dict, oli_dict = read_dorschner_nk()
        pyr_qabs, pyr_qsca, _, _ = calculate_miecoeff(wave, pyr_dict[x], a=a)
        oli_qabs, oli_qsca, _, _ = calculate_miecoeff(wave, oli_dict[y], a=a)

        np.savetxt(os.path.join('..', 'src', 'templates', f'dorschner_wave.txt'), wave)
        np.savetxt(os.path.join('..', 'src', 'templates', f'dorschner_qabs_pyr_{x/100}_{a}.txt'), pyr_qabs)
        np.savetxt(os.path.join('..', 'src', 'templates', f'dorschner_qabs_oli_{y/100}_{a}.txt'), oli_qabs)
        np.savetxt(os.path.join('..', 'src', 'templates', f'dorschner_qsca_pyr_{x/100}_{a}.txt'), pyr_qsca)
        np.savetxt(os.path.join('..', 'src', 'templates', f'dorschner_qsca_oli_{y/100}_{a}.txt'), oli_qsca)
