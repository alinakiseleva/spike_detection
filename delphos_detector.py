import math
import scipy
import numpy as np


def initialize_variables(Fs):
    Oct = [math.floor(math.log(Fs / (4 * 80)) / math.log(2)), math.ceil(math.log(Fs / (4 * 8)) / math.log(2))]

    if Oct[0] < -1:
        Oct[0] = 0

    NbVoi = 12
    VanMom = 20
    # thr = 40

    return Oct, NbVoi, VanMom  # , thr


def DoG(sig, Oct, NbVoi, VanMom, Fs, Nexp=2, scaling=0):
    # Apply DoG Analytic Continuous Wavelet Transform

    siglength = len(sig)
    fff = np.array(range(0, siglength)) * 2 * math.pi / siglength
    Cst = 4 * VanMom / (math.pi * math.pi)
    fsig = scipy.fft.fft(sig)
    NbOct = len(range(Oct[0], Oct[1]))
    wt = np.zeros((NbOct * NbVoi, siglength), dtype=complex)
    freqlist = np.zeros((NbOct * NbVoi, 1))
    j = 0

    for oct in range(Oct[0], Oct[1]):
        for voi in range(0, NbVoi):
            scale = 2 ** (oct + voi / NbVoi)
            freqlist[j] = Fs / (4 * scale)
            tmp = scale * fff
            psi = (tmp ** VanMom) * np.exp(-Cst * tmp ** Nexp / 2)
            fTrans = fsig * psi.T

            if scaling:
                wt[j, :] = scipy.fft.ifft(fTrans).T
            else:
                wt[j, :] = np.sqrt(scale) * scipy.fft.ifft(fTrans).T

            j += 1

    wt = np.flipud(wt)
    freqlist = np.flip(freqlist)

    return wt, freqlist


def z_H0(tf, Fs):
    tf_z = []
    if tf.shape[1] > tf.shape[0]:
        tf = tf.T

    w = np.dot(scipy.signal.tukey(tf.shape[0], 0.25 * Fs / tf.shape[0]).reshape(tf.shape[0], 1),
               np.ones((1, tf.shape[1])))  #
    tf_real_modif = tf.real
    tf_imag_modif = -tf.imag
    tf2 = abs(tf) ** 2
    tf = []
    Nf = tf_real_modif.shape[1]
    sigma_real_N = np.zeros((1, Nf))
    tf_stat = tf_real_modif
    N = tf_stat.shape[0]

    if N > 16000:
        decimate = np.floor(np.linspace(Fs - 1, N - Fs - 1, 15000)).astype(int).T
    else:
        decimate = np.array(np.linspace(int(Fs), int(N + 1))).astype(int) - Fs

    tf_stat = tf_stat[decimate]
    K = 1.5

    for i in range(0, Nf):
        b = tf_stat.T[:][i]
        IQR = scipy.stats.iqr(b)
        q = np.quantile(b, [1 / 4, 3 / 4]) + [-K * IQR, K * IQR]
        b = b[b >= q[0]]
        b = b[b <= q[1]]
        b = np.array(b, dtype=float)
        _, sigma = scipy.stats.norm.fit(b)
        sigma_real_N[0][i] = sigma

    tf_real_modif = tf_real_modif / sigma_real_N
    tf_imag_modif = tf_imag_modif / sigma_real_N
    b = []
    tf_stat = []
    tf_z = w * abs(tf_real_modif + 1j * tf_imag_modif) ** 2
    tf_imag_modif = []

    return tf_z, tf2, tf_real_modif, sigma_real_N