import numpy as np
from typing import Union, Optional, Tuple, Generator, List
import scipy.signal as sig

class SimluatedSignal:

    @staticmethod
    def generate_wave(fs, frequency, amplitude, n_secs):
        time = np.arange(0, n_secs, 1 / fs)
        sim_signal = amplitude * np.sin(2 * np.pi * frequency * time)
        return sim_signal.reshape(sim_signal.shape[0], 1)

    def __init__(
            self,
            n_chs: int,
            n_secs: Union[int, float],
            sfreq: int,
            fs_range: Tuple[int, int],
            amplitude_range: Union[Tuple[int, int], Tuple[float, float]],
            n_fs_bands: Optional[int] = 5,
            noise_scale: Optional[Union[int, float]] = 0.5
    ):

        self.n_chs = n_chs
        self.n_secs = n_secs
        self.sfreq = sfreq
        self.fs = sfreq * 2
        self.min_freq, self.max_freq = fs_range
        self.min_amplitude, self.max_amplitude = amplitude_range
        self.n_fs_bands = n_fs_bands
        self.noise_scale = noise_scale

    def __call__(self) -> Generator[np.ndarray, None, None]:

        while True:
            sim_signal = np.zeros((self.fs * self.n_secs, self.n_chs))
            freqs = np.logspace(
                np.log10(self.min_freq),
                np.log10(self.max_freq),
                self.n_fs_bands,
                base=10.0
            ).astype(int)

            for ch in range(self.n_chs):
                for i in range(freqs.shape[0] - 1):
                    amplitude = np.random.uniform(self.min_amplitude, self.max_amplitude)
                    frequency = np.random.uniform(freqs[i], freqs[i + 1])
                    sim_signal[:, ch, None] += self.generate_wave(self.fs, frequency, amplitude / (i + 1), self.n_secs)

            noise = np.random.normal(0, amplitude * self.noise_scale, [self.n_secs * self.fs, self.n_chs])
            sim_signal += noise
            sim_signal = sig.resample(sim_signal, sim_signal.shape[0] // self.fs * self.sfreq)

            yield sim_signal


class SimluatedSpikeSignal(SimluatedSignal):

    @staticmethod
    def generate_spike(
            fs: int,
            duration: Union[int, Tuple[int, int], Tuple[float, float]],
            spike_function: str
    ):
        spike_duration = duration if isinstance(duration, (int, float)) else np.random.uniform(*duration)
        if spike_function == 'daub':
            spike_shape = np.random.randint(10, 30)
            spike = sig.daub(spike_shape)
        else:
            spike_shape = np.random.randint(20, 100)
            spike = sig.ricker(spike_shape, 1)
        spike = sig.resample(spike, int(spike_duration * fs))
        peak = np.argmax(spike)
        return spike, peak

    @staticmethod
    def chose_spike_inds(
            max_len_spike: int,
            len_signal: int,
            n_spikes: int
    ) -> List[int]:
        if max_len_spike > len_signal:
            raise ValueError(f'Spike length: {max_len_spike} greater than signal length: {len_signal}')
        if len_signal - (n_spikes + 1) * max_len_spike < 0:
            n_spikes = np.floor(len_signal / max_len_spike).astype(int)
        free_samples = len_signal - (n_spikes + 1) * max_len_spike
        last_inds = 0
        spike_inds = []
        for i in range(n_spikes):
            shift = np.random.randint(0, free_samples) if free_samples > 0 else 0
            last_inds += shift + max_len_spike
            free_samples -= shift
            spike_inds.append(int(last_inds))
        return spike_inds

    def __init__(self,
                 n_chs: int,
                 n_secs: Union[int, float],
                 sfreq: int,
                 fs_range: Tuple[int, int],
                 amplitude_range: Union[Tuple[int, int], Tuple[float, float]],
                 spike_num: Union[int, Tuple[int, int]],
                 spike_scale: Optional[Union[int, float, Tuple[int, int], Tuple[float, float]]] = 2,
                 p_spike_shape: Optional[float] = 0.5,
                 spike_duration: Optional[Union[int, float, Tuple[int, int], Tuple[float, float]]] = (0.1, 0.2),
                 n_fs_bands: Optional[int] = 5,
                 noise_scale: Optional[Union[int, float]] = 0.5,
                 ):
        super().__init__(n_chs, n_secs, sfreq, fs_range, amplitude_range, n_fs_bands, noise_scale)
        self.spike_num = spike_num
        self.spike_scale = spike_scale
        self.p_spike_shape = p_spike_shape
        self.spike_duration = spike_duration

    def __call__(self) -> Generator[np.ndarray, None, None]:

        while True:
            sim_signal = np.zeros((self.fs * self.n_secs, self.n_chs))
            freqs = np.logspace(
                np.log10(self.min_freq),
                np.log10(self.max_freq),
                self.n_fs_bands,
                base=10.0
            ).astype(int)
            spike_inds = []

            for ch in range(self.n_chs):
                for i in range(freqs.shape[0] - 1):
                    amplitude = np.random.uniform(self.min_amplitude, self.max_amplitude)
                    frequency = np.random.uniform(freqs[i], freqs[i + 1])
                    sim_signal[:, ch, None] += super().generate_wave(
                        self.fs,
                        frequency,
                        amplitude / (i + 1),
                        self.n_secs
                    )

                ch_spike_num = self.spike_num \
                    if isinstance(self.spike_num, (int, float)) \
                    else np.random.randint(*self.spike_num)

                ch_spike_inds = self.chose_spike_inds(
                    np.ceil(np.max(self.spike_duration) * self.fs),
                    sim_signal.shape[0],
                    ch_spike_num
                )
                ch_max_amp = np.max(sim_signal[:, ch])

                for i in range(len(ch_spike_inds)):
                    spike, peak_time = self.generate_spike(
                        self.fs,
                        self.spike_duration,
                        'daub' \
                            if np.random.rand() > 0.5 \
                            else 'ricker'
                    )

                    spike_scale_ratio = self.spike_scale \
                        if isinstance(self.spike_scale, (int, float)) \
                        else np.random.uniform(*self.spike_scale)
                    spike_amplitude = ch_max_amp / np.max(spike) * spike_scale_ratio

                    sim_signal[ch_spike_inds[i]: ch_spike_inds[i] + spike.shape[0], ch] += spike * spike_amplitude

                    ch_spike_inds[i] += peak_time
                    spike_inds.append([ch, ch_spike_inds[i] // (self.fs / self.sfreq)])

            noise = np.random.normal(
                0,
                amplitude * self.noise_scale,
                [self.n_secs * self.fs, self.n_chs]
            )

            sim_signal += noise
            sim_signal = sig.resample(
                sim_signal,
                sim_signal.shape[0] // self.fs * self.sfreq
            )

            yield sim_signal, np.array(spike_inds).astype(int)