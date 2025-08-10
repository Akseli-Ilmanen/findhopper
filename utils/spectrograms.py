import librosa
import numpy as np
import matplotlib.patches

def compute_spectrogram_db(y, sr, n_fft=1024, hop_length=512, noise_floor_db=-70):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    S_db = np.maximum(S_db, noise_floor_db)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return S_db, freqs


def plot_spectrogram_with_band(ax, S_db, freqs, freq_band=None, sample_idx=None, ylabel=False):
    """
    Plots a spectrogram on the given axis and overlays a red box indicating the frequency band and time window.

    Parameters:
        ax: matplotlib axis to plot on.
        S_db: Spectrogram (2D numpy array).
        freqs: Frequency bins (1D numpy array).
        freq_band: [low, high] frequency band (Hz). If None, uses min/max of freqs.
        sample_idx: (start_idx, end_idx) tuple for time window (columns in S_db). If None, uses full range.
    """
    ax.imshow(S_db, origin='lower', aspect='auto', cmap='viridis')
    
    if ylabel:
        ax.set_yticks(np.linspace(0, S_db.shape[0] - 1, 5))
        ax.set_yticklabels([f"{int(freq)}" for freq in np.linspace(freqs[0], freqs[-1], 5)])
        ax.set_ylabel("Frequency (Hz)")
    else:
        ax.set_yticks([])
    
    # Frequency indices
    if freq_band is not None:
        freq_idx_0 = np.argmin(np.abs(freqs - freq_band[0]))
        freq_idx_1 = np.argmin(np.abs(freqs - freq_band[1]))
    else:
        freq_idx_0 = 0
        freq_idx_1 = len(freqs) - 1
        
    # Time indices
    if sample_idx is not None:
        time_idx_0, time_idx_1 = sample_idx
    else:
        time_idx_0 = 0
        time_idx_1 = S_db.shape[1] - 1
    # Draw red rectangle
    rect = matplotlib.patches.Rectangle(
        (time_idx_0, freq_idx_0),
        time_idx_1 - time_idx_0,
        freq_idx_1 - freq_idx_0,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)