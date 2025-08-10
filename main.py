# Common imports
from scipy.signal import resample, find_peaks, butter, filtfilt
from pathlib import Path
from itertools import groupby
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Benda lab: https://github.com/bendalab
from audioio import load_audio, write_audio
from thunderhopper import configuration, process_signal

# xsrp library: https://github.com/egrinstein/xsrp
from xsrp.conventional_srp import ConventionalSrp
from visualization.grids import plot_uniform_cartesian_grid


# added by Akseli
from utils.signal_processing import filter, find_peak_boundaries
from utils.video_export import fig_to_frame, combine_audio_video, normalize_wav_loudness
from utils.spectrograms import compute_spectrogram_db, plot_spectrogram_with_band



# -------------- CONFIG ---------------

# .wav folders (will be concatanted to one file)
#data_path = Path(r".\data\grid\4-2_recorder\2025-08-01_10-50\test") # can be parent folder or .wav file (specify as str or Path object)
data_path = r".\data\sample_data\micarray4-1-20250801T114937.wav"



# Inspect recordings, and draw min/max frequency for the calls of different species, individuals
freq_bands = [[6000, 9000], [12000, 19000]]

# bandpass filter settingss
lowcut = 3000 
highcut = 28000

# thunderhopper
feat_rate = 3000 

# segments of interest, need to have a peak in the norm > peak_threshold
# then 'go down' left/right from the peak until you reach norm < peak_boundary_threshold
peak_threshold = 0.4
peak_boundary_threshold = 0.2
save_masked_signal = False # Can save the intermediate 'masked_signal.wav' and inspect in audacity


# SRP mic settings
mic_spacing = 1  # 2m spacing
room_dims = [3.0, 3.0]  # 4m x 4m search area
n_grid_cells = 50  # Number of grid cells per dimension
mic_array_origin = [1.0, 1.0]  # bottom-left corner of mic array in room coordinates


# spectrogram 
n_fft = 1024
hop_length = 512


# Images
create_images = True
cols = 3


# Videos
create_video = True
video_fps = 5 



if isinstance(data_path, str):
    data_path = Path(data_path)
    
if data_path.is_file() and data_path.suffix.lower() == ".wav":
    wav_files = [data_path]
    output_base = data_path.parent
elif data_path.is_dir():
    wav_files = list(data_path.glob("*.wav"))
    output_base = data_path
else:
    raise ValueError(f"{data_path} is neither a .wav file nor a directory containing .wav files.")

temps = []
for wav_file in wav_files:
    print(f"Detected audio files: {wav_file}")
    temp, rate = load_audio(str(wav_file)) # sampling rate of original audio file = 'rate', sampling rate of output video = 'video_fps'
    temps.append(temp)

image_path = output_base / "source_pred_images"
bool_path = output_base / 'segment_boolean.npy'
video_path = output_base / "source_pred_videos"
os.makedirs(image_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)


signal = np.concatenate(temps, axis=0)  
n_chan = signal.shape[1] # num channels



# 2x2 spacing
mic_positions = np.array([
    [mic_array_origin[0], mic_array_origin[1] + mic_spacing],           # Top-left
    [mic_array_origin[0] + mic_spacing, mic_array_origin[1] + mic_spacing],  # Top-right
    [mic_array_origin[0] + mic_spacing, mic_array_origin[1]],           # Bottom-right
    [mic_array_origin[0], mic_array_origin[1]]                          # Bottom-left
])



srp = ConventionalSrp(
    fs=rate,
    grid_type="2D",
    n_grid_cells=n_grid_cells,
    mic_positions=mic_positions,
    room_dims=room_dims,
    mode="cross_correlation", # ["gcc_phat_freq", "gcc_phat_time", "cross_correlation"]
    interpolation=True,
    n_average_samples=2
)



# -------------------------------------------

signal  = filter(data=signal, lowcut=lowcut, highcut=highcut, fs=rate, btype='band')
signal_norm_mask = signal.copy()


# returns = ['norm']
# config = configuration(feat_rate=feat_rate, env_rate=feat_rate)
# print(f"Runninng thunderhopper pipeline...")
# data,_  = process_signal(config, signal=signal, rate=rate, returns=returns)


# norm_resampled = resample(data["norm"], signal.shape[0], axis=0)

# print("Segmenting based on feature norm...")
# for i in range(n_chan):
#     print(f"Processing channel {i+1}...")
#     peaks = find_peaks(norm_resampled[:, i], height=peak_threshold)[0]
#     print(f"Found {len(peaks)} peaks in channel {i+1}. If none found, adjust peak_threshold")
#     regions = find_peak_boundaries(norm_resampled[:, i], peaks, boundary_threshold=peak_boundary_threshold)

#     mask = np.zeros(signal_norm_mask.shape[0], dtype=bool)
#     for (start, end) in regions:        
#         mask[start:end] = True

#     signal_norm_mask[~mask, i] = 0


# if save_masked_signal:
#     print("Saving masked signal to 'masked_signal.wav'")
#     write_audio(signal_norm_mask, 'masked_signal.wav', rate=rate)




# # Create boolean mask for samples where there is a segment in any channel
# segment_boolean = np.any(signal_norm_mask != 0, axis=1)
# np.save(bool_path, segment_boolean)


segment_boolean = np.load(bool_path, allow_pickle=True) if bool_path.exists() else None



# Make list of (start, stop) tuples for those segments
seg_indices = []
for k, g in groupby(enumerate(segment_boolean), key=lambda x: x[1]):
    if k:  # k is True for segments
        group = list(g)
        seg_indices.append((group[0][0], group[-1][0] + 1))



for freq_band in freq_bands:
    print()
    print(f"Generating source prediction within frequency band {freq_band} Hz ...")
    

    signal_source_mask = filter(signal, lowcut=freq_band[0], highcut=freq_band[1], fs=rate, order=5, btype='band')


    if create_images:
        print(f"Saving images to {image_path}...")
        cols_counter = 0   
        rows = n_chan+1
    
    est_pos_list = []
    srp_map_list = []    

    print("Computing SRPs...")
    for i, (seg_start, seg_end) in enumerate(seg_indices):
        col = i % cols

        est_pos, srp_map, grid = srp.forward(signal_source_mask[seg_start:seg_end, :].T)
        est_pos_list.append(est_pos)
        srp_map_list.append(srp_map)
        

        if create_images:
            print(f"Processing image for segment {i+1}/{len(seg_indices)}: {seg_start/rate:.2f}s - {seg_end/rate:.2f}s")
                
            if i % cols == 0:
                img_fig, img_axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)



            for ch in range(n_chan):
                S_db, freqs = compute_spectrogram_db(signal[seg_start:seg_end, ch], sr=rate, n_fft=n_fft, hop_length=hop_length)

                if col == 0:
                    plot_spectrogram_with_band(
                        img_axs[ch, col],
                        S_db,
                        freqs,
                        freq_band=freq_band,
                        sample_idx=None,
                        ylabel=True
                    )
                else:
                    plot_spectrogram_with_band(
                        img_axs[ch, col],
                        S_db,
                        freqs,
                        freq_band=freq_band,
                        sample_idx=None,
                        ylabel=False
                    )

                if ch == 0:
                    img_axs[ch, col].set_title(f"{seg_start/rate:.2f}s - {seg_end/rate:.2f}s")

            
            plot_uniform_cartesian_grid(
                grid,
                room_dims,
                srp_map=srp_map,
                mic_positions=mic_positions,
                source_positions=est_pos.reshape(1, -1),
                ax=img_axs[n_chan, col],
                simplistic=(False if col == cols - 1 else True)
            )

            if col == cols - 1 or i == len(seg_indices) - 1: 
                # Save and close figure when row is complete OR at the last segment
                actual_cols = col + 1 if i == len(seg_indices) - 1 and col != cols - 1 else cols
                plt.savefig(Path(image_path, f'source-{freq_band}_seg-{cols_counter}-{cols_counter+actual_cols-1}.jpg'))
                cols_counter += actual_cols
                plt.close(img_fig)
                
    if create_video:
        print()
        print(f"Saving videos to {video_path}...")
        
        # Audio (.wav)
        audio_filename = Path(video_path, f'source_audio-{freq_band}.wav')
        write_audio(str(audio_filename), signal_source_mask, rate=rate)

        # If some frequency bands have very quiet sources, could amplify gain, like done here. 
        # The problem is that background noise is also amplified a lot. Maybe in the future, someone could add a function
        # to only export wav files for certain channels and segments where the call occurs.
        # normalize_wav_loudness(str(audio_filename), str(audio_filename), threshold_db=-40.0)

        # Video (.mp4)
        video_filename = Path(video_path, f'source_video_{freq_band}.mp4')

        
        total_duration = len(signal) / rate  # Total duration in seconds
        total_frames = int(total_duration * video_fps)
     

        height, width = 900, 1600
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_filename), fourcc, video_fps, (width, height))        
     
        

        frames = []

        for frame_idx in range(total_frames):
            current_time = (frame_idx / video_fps)

            # Find the segment whose start is closest to current_time and current_time > seg_start
            valid_indices = [idx for idx, (seg_start, seg_end) in enumerate(seg_indices) if current_time >= (seg_start / rate)]
            if valid_indices:
                closest_seg_idx = min(
                    valid_indices,
                    key=lambda idx: abs((seg_indices[idx][0] / rate) - current_time)
                )
            else:
                # Create black frame when no valid segment
                if frame_idx % 50 == 0:
                    print(f"Processing black frame {frame_idx + 1}/{total_frames}: {current_time:.2f}s")
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frames.append(frame)
                continue
                
            seg_start, seg_end = seg_indices[closest_seg_idx]
            i = closest_seg_idx
            start_t = seg_start / rate

            if frame_idx % 50 == 0:
                print(f"Processing fig frame {frame_idx + 1}/{total_frames}: {current_time:.2f}s")


            video_fig, video_axs = plt.subplots(1, 2, figsize=(16, 9), squeeze=False)

            est_pos = est_pos_list[i]
            srp_map = srp_map_list[i]

            # Find closest mic
            distances = np.linalg.norm(mic_positions - est_pos, axis=1)
            closest_mic_idx = np.argmin(distances)
                                
            S_db, freqs = compute_spectrogram_db(signal[seg_start:seg_end, closest_mic_idx], sr=rate, n_fft=n_fft, hop_length=hop_length)


            time_axis = np.linspace(0, S_db.shape[1] * hop_length / rate, S_db.shape[1])
            time_start_idx = np.argmin(np.abs(time_axis - (current_time - start_t)))
            time_end_idx = np.argmin(np.abs(time_axis - ((current_time + 1 / video_fps) - start_t)))

            plot_spectrogram_with_band(
            video_axs[0, 0],
            S_db,
            freqs,
            freq_band=freq_band,
            sample_idx=(time_start_idx, time_end_idx), 
            ylabel=True
            )  
                        

            video_axs[0, 0].set_title(f"Segment {i+1} ({seg_start/rate:.2f}s - {seg_end/rate:.2f}s) - channel: {closest_mic_idx + 1}")


            plot_uniform_cartesian_grid(
                grid,
                room_dims,
                srp_map=srp_map,
                mic_positions=mic_positions,
                source_positions=est_pos.reshape(1, -1),
                ax=video_axs[0, 1], 
                simplistic=False
            )
            video_axs[0, 1].set_title(f"Closest mic: {closest_mic_idx + 1}")

            frame = fig_to_frame(video_fig)
            frames.append(frame)

        
        for frame in frames:
            video_writer.write(frame)
    
        video_writer.release()
        
        # Combine .wav and .mp4 files into a single file
        try:
            combined_filename = combine_audio_video(
            video_path=video_path,
            audio_filename=audio_filename, 
            video_filename=video_filename,
            delete_originals=True  
            )
            print(f"Combined audio-video file saved to {combined_filename}")
        except Exception as e:
            print(f"Error combining audio and video: {e}")
        






        

