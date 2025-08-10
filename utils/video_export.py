import cv2
import numpy as np
import io
import subprocess
from pathlib import Path
import time
import matplotlib.pyplot as plt
from pydub import AudioSegment

# Convert figures to frames
def fig_to_frame(fig_or_ax):
    """
    Converts a Matplotlib figure or axes object into an OpenCV image frame (NumPy array).

    Parameters
    ----------
    fig_or_ax : matplotlib.figure.Figure or matplotlib.axes.Axes
        The Matplotlib figure or axes to convert. If an axes object is provided, its parent figure is used.
        
    Returns
    -------
    img : numpy.ndarray
        The image frame in OpenCV format (height x width x 3, dtype=uint8, BGR color order).
    """
    # If it's an axes object, get the figure from it
    if hasattr(fig_or_ax, 'get_figure'):
        fig = fig_or_ax.get_figure()
    else:
        fig = fig_or_ax
    
    # Ensure the figure is drawn/rendered
    fig.canvas.draw()
    time.sleep(0.05)  # Add a short delay to ensure the figure is rendered
    
    # Get RGB buffer directly from canvas
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    return img


def combine_audio_video(video_path, audio_filename, video_filename, output_filename=None, delete_originals=False):
    """
    Combine audio and video files using ffmpeg.
    
    Parameters:
    -----------
    video_path : Path or str
        Directory path where files are located
    audio_filename : Path or str  
        Path to the audio file (.wav)
    video_filename : Path or str
        Path to the video file (.mp4)
    output_filename : Path or str, optional
        Output filename. If None, will use video filename with '_combined' suffix
    delete_originals : bool, default False
        Whether to delete the original audio and video files after combining
        
    Returns:
    --------
    Path
        Path to the combined output file
        
    """
    video_path = Path(video_path)
    audio_filename = Path(audio_filename)
    video_filename = Path(video_filename)
    
    
    # Generate output filename if not provided
    if output_filename is None:
        video_stem = video_filename.stem
        video_suffix = video_filename.suffix
        output_filename = video_path / f"{video_stem}_combined{video_suffix}"
        print(output_filename)
    else:
        output_filename = Path(output_filename)
    

    cmd = [
        'ffmpeg',
        '-i', str(video_filename),
        '-i', str(audio_filename), 
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-y',  # Overwrite any existing output file
        str(output_filename)
    ]
    
    try:
        print(f"Combining {video_filename.name} with {audio_filename.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        

        if delete_originals:
            audio_filename.unlink()
            video_filename.unlink()
            
        return output_filename
        
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg command failed with return code {e.returncode}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error output: {e.stderr}")
        raise
    except FileNotFoundError:
        raise FileNotFoundError("ffmpeg not found. Please install ffmpeg and ensure it's in your PATH.")
    
    


def normalize_wav_loudness(input_file, output_file, threshold_db=-40.0):
    """
    Adjusts the loudness of a WAV file to meet a minimum dBFS threshold.

    Args:
        input_file (str): Path to the input .wav file.
        output_file (str): Path where the adjusted .wav will be saved.
        threshold_db (float): Minimum target loudness in dBFS. Default is -40.0.
    """
    
    audio = AudioSegment.from_wav(input_file)


    current_loudness = audio.dBFS
    print(f"Current loudness: {current_loudness:.2f} dBFS")

    # If quieter than threshold, apply gain
    if current_loudness < threshold_db:
        gain_needed = threshold_db - current_loudness
        print(f"Applying gain of {gain_needed:.2f} dB")
        audio = audio.apply_gain(gain_needed)

    audio.export(output_file, format="wav")



