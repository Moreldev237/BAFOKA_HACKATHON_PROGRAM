 import torch

import os
import tempfile
import torch
# from utilities.audio_enhance.enhance_audio_data import make_prediction
import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

import torchaudio
from torch import mean as _mean
from torch import hamming_window, log10, no_grad, exp


def make_prediction(*args, **kwargs):
    raise NotImplementedError("Le module 'utilities' n'existe pas dans ce projet.")



def return_input(user_input):
    if user_input is None:
        return None
    return user_input


def stereo_to_mono_convertion(waveform):
    if waveform.shape[0] > 1:
        waveform = _mean(waveform, dim=0, keepdims=True)
        return waveform
    else:
        return waveform
        
def load_audio(audio_path):

    audio_tensor, sr = torchaudio.load(audio_path)
    audio_tensor = stereo_to_mono_convertion(audio_tensor)
    audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000)
    return audio_tensor

def load_audio_numpy(audio_path):
    audio_tensor, sr = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000)
    audio_array = audio_tensor.numpy()
    return (16000, audio_array.ravel())

def audio_to_spectrogram(audio):
    transform_fn = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=512//4, power=None, window_fn=hamming_window)
    spectrogram = transform_fn(audio)
    return spectrogram

def extract_magnitude_and_phase(spectrogram):
    magnitude, phase = spectrogram.abs(), spectrogram.angle()
    return magnitude, phase

def amplitude_to_db(magnitude_spec):
    max_amplitude = magnitude_spec.max()
    db_spectrogram = torchaudio.functional.amplitude_to_DB(magnitude_spec, 20, 10e-10, log10(max_amplitude), 100.0)
    return db_spectrogram, max_amplitude

def min_max_scaling(spectrogram, scaler):
    # Min-Max scaling (soundness of the math is questionable due to the use of each spectrograms' max value during decibel-scaling)
    spectrogram = scaler.transform(spectrogram)
    return spectrogram

def inverse_min_max(spectrogram, scaler):
    spectrogram = scaler.inverse_transform(spectrogram)
    return spectrogram

def db_to_amplitude(db_spectrogram, max_amplitude):
    return max_amplitude * 10**(db_spectrogram/20)

def reconstruct_complex_spectrogram(magnitude, phase):
    return magnitude * exp(1j*phase)
     
def inverse_fft(spectrogram):
    inverse_fn = torchaudio.transforms.InverseSpectrogram(n_fft=512, hop_length=512//4, window_fn=hamming_window)
    return inverse_fn(spectrogram)

def transform_audio(audio, scaler):
    spectrogram = audio_to_spectrogram(audio)
    magnitude, phase = extract_magnitude_and_phase(spectrogram)
    db_spectrogram, max_amplitude = amplitude_to_db(magnitude)
    db_spectrogram = min_max_scaling(db_spectrogram, scaler)
    return db_spectrogram.unsqueeze(0), phase, max_amplitude

def spectrogram_to_audio(db_spectrogram, scaler, phase, max_amplitude):
    db_spectrogram = db_spectrogram.squeeze(0)
    db_spectrogram = inverse_min_max(db_spectrogram, scaler)
    spectrogram = db_to_amplitude(db_spectrogram, max_amplitude)
    complex_spec = reconstruct_complex_spectrogram(spectrogram, phase)
    audio = inverse_fft(complex_spec)
    return audio

def save_audio(audio, output_path):
    if not output_path:
        output_path = r"enhanced_audio.wav"
    torchaudio.save(output_path, audio, 16000)
    return r"enhanced_audio.wav"

def predict(user_input, model, scaler, output_path):
    audio = load_audio(user_input)
    spectrogram, phase, max_amplitude = transform_audio(audio, scaler)
    
    with no_grad():
        enhanced_spectrogram = model.forward(spectrogram)
    enhanced_audio = spectrogram_to_audio(enhanced_spectrogram, scaler, phase, max_amplitude)
    enhanced_audio_path = save_audio(enhanced_audio, output_path)
    return enhanced_audio_path


class min_max_scaler():
    def __init__(self, upper_bound=1, lower_bound=0):
        
        self.upper = upper_bound
        self.lower = lower_bound
        self.minimum = torch.ones(1) * torch.inf
        self.maximum = - torch.ones(1) *torch.inf

    def fit(self, set_maximum=0.0, set_minimum=-100.0):
        """Find min and max of given subset OR set min and max manually. 
           Since dB-spectrograms are on the scale [-100, 0] by default, default values are set to those values.

        Args:
            set_maximum (float, optional): set maximum value manually. Defaults to 0.0.
            set_minimum (float, optional): set minimum value manually. Defaults to -100.0.

        Returns:
            None: None
        """
        if set_minimum is not None and set_maximum is not None:
            self.minimum = set_minimum
            self.maximum = set_maximum
        return None
    
    def transform(self, spectrogram):
        if self.minimum == torch.inf:
            raise ValueError("Cannot transform before scaler is fitted with min-max-values")
        return (self.upper - self.lower) * (spectrogram - self.minimum) / (self.maximum - self.minimum) + self.lower
        
    def inverse_transform(self, spectrogram):
        if self.minimum == torch.inf:
            raise ValueError("Cannot inverse transform before scaler is fitted with min-max-values")
        return (spectrogram - self.lower) * (self.maximum - self.minimum) / (self.upper - self.lower) + self.minimum

class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        skip_connection = x
        x = self.pool(x)
        return x, skip_connection

class DecodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecodingBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, skip_connection):
        x = self.conv_transpose(x)
        pd = (0, skip_connection.size(-1) - x.size(-1), 0, skip_connection.size(-2) - x.size(-2))
        x = nn.functional.pad(x, pd, mode='constant', value=0)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x

class UNet(nn.Module):
    def __init__(self, init_features=32, bottleneck_size=512):
        super(UNet, self).__init__()
        self.encoding_block1 = EncodingBlock(1, init_features)
        self.encoding_block2 = EncodingBlock(init_features, init_features*2)
        self.encoding_block3 = EncodingBlock(init_features*2, init_features*4)
        self.encoding_block4 = EncodingBlock(init_features*4, init_features*8)

        self.bottleneck_conv1 = nn.Conv2d(init_features*8, bottleneck_size, kernel_size=3, padding=1)
        self.bottleneck_conv2 = nn.Conv2d(bottleneck_size, bottleneck_size, kernel_size=3, padding=1)

        self.decoding_block4 = DecodingBlock(bottleneck_size, init_features*8)
        self.decoding_block3 = DecodingBlock(init_features*8, init_features*4)
        self.decoding_block2 = DecodingBlock(init_features*4, init_features*2)
        self.decoding_block1 = DecodingBlock(init_features*2, init_features)

        self.final_conv = nn.Conv2d(init_features, 1, kernel_size=1)
        
    def forward(self, x):
        x, skip1 = self.encoding_block1(x)
        x, skip2 = self.encoding_block2(x)
        x, skip3 = self.encoding_block3(x)
        x, skip4 = self.encoding_block4(x)

        x = self.bottleneck_conv1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.bottleneck_conv2(x)
        x = nn.ReLU(inplace=True)(x)

        x = self.decoding_block4(x, skip4)
        x = self.decoding_block3(x, skip3)
        x = self.decoding_block2(x, skip2)
        x = self.decoding_block1(x, skip1)

        x = self.final_conv(x)
        return x


class AudioVideoDenoiser:
    """
    Audio & video denoiser using your UNet model.

    - Loads the UNet + scaler once (fast for multiple calls).
    - GPU is used if available.
    - For videos: extracts audio with ffmpeg, denoises it, then remuxes into the original video stream.
    - Simple progress callbacks (0..100) at key steps.
    """

    def __init__(
        self,
        model_path: str = "weights/audio_enhance/model.pth",
        device: Optional[str] = None,         # "cuda" | "cpu" | None -> auto
        ffmpeg: str = "ffmpeg",               # path or just "ffmpeg" if on PATH
        callback=None,                        # callback(progress: float)
        keep_temp: bool = False,              # keep temp files for debugging
        temp_dir: Optional[str] = None,       # custom temp dir (optional)
    ):
        self.ffmpeg = ffmpeg
        self.keep_temp = keep_temp
        self.temp_dir = temp_dir
        self.callback = callback if callable(callback) else (lambda p: None)

        # Device selection (GPU preferred if available)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model once
        self.model = UNet().to(self.device)
        state = torch.load(model_path, map_location=self.device)
        # accept either {"model_state_dict": ...} or a raw state dict
        sd = state.get("model_state_dict", state)
        self.model.load_state_dict(sd)
        self.model.eval()

        # Prepare scaler (as in your script)
        self.scaler = min_max_scaler()
        self.scaler.fit()

    # -------------- utils --------------
    def _cb(self, p: float):
        """Safe progress callback in [0,100]."""
        try:
            p = float(max(0.0, min(100.0, p)))
            self.callback(p)
        except Exception:
            pass

    def _require_ffmpeg(self):
        """Ensure ffmpeg is available on PATH or custom path is valid."""
        from shutil import which
        if which(self.ffmpeg) is None:
            raise RuntimeError(
                "FFmpeg not found. Install ffmpeg or pass a valid path via ffmpeg=..."
            )

    @staticmethod
    def _default_out(in_path: str, suffix: str, fallback_ext: str) -> str:
        p = Path(in_path)
        ext = p.suffix if p.suffix else fallback_ext
        return str(p.with_name(p.stem + suffix + ext))

    # -------------- public API --------------
    def denoise_audio(self, in_path: str, out_path: Optional[str] = None) -> str:
        """
        Denoise a single audio file (any format supported by your 'predict' function).
        If out_path is None, writes '<stem>_denoised.wav' next to the input.
        """
        in_path = str(in_path)
        out_path = out_path or self._default_out(in_path, "_denoised", ".wav")

        self._cb(5.0)
        with torch.no_grad():
            # Your predict signature: predict(input_path, model, scaler, output_path)
            predict(in_path, self.model, self.scaler, out_path)
        self._cb(100.0)
        return out_path

    def denoise_video(
        self,
        in_video: str,
        out_video: Optional[str] = None,
        audio_stream_index: int = 0,
        # Extraction options (set to None to keep original)
        extract_sr: Optional[int] = None,     # e.g., 16000 if your model expects it
        extract_channels: Optional[int] = None,  # e.g., 1 for mono speech
        # Remux/encode options
        audio_codec: str = "aac",
        audio_bitrate: str = "192k",
    ) -> str:
        """
        Denoise the audio track of a video and remux it back:

        Steps: extract audio -> denoise audio -> remux (copy video stream).
        - audio_stream_index: which audio stream to extract from the video.
        - extract_sr / extract_channels: resample/mixdown during extraction if your model benefits from a fixed format.
        - audio_codec/audio_bitrate: codec used for the final muxed audio (video stream is -c:v copy).

        Returns the path to the denoised video.
        """
        self._require_ffmpeg()

        in_video = str(in_video)
        vid_path = Path(in_video)
        out_video = out_video or self._default_out(in_video, "_denoised", vid_path.suffix or ".mp4")

        # temp working dir
        tmp_root = Path(self.temp_dir) if self.temp_dir else None
        tmpdir = Path(tempfile.mkdtemp(dir=str(tmp_root) if tmp_root else None))

        try:
            self._cb(5.0)
            # 1) Extract audio to a WAV (uncompressed for your predict())
            extracted_wav = tmpdir / "audio_in.wav"
            cmd = [
                self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", in_video,
                "-map", f"0:a:{audio_stream_index}",
                "-vn",  # no video
            ]
            if extract_sr:
                cmd += ["-ar", str(extract_sr)]
            if extract_channels:
                cmd += ["-ac", str(extract_channels)]
            cmd += [str(extracted_wav)]
            subprocess.check_call(cmd)

            self._cb(25.0)

            # 2) Denoise the extracted audio
            denoised_wav = tmpdir / "audio_out.wav"
            with torch.no_grad():
                predict(str(extracted_wav), self.model, self.scaler, str(denoised_wav))
            self._cb(85.0)

            # 3) Remux: copy original video stream, replace audio with denoised
            cmd2 = [
                self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", in_video,
                "-i", str(denoised_wav),
                "-map", "0:v:0",     # take the original video stream
                "-map", "1:a:0",     # take the denoised audio
                "-c:v", "copy",      # do not re-encode video
                "-c:a", audio_codec,
                "-b:a", audio_bitrate,
                "-shortest",         # match the shortest stream (avoid trailing black)
                str(out_video),
            ]
            subprocess.check_call(cmd2)

            self._cb(100.0)
            return str(out_video)

        finally:
            if not self.keep_temp:
                shutil.rmtree(tmpdir, ignore_errors=True)


def enhance_video_audio(input_video_path, output_video_path, model_path="model.pth"):
    # Check if input video exists
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"The input video path {input_video_path} does not exist.")

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "extracted_audio.wav")
        
        # Try extracting audio using ffmpeg
        ffmpeg_extract_cmd = f"ffmpeg -y -i \"{input_video_path}\" -q:a 0 -map a \"{audio_path}\""
        result = os.system(ffmpeg_extract_cmd)
        
        # If ffmpeg fails, fallback to moviepy
        if result != 0 or not os.path.exists(audio_path):
            try:
                from moviepy.editor import VideoFileClip
                video_clip = VideoFileClip(input_video_path)
                video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
            except Exception as e:
                raise RuntimeError(f"Failed to extract audio from video: {e}")

        # Enhance audio
        enhanced_audio_path = os.path.join(temp_dir, "enhanced_audio.wav")
        make_prediction(audio_path, enhanced_audio_path, model_path=model_path)

        # Replace original audio with enhanced audio
        ffmpeg_combine_cmd = (
            f"ffmpeg -y -i \"{input_video_path}\" -i \"{enhanced_audio_path}\" "
            f"-c:v copy -c:a aac -strict experimental \"{output_video_path}\""
        )
        result = os.system(ffmpeg_combine_cmd)
        if result != 0:
            raise RuntimeError("Failed to combine video with enhanced audio.")




if __name__ == "__main__":
    def progress(p):
        print(f"Progress: {p:.1f}%")

    denoiser = AudioVideoDenoiser(
        model_path="weights/audio_enhance/model.pth",
        callback=progress,
    )

    # 1) Audio
    out_wav = denoiser.denoise_audio("samples_DNS3_test_set_0072_noisy.wav")
    print("Audio denoised ->", out_wav)

    # 2) Vidéo (extrait la piste audio, débruite, remuxe)
    out_mp4 = denoiser.denoise_video(
        "noisy_clip.mp4",
        extract_sr=16000,         # si ton modèle attend 16 kHz
        extract_channels=1,       # mono pour la parole
        audio_codec="aac",
        audio_bitrate="192k",
    )
    print("Video denoised ->", out_mp4)


