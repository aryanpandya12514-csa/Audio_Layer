import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import wavfile

class DeepfakeDataset(Dataset):
    """
    A PyTorch Dataset for loading 'real_audio' and 'fake_audio' .wav files.
    This bypasses torchaudio completely to avoid codec crashes, utilizing scipy.io.wavfile.
    It strictly ensures a fixed temporal geometry for audio arrays (exactly 3 seconds).
    """
    def __init__(self, root_dir):
        """
        Initializes the dataset by scanning for .wav files in the predefined
        'real_audio' and 'fake_audio' subdirectories.
        
        Args:
            root_dir (str): The base directory path containing the audio class folders.
        """
        self.file_paths = []
        self.labels = []
        
        # 1. Traverse and index Real audio files
        real_dir = os.path.join(root_dir, 'real_audio')
        if os.path.exists(real_dir):
            # Matches all .wav files nested within real_audio
            real_files = glob.glob(os.path.join(real_dir, '**', '*.wav'), recursive=True)
            self.file_paths.extend(real_files)
            # Label Real strictly as 0.0
            self.labels.extend([0.0] * len(real_files))
            
        # 2. Traverse and index Fake audio files
        fake_dir = os.path.join(root_dir, 'fake_audio')
        if os.path.exists(fake_dir):
            # Matches all .wav files nested within fake_audio
            fake_files = glob.glob(os.path.join(fake_dir, '**', '*.wav'), recursive=True)
            self.file_paths.extend(fake_files)
            # Label Fake strictly as 1.0
            self.labels.extend([1.0] * len(fake_files))
            
        # Target constraints: We require exactly 3 seconds at 16000Hz -> 48,000 temporal samples
        self.target_samples = 48000
        
    def __len__(self):
        """Returns the total number of audio samples parsed in the dataset directory."""
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        """
        Retrieves, decodes, normalizes, and geometrically coerces the target audio file.
        
        Args:
            idx (int): The dataloader query index.
            
        Returns:
            tuple: 
                - audio_tensor (torch.Tensor): A strictly 1D tensor [48000] of PyTorch floats.
                - label_tensor (torch.Tensor): A scalar tensor representing 0.0 (Real) or 1.0 (Fake).
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # 3. Read the audio file directly via numpy backend to avoid PyTorch av/FFmpeg crashing
            # This returns the sampling rate (which we expect is 16kHz) and the raw PCM data arrays
            sample_rate, data = wavfile.read(file_path)
            
            # Cast raw integers into float space for numerical manipulation
            data = data.astype(np.float32)
            
            # 4. Handle Stereo constraints (Crush to Mono)
            # If the audio has multiple channels (e.g., shape is [samples, 2]), average them out
            if len(data.shape) > 1 and data.shape[1] > 1:
                data = np.mean(data, axis=1)
                
            # 5. Normalize raw magnitudes to strictly bounded -1.0 to 1.0 float limits
            # Instead of guessing the bit-depth (int16/int32), we dynamically use the signal maximum
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = data / max_val
                
            # 6. Strict Geometry Fix (Enforce explicitly 48,000 samples)
            if len(data) > self.target_samples:
                # Truncate: The file is longer than 3 seconds; keep the first 48,000 samples
                data = data[:self.target_samples]
            elif len(data) < self.target_samples:
                # Pad: The file is shorter; inject zero-arrays seamlessly over the missing duration
                padding_size = self.target_samples - len(data)
                padding = np.zeros(padding_size, dtype=np.float32)
                data = np.concatenate((data, padding))
                
        except Exception:
            # 7. Fallback Safety Intercept
            # If the .wav file is corrupted, empty, or unreadable, we deploy a purely zeroed array
            # to assure the broader training loop won't detonate halfway through an epoch
            data = np.zeros(self.target_samples, dtype=np.float32)
            
        # 8. Transform mathematical bounds to PyTorch tensor constructs
        audio_tensor = torch.from_numpy(data)
        
        # We form an explicitly sized dimension for the binary classification head scalar expectation
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        return audio_tensor, label_tensor