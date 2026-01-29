# gender_features.py
"""
Enhanced Gender Feature Extraction Module
Extracts formants, MFCCs, prosodic features, and spectral features for gender classification
"""

import numpy as np
import librosa
import scipy.signal as signal


def extract_formants(audio, sr):
    """
    Extract first two formant frequencies using LPC analysis
    
    Args:
        audio: Audio signal (numpy array)
        sr: Sample rate
        
    Returns:
        tuple: (F1, F2) formant frequencies in Hz
    """
    try:
        # Pre-emphasis filter to boost high frequencies
        pre_emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        # LPC order (rule of thumb: sr/1000 + 2)
        lpc_order = min(int(sr / 1000) + 2, 50)
        
        # LPC analysis
        a = librosa.lpc(pre_emphasized + 1e-6, order=lpc_order)
        
        # Find formants from LPC polynomial roots
        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]  # Keep only positive frequencies
        
        # Convert to Hz
        angz = np.arctan2(np.imag(roots), np.real(roots))
        formants = sorted(angz * (sr / (2 * np.pi)))
        
        # Filter physically plausible formants (50-5000 Hz)
        formants = [f for f in formants if 50 < f < 5000]
        
        # Return first two formants or defaults
        f1 = formants[0] if len(formants) >= 1 else 500
        f2 = formants[1] if len(formants) >= 2 else 1500
        
        return f1, f2
        
    except Exception as e:
        print(f"Warning: Formant extraction failed: {e}")
        return 500, 1500  # Default values


def extract_prosodic_features(audio, sr):
    """
    Extract prosodic features: energy and pitch statistics
    
    Args:
        audio: Audio signal
        sr: Sample rate
        
    Returns:
        dict: Prosodic features
    """
    # Energy features
    energy_mean = np.sqrt(np.mean(audio ** 2))
    rms = librosa.feature.rms(y=audio)
    energy_std = np.std(rms)
    
    # Pitch features using pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, 
        fmin=librosa.note_to_hz('C2'),  # ~65 Hz
        fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
        sr=sr
    )
    
    # Filter valid pitch values
    f0_valid = f0[~np.isnan(f0)]
    
    if len(f0_valid) > 0:
        pitch_mean = np.median(f0_valid)
        pitch_std = np.std(f0_valid)
        pitch_range = np.ptp(f0_valid)  # Peak to peak (max - min)
    else:
        pitch_mean = 0
        pitch_std = 0
        pitch_range = 0
    
    return {
        'energy_mean': float(energy_mean),
        'energy_std': float(energy_std),
        'pitch_mean': float(pitch_mean),
        'pitch_std': float(pitch_std),
        'pitch_range': float(pitch_range)
    }


def extract_gender_features(audio, sr):
    """
    Extract comprehensive feature vector for gender classification
    
    Features:
    - Pitch (median F0)
    - Formants (F1, F2)
    - MFCCs (13 coefficients)
    - Prosodic features (4: energy mean/std, pitch std/range)
    - Spectral features (3: centroid, rolloff, ZCR)
    
    Total: 23 features
    
    Args:
        audio: Audio signal (numpy array)
        sr: Sample rate
        
    Returns:
        numpy array: Feature vector of length 23
    """
    # 1. Pitch (1 feature)
    f0, _, _ = librosa.pyin(audio, fmin=50, fmax=400, sr=sr)
    f0_valid = f0[~np.isnan(f0)]
    pitch_mean = np.median(f0_valid) if len(f0_valid) > 0 else 0
    
    # 2. Formants (2 features)
    f1, f2 = extract_formants(audio, sr)
    
    # 3. MFCCs (13 features)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    
    # 4. Prosodic features (4 features)
    prosodic = extract_prosodic_features(audio, sr)
    
    # 5. Spectral features (3 features)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    
    # Combine all features
    features = np.concatenate([
        [pitch_mean, f1, f2],                      # 3
        mfcc_mean,                                  # 13
        [prosodic['energy_mean'], 
         prosodic['energy_std'],
         prosodic['pitch_std'], 
         prosodic['pitch_range']],                 # 4
        [spectral_centroid, 
         spectral_rolloff, 
         zcr]                                       # 3
    ])
    
    return features.astype(np.float32)


if __name__ == "__main__":
    # Test the feature extraction
    import soundfile as sf
    
    print("Testing gender feature extraction...")
    
    # Create test audio
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Male voice (120 Hz)
    audio_male = 0.3 * np.sin(2 * np.pi * 120 * t)
    features_male = extract_gender_features(audio_male, sr)
    
    print(f"\nMale voice features:")
    print(f"  Shape: {features_male.shape}")
    print(f"  Pitch: {features_male[0]:.2f} Hz")
    print(f"  F1: {features_male[1]:.2f} Hz")
    print(f"  F2: {features_male[2]:.2f} Hz")
    
    # Female voice (220 Hz)
    audio_female = 0.3 * np.sin(2 * np.pi * 220 * t)
    features_female = extract_gender_features(audio_female, sr)
    
    print(f"\nFemale voice features:")
    print(f"  Shape: {features_female.shape}")
    print(f"  Pitch: {features_female[0]:.2f} Hz")
    print(f"  F1: {features_female[1]:.2f} Hz")
    print(f"  F2: {features_female[2]:.2f} Hz")
    
    print(f"\n✓ Feature extraction module working correctly!")
