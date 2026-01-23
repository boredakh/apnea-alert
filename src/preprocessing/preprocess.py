# File: src/preprocessing/preprocess.py
"""
ECG Preprocessing Module for Sleep Apnea Detection
"""
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis
import pandas as pd
import os

class ECGPreprocessor:
    """Preprocess ECG signals for apnea detection"""
    
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
        
    def bandpass_filter(self, ecg_signal, lowcut=0.5, highcut=40.0):
        """Apply bandpass filter to ECG signal"""
        # Design bandpass filter
        b, a = butter(4, [lowcut/self.nyquist, highcut/self.nyquist], btype='band')
        filtered_signal = filtfilt(b, a, ecg_signal)
        return filtered_signal
    
    def notch_filter(self, ecg_signal, notch_freq=50.0, quality_factor=30.0):
        """Apply notch filter to remove powerline interference"""
        b, a = signal.iirnotch(notch_freq/self.nyquist, quality_factor)
        filtered_signal = filtfilt(b, a, ecg_signal)
        return filtered_signal
    
    def segment_ecg(self, ecg_signal, labels, segment_length_sec=60):
        """Segment ECG signal into 1-minute segments"""
        segment_samples = segment_length_sec * self.sampling_rate
        num_segments = len(ecg_signal) // segment_samples
        
        segments = []
        segment_labels = []
        
        for i in range(num_segments):
            start = i * segment_samples
            end = start + segment_samples
            
            if end <= len(ecg_signal):
                segment = ecg_signal[start:end]
                
                # Only include if we have a label for this minute
                if i < len(labels):
                    label = labels[i]
                    if label in ['A', 'N']:  # Only include apnea/normal
                        segments.append(segment)
                        segment_labels.append(label)
        
        return np.array(segments), np.array(segment_labels)
    
    def extract_features(self, ecg_segment):
        """Extract features from ECG segment"""
        features = {}
        
        # Time-domain features
        features['mean'] = np.mean(ecg_segment)
        features['std'] = np.std(ecg_segment)
        features['min'] = np.min(ecg_segment)
        features['max'] = np.max(ecg_segment)
        features['range'] = np.ptp(ecg_segment)
        features['rms'] = np.sqrt(np.mean(ecg_segment**2))
        
        # Statistical features
        features['skewness'] = skew(ecg_segment)
        features['kurtosis'] = kurtosis(ecg_segment)
        
        # Heart rate features (simplified)
        r_peaks = self.detect_r_peaks(ecg_segment)
        if len(r_peaks) > 1:
            rr_intervals = np.diff(r_peaks) / self.sampling_rate
            features['hr_mean'] = 60 / np.mean(rr_intervals)
            features['hr_std'] = np.std(60 / rr_intervals)
            features['rr_mean'] = np.mean(rr_intervals)
            features['rr_std'] = np.std(rr_intervals)
        else:
            features['hr_mean'] = 0
            features['hr_std'] = 0
            features['rr_mean'] = 0
            features['rr_std'] = 0
        
        # Frequency-domain features
        frequencies, psd = welch(ecg_segment, fs=self.sampling_rate, nperseg=256)
        
        # Low frequency (0.04-0.15 Hz) - associated with sympathetic activity
        lf_mask = (frequencies >= 0.04) & (frequencies <= 0.15)
        features['lf_power'] = np.sum(psd[lf_mask]) if np.any(lf_mask) else 0
        
        # High frequency (0.15-0.4 Hz) - associated with parasympathetic activity
        hf_mask = (frequencies >= 0.15) & (frequencies <= 0.4)
        features['hf_power'] = np.sum(psd[hf_mask]) if np.any(hf_mask) else 0
        
        # LF/HF ratio (sympathovagal balance)
        features['lf_hf_ratio'] = features['lf_power'] / features['hf_power'] if features['hf_power'] > 0 else 0
        
        return features
    
    def detect_r_peaks(self, ecg_segment, distance=None, prominence=None):
        """Detect R-peaks in ECG segment"""
        if distance is None:
            distance = self.sampling_rate * 0.5  # Minimum 0.5s between beats
        if prominence is None:
            prominence = 0.1  # Minimum peak prominence
        
        peaks, _ = signal.find_peaks(ecg_segment, distance=distance, prominence=prominence)
        return peaks
    
    def process_record(self, ecg_signal, labels, record_name=""):
        """Process a complete record"""
        print(f"Processing {record_name}..." if record_name else "Processing record...")
        
        # Step 1: Filtering
        print("  Step 1: Filtering...")
        filtered_ecg = self.bandpass_filter(ecg_signal)
        filtered_ecg = self.notch_filter(filtered_ecg)
        
        # Step 2: Segmentation
        print(f"  Step 2: Segmenting...")
        segments, segment_labels = self.segment_ecg(filtered_ecg, labels)
        print(f"    Created {len(segments)} segments")
        print(f"    Apnea segments: {np.sum(segment_labels == 'A')}")
        print(f"    Normal segments: {np.sum(segment_labels == 'N')}")
        
        # Step 3: Feature extraction
        print(f"  Step 3: Extracting features...")
        features_list = []
        for i, segment in enumerate(segments):
            features = self.extract_features(segment)
            features['label'] = segment_labels[i]
            features['segment_id'] = i
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        return {
            'segments': segments,
            'labels': segment_labels,
            'features': features_df,
            'filtered_ecg': filtered_ecg
        }


def test_preprocessor():
    """Test function for the preprocessor"""
    print("=" * 60)
    print("TESTING ECG PREPROCESSOR")
    print("=" * 60)
    
    # Create a synthetic ECG signal for testing
    sampling_rate = 100
    duration = 60  # 1 minute
    t = np.linspace(0, duration, sampling_rate * duration)
    
    # Synthetic ECG with heartbeat at ~1 Hz (60 BPM)
    ecg_signal = 0.5 * np.sin(2 * np.pi * 1 * t)  # 1 Hz baseline
    ecg_signal += 0.1 * np.sin(2 * np.pi * 5 * t)  # 5 Hz component
    ecg_signal += 0.05 * np.sin(2 * np.pi * 20 * t)  # 20 Hz component
    ecg_signal += 0.01 * np.random.randn(len(t))  # Noise
    
    # Synthetic labels (first 30s apnea, last 30s normal)
    labels = np.array(['A', 'N'])
    
    # Initialize and test preprocessor
    preprocessor = ECGPreprocessor(sampling_rate=sampling_rate)
    
    print(f"Test ECG shape: {ecg_signal.shape}")
    print(f"Test labels: {labels}")
    
    # Test individual methods
    print("\nTesting bandpass filter...")
    filtered = preprocessor.bandpass_filter(ecg_signal)
    print(f"  Original std: {ecg_signal.std():.4f}")
    print(f"  Filtered std: {filtered.std():.4f}")
    
    print("\nTesting segmentation...")
    segments, seg_labels = preprocessor.segment_ecg(ecg_signal, labels)
    print(f"  Created {len(segments)} segments")
    print(f"  Segment shape: {segments[0].shape}")
    print(f"  Labels: {seg_labels}")
    
    print("\nTesting feature extraction...")
    features = preprocessor.extract_features(segments[0])
    print(f"  Extracted {len(features)} features")
    print(f"  Sample features: mean={features['mean']:.4f}, hr_mean={features.get('hr_mean', 0):.1f} BPM")
    
    print("\nTesting complete process...")
    result = preprocessor.process_record(ecg_signal, labels, "test_record")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    test_preprocessor()