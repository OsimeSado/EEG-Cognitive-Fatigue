"""
Feature Extraction Script 
Saves results to CSV for visualization in subject-specific folders
"""

import numpy as np
import pandas as pd
from scipy.signal import welch, coherence, spectrogram
from scipy.stats import entropy
import antropy as ant
import warnings
warnings.filterwarnings('ignore')

from data_loader import EEGDataLoader

class ComprehensiveFeatureExtractor:
    """Extract all features for pre vs post fatigue analysis"""
    
    def __init__(self, sampling_rate):
        self.fs = sampling_rate
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 80)
        }
    
    # 1. SPECTRAL FEATURES (PSD & Band Powers)

    
    def extract_psd_features(self, data, channels):
        """
        Extract Power Spectral Density for each channel
        Returns: DataFrame with PSD values and band powers
        """
        results = []
        
        for ch_idx, ch_name in enumerate(channels):
            signal = data[ch_idx]
            
            # Compute PSD
            freqs, psd = welch(signal, self.fs, nperseg=int(2*self.fs))
            
            # Limit to 0.5-40 Hz
            freq_mask = (freqs >= 0.5) & (freqs <= 40)
            freqs_limited = freqs[freq_mask]
            psd_limited = psd[freq_mask]
            
            # Extract band powers
            band_powers = {}
            total_power = np.sum(psd_limited)
            
            for band_name, (low, high) in self.bands.items():
                idx = (freqs_limited >= low) & (freqs_limited <= high)
                band_power_abs = np.sum(psd_limited[idx])
                band_power_rel = band_power_abs / total_power if total_power > 0 else 0
                
                band_powers[f'{band_name}_abs'] = band_power_abs
                band_powers[f'{band_name}_rel'] = band_power_rel
            
            # Ratios
            band_powers['theta_beta_ratio'] = (band_powers['theta_abs'] / 
                                               band_powers['beta_abs'] 
                                               if band_powers['beta_abs'] > 0 else 0)
            band_powers['alpha_theta_ratio'] = (band_powers['alpha_abs'] / 
                                                band_powers['theta_abs'] 
                                                if band_powers['theta_abs'] > 0 else 0)
            
            result = {
                'channel': ch_name,
                **band_powers
            }
            results.append(result)
        
        return pd.DataFrame(results)
    

    # 2. COMPLEXITY FEATURES (Entropy & Hjorth)

    
    def extract_complexity_features(self, data, channels):
        """
        Extract nonlinear complexity measures
        Returns: DataFrame with entropy and Hjorth parameters
        """
        results = []
        
        for ch_idx, ch_name in enumerate(channels):
            signal = data[ch_idx]
            
            features = {'channel': ch_name}
            
            # Sample Entropy
            try:
                features['sample_entropy'] = ant.sample_entropy(signal)
            except:
                features['sample_entropy'] = np.nan
            
            # Spectral Entropy
            try:
                features['spectral_entropy'] = ant.spectral_entropy(signal, sf=self.fs, method='welch')
            except:
                features['spectral_entropy'] = np.nan
            
            # Hjorth Parameters
            try:
                activity, mobility, complexity = self._compute_hjorth(signal)
                features['hjorth_activity'] = activity
                features['hjorth_mobility'] = mobility
                features['hjorth_complexity'] = complexity
            except:
                features['hjorth_activity'] = np.nan
                features['hjorth_mobility'] = np.nan
                features['hjorth_complexity'] = np.nan
            
            # Higuchi Fractal Dimension
            try:
                features['fractal_dimension'] = ant.higuchi_fd(signal)
            except:
                features['fractal_dimension'] = np.nan
            
            results.append(features)
        
        return pd.DataFrame(results)
    
    def _compute_hjorth(self, signal):
        """Compute Hjorth parameters"""
        activity = np.var(signal)
        diff1 = np.diff(signal)
        mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
        diff2 = np.diff(diff1)
        complexity = (np.sqrt(np.var(diff2) / np.var(diff1)) / mobility 
                     if mobility > 0 and np.var(diff1) > 0 else 0)
        return activity, mobility, complexity
    
    # 3. SLIDING WINDOW FEATURES (Temporal Dynamics)
    
    def extract_sliding_window_features(self, data, channels, window_sec=10, overlap=0.5):
        """
        Extract band powers over sliding windows to show temporal evolution
        Returns: DataFrame with time-series band powers
        """
        window_samples = int(window_sec * self.fs)
        step = int(window_samples * (1 - overlap))
        
        n_windows = (data.shape[1] - window_samples) // step + 1
        
        results = []
        
        for win_idx in range(n_windows):
            start = win_idx * step
            end = start + window_samples
            window_data = data[:, start:end]
            
            # Average across channels for global measure
            window_avg = np.mean(window_data, axis=0)
            
            freqs, psd = welch(window_avg, self.fs, nperseg=min(len(window_avg), int(2*self.fs)))
            
            window_features = {
                'window': win_idx,
                'time_sec': start / self.fs
            }
            
            for band_name, (low, high) in self.bands.items():
                idx = (freqs >= low) & (freqs <= high)
                window_features[f'{band_name}_power'] = np.sum(psd[idx])
            
            results.append(window_features)
        
        return pd.DataFrame(results)
    
    # 4. TIME-FREQUENCY FEATURES (Spectrogram Data) - for Visualization
    
    def extract_time_frequency_features(self, data, channel_idx=0):
        """
        Extract time-frequency representation for one channel
        Returns: freqs, times, spectrogram array
        """
        signal = data[channel_idx]
        
        # Convert to integers to avoid slice error
        nperseg = int(2 * self.fs)
        noverlap = int(1.5 * self.fs)
        
        f, t, Sxx = spectrogram(signal, self.fs, nperseg=nperseg, noverlap=noverlap)
        
        # Limit to 0.5-40 Hz
        freq_mask = (f >= 0.5) & (f <= 40)
        
        return f[freq_mask], t, Sxx[freq_mask, :]
    
    # 5. CONNECTIVITY FEATURES (Coherence)
    
    def extract_connectivity_features(self, data, channels):
        """
        Extract functional connectivity using coherence
        Returns: connectivity matrix (n_channels x n_channels)
        """
        n_channels = len(channels)
        conn_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                # Compute coherence between channels i and j
                f, Cxy = coherence(data[i], data[j], self.fs, nperseg=int(2*self.fs))
                
                # Average coherence in alpha band (8-13 Hz)
                alpha_mask = (f >= 8) & (f <= 13)
                conn_matrix[i, j] = np.mean(Cxy[alpha_mask])
                conn_matrix[j, i] = conn_matrix[i, j]
        
        return conn_matrix
    
    # 6. GRAPH THEORY FEATURES
    
    def extract_graph_metrics(self, conn_matrix, threshold=0.5):
        """
        Extract graph theory metrics from connectivity matrix
        Returns: dict with graph metrics
        """
        import networkx as nx
        
        # Threshold connectivity matrix
        binary_matrix = (conn_matrix > threshold).astype(int)
        np.fill_diagonal(binary_matrix, 0)
        
        # Create graph
        G = nx.from_numpy_array(binary_matrix)
        
        metrics = {}
        
        try:
            metrics['global_efficiency'] = nx.global_efficiency(G)
        except:
            metrics['global_efficiency'] = np.nan
        
        try:
            metrics['local_efficiency'] = nx.local_efficiency(G)
        except:
            metrics['local_efficiency'] = np.nan
        
        try:
            metrics['clustering_coefficient'] = nx.average_clustering(G)
        except:
            metrics['clustering_coefficient'] = np.nan
        
        try:
            if nx.is_connected(G):
                metrics['average_path_length'] = nx.average_shortest_path_length(G)
            else:
                metrics['average_path_length'] = np.nan
        except:
            metrics['average_path_length'] = np.nan
        
        return metrics
    
    # 7. MULTISCALE ENTROPY
    def extract_multiscale_entropy(self, data, channel_idx=0, max_scale=20):
        """
        Compute multiscale entropy
        Returns: DataFrame with entropy at each scale
        """
        signal = data[channel_idx]
        
        scales = []
        entropies = []
        
        for scale in range(1, max_scale + 1):
            try:
                # Coarse-grain signal
                coarse = self._coarse_grain(signal, scale)
                # Compute sample entropy
                se = ant.sample_entropy(coarse)
                scales.append(scale)
                entropies.append(se)
            except:
                scales.append(scale)
                entropies.append(np.nan)
        
        return pd.DataFrame({'scale': scales, 'entropy': entropies})
    
    def _coarse_grain(self, signal, scale):
        """Coarse-grain signal for MSE"""
        n = len(signal) // scale
        coarse = np.mean(signal[:n*scale].reshape(n, scale), axis=1)
        return coarse


# MAIN EXTRACTION PIPELINE
def extract_all_features_for_subject(data_dir, subject_id, output_dir='features_output'):
    """
    Extract all features for a single subject
    Saves separate CSVs for each feature type in subject-specific folder
    """
    from pathlib import Path
    
    # Create main output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create subject-specific subdirectory
    subject_path = output_path / subject_id
    subject_path.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"EXTRACTING FEATURES: {subject_id}")
    print(f"{'='*70}\n")
    
    # Load data
    loader = EEGDataLoader(data_dir)
    subject_data = loader.load_subject(subject_id, return_both=True)
    
    if not subject_data:
        print(f"Failed to load {subject_id}")
        return None
    
    pre_data = subject_data['baseline_data']
    post_data = subject_data['task_data']
    channels = subject_data['channels']
    fs = subject_data['fs']
    
    if pre_data is None or post_data is None:
        print(f"Missing pre or post data for {subject_id}")
        return None
    
    # Initialize extractor
    extractor = ComprehensiveFeatureExtractor(fs)
    
    # 1. PSD & Band Powers
    print("1. Extracting PSD and band powers...")
    psd_pre = extractor.extract_psd_features(pre_data, channels)
    psd_pre['condition'] = 'pre'
    psd_pre['subject'] = subject_id
    
    psd_post = extractor.extract_psd_features(post_data, channels)
    psd_post['condition'] = 'post'
    psd_post['subject'] = subject_id
    
    psd_combined = pd.concat([psd_pre, psd_post], ignore_index=True)
    psd_path = subject_path / f'{subject_id}_band_powers.csv'
    psd_combined.to_csv(psd_path, index=False)
    print(f"    Saved to {subject_id}/{psd_path.name}")
    
    # 2. Complexity Features
    print("2. Extracting complexity features...")
    complexity_pre = extractor.extract_complexity_features(pre_data, channels)
    complexity_pre['condition'] = 'pre'
    complexity_pre['subject'] = subject_id
    
    complexity_post = extractor.extract_complexity_features(post_data, channels)
    complexity_post['condition'] = 'post'
    complexity_post['subject'] = subject_id
    
    complexity_combined = pd.concat([complexity_pre, complexity_post], ignore_index=True)
    complexity_path = subject_path / f'{subject_id}_complexity.csv'
    complexity_combined.to_csv(complexity_path, index=False)
    print(f"   Saved to {subject_id}/{complexity_path.name}")
    
    # 3. Sliding Window Features (only for post/task data)
    print("3. Extracting sliding window features...")
    sliding_post = extractor.extract_sliding_window_features(post_data, channels)
    sliding_post['subject'] = subject_id
    sliding_path = subject_path / f'{subject_id}_sliding_window.csv'
    sliding_post.to_csv(sliding_path, index=False)
    print(f"   Saved to {subject_id}/{sliding_path.name}")
    
    # 4. Time-Frequency Data
    print("4. Extracting time-frequency features...")
    f_post, t_post, Sxx_post = extractor.extract_time_frequency_features(post_data, channel_idx=0)
    
    # Save as numpy arrays
    tf_path = subject_path / f'{subject_id}_timefreq.npz'
    np.savez(tf_path, freqs=f_post, times=t_post, spectrogram=Sxx_post)
    print(f"   Saved to {subject_id}/{tf_path.name}")
    
    # 5. Connectivity
    print("5. Extracting connectivity features...")
    conn_pre = extractor.extract_connectivity_features(pre_data, channels)
    conn_post = extractor.extract_connectivity_features(post_data, channels)
    
    conn_path_pre = subject_path / f'{subject_id}_connectivity_pre.npy'
    conn_path_post = subject_path / f'{subject_id}_connectivity_post.npy'
    np.save(conn_path_pre, conn_pre)
    np.save(conn_path_post, conn_post)
    print(f"   Saved to {subject_id}/{conn_path_pre.name} and {conn_path_post.name}")
    
    # 6. Graph Metrics
    print("6. Extracting graph metrics...")
    graph_pre = extractor.extract_graph_metrics(conn_pre)
    graph_pre['condition'] = 'pre'
    graph_pre['subject'] = subject_id
    
    graph_post = extractor.extract_graph_metrics(conn_post)
    graph_post['condition'] = 'post'
    graph_post['subject'] = subject_id
    
    graph_df = pd.DataFrame([graph_pre, graph_post])
    graph_path = subject_path / f'{subject_id}_graph_metrics.csv'
    graph_df.to_csv(graph_path, index=False)
    print(f"   Saved to {subject_id}/{graph_path.name}")
    
    # 7. Multiscale Entropy
    print("7. Extracting multiscale entropy...")
    mse_pre = extractor.extract_multiscale_entropy(pre_data, channel_idx=0)
    mse_pre['condition'] = 'pre'
    mse_pre['subject'] = subject_id
    
    mse_post = extractor.extract_multiscale_entropy(post_data, channel_idx=0)
    mse_post['condition'] = 'post'
    mse_post['subject'] = subject_id
    
    mse_combined = pd.concat([mse_pre, mse_post], ignore_index=True)
    mse_path = subject_path / f'{subject_id}_multiscale_entropy.csv'
    mse_combined.to_csv(mse_path, index=False)
    print(f"   Saved to {subject_id}/{mse_path.name}")
    
    print(f"\n{'='*70}")
    print(f"✓ FEATURE EXTRACTION COMPLETE: {subject_id}")
    print(f"{'='*70}\n")
    
    return {
        'subject': subject_id,
        'psd_path': str(psd_path),
        'complexity_path': str(complexity_path),
        'sliding_path': str(sliding_path),
        'graph_path': str(graph_path),
        'mse_path': str(mse_path)
    }


def batch_extract_all_subjects(data_dir, output_dir='features_output', n_subjects=8):
    """
    Extract features for random subset of subjects
    
    Parameters:
    -----------
    data_dir : str
        Path to dataset directory
    output_dir : str
        Output directory for features
    n_subjects : int
        Number of random subjects to process (default: 8)
        Set to None or very large number to process all subjects
    """
    import random
    
    loader = EEGDataLoader(data_dir)
    all_subjects = loader.get_all_subjects()
    
    # Select random subjects
    if n_subjects is None or n_subjects >= len(all_subjects):
        subjects = all_subjects
        print(f"\nProcessing all {len(subjects)} subjects")
    else:
        subjects = random.sample(all_subjects, n_subjects)
        print(f"\nRandomly selected {n_subjects} subjects from {len(all_subjects)} total")
    
    print(f"Selected subjects: {subjects}")
    print("Starting batch feature extraction...\n")
    
    results = []
    successful = 0
    failed = 0
    
    for idx, subject_id in enumerate(subjects):
        print(f"\n{'#'*70}")
        print(f"[{idx+1}/{len(subjects)}] Processing {subject_id}")
        print(f"{'#'*70}")
        
        try:
            result = extract_all_features_for_subject(data_dir, subject_id, output_dir)
            if result:
                results.append(result)
                successful += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ Error processing {subject_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"BATCH EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successfully processed: {successful}/{len(subjects)} subjects")
    if failed > 0:
        print(f"✗ Failed: {failed}/{len(subjects)} subjects")
    print(f"{'='*70}\n")
    
    return results



if __name__ == "__main__":
    # UPDATE THIS PATH
    DATA_DIR = ('Insert/Your/Path/Here')
    OUTPUT_DIR = "features_output"
    
   
    # To process all subjects: set n_subjects=None or n_subjects=999
    batch_extract_all_subjects(DATA_DIR, OUTPUT_DIR, n_subjects=None)\
    
    print("\n Feature extraction complete!")
    print(f" Results saved to: {OUTPUT_DIR}/")
    print("\nNext step: Run visualization.py to generate all plots")