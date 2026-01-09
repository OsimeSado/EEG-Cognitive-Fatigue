"""
EEG Data Loading Module
Loads EDF files for cognitive fatigue analysis
"""

import numpy as np
import pandas as pd
import pyedflib
from pathlib import Path
import matplotlib.pyplot as plt

class EEGDataLoader:
    """Load and manage EEG data from EDF files"""
    
    def __init__(self, data_directory):
        """
        Initialize data loader
        
        Parameters:
        -----------
        data_directory : str
            Path to directory containing EDF files and subject-info.csv
        """
        self.data_dir = Path(data_directory)
        self.subject_info = None
        self.load_subject_info()
        
    def load_subject_info(self):
        """Load subject information CSV"""
        info_path = self.data_dir / 'subject-info.csv'
        if info_path.exists():
            self.subject_info = pd.read_csv(info_path)
            print(f" Loaded info for {len(self.subject_info)} subjects")
            # Determine group based on 'Count quality'
            if 'Count quality' in self.subject_info.columns:
                print(f"  - Good performers (Group G): {(self.subject_info['Count quality'] == 1).sum()}")
                print(f"  - Poor performers (Group B): {(self.subject_info['Count quality'] == 0).sum()}")
        else:
            print(f" Warning: subject-info.csv not found at {info_path}")
            
    def load_eeg_file(self, filepath):
        """Load single EDF file"""
        try:
            f = pyedflib.EdfReader(str(filepath))
            
            # Get metadata
            n_channels = f.signals_in_file
            channel_names = f.getSignalLabels()
            sampling_rate = f.getSampleFrequency(0)
            n_samples = f.getNSamples()[0]
            
            # Read all signals
            signals = np.zeros((n_channels, n_samples))
            for i in range(n_channels):
                signals[i, :] = f.readSignal(i)
            
            f.close()
            
            duration = n_samples / sampling_rate
            
            return signals, channel_names, sampling_rate, duration
            
        except Exception as e:
            print(f" Error loading {filepath.name}: {str(e)}")
            return None, None, None, None
    
    def load_subject(self, subject_id, return_both=True):
        """
        Load both baseline and task EEG for a subject
        
        Parameters:
        -----------
        subject_id : str or int
            Subject ID. 
            If int (e.g., 0), converts to 'Subject00'. 
            If str (e.g., 'Subject00'), uses as is.
        """
        if isinstance(subject_id, int):
            # Dataset uses Subject00, Subject01, etc.
            subject_id = f"Subject{subject_id:02d}"
        
        # File paths (Using the _1 and _2 suffixes correctly)
        baseline_file = self.data_dir / f"{subject_id}_1.edf"
        task_file = self.data_dir / f"{subject_id}_2.edf"
        
        result = {
            'subject_id': subject_id,
            'baseline_data': None,
            'task_data': None,
            'channels': None,
            'fs': None,
            'baseline_duration': None,
            'task_duration': None,
            'subject_info': None
        }
        
        # Load baseline
        if return_both:
            if baseline_file.exists():
                baseline_data, channels, fs, duration = self.load_eeg_file(baseline_file)
                if baseline_data is not None:
                    result['baseline_data'] = baseline_data
                    result['channels'] = channels
                    result['fs'] = fs
                    result['baseline_duration'] = duration
                    print(f" Loaded baseline ({baseline_file.name}): {duration:.1f}s")
            else:
                print(f" Baseline file not found: {baseline_file.name}")
        
        # Load task
        if task_file.exists():
            task_data, channels, fs, duration = self.load_eeg_file(task_file)
            if task_data is not None:
                result['task_data'] = task_data
                # If baseline wasn't loaded, get channels/fs from here
                if result['channels'] is None:
                    result['channels'] = channels
                    result['fs'] = fs
                result['task_duration'] = duration
                print(f" Loaded task ({task_file.name}): {duration:.1f}s")
        else:
            print(f" Task file not found: {task_file}")
            return None
        
        # Get subject info
        if self.subject_info is not None:
            # Check if 'Subject' column exists and match
            if 'Subject' in self.subject_info.columns:
                subject_row = self.subject_info[self.subject_info['Subject'] == subject_id]
                if not subject_row.empty:
                    result['subject_info'] = subject_row.iloc[0].to_dict()
        
        return result
    
    def get_all_subjects(self):
        """Get list of all available subject IDs from file system"""
        # Look for files matching Subject*_2.edf
        edf_files = list(self.data_dir.glob("Subject*_2.edf"))
        
        # Extract "Subject00", "Subject01", etc.
        subjects = sorted([f.stem.split('_')[0] for f in edf_files])
        return subjects
    
    def plot_raw_eeg(self, data, channels, fs, duration_sec=10, title="Raw EEG"):
        """Plot raw EEG signals"""
        if data is None:
            print("No data to plot.")
            return
            
        n_samples_plot = int(duration_sec * fs)
        # Ensure we don't try to plot more samples than exist
        n_samples_plot = min(n_samples_plot, data.shape[1])
        
        data_plot = data[:, :n_samples_plot]
        time = np.arange(n_samples_plot) / fs
        
        # Determine how many channels to plot (max 8)
        num_channels_to_plot = min(8, len(channels))
        
        fig, axes = plt.subplots(num_channels_to_plot, 1, figsize=(15, 12), sharex=True)
        if num_channels_to_plot == 1:
            axes = [axes]
        
        for i in range(num_channels_to_plot):
            ax = axes[i]
            # Offset the data for better visibility if needed, or plot raw
            ax.plot(time, data_plot[i], linewidth=0.8, color='#2c3e50')
            ax.set_ylabel(channels[i], rotation=0, ha='right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Remove spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if i == 0:
                ax.set_title(title, fontsize=12, fontweight='bold')
            if i == num_channels_to_plot - 1:
                ax.set_xlabel('Time (s)', fontsize=10)
        
        plt.tight_layout()
        return fig

    def print_summary(self, subject_data):
        """Print summary of loaded data"""
        if not subject_data:
            return

        print("\n" + "="*60)
        print(f"SUBJECT: {subject_data['subject_id']}")
        print("="*60)
        
        if subject_data['subject_info']:
            info = subject_data['subject_info']
            print(f"Gender: {info.get('Gender', 'N/A')}")
            print(f"Age: {info.get('Age', 'N/A')}")
            
            # Helper to interpret group
            perf = info.get('Count quality')
            if perf == 1:
                perf_str = "Good (Group G)"
            elif perf == 0:
                perf_str = "Poor (Group B)"
            else:
                perf_str = "Unknown"
            print(f"Performance: {perf_str}")
        
        if subject_data.get('channels'):
            print(f"\nChannels: {len(subject_data['channels'])}")
        
        if subject_data.get('fs'):
            print(f"Sampling Rate: {subject_data['fs']} Hz")
        
        if subject_data['baseline_data'] is not None:
            print(f"\nBaseline:")
            print(f"  Duration: {subject_data['baseline_duration']:.1f}s")
            print(f"  Shape: {subject_data['baseline_data'].shape}")
        
        if subject_data['task_data'] is not None:
            print(f"\nTask:")
            print(f"  Duration: {subject_data['task_duration']:.1f}s")
            print(f"  Shape: {subject_data['task_data'].shape}")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    # 1. UPDATE THIS PATH TO YOUR ACTUAL FOLDER
    data_path = ('Insert/Your/Path/Here')
    
    loader = EEGDataLoader(data_path)
    
    # 2. Get available subjects automatically
    subjects = loader.get_all_subjects()
    
    if len(subjects) > 0:
        print(f"Found {len(subjects)} subjects. First 5: {subjects[:5]}")
        
        # 3. Load the FIRST available subject dynamically
        subject_to_load = subjects[0] 
        print(f"\nAttempting to load: {subject_to_load}")
        
        subject_data = loader.load_subject(subject_to_load)
        
        if subject_data:
            loader.print_summary(subject_data)
            
            if subject_data['task_data'] is not None:
                fig = loader.plot_raw_eeg(
                    subject_data['task_data'],
                    subject_data['channels'],
                    subject_data['fs'],
                    duration_sec=5,
                    title=f"Raw EEG - {subject_data['subject_id']} - Task"
                )
                plt.show()
    else:
        print("No subjects found in directory. Check your path or filenames.")