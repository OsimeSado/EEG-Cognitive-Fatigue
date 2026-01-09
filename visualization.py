"""
Visualization Script
Generates all publication-quality pre vs post fatigue comparison plots
Reads CSV files from feature extraction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class FatigueVisualizer:
    """Generate all pre vs post fatigue visualization plots"""
    
    def __init__(self, features_dir='features_output', output_dir='visualization_output'):
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.colors = {
            'pre': '#2ecc71',   # Green
            'post': '#e74c3c'   # Red
        }
    

    # PLOT 1: BAND POWER STATISTICS (Bar Chart with Significance)

    
    def plot_band_power_statistics(self, subject_id=None, group_level=False):
        """
        Plot 1: Band Power Statistics
        Boxplots with significance stars and effect sizes
        """
        if group_level:
            # Load all subjects from subdirectories
            all_files = []
            for subject_dir in self.features_dir.iterdir():
                if subject_dir.is_dir():
                    band_file = subject_dir / f'{subject_dir.name}_band_powers.csv'
                    if band_file.exists():
                        all_files.append(band_file)
            
            df_list = [pd.read_csv(f) for f in all_files]
            df = pd.concat(df_list, ignore_index=True)
            title_prefix = "Group"
            save_name = "group_band_power_stats.png"
        else:
            # Single subject - look in subject subdirectory
            subject_dir = self.features_dir / subject_id
            file_path = subject_dir / f'{subject_id}_band_powers.csv'
            df = pd.read_csv(file_path)
            title_prefix = subject_id
            save_name = f"{subject_id}_band_power_stats.png"
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Select bands to plot
        bands = ['delta', 'theta', 'alpha', 'beta']
        
        # Prepare data for plotting
        plot_data = []
        for band in bands:
            for _, row in df.iterrows():
                plot_data.append({
                    'Band': band.capitalize(),
                    'Power': row[f'{band}_abs'],
                    'Condition': row['condition'].capitalize()
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Plot boxplot
        sns.boxplot(data=plot_df, x='Band', y='Power', hue='Condition',
                   palette={'Pre': self.colors['pre'], 'Post': self.colors['post']},
                   ax=axes[0])
        axes[0].set_ylabel('Absolute Power (µV²)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
        axes[0].set_title(f'{title_prefix} - Band Power Comparison', 
                         fontsize=13, fontweight='bold')
        axes[0].legend(title='', fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add significance stars
        for idx, band in enumerate(bands):
            pre_vals = df[df['condition'] == 'pre'][f'{band}_abs'].values
            post_vals = df[df['condition'] == 'post'][f'{band}_abs'].values
            
            t_stat, p_val = stats.ttest_ind(pre_vals, post_vals)
            
            y_max = max(plot_df[plot_df['Band'] == band.capitalize()]['Power'])
            
            if p_val < 0.001:
                axes[0].text(idx, y_max * 1.1, '***', ha='center', 
                           fontsize=14, fontweight='bold')
            elif p_val < 0.01:
                axes[0].text(idx, y_max * 1.1, '**', ha='center', 
                           fontsize=14, fontweight='bold')
            elif p_val < 0.05:
                axes[0].text(idx, y_max * 1.1, '*', ha='center', 
                           fontsize=14, fontweight='bold')
        
        # Plot effect sizes
        effect_sizes = []
        for band in bands:
            pre_vals = df[df['condition'] == 'pre'][f'{band}_abs'].values
            post_vals = df[df['condition'] == 'post'][f'{band}_abs'].values
            
            pooled_std = np.sqrt((np.var(pre_vals) + np.var(post_vals)) / 2)
            cohens_d = (np.mean(post_vals) - np.mean(pre_vals)) / pooled_std
            effect_sizes.append(cohens_d)
        
        colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in effect_sizes]
        axes[1].barh([b.capitalize() for b in bands], effect_sizes, 
                    color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
        axes[1].set_xlabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency Band', fontsize=12, fontweight='bold')
        axes[1].set_title('Effect Sizes (Post - Pre)', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved: {save_name}")
    

    # PLOT 2: TOPOGRAPHIC BAND POWER MAPS (with proper topoplot style)

    
    def plot_topographic_maps(self, subject_id):
        """
        Plot 2: Topographic band power distribution (proper topoplots)
        """
        subject_dir = self.features_dir / subject_id
        file_path = subject_dir / f'{subject_id}_band_powers.csv'
        df = pd.read_csv(file_path)
        
        bands = ['delta', 'theta', 'alpha', 'beta']
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        
        for band_idx, band in enumerate(bands):
            # Pre-fatigue (top row)
            ax_pre = fig.add_subplot(gs[0, band_idx])
            pre_data = df[df['condition'] == 'pre']
            pre_powers = pre_data[f'{band}_abs'].values
            
            # Create circular topoplot
            self._plot_topomap_circle(ax_pre, pre_powers, 
                                     title=f'{band.capitalize()} - Pre',
                                     cmap='RdYlBu_r')
            
            # Post-fatigue (bottom row)
            ax_post = fig.add_subplot(gs[1, band_idx])
            post_data = df[df['condition'] == 'post']
            post_powers = post_data[f'{band}_abs'].values
            
            self._plot_topomap_circle(ax_post, post_powers,
                                     title=f'{band.capitalize()} - Post',
                                     cmap='RdYlBu_r')
        
        plt.suptitle(f'{subject_id} - Topographic Band Power Distribution',
                    fontsize=16, fontweight='bold', y=0.98)
        
        save_name = f"{subject_id}_topographic_maps.png"
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved: {save_name}")
    
    def _plot_topomap_circle(self, ax, values, title='', cmap='RdYlBu_r'):
        # Normalize values
        vmin, vmax = np.min(values), np.max(values)
        norm_values = (values - vmin) / (vmax - vmin) if vmax > vmin else values
        
        # Create circular arrangement (simple 23-channel layout)
        n = len(values)
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radius = 0.8
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        # Create interpolated surface
        from scipy.interpolate import griddata
        
        # Create grid for interpolation
        xi = np.linspace(-1, 1, 100)
        yi = np.linspace(-1, 1, 100)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate
        Zi = griddata((x, y), values, (Xi, Yi), method='cubic')
        
        # Mask outside circle
        mask = Xi**2 + Yi**2 > 1
        Zi[mask] = np.nan
        
        # Plot
        im = ax.contourf(Xi, Yi, Zi, levels=20, cmap=cmap)
        
        # Add electrode positions
        ax.scatter(x, y, c='black', s=20, zorder=10, edgecolors='white', linewidths=0.5)
        
        # Draw head outline
        circle = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Draw nose
        nose_width = 0.1
        nose_height = 0.15
        nose = plt.Polygon([[-nose_width, 1], [0, 1 + nose_height], [nose_width, 1]], 
                          closed=True, edgecolor='black', facecolor='white', linewidth=2)
        ax.add_patch(nose)
        
        # Draw ears
        ear_left = plt.Circle((-1, 0), 0.1, fill=False, edgecolor='black', linewidth=2)
        ear_right = plt.Circle((1, 0), 0.1, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(ear_left)
        ax.add_patch(ear_right)
        
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        # Add colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
    

    # PLOT 2B: PRE VS POST BAR CHART COMPARISON (like your image)

    
    def plot_pre_post_bar_comparison(self, subject_id=None, group_level=False):
        """Plot 2B: Pre vs Post Fatigue Band Power Comparison (Bar Chart)"""
        if group_level:
            all_files = []
            for subject_dir in self.features_dir.iterdir():
                if subject_dir.is_dir():
                    band_file = subject_dir / f'{subject_dir.name}_band_powers.csv'
                    if band_file.exists():
                        all_files.append(band_file)
            
            df_list = [pd.read_csv(f) for f in all_files]
            df = pd.concat(df_list, ignore_index=True)
            title_prefix = "Group"
            save_name = "group_pre_post_comparison.png"
        else:
            subject_dir = self.features_dir / subject_id
            file_path = subject_dir / f'{subject_id}_band_powers.csv'
            df = pd.read_csv(file_path)
            title_prefix = subject_id
            save_name = f"{subject_id}_pre_post_comparison.png"
        
        # Calculate means per condition
        bands = ['alpha', 'theta', 'beta', 'gamma']
        
        # Add theta/beta ratio
        pre_data = df[df['condition'] == 'pre']
        post_data = df[df['condition'] == 'post']
        
        pre_means = []
        post_means = []
        labels = []
        
        for band in bands:
            pre_means.append(pre_data[f'{band}_abs'].mean())
            post_means.append(post_data[f'{band}_abs'].mean())
            labels.append(f'{band}_abs')
        
        # Add theta/beta ratio
        pre_means.append(pre_data['theta_beta_ratio'].mean())
        post_means.append(post_data['theta_beta_ratio'].mean())
        labels.append('theta_beta_ratio')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(labels))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, pre_means, width, 
                      label='Pre-Fatigue', color='#2ecc71', alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, post_means, width,
                      label='Post-Fatigue', color='#e74c3c', alpha=0.8,
                      edgecolor='black', linewidth=1.5)
        
        # Customize
        ax.set_ylabel('Power (µV²)', fontsize=14, fontweight='bold')
        ax.set_title(f'{title_prefix} - Pre vs Post Fatigue Comparison',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=12)
        ax.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved: {save_name}")
    

    # PLOT 3: COMPLEXITY FEATURES (Boxplots)

    
    def plot_complexity_features(self, subject_id=None, group_level=False):
        """Plot 3: Nonlinear complexity measures"""
        if group_level:
            all_files = []
            for subject_dir in self.features_dir.iterdir():
                if subject_dir.is_dir():
                    complex_file = subject_dir / f'{subject_dir.name}_complexity.csv'
                    if complex_file.exists():
                        all_files.append(complex_file)
            
            df_list = [pd.read_csv(f) for f in all_files]
            df = pd.concat(df_list, ignore_index=True)
            title_prefix = "Group"
            save_name = "group_complexity_features.png"
        else:
            subject_dir = self.features_dir / subject_id
            file_path = subject_dir / f'{subject_id}_complexity.csv'
            df = pd.read_csv(file_path)
            title_prefix = subject_id
            save_name = f"{subject_id}_complexity_features.png"
        
        features = ['sample_entropy', 'spectral_entropy', 'hjorth_complexity', 'fractal_dimension']
        feature_labels = ['Sample Entropy', 'Spectral Entropy', 'Hjorth Complexity', 'Fractal Dimension']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (feat, label) in enumerate(zip(features, feature_labels)):
            plot_data = []
            for _, row in df.iterrows():
                if not np.isnan(row[feat]):
                    plot_data.append({
                        'Value': row[feat],
                        'Condition': row['condition'].capitalize()
                    })
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                
                sns.boxplot(data=plot_df, x='Condition', y='Value',
                           palette={'Pre': self.colors['pre'], 'Post': self.colors['post']},
                           ax=axes[idx])
                axes[idx].set_ylabel(label, fontsize=11, fontweight='bold')
                axes[idx].set_xlabel('')
                axes[idx].set_title(label, fontsize=12, fontweight='bold')
                axes[idx].grid(True, alpha=0.3, axis='y')
                
                # Statistical test
                pre_vals = plot_df[plot_df['Condition'] == 'Pre']['Value'].values
                post_vals = plot_df[plot_df['Condition'] == 'Post']['Value'].values
                
                if len(pre_vals) > 0 and len(post_vals) > 0:
                    t_stat, p_val = stats.ttest_ind(pre_vals, post_vals)
                    
                    y_max = plot_df['Value'].max()
                    if p_val < 0.05:
                        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                        axes[idx].text(0.5, y_max * 1.05, sig, ha='center',
                                     fontsize=14, fontweight='bold')
        
        plt.suptitle(f'{title_prefix} - Complexity Features (Pre vs Post)',
                    fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_name}")
    

    # PLOT 4: TIME-FREQUENCY SPECTROGRAM

    
    def plot_time_frequency_spectrogram(self, subject_id):
        """Plot 4: Spectrogram showing temporal evolution"""
        subject_dir = self.features_dir / subject_id
        tf_path = subject_dir / f'{subject_id}_timefreq.npz'
        
        if not tf_path.exists():
            print(f" Time-frequency data not found for {subject_id}")
            return
        
        data = np.load(tf_path)
        freqs = data['freqs']
        times = data['times']
        Sxx = data['spectrogram']
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        im = ax.pcolormesh(times, freqs, 10 * np.log10(Sxx),
                          shading='gouraud', cmap='jet')
        
        ax.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_title(f'{subject_id} - Time-Frequency Spectrogram (Task)',
                    fontsize=14, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontsize=11, fontweight='bold')
        
        # Mark band boundaries
        bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30)}
        for band_name, (low, high) in bands.items():
            ax.axhline(y=low, color='white', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(times[-1] * 0.95, (low + high) / 2, band_name,
                   color='white', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        plt.tight_layout()
        
        save_name = f"{subject_id}_spectrogram.png"
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved: {save_name}")
    

    # PLOT 5: SLIDING WINDOW DYNAMICS

    
    def plot_sliding_window_dynamics(self, subject_id):
        """Plot 5: Band power evolution over time"""
        subject_dir = self.features_dir / subject_id
        file_path = subject_dir / f'{subject_id}_sliding_window.csv'
        df = pd.read_csv(file_path)
        
        bands = ['delta', 'theta', 'alpha', 'beta']
        colors_band = ['#9b59b6', '#e67e22', '#3498db', '#1abc9c']
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        for idx, (band, color) in enumerate(zip(bands, colors_band)):
            axes[idx].plot(df['time_sec'], df[f'{band}_power'],
                          linewidth=2.5, color=color, alpha=0.8, label=band.capitalize())
            axes[idx].fill_between(df['time_sec'], df[f'{band}_power'],
                                  alpha=0.2, color=color)
            
            # Add trend line
            z = np.polyfit(df['time_sec'], df[f'{band}_power'], 1)
            p = np.poly1d(z)
            axes[idx].plot(df['time_sec'], p(df['time_sec']), "--",
                          color='black', linewidth=1.5, alpha=0.5, label='Trend')
            
            axes[idx].set_ylabel(f'{band.capitalize()} Power', fontsize=11, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend(loc='best', fontsize=9)
            
            if idx == len(bands) - 1:
                axes[idx].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        
        plt.suptitle(f'{subject_id} - Sliding Window Band Power Dynamics',
                    fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_name = f"{subject_id}_sliding_window_dynamics.png"
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved: {save_name}")
    

    # PLOT 6: FUNCTIONAL CONNECTIVITY HEATMAPS

    
    def plot_connectivity_heatmaps(self, subject_id):
        """Plot 6: Connectivity matrices"""
        subject_dir = self.features_dir / subject_id
        conn_pre_path = subject_dir / f'{subject_id}_connectivity_pre.npy'
        conn_post_path = subject_dir / f'{subject_id}_connectivity_post.npy'
        
        if not (conn_pre_path.exists() and conn_post_path.exists()):
            print(f" Connectivity data not found for {subject_id}")
            return
        
        conn_pre = np.load(conn_pre_path)
        conn_post = np.load(conn_post_path)
        conn_diff = conn_post - conn_pre
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        mask = np.eye(conn_pre.shape[0], dtype=bool)
        
        # Pre-fatigue
        sns.heatmap(conn_pre, mask=mask, cmap='RdYlBu_r', center=0.5,
                   square=True, linewidths=0.5, ax=axes[0],
                   cbar_kws={"label": "Coherence", "shrink": 0.8},
                   vmin=0, vmax=1)
        axes[0].set_title('Pre-Fatigue Connectivity', fontsize=13, fontweight='bold')
        
        # Post-fatigue
        sns.heatmap(conn_post, mask=mask, cmap='RdYlBu_r', center=0.5,
                   square=True, linewidths=0.5, ax=axes[1],
                   cbar_kws={"label": "Coherence", "shrink": 0.8},
                   vmin=0, vmax=1)
        axes[1].set_title('Post-Fatigue Connectivity', fontsize=13, fontweight='bold')
        
        # Difference
        vmax = np.max(np.abs(conn_diff))
        sns.heatmap(conn_diff, mask=mask, cmap='RdBu_r', center=0,
                   vmin=-vmax, vmax=vmax,
                   square=True, linewidths=0.5, ax=axes[2],
                   cbar_kws={"label": "Δ Coherence", "shrink": 0.8})
        axes[2].set_title('Connectivity Change', fontsize=13, fontweight='bold')
        
        plt.suptitle(f'{subject_id} - Functional Connectivity Analysis',
                    fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_name = f"{subject_id}_connectivity_heatmaps.png"
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved: {save_name}")
    

    # PLOT 7: GRAPH THEORY METRICS

    
    def plot_graph_metrics(self, subject_id=None, group_level=False):
        """Plot 7: Graph theory metrics comparison"""
        if group_level:
            all_files = []
            for subject_dir in self.features_dir.iterdir():
                if subject_dir.is_dir():
                    graph_file = subject_dir / f'{subject_dir.name}_graph_metrics.csv'
                    if graph_file.exists():
                        all_files.append(graph_file)
            
            df_list = [pd.read_csv(f) for f in all_files]
            df = pd.concat(df_list, ignore_index=True)
            title_prefix = "Group"
            save_name = "group_graph_metrics.png"
        else:
            subject_dir = self.features_dir / subject_id
            file_path = subject_dir / f'{subject_id}_graph_metrics.csv'
            df = pd.read_csv(file_path)
            title_prefix = subject_id
            save_name = f"{subject_id}_graph_metrics.png"
        
        metrics = ['global_efficiency', 'local_efficiency', 'clustering_coefficient', 'average_path_length']
        metric_labels = ['Global Efficiency', 'Local Efficiency', 'Clustering Coefficient', 'Average Path Length']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            pre_val = df[df['condition'] == 'pre'][metric].values
            post_val = df[df['condition'] == 'post'][metric].values
            
            pre_val = pre_val[~np.isnan(pre_val)]
            post_val = post_val[~np.isnan(post_val)]
            
            if len(pre_val) > 0 and len(post_val) > 0:
                axes[idx].bar(['Pre', 'Post'], [np.mean(pre_val), np.mean(post_val)],
                             color=[self.colors['pre'], self.colors['post']], alpha=0.7,
                             edgecolor='black', linewidth=2)
                axes[idx].set_title(label, fontsize=11, fontweight='bold')
                axes[idx].grid(True, alpha=0.3, axis='y')
                
                pct_change = ((np.mean(post_val) - np.mean(pre_val)) / np.mean(pre_val)) * 100
                axes[idx].text(1, np.mean(post_val) * 1.05, f'{pct_change:+.1f}%',
                             ha='center', fontsize=10, fontweight='bold')
        
        plt.suptitle(f'{title_prefix} - Graph Theory Metrics',
                    fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved: {save_name}")
    

    # PLOT 8: MULTISCALE ENTROPY CURVES
 
    
    def plot_multiscale_entropy(self, subject_id=None, group_level=False):
        """Plot 8: Multiscale entropy curves"""
        if group_level:
            all_files = []
            for subject_dir in self.features_dir.iterdir():
                if subject_dir.is_dir():
                    mse_file = subject_dir / f'{subject_dir.name}_multiscale_entropy.csv'
                    if mse_file.exists():
                        all_files.append(mse_file)
            
            df_list = [pd.read_csv(f) for f in all_files]
            df = pd.concat(df_list, ignore_index=True)
            title_prefix = "Group"
            save_name = "group_multiscale_entropy.png"
        else:
            subject_dir = self.features_dir / subject_id
            file_path = subject_dir / f'{subject_id}_multiscale_entropy.csv'
            df = pd.read_csv(file_path)
            title_prefix = subject_id
            save_name = f"{subject_id}_multiscale_entropy.png"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pre_data = df[df['condition'] == 'pre']
        ax.plot(pre_data['scale'], pre_data['entropy'], 'o-',
               linewidth=2.5, markersize=8, label='Pre-Fatigue',
               color=self.colors['pre'], alpha=0.8)
        
        post_data = df[df['condition'] == 'post']
        ax.plot(post_data['scale'], post_data['entropy'], 's-',
               linewidth=2.5, markersize=8, label='Post-Fatigue',
               color=self.colors['post'], alpha=0.8)
        
        ax.set_xlabel('Scale Factor', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sample Entropy', fontsize=12, fontweight='bold')
        ax.set_title(f'{title_prefix} - Multiscale Entropy Analysis',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved: {save_name}")
    
   
    # PLOT 9: FEATURE SPACE CLUSTERING (PCA)
   
    
    def plot_feature_space_clustering(self, group_level=True):
        """Plot 9: PCA visualization"""
        if not group_level:
            print(" Feature space clustering requires group-level data")
            return
        
        all_files = []
        for subject_dir in self.features_dir.iterdir():
            if subject_dir.is_dir():
                band_file = subject_dir / f'{subject_dir.name}_band_powers.csv'
                if band_file.exists():
                    all_files.append(band_file)
        
        df_list = [pd.read_csv(f) for f in all_files]
        df = pd.concat(df_list, ignore_index=True)
        
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        feature_cols = ['delta_abs', 'theta_abs', 'alpha_abs', 'beta_abs',
                       'theta_beta_ratio', 'alpha_theta_ratio']
        
        agg_data = df.groupby(['subject', 'condition'])[feature_cols].mean().reset_index()
        
        X = agg_data[feature_cols].values
        labels = agg_data['condition'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for label, color in [('pre', self.colors['pre']), ('post', self.colors['post'])]:
            mask = labels == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      label=label.capitalize(), alpha=0.6, s=150,
                      color=color, edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                     fontsize=11, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                     fontsize=11, fontweight='bold')
        ax.set_title('PCA Feature Space: Pre vs Post Clustering',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_name = "group_pca_clustering.png"
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved: {save_name}")
    

    # GENERATE ALL PLOTS

    
    def generate_all_plots(self, subject_id):
        """Generate all plots for a single subject"""
        print(f"\n{'='*70}")
        print(f"GENERATING PLOTS: {subject_id}")
        print(f"{'='*70}\n")
        
        self.plot_band_power_statistics(subject_id=subject_id)
        self.plot_topographic_maps(subject_id)
        self.plot_pre_post_bar_comparison(subject_id=subject_id)
        self.plot_complexity_features(subject_id=subject_id)
        self.plot_time_frequency_spectrogram(subject_id)
        self.plot_sliding_window_dynamics(subject_id)
        self.plot_connectivity_heatmaps(subject_id)
        self.plot_graph_metrics(subject_id=subject_id)
        self.plot_multiscale_entropy(subject_id=subject_id)
        
        print(f"\n{'='*70}")
        print(f"COMPLETE: {subject_id}")
        print(f"{'='*70}\n")
    
    def generate_group_plots(self):
        """Generate group-level plots"""
        print(f"\n{'='*70}")
        print(f"GENERATING GROUP-LEVEL PLOTS")
        print(f"{'='*70}\n")
        
        self.plot_band_power_statistics(group_level=True)
        self.plot_pre_post_bar_comparison(group_level=True)
        self.plot_complexity_features(group_level=True)
        self.plot_graph_metrics(group_level=True)
        self.plot_multiscale_entropy(group_level=True)
        self.plot_feature_space_clustering(group_level=True)
        
        print(f"\n{'='*70}")
        print(f"GROUP PLOTS COMPLETE")
        print(f"{'='*70}\n")


# MAIN EXECUTION

if __name__ == "__main__":
    FEATURES_DIR = "features_output"
    OUTPUT_DIR = "visualization_output"
    
    visualizer = FatigueVisualizer(FEATURES_DIR, OUTPUT_DIR)
    
    # Get all subjects from subdirectories
    subject_dirs = [d for d in Path(FEATURES_DIR).iterdir() if d.is_dir() and d.name.startswith('Subject')]
    subjects = sorted([d.name for d in subject_dirs])
    
    if subjects:
        print(f"\nFound {len(subjects)} subjects with extracted features")
        
        # Generate plots for each subject
        for subject_id in subjects:
            visualizer.generate_all_plots(subject_id)
        
        # Generate group-level plots
        visualizer.generate_group_plots()
        
        print(f"\n All visualizations saved to {OUTPUT_DIR}/")
    else:
        print("\n No extracted features found. Run feature_extraction.py first.")