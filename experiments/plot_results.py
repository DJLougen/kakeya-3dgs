#!/usr/bin/env python3
"""Generate figures for Paper 1: Kakeya-Inspired Load Balancing for 3D Gaussian Splatting."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import matplotlib.pyplot as plt
import numpy as np

# Paths
REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / 'results'
FIGURES_DIR = REPO_ROOT / 'paper' / 'figures'

# Set style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


def plot_load_balance():
    """Figure 1: Load balance comparison across scene types."""
    with open('results/exp1_load_balance.json') as f:
        data = json.load(f)
    
    scenes = list(data.keys())
    baseline = [data[s]['baseline_balance'] for s in scenes]
    polynomial = [data[s]['polynomial_balance'] for s in scenes]
    
    x = np.arange(len(scenes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline', color='#d62728', alpha=0.8)
    bars2 = ax.bar(x + width/2, polynomial, width, label='Polynomial Partitioning', color='#2ca02c', alpha=0.8)
    
    ax.set_xlabel('Scene Type', fontsize=12)
    ax.set_ylabel('Load Balance Ratio (lower is better)', fontsize=12)
    ax.set_title('Load Balance Improvement: Baseline vs Polynomial Partitioning', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in scenes])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(baseline) * 1.2)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}x', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('paper/figures/load_balance.png', bbox_inches='tight')
    plt.close()
    print("Saved: load_balance.png")


def plot_algebraic_detection():
    """Figure 2: Algebraic case detection results."""
    with open('results/exp2_algebraic_case.json') as f:
        data = json.load(f)
    
    scenes = list(data.keys())
    concentrations = [data[s]['concentration'] * 100 for s in scenes]
    correct = [data[s]['correct'] for s in scenes]
    colors = ['#2ca02c' if c else '#d62728' for c in correct]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(scenes, concentrations, color=colors, alpha=0.8, edgecolor='black')
    
    ax.axhline(y=65, color='gray', linestyle='--', linewidth=2, label='Detection Threshold (65%)')
    ax.set_xlabel('Scene Type', fontsize=12)
    ax.set_ylabel('Concentration Score (%)', fontsize=12)
    ax.set_title('Algebraic Case Detection: Concentration on Polynomial Surfaces', fontsize=14, fontweight='bold')
    ax.set_xticklabels([s.upper() for s in scenes])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 110)
    
    # Add value labels and markers
    for i, (bar, conc, corr) in enumerate(zip(bars, concentrations, correct)):
        height = bar.get_height()
        marker = '✓' if corr else '✗'
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{conc:.1f}%\n{marker}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('paper/figures/algebraic_detection.png', bbox_inches='tight')
    plt.close()
    print("Saved: algebraic_detection.png")


def plot_scalability():
    """Figure 3: Scalability with Gaussian count."""
    with open('results/exp3_scalability.json') as f:
        data = json.load(f)
    
    counts = [int(k) for k in data.keys()]
    baseline = [data[str(c)]['baseline_balance'] for c in counts]
    polynomial = [data[str(c)]['polynomial_balance'] for c in counts]
    improvement = [data[str(c)]['improvement_ratio'] for c in counts]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: balance ratios
    ax1.plot(counts, baseline, 'o-', label='Baseline', color='#d62728', linewidth=2, markersize=8)
    ax1.plot(counts, polynomial, 's-', label='Polynomial', color='#2ca02c', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Gaussians', fontsize=12)
    ax1.set_ylabel('Load Balance Ratio', fontsize=12)
    ax1.set_title('Balance Ratio vs Scene Size', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Right: improvement ratio
    ax2.plot(counts, improvement, 'D-', color='#1f77b4', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Gaussians', fontsize=12)
    ax2.set_ylabel('Improvement Factor', fontsize=12)
    ax2.set_title('Balance Improvement Factor', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Add annotations
    for i, (c, imp) in enumerate(zip(counts, improvement)):
        ax2.annotate(f'{imp:.1f}x', (c, imp), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('paper/figures/scalability.png', bbox_inches='tight')
    plt.close()
    print("Saved: scalability.png")


def plot_overhead_tradeoff():
    """Figure 4: Overhead vs balance improvement tradeoff."""
    with open('results/exp4_overhead.json') as f:
        data = json.load(f)
    
    scenes = list(data.keys())
    overhead = [data[s]['overhead_ratio'] for s in scenes]
    balance_gain = [data[s]['balance_improvement'] for s in scenes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(scenes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, overhead, width, label='Overhead (slower)', color='#ff7f0e', alpha=0.8)
    bars2 = ax.bar(x + width/2, balance_gain, width, label='Balance Gain (better)', color='#9467bd', alpha=0.8)
    
    ax.set_xlabel('Scene Type', fontsize=12)
    ax.set_ylabel('Factor', fontsize=12)
    ax.set_title('Tradeoff: Partitioning Overhead vs Balance Improvement', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in scenes])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(overhead + balance_gain) * 1.2)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}x', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('paper/figures/overhead_tradeoff.png', bbox_inches='tight')
    plt.close()
    print("Saved: overhead_tradeoff.png")


def plot_degree_sensitivity():
    """Figure 5: Degree sensitivity (shows robustness)."""
    with open('results/exp5_degree.json') as f:
        data = json.load(f)
    
    scenes = list(data.keys())
    degrees = [2, 3, 4, 5]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, scene in enumerate(scenes):
        balances = [data[scene][f'degree_{d}'] for d in degrees]
        ax.plot(degrees, balances, 'o-', label=scene.upper(), color=colors[i], 
                linewidth=2, markersize=8)
    
    ax.set_xlabel('Polynomial Degree', fontsize=12)
    ax.set_ylabel('Load Balance Ratio', fontsize=12)
    ax.set_title('Degree Sensitivity: Balance is Robust to Polynomial Degree', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks(degrees)
    ax.set_ylim(1.0, 2.0)
    
    plt.tight_layout()
    plt.savefig('paper/figures/degree_sensitivity.png', bbox_inches='tight')
    plt.close()
    print("Saved: degree_sensitivity.png")


def main():
    """Generate all figures."""
    print("Generating figures for Paper 1...")
    print("=" * 60)
    
    figures_dir = Path('paper/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plot_load_balance()
    plot_algebraic_detection()
    plot_scalability()
    plot_overhead_tradeoff()
    plot_degree_sensitivity()
    
    print("=" * 60)
    print("All figures generated in paper/figures/")


if __name__ == '__main__':
    main()
