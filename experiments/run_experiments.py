#!/usr/bin/env python3
"""
Run experiments for Paper 1: Kakeya-Inspired Load Balancing for 3D Gaussian Splatting

Experiments:
1. Load balance ratio: baseline vs polynomial partitioning across scene types
2. Algebraic case detection: planar/cylindrical scenes trigger surface concentration
3. Scalability: how balance scales with Gaussian count (10k → 100k)
4. Overhead analysis: partitioning cost vs balance improvement tradeoff
5. Degree sensitivity: how polynomial degree affects balance quality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import time
from typing import Dict, List, Tuple
from src.gaussian_splatting import (
    GaussianSplatRenderer, generate_synthetic_scene, polynomial_partition_2d
)


def experiment_1_load_balance():
    """Compare load balance ratio: baseline vs polynomial partitioning."""
    print("\n=== Experiment 1: Load Balance Ratio ===")
    print("Primary metric: max(tile_loads) / mean(tile_loads)")
    print("Lower is better (1.0 = perfect balance)\n")
    
    scene_types = ['uniform', 'clustered', 'planar', 'cylindrical']
    num_gaussians = 10000
    image_size = (1024, 1024)
    tile_size = 16
    
    results = {}
    
    for scene_type in scene_types:
        print(f"{scene_type.upper()} scene ({num_gaussians} Gaussians):")
        renderer = GaussianSplatRenderer(image_size[0], image_size[1], tile_size)
        gaussians = generate_synthetic_scene(scene_type, num_gaussians, seed=42)
        renderer.add_gaussians(gaussians)
        
        camera = torch.tensor([
            [500, 0, 512, 0],
            [0, 500, 512, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32)
        
        screen_positions = renderer.project_to_screen(camera)
        
        # Baseline
        baseline_assign = renderer.assign_gaussians_to_tiles_baseline(screen_positions)
        baseline_balance = renderer.compute_load_balance_ratio(baseline_assign)
        baseline_tiles = len(baseline_assign)
        
        # Polynomial partitioning (degree 3)
        poly_assign = renderer.assign_gaussians_to_tiles_polynomial(screen_positions, degree=3)
        poly_balance = renderer.compute_load_balance_ratio(poly_assign)
        poly_tiles = len(poly_assign)
        
        improvement = baseline_balance / poly_balance
        
        results[scene_type] = {
            'baseline_balance': baseline_balance,
            'polynomial_balance': poly_balance,
            'improvement_ratio': improvement,
            'baseline_tiles': baseline_tiles,
            'polynomial_tiles': poly_tiles
        }
        
        print(f"  Baseline: {baseline_balance:.2f}x ({baseline_tiles} tiles)")
        print(f"  Polynomial: {poly_balance:.2f}x ({poly_tiles} tiles)")
        print(f"  Improvement: {improvement:.2f}x better balance\n")
    
    return results


def experiment_2_algebraic_case():
    """Test algebraic case detection on different scene types."""
    print("\n=== Experiment 2: Algebraic Case Detection ===")
    print("Detect when Gaussians concentrate on algebraic surfaces\n")
    
    scene_types = ['uniform', 'clustered', 'planar', 'cylindrical']
    num_gaussians = 10000
    image_size = (1024, 1024)
    tile_size = 16
    
    results = {}
    
    for scene_type in scene_types:
        print(f"{scene_type.upper()} scene:")
        renderer = GaussianSplatRenderer(image_size[0], image_size[1], tile_size)
        gaussians = generate_synthetic_scene(scene_type, num_gaussians, seed=42)
        renderer.add_gaussians(gaussians)
        
        camera = torch.tensor([
            [500, 0, 512, 0],
            [0, 500, 512, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32)
        
        screen_positions = renderer.project_to_screen(camera)
        is_algebraic, concentration = renderer.detect_algebraic_concentration(
            screen_positions, threshold=0.65
        )
        
        expected = scene_type in ['planar', 'cylindrical']
        correct = is_algebraic == expected
        
        results[scene_type] = {
            'is_algebraic': is_algebraic,
            'concentration': concentration,
            'expected_algebraic': expected,
            'correct': correct
        }
        
        print(f"  Detected: {is_algebraic}")
        print(f"  Concentration: {concentration:.1%}")
        print(f"  Expected algebraic: {expected}")
        print(f"  Correct: {correct}\n")
    
    return results


def experiment_3_scalability():
    """Test how balance scales with Gaussian count."""
    print("\n=== Experiment 3: Scalability ===")
    print("Balance ratio at different scene sizes\n")
    
    scene_type = 'clustered'  # Most realistic
    image_size = (1024, 1024)
    tile_size = 16
    gaussian_counts = [10000, 25000, 50000, 75000, 100000]
    
    results = {}
    
    for num_gaussians in gaussian_counts:
        print(f"{num_gaussians} Gaussians:")
        renderer = GaussianSplatRenderer(image_size[0], image_size[1], tile_size)
        gaussians = generate_synthetic_scene(scene_type, num_gaussians, seed=42)
        renderer.add_gaussians(gaussians)
        
        camera = torch.tensor([
            [500, 0, 512, 0],
            [0, 500, 512, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32)
        
        screen_positions = renderer.project_to_screen(camera)
        
        # Baseline
        baseline_assign = renderer.assign_gaussians_to_tiles_baseline(screen_positions)
        baseline_balance = renderer.compute_load_balance_ratio(baseline_assign)
        
        # Polynomial partitioning
        poly_assign = renderer.assign_gaussians_to_tiles_polynomial(screen_positions, degree=3)
        poly_balance = renderer.compute_load_balance_ratio(poly_assign)
        
        improvement = baseline_balance / poly_balance
        
        results[f'{num_gaussians}'] = {
            'baseline_balance': baseline_balance,
            'polynomial_balance': poly_balance,
            'improvement_ratio': improvement
        }
        
        print(f"  Baseline: {baseline_balance:.2f}x")
        print(f"  Polynomial: {poly_balance:.2f}x")
        print(f"  Improvement: {improvement:.2f}x\n")
    
    return results


def experiment_4_overhead_analysis():
    """Analyze partitioning overhead vs balance improvement tradeoff."""
    print("\n=== Experiment 4: Overhead Analysis ===")
    print("Tradeoff: partitioning cost vs balance improvement\n")
    
    scene_types = ['uniform', 'clustered', 'planar', 'cylindrical']
    num_gaussians = 50000
    image_size = (1024, 1024)
    tile_size = 16
    
    results = {}
    
    for scene_type in scene_types:
        print(f"{scene_type.upper()} scene ({num_gaussians} Gaussians):")
        renderer = GaussianSplatRenderer(image_size[0], image_size[1], tile_size)
        gaussians = generate_synthetic_scene(scene_type, num_gaussians, seed=42)
        renderer.add_gaussians(gaussians)
        
        camera = torch.tensor([
            [500, 0, 512, 0],
            [0, 500, 512, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32)
        
        screen_positions = renderer.project_to_screen(camera)
        
        # Baseline timing
        start = time.time()
        baseline_assign = renderer.assign_gaussians_to_tiles_baseline(screen_positions)
        baseline_time = (time.time() - start) * 1000
        baseline_balance = renderer.compute_load_balance_ratio(baseline_assign)
        
        # Polynomial timing
        start = time.time()
        poly_assign = renderer.assign_gaussians_to_tiles_polynomial(screen_positions, degree=3)
        poly_time = (time.time() - start) * 1000
        poly_balance = renderer.compute_load_balance_ratio(poly_assign)
        
        balance_improvement = baseline_balance / poly_balance
        overhead_ratio = poly_time / baseline_time
        
        results[scene_type] = {
            'baseline_time_ms': baseline_time,
            'polynomial_time_ms': poly_time,
            'overhead_ratio': overhead_ratio,
            'baseline_balance': baseline_balance,
            'polynomial_balance': poly_balance,
            'balance_improvement': balance_improvement
        }
        
        print(f"  Baseline: {baseline_time:.1f}ms -> {baseline_balance:.2f}x balance")
        print(f"  Polynomial: {poly_time:.1f}ms -> {poly_balance:.2f}x balance")
        print(f"  Overhead: {overhead_ratio:.1f}x slower")
        print(f"  Balance gain: {balance_improvement:.2f}x better\n")
    
    return results


def experiment_5_degree_sensitivity():
    """Test how polynomial degree affects balance quality."""
    print("\n=== Experiment 5: Degree Sensitivity ===")
    print("Balance ratio at different polynomial degrees\n")
    
    scene_types = ['uniform', 'clustered', 'planar']
    num_gaussians = 10000
    image_size = (1024, 1024)
    tile_size = 16
    degrees = [2, 3, 4, 5]
    
    results = {}
    
    for scene_type in scene_types:
        print(f"{scene_type.upper()} scene:")
        renderer = GaussianSplatRenderer(image_size[0], image_size[1], tile_size)
        gaussians = generate_synthetic_scene(scene_type, num_gaussians, seed=42)
        renderer.add_gaussians(gaussians)
        
        camera = torch.tensor([
            [500, 0, 512, 0],
            [0, 500, 512, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32)
        
        screen_positions = renderer.project_to_screen(camera)
        
        degree_results = {}
        for degree in degrees:
            poly_assign = renderer.assign_gaussians_to_tiles_polynomial(
                screen_positions, degree=degree
            )
            poly_balance = renderer.compute_load_balance_ratio(poly_assign)
            
            degree_results[f'degree_{degree}'] = poly_balance
            print(f"  Degree {degree}: {poly_balance:.2f}x balance")
        
        results[scene_type] = degree_results
        print()
    
    return results


def main():
    """Run all experiments and save results."""
    print("=" * 70)
    print("Paper 1: Kakeya-Inspired Load Balancing for 3D Gaussian Splatting")
    print("=" * 70)
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Run experiments
    exp1_results = experiment_1_load_balance()
    with open(results_dir / 'exp1_load_balance.json', 'w') as f:
        json.dump(exp1_results, f, indent=2)
    print(f"Saved: exp1_load_balance.json")
    
    exp2_results = experiment_2_algebraic_case()
    with open(results_dir / 'exp2_algebraic_case.json', 'w') as f:
        json.dump(exp2_results, f, indent=2)
    print(f"Saved: exp2_algebraic_case.json")
    
    exp3_results = experiment_3_scalability()
    with open(results_dir / 'exp3_scalability.json', 'w') as f:
        json.dump(exp3_results, f, indent=2)
    print(f"Saved: exp3_scalability.json")
    
    exp4_results = experiment_4_overhead_analysis()
    with open(results_dir / 'exp4_overhead.json', 'w') as f:
        json.dump(exp4_results, f, indent=2)
    print(f"Saved: exp4_overhead.json")
    
    exp5_results = experiment_5_degree_sensitivity()
    with open(results_dir / 'exp5_degree.json', 'w') as f:
        json.dump(exp5_results, f, indent=2)
    print(f"Saved: exp5_degree.json")
    
    print("\n" + "=" * 70)
    print("All experiments complete. Results saved to results/")
    print("=" * 70)


if __name__ == '__main__':
    main()
