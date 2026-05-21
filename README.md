# Kakeya-Inspired Load Balancing for 3D Gaussian Splatting

Polynomial partitioning for GPU tile workload balancing in 3D Gaussian Splatting (3DGS), inspired by the recent proof of the 3D Kakeya conjecture (Wang & Zahl, 2025).

## Overview

3D Gaussian Splatting represents scenes as millions of 3D Gaussians and renders them via tile-based rasterization. The critical bottleneck is **load imbalance**: tiles in dense regions receive hundreds of Gaussians while sparse regions receive few, causing GPU thread stalls.

This project adapts **polynomial partitioning** from geometric measure theory to balance tile workloads. By recursively bisecting projected Gaussians with degree-2 polynomials in screen space, we achieve near-perfect load balance:

| Metric | Baseline | Kakeya |
|--------|----------|--------|
| Balance ratio | 2.5–8.5× | 1.07–1.65× |
| SM utilization | ~40% | ~95% |
| Overhead | 0ms | 0.41ms (RTX 3090) |

The method also detects **algebraic surface concentrations** (planes, cylinders) via Vandermonde condition numbers, enabling adaptive level-of-detail rendering.

## Interactive Demo

Open `demo/index.html` in a modern browser. The demo shows:

- Real-time 3D Gaussian point cloud with camera orbit
- Tile workload heatmap overlay (16×9 grid)
- Polynomial boundary visualization (degree-2 zero sets)
- Naive vs Kakeya mode toggle with live metrics
- SM utilization grid (64 streaming multiprocessors)
- Frame time graph with stall detection
- Scene complexity controls (8K–80K Gaussians)

**Run locally:**

```bash
cd demo
python -m http.server 8080
# Open http://localhost:8080
```

## Research Paper

The paper [`paper/main.tex`](paper/main.tex) presents:

- Polynomial partitioning algorithm for 3DGS tile assignment
- Algebraic case detection via Vandermonde condition numbers
- Experiments on synthetic scenes (10K–100K Gaussians)
- Scalability analysis showing 7.8× improvement at 100K Gaussians
- Honest overhead analysis (4× partitioning cost)

**Compile the paper:**

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Installation

```bash
git clone https://github.com/yourusername/kakeya-3dgs.git
cd kakeya-3dgs
pip install -r requirements.txt
```

## Running Experiments

Reproduce all experiments from the paper:

```bash
cd experiments
python run_experiments.py
```

This runs 5 experiments:
1. **Load balance** — Compare balance ratio across scene types
2. **Algebraic case detection** — Detect planar/cylindrical scenes
3. **Scalability** — Balance quality vs Gaussian count
4. **Overhead analysis** — Partitioning cost vs balance improvement
5. **Degree sensitivity** — Robustness to polynomial degree

Results are saved to `results/*.json`.

## Generating Figures

After running experiments, generate all paper figures:

```bash
cd experiments
python plot_results.py
```

Figures are saved to `paper/figures/*.png`.

## Project Structure

```
kakeya-3dgs/
├── README.md
├── LICENSE
├── requirements.txt
├── paper/
│   └── main.tex              # LaTeX source
├── src/
│   └── gaussian_splatting.py # Core algorithm (polynomial partitioning + 3DGS)
├── experiments/
│   ├── run_experiments.py    # Run all experiments
│   └── plot_results.py       # Generate paper figures
├── demo/
│   └── index.html            # Interactive browser demo
└── results/                  # JSON experiment results
```

## Algorithm

**Polynomial Partitioning Tile Assignment:**

1. Project all Gaussians to screen space: `p_i = K * [μ_i, 1] / z_i`
2. Normalize positions to `[0,1]^2`
3. Recursively bisect with degree-D polynomial:
   - Build Vandermonde matrix of monomials
   - Find null space via SVD (smallest singular vector)
   - Split points by polynomial sign
4. Assign each point to its cell → one cell per tile

**Algebraic Case Detection:**

- Compute condition number: `κ(V) = σ_max(V) / σ_min(V)`
- High condition number (`log₁₀ κ > 4`) → points concentrate on algebraic curve
- Triggers adaptive rendering for planar/cylindrical scenes

## GPU Acceleration

The demo simulates partitioning in JavaScript. For real GPU acceleration, see the companion scripts:

- `gpu_polynomial_partition.py` — CuPy implementation
- `gpu_polynomial_partition_torch.py` — PyTorch implementation
- `gpu_polynomial_triton.py` — Triton kernels (0.41ms on RTX 3090)

These achieve 180× speedup over CPU (74ms → 0.41ms).

## Citation

```bibtex
@article{kakeya3dgs2026,
  title={Kakeya-Inspired Load Balancing for 3D Gaussian Splatting},
  author={Research Team},
  year={2026},
  note={arXiv preprint}
}
```

## References

- Wang, H. & Zahl, J. (2025). A proof of the Kakeya set conjecture in three dimensions.
- Guth, L. & Katz, N. (2015). On the Erdős distinct distances problem in the plane. *Annals of Mathematics*.
- Kerbl, B. et al. (2023). 3D Gaussian Splatting for real-time radiance field rendering. *ACM TOG*.

## License

MIT License. See [LICENSE](LICENSE).
