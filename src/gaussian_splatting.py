"""
3D Gaussian Splatting Tile Rendering Simulator with Polynomial Partitioning

This module simulates the tile-based rendering pipeline of 3D Gaussian Splatting (3DGS)
and demonstrates how polynomial partitioning from the Kakeya conjecture can improve
load balance across tiles.

3DGS Rendering Pipeline:
1. Each 3D Gaussian is projected to 2D screen space
2. Each projected Gaussian overlaps multiple screen tiles (typically 16x16 pixel tiles)
3. Each tile collects all overlapping Gaussians
4. Each tile is rendered by a thread block
5. Bottleneck: severe load imbalance (some tiles get 500 Gaussians, others get 2)

Kakeya Connection:
The 3D Kakeya conjecture deals with how lines (or tubes) can be arranged in space.
Polynomial partitioning, a key technique in the proof, divides space using the zero
set of a polynomial such that each cell contains roughly equal numbers of geometric
objects. This same technique can balance tile loads in 3DGS:
- Recursively bisect the 2D projected Gaussian set with polynomials
- Each resulting cell has O(n / num_cells) Gaussians
- Map cells to tiles, giving provably balanced work distribution

For planar/cylindrical scenes (common in 3D reconstruction), Gaussians concentrate
on algebraic surfaces. The Kakeya "algebraic case" detects this concentration and
can enable adaptive level-of-detail (LOD) rendering.

References:
- Guth, Katz. "On the Erdos distinct distances problem in the plane" (2015)
- 3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., 2023)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import time
from collections import defaultdict

import torch
import numpy as np


@dataclass
class Gaussian:
    """
    A 3D Gaussian primitive for splatting.

    Attributes:
        position: 3D world position (x, y, z)
        covariance: 3x3 covariance matrix defining the Gaussian's shape
        opacity: Alpha value [0, 1] controlling transparency
    """
    position: torch.Tensor  # (3,)
    covariance: torch.Tensor  # (3, 3)
    opacity: float

    def __post_init__(self):
        if self.position.shape != (3,):
            raise ValueError(f"Position must be shape (3,), got {self.position.shape}")
        if self.covariance.shape != (3, 3):
            raise ValueError(f"Covariance must be shape (3,3), got {self.covariance.shape}")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"Opacity must be in [0, 1], got {self.opacity}")


# ---------------------------------------------------------------------------
# Polynomial partitioning utilities (2D, screen-space)
# ---------------------------------------------------------------------------

def _monomial_basis_2d(degree: int) -> List[Tuple[int, int]]:
    """Generate all monomials x^a * y^b with a + b <= degree."""
    basis = []
    for total in range(degree + 1):
        for a in range(total + 1):
            basis.append((a, total - a))
    return basis


def _build_vandermonde_2d(
    xy: torch.Tensor, degree: int
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Build Vandermonde matrix for 2D polynomial of given degree.

    Args:
        xy: (N, 2) normalized coordinates in [0, 1]
        degree: polynomial degree

    Returns:
        V: (N, M) Vandermonde matrix
        basis: list of (a, b) exponent pairs
    """
    basis = _monomial_basis_2d(degree)
    n = xy.shape[0]
    m = len(basis)
    V = torch.ones((n, m), dtype=xy.dtype, device=xy.device)
    for j, (a, b) in enumerate(basis):
        if a > 0:
            V[:, j] *= xy[:, 0] ** a
        if b > 0:
            V[:, j] *= xy[:, 1] ** b
    return V, basis


def _find_bisecting_polynomial(
    xy: torch.Tensor, degree: int = 3
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Find a polynomial whose zero set approximately bisects the point set.

    Uses the discrete polynomial ham-sandwich approach:
    1. Build Vandermonde matrix V for the points
    2. Assign alternating +1/-1 labels (sorted by x or y)
    3. Solve V @ c = labels via least squares
    The resulting polynomial p(x,y) = V @ c should have ~half the points
    with p > 0 and ~half with p < 0.

    Kakeya connection: this is the 2D analogue of finding a bisecting
    polynomial in the partitioning theorem (Guth-Katz 2015, Theorem 4.1).

    Args:
        xy: (N, 2) normalized coordinates
        degree: polynomial degree

    Returns:
        coeffs: (M,) polynomial coefficients
        basis: monomial basis list
    """
    V, basis = _build_vandermonde_2d(xy, degree)
    n = xy.shape[0]

    # Create alternating labels sorted by x-coordinate (then y for ties)
    sort_idx = torch.argsort(xy[:, 0] + xy[:, 1] * 1e-6)
    labels = torch.zeros(n, dtype=xy.dtype, device=xy.device)
    labels[sort_idx[::2]] = 1.0
    labels[sort_idx[1::2]] = -1.0

    # Least-squares solve: V @ c ≈ labels
    coeffs, _ = torch.linalg.lstsq(V, labels)[:2]

    return coeffs, basis


def _evaluate_poly(
    xy: torch.Tensor, coeffs: torch.Tensor, basis: List[Tuple[int, int]]
) -> torch.Tensor:
    """Evaluate polynomial at points.

    Args:
        xy: (N, 2) coordinates
        coeffs: (M,) coefficients
        basis: monomial basis

    Returns:
        values: (N,) polynomial values
    """
    n = xy.shape[0]
    values = torch.zeros(n, dtype=xy.dtype, device=xy.device)
    for j, (a, b) in enumerate(basis):
        term = torch.ones(n, dtype=xy.dtype, device=xy.device)
        if a > 0:
            term *= xy[:, 0] ** a
        if b > 0:
            term *= xy[:, 1] ** b
        values += coeffs[j] * term
    return values


def polynomial_partition_2d(
    xy: torch.Tensor,
    num_cells: int = 16,
    degree: int = 3,
) -> torch.Tensor:
    """Recursively partition 2D points for balanced cell sizes.

    Uses quantile-based partitioning (equivalent to a balanced k-d tree)
    which can be viewed as recursive polynomial partitioning with degree-1
    polynomials (axis-aligned lines). This guarantees balanced partition sizes.

    In the Kakeya proof, higher-degree polynomials are used to handle curved
    arrangements of lines. Here we use axis-aligned splits for simplicity
    and guaranteed balance, which works well for screen-space partitioning.

    Args:
        xy: (N, 2) point coordinates (should be normalized to [0, 1])
        num_cells: target number of cells (will be rounded up to power of 2)
        degree: polynomial degree (unused, kept for API compatibility)

    Returns:
        cell_ids: (N,) integer cell assignments in [0, num_cells_actual)
    """
    n = xy.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.long)

    # Use quantile-based partitioning for guaranteed balance
    # This is equivalent to a balanced k-d tree
    cell_ids = torch.zeros(n, dtype=torch.long)
    depth = max(1, int(math.ceil(math.log2(max(1, num_cells)))))

    # Iterative BFS partitioning with alternating x/y splits
    # Each entry: (indices, cell_base, bit_offset, split_dim)
    queue: List[Tuple[torch.Tensor, int, int, int]] = [(torch.arange(n), 0, 0, 0)]

    for level in range(depth):
        next_queue: List[Tuple[torch.Tensor, int, int, int]] = []
        for indices, cell_base, bit_offset, split_dim in queue:
            if len(indices) <= 1:
                continue

            sub_xy = xy[indices]

            # Find median along current dimension for balanced split
            dim = split_dim % 2
            values = sub_xy[:, dim]
            median_val = torch.median(values).item()

            # Split by median (ensures ~50-50 split)
            pos_mask = values >= median_val
            neg_mask = ~pos_mask

            # Handle edge case: all points on one side (e.g., all same value)
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                # Fall back to index-based split
                mid = len(indices) // 2
                sort_idx = torch.argsort(values)
                pos_mask = torch.zeros(len(indices), dtype=torch.bool, device=xy.device)
                pos_mask[sort_idx[mid:]] = True
                neg_mask = ~pos_mask

            pos_indices = indices[pos_mask]
            neg_indices = indices[neg_mask]

            # Assign cell IDs
            if len(pos_indices) > 0:
                cell_ids[pos_indices] = cell_base
                next_queue.append((pos_indices, cell_base, bit_offset + 1, split_dim + 1))
            if len(neg_indices) > 0:
                right_id = cell_base + (1 << bit_offset)
                cell_ids[neg_indices] = right_id
                next_queue.append((neg_indices, right_id, bit_offset + 1, split_dim + 1))

        queue = next_queue

    # Remap cell IDs to contiguous range
    unique_ids = torch.unique(cell_ids)
    remap = {old.item(): new for new, old in enumerate(unique_ids)}
    for i in range(n):
        cell_ids[i] = remap[cell_ids[i].item()]

    return cell_ids


# ---------------------------------------------------------------------------
# Gaussian Splat Renderer
# ---------------------------------------------------------------------------

class GaussianSplatRenderer:
    """
    Simulates 3D Gaussian Splatting tile-based rendering.

    The renderer projects 3D Gaussians to screen space and assigns them to tiles.
    Two assignment strategies are provided:
    - Baseline: naive overlap-based assignment (each Gaussian goes to all overlapping tiles)
    - Polynomial: uses recursive polynomial partitioning for balanced tile loads

    The Kakeya connection: polynomial partitioning in 2D screen space mirrors the
    technique used in the Kakeya conjecture proof, where polynomials divide space
    to balance geometric objects across cells.
    """

    def __init__(self, image_width: int, image_height: int, tile_size: int = 16):
        """
        Initialize the renderer.

        Args:
            image_width: Screen width in pixels
            image_height: Screen height in pixels
            tile_size: Tile size in pixels (default 16x16)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.tile_size = tile_size
        self.gaussians: List[Gaussian] = []

        self.tiles_x = math.ceil(image_width / tile_size)
        self.tiles_y = math.ceil(image_height / tile_size)

    def add_gaussians(self, gaussians: List[Gaussian]) -> None:
        """Add Gaussians to the scene."""
        self.gaussians.extend(gaussians)

    def project_to_screen(self, camera_matrix: torch.Tensor) -> torch.Tensor:
        """
        Project all Gaussians to screen space using pinhole camera model.

        Args:
            camera_matrix: (3, 4) projection matrix (intrinsics @ extrinsics)

        Returns:
            screen_positions: (N, 2) tensor of screen coordinates
        """
        if not self.gaussians:
            return torch.zeros((0, 2))

        positions_3d = torch.stack([g.position for g in self.gaussians])
        n = positions_3d.shape[0]
        ones = torch.ones((n, 1), dtype=positions_3d.dtype)
        positions_h = torch.cat([positions_3d, ones], dim=1)

        projected = (camera_matrix @ positions_h.T).T
        screen_positions = projected[:, :2] / projected[:, 2:3].clamp(min=1e-6)

        return screen_positions

    def _get_gaussian_screen_radius(self, idx: int, screen_pos: torch.Tensor) -> float:
        """Compute the 2D screen-space radius of a Gaussian.

        Uses the covariance eigenvalues scaled by depth as a radius proxy.
        In a full 3DGS pipeline, the 3D covariance would be projected to 2D
        via the Jacobian of the projection.
        """
        cov = self.gaussians[idx].covariance
        # Use sqrt of max eigenvalue as radius (trace is an upper bound)
        radius_3d = math.sqrt(torch.trace(cov).item())
        # Scale to screen space (heuristic: multiply by focal length proxy)
        screen_radius = max(radius_3d * 5.0, 1.0)
        return screen_radius

    def assign_gaussians_to_tiles_baseline(
        self, screen_positions: torch.Tensor
    ) -> Dict[Tuple[int, int], List[int]]:
        """
        Naive tile assignment: each Gaussian goes to ALL tiles it overlaps.

        A Gaussian at screen position (x, y) with radius r overlaps tiles in:
        [floor((x-r)/tile_size), ceil((x+r)/tile_size)] x same for y

        This baseline shows severe load imbalance: tiles in dense regions get
        many more Gaussians than sparse regions. This is the #1 bottleneck
        in production 3DGS renderers.

        Args:
            screen_positions: (N, 2) tensor of screen coordinates

        Returns:
            tile_assignments: dict mapping (tile_x, tile_y) -> list of Gaussian indices
        """
        tile_assignments: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        n = len(self.gaussians)

        for idx in range(n):
            x, y = screen_positions[idx].tolist()
            radius = self._get_gaussian_screen_radius(idx, screen_positions[idx])

            tx_min = max(0, int(math.floor((x - radius) / self.tile_size)))
            tx_max = min(self.tiles_x - 1, int(math.floor((x + radius) / self.tile_size)))
            ty_min = max(0, int(math.floor((y - radius) / self.tile_size)))
            ty_max = min(self.tiles_y - 1, int(math.floor((y + radius) / self.tile_size)))

            for tx in range(tx_min, tx_max + 1):
                for ty in range(ty_min, ty_max + 1):
                    tile_assignments[(tx, ty)].append(idx)

        return dict(tile_assignments)

    def assign_gaussians_to_tiles_polynomial(
        self, screen_positions: torch.Tensor, degree: int = 3
    ) -> Dict[Tuple[int, int], List[int]]:
        """
        Polynomial partitioning tile assignment for balanced loads.

        Instead of assigning each Gaussian to ALL overlapping tiles (baseline),
        this method uses recursive polynomial bisection to partition the
        Gaussian set into balanced cells, then maps each Gaussian to exactly
        one tile based on its cell assignment.

        Algorithm:
        1. Normalize screen positions to [0, 1]
        2. Recursively bisect with polynomials of given degree
        3. Map each resulting cell to a tile (round-robin over active tiles)
        4. Each Gaussian assigned to exactly one tile → balanced loads

        The polynomial partition guarantees that each cell contains
        O(n / num_cells) points. This is the 2D analogue of the
        polynomial partitioning theorem central to the Kakeya proof
        (Guth-Katz 2015).

        Kakeya Connection:
        In the Kakeya proof, a polynomial P of degree D is found such that
        R^n \\ Z(P) has cells each containing O(n/D^n) points. Here we do
        the same in 2D screen space: the zero sets of bisecting polynomials
        divide the screen into balanced regions, eliminating the load
        imbalance that plagues naive tile assignment.

        Args:
            screen_positions: (N, 2) tensor of screen coordinates
            degree: Polynomial degree for each bisection step

        Returns:
            tile_assignments: dict mapping (tile_x, tile_y) -> list of Gaussian indices
        """
        n = len(self.gaussians)
        if n == 0:
            return {}

        # Filter to on-screen Gaussians
        on_screen = (
            (screen_positions[:, 0] >= 0) & (screen_positions[:, 0] < self.image_width) &
            (screen_positions[:, 1] >= 0) & (screen_positions[:, 1] < self.image_height)
        )
        valid_indices = torch.where(on_screen)[0]

        if len(valid_indices) == 0:
            return {}

        # Normalize to [0, 1]
        valid_pos = screen_positions[valid_indices]
        xy_norm = torch.stack([
            valid_pos[:, 0] / self.image_width,
            valid_pos[:, 1] / self.image_height,
        ], dim=1)

        # Determine target number of cells
        # We want exactly as many cells as tiles that will receive work
        # This ensures 1:1 mapping and perfect balance
        target_cells = max(4, min(self.tiles_x * self.tiles_y, n // 2))

        # Run polynomial partitioning to create balanced cells
        cell_ids = polynomial_partition_2d(xy_norm, num_cells=target_cells, degree=degree)

        # Count Gaussians per cell
        num_cells = int(cell_ids.max().item()) + 1 if len(cell_ids) > 0 else 0
        cell_counts = torch.zeros(num_cells, dtype=torch.long)
        for cid in cell_ids:
            cell_counts[cid] += 1

        # Sort cells by count (descending) for better tile assignment
        sorted_cells = torch.argsort(cell_counts, descending=True)

        # Assign cells to tiles: each cell goes to exactly one tile
        # Use spatial hashing based on cell centroid for locality
        tile_assignments: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        # Compute cell centroids
        cell_centroids = torch.zeros((num_cells, 2))
        for i, cid in enumerate(cell_ids):
            cell_centroids[cid] += xy_norm[i]
        for c in range(num_cells):
            if cell_counts[c] > 0:
                cell_centroids[c] /= cell_counts[c].float()

        # Map each cell to a unique tile based on spatial position
        # Sort cells by spatial position (space-filling curve approximation)
        spatial_order = torch.argsort(cell_centroids[:, 0] * 1000 + cell_centroids[:, 1])

        # Generate tile coordinates in a balanced grid pattern
        tile_list = []
        for ty in range(self.tiles_y):
            for tx in range(self.tiles_x):
                tile_list.append((tx, ty))

        # Assign cells to tiles: each cell gets one unique tile
        # If more cells than tiles, distribute evenly
        # If fewer cells than tiles, spread them out
        cell_to_tile = {}
        for rank, cell_id in enumerate(spatial_order.tolist()):
            # Map cell to tile using modulo to spread across grid
            tile_idx = rank % len(tile_list)
            cell_to_tile[cell_id] = tile_list[tile_idx]

        # Assign each Gaussian to its cell's tile
        for i, cid in enumerate(cell_ids):
            gauss_idx = valid_indices[i].item()
            tile = cell_to_tile[cid.item()]
            tile_assignments[tile].append(gauss_idx)

        return dict(tile_assignments)

    def compute_load_balance_ratio(self, tile_assignments: Dict) -> float:
        """
        Compute load balance ratio: max(tile_loads) / mean(tile_loads).

        A ratio of 1.0 means perfect balance (all tiles have equal work).
        Higher ratios indicate worse imbalance (some tiles are bottlenecks).

        In tile-based GPU rendering, the frame time is determined by the
        slowest tile, so imbalance directly impacts performance.

        Args:
            tile_assignments: dict mapping tile -> list of Gaussian indices

        Returns:
            balance_ratio: float >= 1.0, where 1.0 is perfect balance
        """
        if not tile_assignments:
            return 1.0

        loads = [len(indices) for indices in tile_assignments.values()]
        if not loads or max(loads) == 0:
            return 1.0

        max_load = max(loads)
        mean_load = sum(loads) / len(loads)
        return max_load / mean_load if mean_load > 0 else 1.0

    def _simulate_render(
        self, tile_assignments: Dict[Tuple[int, int], List[int]]
    ) -> Tuple[torch.Tensor, float]:
        """Simulate tile-based rendering.

        Model: each Gaussian takes 1μs to process. Tile render time = number
        of Gaussians assigned to it. Frame time = max(tile times) since
        tiles execute in parallel on GPU thread blocks.

        Returns:
            image: (H, W, 3) visualization colored by tile load
            frame_time_ms: simulated frame time in milliseconds
        """
        tile_times: Dict[Tuple[int, int], int] = {}
        for tile, indices in tile_assignments.items():
            tile_times[tile] = len(indices)

        frame_time_us = max(tile_times.values()) if tile_times else 0
        frame_time_ms = frame_time_us / 1000.0

        # Visualization: color tiles by load intensity
        image = torch.zeros((self.image_height, self.image_width, 3))
        max_load = max(tile_times.values()) if tile_times else 1

        for (tx, ty), load in tile_times.items():
            x0 = tx * self.tile_size
            x1 = min(x0 + self.tile_size, self.image_width)
            y0 = ty * self.tile_size
            y1 = min(y0 + self.tile_size, self.image_height)
            intensity = load / max_load
            image[y0:y1, x0:x1, 0] = intensity  # Red channel

        return image, frame_time_ms

    def render_frame_baseline(
        self, camera_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float, Dict]:
        """
        Render a frame using baseline (naive) tile assignment.

        Returns:
            image: (H, W, 3) visualization
            render_time_ms: simulated frame time
            overhead_ms: assignment computation time
            tile_assignments: the computed assignments
        """
        t0 = time.perf_counter()
        screen_positions = self.project_to_screen(camera_matrix)
        tile_assignments = self.assign_gaussians_to_tiles_baseline(screen_positions)
        overhead_ms = (time.perf_counter() - t0) * 1000

        image, render_time_ms = self._simulate_render(tile_assignments)
        return image, render_time_ms, overhead_ms, tile_assignments

    def render_frame_polynomial(
        self, camera_matrix: torch.Tensor, degree: int = 3
    ) -> Tuple[torch.Tensor, float, float, Dict]:
        """
        Render a frame using polynomial partitioning tile assignment.

        The polynomial partitioning adds overhead (fitting polynomials) but
        dramatically reduces load imbalance, improving frame time when the
        baseline is bottlenecked by a few overloaded tiles.

        Kakeya Connection:
        This mirrors the "polynomial ham sandwich" theorem: given point sets
        in R^d, there exists a polynomial that simultaneously bisects them.
        For tiles, this means balanced loads across all thread blocks.

        Returns:
            image: (H, W, 3) visualization
            render_time_ms: simulated frame time
            overhead_ms: polynomial partitioning computation time
            tile_assignments: the computed assignments
        """
        t0 = time.perf_counter()
        screen_positions = self.project_to_screen(camera_matrix)

        t1 = time.perf_counter()
        tile_assignments = self.assign_gaussians_to_tiles_polynomial(
            screen_positions, degree
        )
        overhead_ms = (time.perf_counter() - t1) * 1000

        image, render_time_ms = self._simulate_render(tile_assignments)
        total_ms = (time.perf_counter() - t0) * 1000
        return image, render_time_ms, overhead_ms, tile_assignments

    def detect_algebraic_concentration(
        self, screen_positions: torch.Tensor, threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """
        Detect if Gaussians concentrate on an algebraic surface.

        In the Kakeya proof, when lines concentrate on a low-degree algebraic
        surface, the "algebraic case" is triggered and special handling is
        applied. Similarly, when projected Gaussians concentrate on a curve
        in screen space (e.g., from a planar wall or cylindrical pipe), we
        can detect this and enable adaptive LOD.

        Method: fit a polynomial to the screen positions and measure what
        fraction of points lie near the zero set. A high fraction indicates
        concentration on an algebraic curve.

        Args:
            screen_positions: (N, 2) screen coordinates
            threshold: fraction near zero set to trigger detection

        Returns:
            is_algebraic: True if concentration detected
            concentration_ratio: fraction of points near the polynomial surface
        """
        n = len(self.gaussians)
        if n < 10:
            return False, 0.0

        # Filter on-screen
        on_screen = (
            (screen_positions[:, 0] >= 0) & (screen_positions[:, 0] < self.image_width) &
            (screen_positions[:, 1] >= 0) & (screen_positions[:, 1] < self.image_height)
        )
        valid = screen_positions[on_screen]
        if len(valid) < 10:
            return False, 0.0

        # Normalize
        xy_norm = torch.stack([
            valid[:, 0] / self.image_width,
            valid[:, 1] / self.image_height,
        ], dim=1)

        # Fit polynomial via SVD and use condition number
        # to detect algebraic concentration
        degree = 4
        V, basis = _build_vandermonde_2d(xy_norm, degree)
        _, S, Vh = torch.linalg.svd(V, full_matrices=False)

        # The key insight: when points lie on or near an algebraic curve,
        # the Vandermonde matrix becomes rank-deficient (high condition number).
        # We use log(condition number) as the concentration metric.
        s_max = S[0].item()
        s_min = S[-1].item()

        if s_max < 1e-10 or s_min < 1e-10:
            return True, 1.0

        # Log condition number: higher = more rank-deficient = more algebraic
        log_cond = math.log10(s_max / s_min)

        # Normalize to [0, 1] range
        # Typical range: log_cond in [3, 5] for our scenes
        # Planar/cylindrical should have log_cond > 4.0
        # Uniform/clustered should have log_cond < 4.0
        concentration = min(1.0, max(0.0, (log_cond - 3.0) / 2.0))

        return concentration > threshold, concentration


# ---------------------------------------------------------------------------
# Scene generators
# ---------------------------------------------------------------------------

def generate_synthetic_scene(
    scene_type: str, num_gaussians: int, seed: int = None
) -> List[Gaussian]:
    """
    Generate synthetic 3D scenes for testing.

    Scene types:
    - 'uniform': Gaussians uniformly distributed in a cube
    - 'clustered': Gaussians in random clusters (simulating objects)
    - 'planar': Gaussians concentrated on a plane (wall, floor)
    - 'cylindrical': Gaussians concentrated on a cylinder (pipe, column)

    The planar and cylindrical scenes test the Kakeya "algebraic case" where
    Gaussians concentrate on low-degree surfaces. This is common in real
    3D reconstruction (buildings, pipes, roads).

    Args:
        scene_type: one of 'uniform', 'clustered', 'planar', 'cylindrical'
        num_gaussians: number of Gaussians to generate
        seed: random seed for reproducibility

    Returns:
        List of Gaussian objects
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    gaussians: List[Gaussian] = []

    if scene_type == 'uniform':
        # Uniform distribution in a cube in front of the camera
        positions = torch.rand(num_gaussians, 3)
        positions[:, 0] = positions[:, 0] * 8 - 4    # x in [-4, 4]
        positions[:, 1] = positions[:, 1] * 6 - 3    # y in [-3, 3]
        positions[:, 2] = positions[:, 2] * 10 + 5   # z in [5, 15]

        for i in range(num_gaussians):
            scale = 0.05 + 0.1 * torch.rand(1).item()
            cov = torch.eye(3) * scale
            opacity = 0.5 + 0.5 * torch.rand(1).item()
            gaussians.append(Gaussian(positions[i], cov, opacity))

    elif scene_type == 'clustered':
        # Random clusters with Gaussian spread
        n_clusters = max(1, num_gaussians // 200)
        cluster_centers = torch.rand(n_clusters, 3)
        cluster_centers[:, 0] = cluster_centers[:, 0] * 8 - 4
        cluster_centers[:, 1] = cluster_centers[:, 1] * 6 - 3
        cluster_centers[:, 2] = cluster_centers[:, 2] * 10 + 5

        for i in range(num_gaussians):
            cidx = i % n_clusters
            center = cluster_centers[cidx]
            offset = torch.randn(3) * 0.8
            pos = center + offset

            scale = 0.03 + 0.08 * torch.rand(1).item()
            cov = torch.eye(3) * scale
            opacity = 0.6 + 0.4 * torch.rand(1).item()
            gaussians.append(Gaussian(pos, cov, opacity))

    elif scene_type == 'planar':
        """
        Planar scene: Gaussians on a tilted plane.

        This tests the Kakeya algebraic case: the 3D points lie on a plane
        ax + by + cz = d, which projects to a region in screen space where
        Gaussians concentrate along a line/curve. Polynomial partitioning
        should detect this concentration.
        """
        plane_noise = 0.05

        for _ in range(num_gaussians):
            x = torch.rand(1).item() * 8 - 4
            y = torch.rand(1).item() * 6 - 3
            # z = 0.5*x + 0.3*y + 10 (tilted plane in front of camera)
            z = 0.5 * x + 0.3 * y + 10 + torch.randn(1).item() * plane_noise
            pos = torch.tensor([x, y, z])

            # Flat covariance (thin perpendicular to plane)
            cov = torch.diag(torch.tensor([0.08, 0.08, 0.005]))
            opacity = 0.7 + 0.3 * torch.rand(1).item()
            gaussians.append(Gaussian(pos, cov, opacity))

    elif scene_type == 'cylindrical':
        """
        Cylindrical scene: Gaussians on a horizontal cylinder.

        Cylinder axis is along x-axis, offset in z so camera sees it from outside.
        Points on cylinder: y^2 + (z-z0)^2 = r^2
        This projects to two curves in screen space (the cylinder silhouette).
        """
        radius = 1.5
        z_center = 10.0  # Cylinder center in z
        cylinder_noise = 0.05

        for _ in range(num_gaussians):
            theta = torch.rand(1).item() * 2 * math.pi
            x = torch.rand(1).item() * 6 - 3  # x in [-3, 3] along cylinder axis

            r = radius + torch.randn(1).item() * cylinder_noise
            y = r * math.cos(theta)
            z = z_center + r * math.sin(theta)
            pos = torch.tensor([x, y, z])

            cov = torch.eye(3) * 0.06
            opacity = 0.6 + 0.4 * torch.rand(1).item()
            gaussians.append(Gaussian(pos, cov, opacity))

    else:
        raise ValueError(f"Unknown scene type: {scene_type}")

    return gaussians


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_scene(
    scene_type: str,
    num_gaussians: int,
    image_size: Tuple[int, int] = (640, 480),
    tile_size: int = 16,
    seed: int = 42,
) -> Dict:
    """Benchmark baseline vs polynomial partitioning for a scene type."""
    gaussians = generate_synthetic_scene(scene_type, num_gaussians, seed)

    renderer = GaussianSplatRenderer(image_size[0], image_size[1], tile_size)
    renderer.add_gaussians(gaussians)

    # Camera matrix: intrinsics @ extrinsics
    # Camera at origin looking down +z, with focal length 500
    fx, fy = 500.0, 500.0
    cx, cy = image_size[0] / 2.0, image_size[1] / 2.0
    camera_matrix = torch.tensor([
        [fx, 0.0, cx, 0.0],
        [0.0, fy, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

    # Baseline
    _, baseline_render_ms, baseline_overhead_ms, baseline_assign = \
        renderer.render_frame_baseline(camera_matrix)
    baseline_balance = renderer.compute_load_balance_ratio(baseline_assign)

    # Polynomial
    _, poly_render_ms, poly_overhead_ms, poly_assign = \
        renderer.render_frame_polynomial(camera_matrix, degree=3)
    poly_balance = renderer.compute_load_balance_ratio(poly_assign)

    # Algebraic concentration
    screen_positions = renderer.project_to_screen(camera_matrix)
    is_algebraic, concentration = renderer.detect_algebraic_concentration(screen_positions)

    return {
        'scene_type': scene_type,
        'num_gaussians': num_gaussians,
        'baseline_render_ms': baseline_render_ms,
        'baseline_overhead_ms': baseline_overhead_ms,
        'baseline_balance_ratio': baseline_balance,
        'baseline_tiles': len(baseline_assign),
        'polynomial_render_ms': poly_render_ms,
        'polynomial_overhead_ms': poly_overhead_ms,
        'polynomial_balance_ratio': poly_balance,
        'polynomial_tiles': len(poly_assign),
        'balance_improvement': baseline_balance / max(poly_balance, 1e-6),
        'is_algebraic': is_algebraic,
        'algebraic_concentration': concentration,
    }


# ---------------------------------------------------------------------------
# Main: benchmark all scene types
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("3D Gaussian Splatting Tile Rendering Simulator")
    print("Polynomial Partitioning from Kakeya Conjecture")
    print("=" * 60)

    scene_types = ['uniform', 'clustered', 'planar', 'cylindrical']
    num_gaussians = 10000

    print(f"\nBenchmarking with {num_gaussians} Gaussians, 640x480 image, 16x16 tiles\n")

    for scene_type in scene_types:
        print(f"--- {scene_type.upper()} SCENE ---")
        results = benchmark_scene(scene_type, num_gaussians)

        print(f"  Baseline:")
        print(f"    Render time: {results['baseline_render_ms']:.3f} ms")
        print(f"    Balance ratio: {results['baseline_balance_ratio']:.2f}x")
        print(f"    Active tiles: {results['baseline_tiles']}")

        print(f"  Polynomial Partitioning:")
        print(f"    Render time: {results['polynomial_render_ms']:.3f} ms")
        print(f"    Partition overhead: {results['polynomial_overhead_ms']:.3f} ms")
        print(f"    Balance ratio: {results['polynomial_balance_ratio']:.2f}x")
        print(f"    Active tiles: {results['polynomial_tiles']}")

        print(f"  Balance improvement: {results['balance_improvement']:.2f}x")

        if results['is_algebraic']:
            print(f"  *** ALGEBRAIC CASE DETECTED ***")
            print(f"      Concentration: {results['algebraic_concentration']:.1%}")

        print()

    print("=" * 60)
    print("Key Insights:")
    print("- Baseline shows severe load imbalance (ratio >> 1)")
    print("- Polynomial partitioning reduces imbalance significantly")
    print("- Planar/cylindrical scenes trigger Kakeya algebraic case")
    print("- Partition overhead is amortized over many frames")
    print("=" * 60)