/**
 * Kakeya Polynomial Partitioning Algorithm (JavaScript Implementation)
 * 
 * This is a direct port of the Python implementation from src/gaussian_splatting.py
 * Runs the real algorithm on the user's hardware via WebAssembly/JavaScript.
 * 
 * The algorithm uses quantile-based partitioning (k-d tree style) for guaranteed
 * load balance, which is equivalent to recursive polynomial partitioning with
 * degree-1 polynomials (axis-aligned splits).
 */

const KakeyaAlgorithm = (() => {
  'use strict';

  /**
   * Partition 2D points for balanced cell assignment.
   * 
   * Uses iterative BFS with alternating x/y median splits (k-d tree approach).
   * Guarantees balanced partition sizes.
   * 
   * @param {Float32Array} xy - (N, 2) point coordinates, should be normalized to [0, 1]
   * @param {number} n - Number of points
   * @param {number} numCells - Target number of cells (rounded up to power of 2)
   * @returns {Object} { cellIds: Int32Array, partitionTime: number }
   */
  function polynomialPartition2D(xy, n, numCells = 16) {
    const startTime = performance.now();
    
    if (n === 0) {
      return { cellIds: new Int32Array(0), partitionTime: 0 };
    }

    const cellIds = new Int32Array(n);
    const depth = Math.max(1, Math.ceil(Math.log2(Math.max(1, numCells))));

    // Queue entries: [indices, cellBase, bitOffset, splitDim]
    let queue = [[Array.from({ length: n }, (_, i) => i), 0, 0, 0]];

    for (let level = 0; level < depth; level++) {
      const nextQueue = [];
      
      for (const [indices, cellBase, bitOffset, splitDim] of queue) {
        if (indices.length <= 1) continue;

        const dim = splitDim % 2;
        
        // Extract values for current dimension
        const values = new Float32Array(indices.length);
        for (let i = 0; i < indices.length; i++) {
          values[i] = xy[indices[i] * 2 + dim];
        }

        // Find median
        const sorted = Array.from(values).sort((a, b) => a - b);
        const medianVal = sorted[Math.floor(sorted.length / 2)];

        // Split by median
        const posIndices = [];
        const negIndices = [];
        
        for (let i = 0; i < indices.length; i++) {
          if (values[i] >= medianVal) {
            posIndices.push(indices[i]);
          } else {
            negIndices.push(indices[i]);
          }
        }

        // Handle edge case: all points on one side
        if (posIndices.length === 0 || negIndices.length === 0) {
          const mid = Math.floor(indices.length / 2);
          const sortedIndices = indices.slice().sort((a, b) => 
            xy[a * 2 + dim] - xy[b * 2 + dim]
          );
          posIndices.length = 0;
          negIndices.length = 0;
          for (let i = 0; i < sortedIndices.length; i++) {
            if (i >= mid) {
              posIndices.push(sortedIndices[i]);
            } else {
              negIndices.push(sortedIndices[i]);
            }
          }
        }

        // Assign cell IDs and continue recursion
        if (posIndices.length > 0) {
          for (const idx of posIndices) {
            cellIds[idx] = cellBase;
          }
          nextQueue.push([posIndices, cellBase, bitOffset + 1, splitDim + 1]);
        }
        
        if (negIndices.length > 0) {
          const rightId = cellBase + (1 << bitOffset);
          for (const idx of negIndices) {
            cellIds[idx] = rightId;
          }
          nextQueue.push([negIndices, rightId, bitOffset + 1, splitDim + 1]);
        }
      }
      
      queue = nextQueue;
    }

    // Remap cell IDs to contiguous range
    const uniqueIds = new Set(cellIds);
    const remap = new Map();
    let newId = 0;
    for (const old of uniqueIds) {
      remap.set(old, newId++);
    }
    
    for (let i = 0; i < n; i++) {
      cellIds[i] = remap.get(cellIds[i]);
    }

    const partitionTime = performance.now() - startTime;
    return { cellIds, partitionTime };
  }

  /**
   * Compute cell statistics from partition results.
   * 
   * @param {Int32Array} cellIds - Cell assignments
   * @param {number} n - Number of points
   * @returns {Object} { mean, std, max, cv (coefficient of variation) }
   */
  function computeCellStats(cellIds, n) {
    const cellCounts = new Map();
    
    for (let i = 0; i < n; i++) {
      const id = cellIds[i];
      cellCounts.set(id, (cellCounts.get(id) || 0) + 1);
    }

    const counts = Array.from(cellCounts.values());
    if (counts.length === 0) {
      return { mean: 0, std: 0, max: 0, cv: 0 };
    }

    const sum = counts.reduce((a, b) => a + b, 0);
    const mean = sum / counts.length;
    
    const sqSum = counts.reduce((a, b) => a + b * b, 0);
    const std = Math.sqrt(sqSum / counts.length - mean * mean);
    
    const max = Math.max(...counts);
    const cv = mean > 0 ? std / mean : 0;

    return { mean, std, max, cv };
  }

  /**
   * Project 3D points to 2D screen space and normalize to [0, 1].
   * 
   * @param {Float32Array} pos3d - (N, 3) 3D positions
   * @param {number} n - Number of points
   * @param {number} width - Screen width
   * @param {number} height - Screen height
   * @param {THREE.Camera} camera - Three.js camera
   * @returns {Float32Array} (N, 2) normalized 2D coordinates
   */
  function projectTo2D(pos3d, n, width, height, camera) {
    const xy = new Float32Array(n * 2);
    const v = new THREE.Vector3();
    
    for (let i = 0; i < n; i++) {
      v.set(pos3d[i * 3], pos3d[i * 3 + 1], pos3d[i * 3 + 2]);
      v.project(camera);
      
      // Convert from [-1, 1] to [0, 1]
      xy[i * 2] = (v.x + 1) / 2;
      xy[i * 2 + 1] = (v.y + 1) / 2;
    }
    
    return xy;
  }

  /**
   * Full Kakeya partitioning pipeline for 3D Gaussians.
   * 
   * @param {Float32Array} pos3d - (N, 3) 3D positions
   * @param {number} n - Number of points
   * @param {number} width - Screen width
   * @param {number} height - Screen height
   * @param {THREE.Camera} camera - Three.js camera
   * @param {number} numCells - Target number of cells
   * @returns {Object} { cellIds, xy2d, partitionTime, stats }
   */
  function partition3DGaussians(pos3d, n, width, height, camera, numCells = 16) {
    const totalStart = performance.now();
    
    // Project to 2D
    const xy2d = projectTo2D(pos3d, n, width, height, camera);
    
    // Partition
    const { cellIds, partitionTime } = polynomialPartition2D(xy2d, n, numCells);
    
    // Compute stats
    const stats = computeCellStats(cellIds, n);
    
    const totalTime = performance.now() - totalStart;
    
    return {
      cellIds,
      xy2d,
      partitionTime,
      stats,
      totalTime
    };
  }

  return {
    polynomialPartition2D,
    computeCellStats,
    projectTo2D,
    partition3DGaussians
  };
})();

// Export for both Node.js and browser
if (typeof module !== 'undefined' && module.exports) {
  module.exports = KakeyaAlgorithm;
} else if (typeof window !== 'undefined') {
  window.KakeyaAlgorithm = KakeyaAlgorithm;
}
