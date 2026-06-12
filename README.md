# SpectralIslandsV3

**The Islands Were the Spectrum All Along**

> *Do not hype. Do not lie. Just show.*

## 1. The Core Problem
In previous iterations of the Geometric Neuron (up to v8), the directed edges and "spectral islands" used to process temporal data were hand-assigned. Attempts to make them emergent using a Ky Fan flow on the lag covariance stalled at a hard ceiling. 

The reason? The system was only looking at the **Symmetric** part of the lag covariance matrix, which represents raw power and is completely blind to time, direction, and phase rotation.

## 2. The Breakthrough

This repository resolves the ceiling by splitting the lag covariance matrix $C$ into its two natural halves:
* **Symmetric Part ($S$)**: Measures raw power and amplitude (direction-blind).
* **Skew/Antisymmetric Part ($A$)**: Captures rotation, chirality, and the arrow of time.

By diagonalizing the Skew operator ($A$), the spectral islands emerge organically. The conjugate eigenpairs of $A$ *are* the spectral islands, and the sign of their frequency ($\text{sign}(\omega_j)$) dictates their chirality. The read/write asymmetry is closed, and no hand-built structures remain.

## 3. Repository Structure

* `spectral_islands.py`: The core proof. Demonstrates that diagonalizing the skew matrix provides the islands for free, and mathematically explains why the symmetric v8 approach failed.
* `probe2.py`: Tracks rotation modes across forward and reverse time. It verifies that the older, hand-built angular momentum ($L_k$) calculations are just one basis-dependent shadow of this exact same skew operator.
* `coverage_test2.py`: A rigorous benchmark proving that when restricted to a limited plane budget, skew eigenmodes optimally capture directed rotational energy, whereas symmetric eigenvectors fail entirely at capturing direction.
* `THESIS.md`: The complete, honest mathematical ledger detailing the gap in v8 and the linear algebra that closes it.

## 4. Getting Started
The codebase relies purely on standard matrix operations. 

**Prerequisites:**
```bash
pip install numpy
```
Running the proofs:

```Bash
python spectral_islands.py
python probe2.py
python coverage_test2.py
```
