#!/usr/bin/env python3
"""
Sensitivity Analysis Module for DeepSDF.

Provides Sobol spatial sensitivity analysis and PCA-decorrelated Sobol
latent sensitivity analysis for trained DeepSDF decoders. All analysis
functions, batched inference, convergence diagnostics, visualizations,
and CSV output live here.

Usage:
    from deep_sdf.sensitivity import (
        SpatialSobolAnalyzer,
        LatentPCASobolAnalyzer,
        load_decoder_for_analysis,
        load_latent_codes_numpy,
        batched_sdf_predict,
        check_convergence,
        plot_sobol_spatial_bar,
        plot_pca_sobol_bar,
        plot_pc_importance_ranking,
        plot_spatial_vs_pc_importance,
        plot_convergence_diagnostic,
        save_results_csv,
    )
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import importlib.metadata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import deep_sdf.utils
import deep_sdf.workspace as ws

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Component 2a — Decoder loading
# ---------------------------------------------------------------------------

def load_decoder_for_analysis(
    experiment_dir: str, checkpoint: str
) -> Tuple[torch.nn.Module, int, int, Dict[str, Any]]:
    """Load a trained decoder ready for sensitivity analysis.

    Args:
        experiment_dir: Path to the experiment directory.
        checkpoint: Checkpoint identifier (e.g. ``"2000"``).

    Returns:
        ``(decoder, latent_size, xyz_dim, specs)`` where *decoder* has been
        moved to CUDA and set to ``eval()`` mode.
    """
    specs = ws.load_experiment_specifications(experiment_dir)

    xyz_dim_from_specs: int = specs["NetworkSpecs"].get("xyz_dim", 3)
    deep_sdf.utils.set_xyz_dim(xyz_dim_from_specs)

    arch = __import__(
        "networks." + specs["NetworkArch"],
        fromlist=["Decoder"],
    )

    latent_size: int = specs["CodeLength"]
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    saved_model_state = torch.load(
        os.path.join(
            experiment_dir,
            ws.model_params_subdir,
            f"{checkpoint}.pth",
        ),
        map_location="cpu",
    )
    state_dict = saved_model_state["model_state_dict"]

    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module_prefix:
        state_dict = {"module." + k: v for k, v in state_dict.items()}

    decoder = torch.nn.DataParallel(decoder)
    decoder.load_state_dict(state_dict)
    decoder = decoder.module.cuda()
    decoder.eval()

    logger.info(
        "Loaded decoder | latent_size=%d xyz_dim=%d arch=%s",
        latent_size,
        xyz_dim_from_specs,
        specs["NetworkArch"],
    )

    return decoder, latent_size, xyz_dim_from_specs, specs


# ---------------------------------------------------------------------------
#  Component 2b — Latent code loading
# ---------------------------------------------------------------------------

def load_latent_codes_numpy(
    experiment_dir: str, checkpoint: str
) -> Tuple[np.ndarray, int]:
    """Load trained latent codes as a flat (N, D) NumPy array.

    Args:
        experiment_dir: Path to the experiment directory.
        checkpoint: Checkpoint identifier.

    Returns:
        ``(codes, code_length)`` where *codes* has shape ``(num_shapes, D)``.
    """
    latent_path = os.path.join(
        experiment_dir, ws.latent_codes_subdir, f"{checkpoint}.pth"
    )

    if not os.path.isfile(latent_path):
        raise FileNotFoundError(f"Latent codes not found: {latent_path}")

    data = torch.load(latent_path, map_location="cpu")

    if isinstance(data.get("latent_codes"), torch.Tensor):
        codes = data["latent_codes"].numpy()
    elif isinstance(data.get("latent_codes"), dict):
        codes = data["latent_codes"]["weight"].numpy()
    else:
        raise ValueError(f"Unknown latent code format in {latent_path}")

    if codes.ndim == 1:
        codes = codes.reshape(1, -1)

    code_length = codes.shape[1]
    logger.info("Loaded %d latent codes of length %d", codes.shape[0], code_length)
    return codes, code_length


# ---------------------------------------------------------------------------
#  Component 2c — Batched SDF prediction
# ---------------------------------------------------------------------------

def batched_sdf_predict(
    decoder: torch.nn.Module,
    input_matrix: np.ndarray,
    max_batch: int = 2**18,
    device: str = "cuda",
) -> torch.Tensor:
    """Run the decoder on a large input matrix in bounded-memory chunks.

    **Unlike** ``decode_sdf()`` (which broadcasts ONE latent), this function
    accepts an input matrix whose **every row** carries its own latent code
    AND spatial coordinates.  The first ``latent_size`` columns are latent
    components; the last ``xyz_dim`` columns are spatial features (already
    engineered if ``xyz_dim > 3``).

    Args:
        decoder: Trained ``Decoder`` in ``eval()`` mode.
        input_matrix: ``(N, latent_size + xyz_dim)`` array (NumPy or torch).
        max_batch: Maximum rows per forward pass (default ``2**18``).
        device: Torch device string.

    Returns:
        Predicted SDF values as a ``(N, 1)`` tensor on CPU.
    """
    if isinstance(input_matrix, torch.Tensor):
        input_tensor = input_matrix.cpu().detach()
        if input_tensor.dtype != torch.float32:
            input_tensor = input_tensor.float()
    else:
        input_tensor = torch.from_numpy(np.asarray(input_matrix, dtype=np.float32))

    N = input_tensor.shape[0]
    outputs: List[torch.Tensor] = []

    with torch.no_grad():
        head = 0
        while head < N:
            tail = min(head + max_batch, N)
            chunk = input_tensor[head:tail].to(device)

            out: torch.Tensor = decoder(chunk)

            if torch.isnan(out).any() or torch.isinf(out).any():
                logger.warning(
                    "NaN or Inf detected in decoder output for chunk [%d:%d]. "
                    "Replacing with 0.0.",
                    head,
                    tail,
                )
                out = torch.where(
                    torch.isnan(out) | torch.isinf(out),
                    torch.zeros_like(out),
                    out,
                )

            outputs.append(out.cpu())
            head = tail

    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
#  Component 2d — Spatial Sobol analyzer
# ---------------------------------------------------------------------------

class SpatialSobolAnalyzer:
    """Sobol' global sensitivity analysis on the *spatial* (xyz) dimensions.

    The spatial dimensions are treated as **independent** inputs.  For each
    Sobol sample a different trained latent code is drawn randomly so that
    the sensitivity estimates are marginalised over the shape population.
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        latent_codes: np.ndarray,
        latent_size: int,
        xyz_dim: int,
        device: str = "cuda",
        max_batch: int = 2**18,
    ) -> None:
        """
        Args:
            decoder: Trained DeepSDF decoder in ``eval()`` mode.
            latent_codes: ``(num_shapes, latent_size)`` NumPy array.
            latent_size: Dimensionality of a single latent code.
            xyz_dim: Number of spatial dimensions (typically 3).
            device: Torch device string.
            max_batch: Max rows per decoder forward pass.
        """
        self.decoder = decoder
        self.latent_codes = latent_codes
        self.latent_size = latent_size
        self.xyz_dim = xyz_dim
        self.device = device
        self.max_batch = max_batch
        self.problem: Optional[Dict[str, Any]] = None
        self._N: Optional[int] = None

    # ------------------------------------------------------------------
    def define_problem(
        self,
        spatial_bounds: List[List[float]],
    ) -> Dict[str, Any]:
        """Create the SALib problem dict for spatial sensitivity.

        Args:
            spatial_bounds: Per-dimension ``[[low, high], ...]`` (length *xyz_dim*).

        Returns:
            SALib problem dict.
        """
        # Sobol always operates on the 3 raw spatial coordinates,
        # regardless of whether feat_eng() later expands them to xyz_dim.
        # Sampling in the engineered feature space (e.g. 51-D Fourier
        # features) is meaningless — those dimensions are deterministic
        # functions of (x,y,z), not independent variables.
        num_vars = 3
        names = ["x", "y", "z"]

        self.problem = {
            "num_vars": num_vars,
            "names": names,
            "bounds": spatial_bounds,
        }
        return self.problem

    # ------------------------------------------------------------------
    def generate_samples(self, N: int) -> np.ndarray:
        """Draw Saltelli (Sobol') samples.

        Args:
            N: Base sample size (total samples = ``N * (2 * D + 2)``).

        Returns:
            Array of shape ``(N_total, xyz_dim)``.
        """
        if self.problem is None:
            raise RuntimeError("Call define_problem() before generate_samples().")

        from SALib.sample import saltelli

        self._N = N
        samples = saltelli.sample(self.problem, N, calc_second_order=True)
        logger.info("Generated %d Sobol spatial samples (N=%d)", samples.shape[0], N)
        return samples

    # ------------------------------------------------------------------
    def evaluate_samples(
        self,
        samples: np.ndarray,
        output_metric: str = "both",
    ) -> Dict[str, np.ndarray]:
        """Evaluate the decoder on every Sobol sample.

        For each sample row a latent code is drawn uniformly at random from
        the available training codes.

        Args:
            samples: ``(N_total, xyz_dim)`` array from :meth:`generate_samples`.
            output_metric:
                - ``"sdf"``     — raw SDF values.
                - ``"abs_sdf"`` — absolute SDF values.
                - ``"both"``    — dict with ``"sdf"`` and ``"abs_sdf"``.

        Returns:
            Dict mapping output_metric key(s) to ``(N_total,)`` NumPy arrays.
        """
        num_samples = samples.shape[0]
        num_shapes = self.latent_codes.shape[0]

        idx = np.random.randint(0, num_shapes, size=num_samples)
        chosen_latents = self.latent_codes[idx]

        # Expand raw (x,y,z) Sobol samples through feat_eng() to match
        # the decoder's expected input dimensionality.  Identity when
        # xyz_dim==3; NeRF Fourier features when xyz_dim==51.
        samples_tensor = torch.from_numpy(samples.astype(np.float32))
        engineered_spatial = deep_sdf.utils.feat_eng(samples_tensor).numpy()
        input_matrix = np.concatenate([chosen_latents, engineered_spatial], axis=1)

        sdf_tensor = batched_sdf_predict(
            self.decoder,
            input_matrix,
            max_batch=self.max_batch,
            device=self.device,
        )
        sdf_vals = sdf_tensor.numpy().ravel()

        logger.info("Evaluated %d Sobol samples", num_samples)

        if output_metric == "sdf":
            return {"sdf": sdf_vals}
        elif output_metric == "abs_sdf":
            return {"abs_sdf": np.abs(sdf_vals)}
        else:
            return {"sdf": sdf_vals, "abs_sdf": np.abs(sdf_vals)}

    # ------------------------------------------------------------------
    def compute_indices(
        self, Y: np.ndarray
    ) -> Dict[str, Any]:
        """Run SALib Sobol analysis.

        Args:
            Y: ``(N_total,)`` output values from :meth:`evaluate_samples`.

        Returns:
            Dict with keys: ``S1``, ``ST``, ``S1_conf``, ``ST_conf``, ``names``,
            ``N``, ``output_metric``, ``num_vars``.
        """
        if self.problem is None:
            raise RuntimeError("Call define_problem() before compute_indices().")

        from SALib.analyze import sobol as sobol_analyze

        Si = sobol_analyze.analyze(self.problem, Y, calc_second_order=True)

        # SALib 1.5.x does not include 'names' in Sobol results — take from problem
        names = Si.get("names", self.problem["names"])

        return {
            "S1": Si["S1"],
            "ST": Si["ST"],
            "S1_conf": Si["S1_conf"],
            "ST_conf": Si["ST_conf"],
            "S2": Si.get("S2", None),
            "S2_conf": Si.get("S2_conf", None),
            "names": names,
            "N": self._N,
            "num_vars": self.xyz_dim,
        }


# ---------------------------------------------------------------------------
#  Component 2e — PCA Sobol latent analyzer
# ---------------------------------------------------------------------------

class LatentPCASobolAnalyzer:
    """Sobol' sensitivity analysis on PCA-decorrelated latent space.

    Strategy:
    1. Fit PCA on training latent codes → find k PCs explaining the target
       variance fraction (default 95%).
    2. The k PC scores are **uncorrelated by construction** — Sobol's
       independence assumption holds.
    3. Saltelli sampling in the k-dimensional PC score space.
    4. Each sample reconstructs a full latent code via inverse PCA, then
       passes through the decoder with random spatial coordinates.
    5. Sobol' indices (S₁, S_T) quantify the influence of each PC direction.
    6. PC loadings matrix maps results back to original latent dimensions:
       which original dims load heavily on influential PCs?
    """

    def __init__(self, decoder, latent_codes, latent_size, xyz_dim,
                 device="cuda", max_batch=2**18, pca_variance=0.95):
        self.decoder = decoder
        self.latent_codes = latent_codes  # (num_shapes, latent_size)
        self.latent_size = latent_size
        self.xyz_dim = xyz_dim
        self.device = device
        self.max_batch = max_batch
        self.pca_variance = pca_variance
        self.problem = None
        self._N = None
        self._xyz_samples = None  # pre-generated spatial context pool

        from sklearn.decomposition import PCA
        self.pca = PCA().fit(latent_codes)
        cumulative = np.cumsum(self.pca.explained_variance_ratio_)
        self.k = int(np.searchsorted(cumulative, pca_variance) + 1)
        self.k = max(self.k, 2)  # need at least 2 for Sobol

        self.pc_scores = self.pca.transform(latent_codes)[:, :self.k]
        self.pc_min = self.pc_scores.min(axis=0)
        self.pc_max = self.pc_scores.max(axis=0)

        logger.info(
            "PCA latent reduction: %d dims → %d PCs (%.1f%% variance)",
            latent_size, self.k, 100 * cumulative[self.k - 1],
        )

    def define_problem(self):
        names = [f"PC_{i+1}" for i in range(self.k)]
        bounds = [[float(self.pc_min[i]), float(self.pc_max[i])] for i in range(self.k)]
        self.problem = {"num_vars": self.k, "names": names, "bounds": bounds}

        rng = np.random.default_rng(42)
        self._xyz_samples = rng.uniform(-1, 1, size=(10000, 3)).astype(np.float32)
        logger.debug("Pre-generated %d spatial context points", self._xyz_samples.shape[0])

        return self.problem

    def generate_samples(self, N):
        from SALib.sample import saltelli
        self._N = N
        samples = saltelli.sample(self.problem, N, calc_second_order=True)
        logger.info(
            "Generated %d Sobol PC samples (N=%d, k=%d)",
            samples.shape[0], N, self.k,
        )
        return samples  # (N_total, k)

    def evaluate_samples(self, pc_samples, output_metric="both"):
        num_samples = pc_samples.shape[0]

        pc_rescaled = pc_samples * (self.pc_max - self.pc_min) + self.pc_min

        latent_reconstructed = (
            pc_rescaled @ self.pca.components_[:self.k, :]
            + self.pca.mean_
        ).astype(np.float32)

        idx = np.random.randint(0, self._xyz_samples.shape[0], size=num_samples)
        chosen_xyz = self._xyz_samples[idx]

        xyz_tensor = torch.from_numpy(chosen_xyz.astype(np.float32))
        engineered_xyz = deep_sdf.utils.feat_eng(xyz_tensor).numpy()

        input_matrix = np.concatenate([latent_reconstructed, engineered_xyz], axis=1)

        sdf_tensor = batched_sdf_predict(
            self.decoder, input_matrix,
            max_batch=self.max_batch, device=self.device,
        )
        sdf_vals = sdf_tensor.numpy().ravel()
        logger.info("Evaluated %d Sobol PC samples", num_samples)

        if output_metric == "sdf":
            return {"sdf": sdf_vals}
        elif output_metric == "abs_sdf":
            return {"abs_sdf": np.abs(sdf_vals)}
        else:
            return {"sdf": sdf_vals, "abs_sdf": np.abs(sdf_vals)}

    def compute_indices(self, Y):
        from SALib.analyze import sobol as sobol_analyze

        Si = sobol_analyze.analyze(self.problem, Y, calc_second_order=True)
        names = Si.get("names", self.problem["names"])

        loadings = self.pca.components_[:self.k, :]

        return {
            "S1": Si["S1"],
            "ST": Si["ST"],
            "S1_conf": Si["S1_conf"],
            "ST_conf": Si["ST_conf"],
            "S2": Si.get("S2", None),
            "S2_conf": Si.get("S2_conf", None),
            "names": names,
            "N": self._N,
            "k": self.k,
            "explained_variance": float(
                np.sum(self.pca.explained_variance_ratio_[:self.k])
            ),
            "pc_loadings": loadings.tolist(),
            "latent_dim_names": [f"latent_{i}" for i in range(self.latent_size)],
            "pca_variance_threshold": self.pca_variance,
        }


# ---------------------------------------------------------------------------
#  Component 2f — Convergence diagnostic
# ---------------------------------------------------------------------------

def check_convergence(
    results: Dict[str, Any],
) -> List[Tuple[str, float, float]]:
    """Flag dimensions whose confidence interval exceeds the point estimate.

    Works with Sobol-format results dicts (S1/S1_conf/ST/ST_conf). Both
    spatial and PCA Sobol analyzers produce this format.

    Args:
        results: Dict from ``SpatialSobolAnalyzer.compute_indices()``
            or ``LatentPCASobolAnalyzer.compute_indices()``.

    Returns:
        List of ``(dim_name, point_estimate, confidence_width)`` tuples for
        unreliable dimensions.
    """
    unreliable: List[Tuple[str, float, float]] = []
    names = results.get("names", [])

    for i, name in enumerate(names):
        s1 = float(results["S1"][i])
        s1_ci = float(results["S1_conf"][i])
        st = float(results["ST"][i])
        st_ci = float(results["ST_conf"][i])

        if s1_ci > abs(s1):
            unreliable.append((name, s1, s1_ci))
            logger.warning(
                "Sobol S1 for '%s': CI (%.4f) > point estimate (%.4f) — unreliable",
                name,
                s1_ci,
                s1,
            )
        if st_ci > abs(st):
            unreliable.append((name, st, st_ci))
            logger.warning(
                "Sobol ST for '%s': CI (%.4f) > point estimate (%.4f) — unreliable",
                name,
                st_ci,
                st,
            )

    if not unreliable:
        logger.info("All dimensions have acceptable confidence intervals.")
    else:
        logger.warning("%d dimension(s) flagged as unreliable.", len(unreliable))

    return unreliable


# ---------------------------------------------------------------------------
#  Component 2g — Visualization functions
# ---------------------------------------------------------------------------

def plot_sobol_spatial_bar(
    results: Dict[str, Any],
    output_path: str,
) -> None:
    """Grouped bar chart of S₁ and S_T for spatial dimensions with error bars.

    Args:
        results: Dict from ``SpatialSobolAnalyzer.compute_indices()``.
        output_path: File path for the saved figure (PNG).
    """
    names = results["names"]
    S1 = np.array(results["S1"])
    ST = np.array(results["ST"])
    S1_conf = np.array(results["S1_conf"])
    ST_conf = np.array(results["ST_conf"])

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, S1, width, yerr=S1_conf, label="S₁ (first-order)",
           capsize=4, color="steelblue")
    ax.bar(x + width / 2, ST, width, yerr=ST_conf, label="S_T (total-order)",
           capsize=4, color="darkorange")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Sobol' index")
    ax.set_title("Spatial Sobol' Sensitivity — S₁ and S_T")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Sobol spatial bar chart saved to %s", output_path)


def plot_pca_sobol_bar(results, output_path, top_k=20):
    """Grouped bar chart of S₁ and S_T for top-k PC dimensions."""
    names = results["names"]
    S1 = np.array(results["S1"])
    ST = np.array(results["ST"])
    S1_conf = np.array(results["S1_conf"])
    ST_conf = np.array(results["ST_conf"])

    top_idx = np.argsort(ST)[::-1][:min(top_k, len(names))]
    names = [names[i] for i in top_idx]
    S1 = S1[top_idx]
    ST = ST[top_idx]
    S1_conf = S1_conf[top_idx]
    ST_conf = ST_conf[top_idx]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.4), 5))
    ax.bar(x - width / 2, S1, width, yerr=S1_conf, label="S₁ (first-order)",
           capsize=3, color="steelblue")
    ax.bar(x + width / 2, ST, width, yerr=ST_conf, label="S_T (total-order)",
           capsize=3, color="darkorange")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45 if len(names) > 10 else 0, ha="right")
    ax.set_ylabel("Sobol' index")
    ax.set_title(f"PCA Sobol' Sensitivity — Top-{len(names)} PCs (S₁ and S_T)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("PCA Sobol bar chart saved to %s", output_path)


def plot_pc_importance_ranking(sobol_results, output_path, top_k=20):
    """Horizontal bar chart of top-k PC dimensions ranked by S_T."""
    ST = np.array(sobol_results["ST"])
    ST_conf = np.array(sobol_results["ST_conf"])
    names = np.array(sobol_results["names"])

    top_idx = np.argsort(ST)[::-1][:top_k]
    top_st = ST[top_idx]
    top_conf = ST_conf[top_idx]
    top_names = names[top_idx]

    y = np.arange(len(top_names))

    fig, ax = plt.subplots(figsize=(8, max(5, top_k * 0.3)))
    ax.barh(y, top_st, xerr=top_conf, capsize=3, color="mediumseagreen")
    ax.set_yticks(y)
    ax.set_yticklabels(top_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("S_T (total-effect index)")
    ax.set_title(f"Top-{top_k} PC Directions by S_T")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("PC importance ranking saved to %s", output_path)


def plot_spatial_vs_pc_importance(spatial_results, pc_results, output_path):
    """Bar chart comparing spatial (max S_T) vs PC (max/mean S_T)."""
    max_spatial_ST = float(np.max(spatial_results["ST"]))
    max_pc_ST = float(np.max(pc_results["ST"]))
    mean_pc_ST = float(np.mean(pc_results["ST"]))

    labels = ["Max S_T\n(spatial)", "Max S_T\n(PCs)", "Mean S_T\n(PCs)"]
    values = [max_spatial_ST, max_pc_ST, mean_pc_ST]
    colors = ["steelblue", "mediumseagreen", "mediumseagreen"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    ax.set_ylabel("Sobol' total-effect index (S_T)")
    ax.set_title("Spatial vs PCA Latent Sensitivity Comparison")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Spatial vs PC comparison saved to %s", output_path)


def plot_convergence_diagnostic(spatial_results, pc_results, output_dir):
    """Convergence diagnostics for spatial Sobol and PCA Sobol.

    Produces: convergence_sobol_spatial.png, convergence_sobol_pca.png
    """
    os.makedirs(output_dir, exist_ok=True)

    def _plot_sobol_convergence(results, filename, title_prefix):
        unreliable = check_convergence(results)
        unreliable_names = {name for name, _, _ in unreliable}
        S1 = np.abs(np.array(results["S1"]))
        S1_conf = np.array(results["S1_conf"])
        ST = np.abs(np.array(results["ST"]))
        ST_conf = np.array(results["ST_conf"])
        names = results["names"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        colors_s1 = ["red" if n in unreliable_names else "steelblue" for n in names]
        ax1.scatter(S1, S1_conf, c=colors_s1, alpha=0.8, edgecolors="k", linewidth=0.3)
        for i, n in enumerate(names):
            if n in unreliable_names:
                ax1.annotate(n, (S1[i], S1_conf[i]), fontsize=7, color="red")
        ax1.set_xlabel("|S₁|")
        ax1.set_ylabel("S₁ confidence interval")
        ax1.set_title(f"{title_prefix} S₁ — Convergence")
        ax1.grid(alpha=0.3)
        ax1max = max(ax1.get_xlim()[1], ax1.get_ylim()[1])
        ax1.plot([0, ax1max], [0, ax1max], "k--", alpha=0.3, label="CI = |S₁|")
        ax1.legend(fontsize=8)

        colors_st = ["red" if n in unreliable_names else "darkorange" for n in names]
        ax2.scatter(ST, ST_conf, c=colors_st, alpha=0.8, edgecolors="k", linewidth=0.3)
        for i, n in enumerate(names):
            if n in unreliable_names:
                ax2.annotate(n, (ST[i], ST_conf[i]), fontsize=7, color="red")
        ax2.set_xlabel("|S_T|")
        ax2.set_ylabel("S_T confidence interval")
        ax2.set_title(f"{title_prefix} S_T — Convergence")
        ax2.grid(alpha=0.3)
        ax2max = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
        ax2.plot([0, ax2max], [0, ax2max], "k--", alpha=0.3, label="CI = |S_T|")
        ax2.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close(fig)

    _plot_sobol_convergence(spatial_results, "convergence_sobol_spatial.png", "Spatial")
    _plot_sobol_convergence(pc_results, "convergence_sobol_pca.png", "PC")
    logger.info("Convergence diagnostics saved to %s", output_dir)


# ---------------------------------------------------------------------------
#  Component 2h — CSV / JSON output
# ---------------------------------------------------------------------------

def save_results_csv(spatial_results, pc_results, output_dir):
    """Save Sobol (spatial + PCA) results as CSV files plus summary JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # ---- Spatial Sobol CSV ----
    with open(os.path.join(output_dir, "sobol_spatial_indices.csv"), "w") as f:
        f.write("name,S1,S1_conf,ST,ST_conf\n")
        for i, name in enumerate(spatial_results["names"]):
            f.write(f"{name},{spatial_results['S1'][i]:.8f},"
                    f"{spatial_results['S1_conf'][i]:.8f},"
                    f"{spatial_results['ST'][i]:.8f},"
                    f"{spatial_results['ST_conf'][i]:.8f}\n")
    logger.info("Spatial Sobol results saved")

    # ---- PCA Sobol CSV ----
    with open(os.path.join(output_dir, "pca_sobol_indices.csv"), "w") as f:
        f.write("name,S1,S1_conf,ST,ST_conf\n")
        for i, name in enumerate(pc_results["names"]):
            f.write(f"{name},{pc_results['S1'][i]:.8f},"
                    f"{pc_results['S1_conf'][i]:.8f},"
                    f"{pc_results['ST'][i]:.8f},"
                    f"{pc_results['ST_conf'][i]:.8f}\n")
    logger.info("PCA Sobol results saved")

    # ---- PC Loadings CSV (k PCs × latent_size original dims) ----
    if "pc_loadings" in pc_results:
        loadings = np.array(pc_results["pc_loadings"])
        latent_names = pc_results.get("latent_dim_names",
                                       [f"latent_{i}" for i in range(loadings.shape[1])])
        with open(os.path.join(output_dir, "pc_loadings.csv"), "w") as f:
            f.write("pc," + ",".join(latent_names) + "\n")
            for i in range(loadings.shape[0]):
                row = ",".join(f"{v:.6f}" for v in loadings[i])
                f.write(f"PC_{i+1},{row}\n")
        logger.info("PC loadings saved")

    # ---- Summary JSON ----
    try:
        salib_version = importlib.metadata.version("SALib")
    except importlib.metadata.PackageNotFoundError:
        salib_version = "unknown"

    summary = {
        "tool": "SALib",
        "salib_version": salib_version,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "spatial_sobol": {
            "N_base": spatial_results.get("N"),
            "num_vars": 3,
            "names": spatial_results.get("names"),
        },
        "pca_sobol": {
            "N_base": pc_results.get("N"),
            "num_vars": pc_results.get("k"),
            "explained_variance": pc_results.get("explained_variance"),
            "pca_variance_threshold": pc_results.get("pca_variance_threshold"),
            "names": pc_results.get("names"),
        },
    }
    with open(os.path.join(output_dir, "sensitivity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary JSON saved")
