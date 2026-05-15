#!/usr/bin/env python3
"""
Sobol' Sensitivity Analysis CLI for DeepSDF.

Orchestrates spatial Sobol' and PCA-decorrelated Sobol latent sensitivity
analysis on a trained DeepSDF model. All analysis logic lives in
``deep_sdf.sensitivity`` — this script only handles CLI argument parsing,
I/O, and orchestration.

Usage:
    python sobol_sensitivity.py -e experiments/my_exp -c 2000
    python sobol_sensitivity.py -e experiments/my_exp -c 2000 --N 1024
    python sobol_sensitivity.py -e experiments/my_exp -c 2000 --device cpu
    python sobol_sensitivity.py -e experiments/my_exp -c 2000 \
        --num-latent-samples 100 --output-metric abs_sdf

Output:
    <experiment_dir>/SensitivityAnalysis/<checkpoint>/
    ├── sobol_spatial_indices.csv
    ├── pca_sobol_indices.csv
    ├── pc_loadings.csv
    ├── sensitivity_summary.json
    └── plots/
        ├── sobol_S1_ST_bar.png
        ├── pca_sobol_bar.png
        ├── pc_importance_ranking.png
        ├── spatial_vs_pc_importance.png
        ├── convergence_sobol_spatial.png
        └── convergence_sobol_pca.png
"""

import argparse
import logging
import os
import sys

import numpy as np

import deep_sdf
from deep_sdf.sensitivity import (
    SpatialSobolAnalyzer,
    LatentPCASobolAnalyzer,
    load_decoder_for_analysis,
    load_latent_codes_numpy,
    check_convergence,
    plot_sobol_spatial_bar,
    plot_pca_sobol_bar,
    plot_pc_importance_ranking,
    plot_spatial_vs_pc_importance,
    plot_convergence_diagnostic,
    save_results_csv,
)

logger = logging.getLogger(__name__)


def main():
    arg_parser = argparse.ArgumentParser(
        description="Run Sobol' and PCA Sobol sensitivity analysis on a trained DeepSDF model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python sobol_sensitivity.py -e experiments/abc_128_64 -c 2000
    python sobol_sensitivity.py -e experiments/abc_128_64 -c 2000 --N 1024 --pca-variance 0.95
    python sobol_sensitivity.py -e experiments/abc_128_64 -c 2000 --device cpu
    python sobol_sensitivity.py -e experiments/abc_128_64 -c 2000 \\
        --num-latent-samples 100 --output-metric sdf
        """,
    )

    # --- Required / core args ---
    arg_parser.add_argument(
        "--experiment", "-e",
        dest="experiment_directory",
        required=True,
        help="Path to the experiment directory (must contain specs.json).",
    )
    arg_parser.add_argument(
        "--checkpoint", "-c",
        dest="checkpoint",
        default="latest",
        help="Checkpoint epoch number, or 'latest' (default: latest).",
    )

    # --- Analysis parameters ---
    arg_parser.add_argument(
        "--N",
        dest="N",
        type=int,
        default=512,
        help="Base sample count for Sobol' sampling (default: 512).",
    )
    arg_parser.add_argument(
        "--max-batch",
        dest="max_batch",
        type=int,
        default=262144,
        help="GPU batch size for inference (default: 262144).",
    )
    arg_parser.add_argument(
        "--output-metric",
        dest="output_metric",
        choices=["both", "sdf", "abs_sdf"],
        default="both",
        help="Sensitivity output metric: 'both', 'sdf', or 'abs_sdf' (default: both).",
    )
    arg_parser.add_argument(
        "--spatial-bounds",
        dest="spatial_bounds",
        nargs=6,
        type=float,
        default=[-1, 1, -1, 1, -1, 1],
        help="Six floats for xyz bounds: xmin xmax ymin ymax zmin zmax "
        "(default: -1 1 -1 1 -1 1).",
    )
    arg_parser.add_argument(
        "--pca-variance",
        dest="pca_variance",
        type=float,
        default=0.95,
        help="Fraction of latent variance to retain via PCA (default: 0.95).",
    )
    arg_parser.add_argument(
        "--num-latent-samples",
        dest="num_latent_samples",
        type=int,
        default=0,
        help="Number of random latent codes to sample (default: 0 = all).",
    )
    arg_parser.add_argument(
        "--device",
        dest="device",
        default="cuda",
        help="Torch device string (default: cuda).",
    )
    arg_parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Override output directory path "
        "(default: <experiment_dir>/SensitivityAnalysis/<checkpoint>/).",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()
    deep_sdf.configure_logging(args)

    # ---- Validate experiment directory -----------------------------------
    exp_dir = args.experiment_directory
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # ---- Load decoder ----------------------------------------------------
    logger.info(
        "Loading decoder from %s (checkpoint: %s)", exp_dir, args.checkpoint
    )
    try:
        decoder, latent_size, xyz_dim, specs = load_decoder_for_analysis(
            exp_dir, args.checkpoint
        )
    except RuntimeError as e:
        if "CUDA" in str(e) or "cuda" in str(e).lower():
            logger.error(
                "CUDA error: %s\nTry --device cpu if no GPU is available.", e
            )
            sys.exit(1)
        raise

    # ---- Load latent codes ------------------------------------------------
    logger.info(
        "Loading latent codes from %s (checkpoint: %s)", exp_dir, args.checkpoint
    )
    codes_array, code_length = load_latent_codes_numpy(exp_dir, args.checkpoint)

    if code_length != latent_size:
        raise ValueError(
            f"Latent code length ({code_length}) does not match "
            f"decoder latent size ({latent_size})."
        )

    # ---- Optional subsampling --------------------------------------------
    if (
        args.num_latent_samples > 0
        and args.num_latent_samples < codes_array.shape[0]
    ):
        rng = np.random.default_rng(42)
        indices = rng.choice(
            codes_array.shape[0], size=args.num_latent_samples, replace=False
        )
        codes_array = codes_array[indices]
        logger.info(
            "Subsampled %d latent codes from %d total",
            codes_array.shape[0],
            args.num_latent_samples,
        )

    if xyz_dim != 3:
        logger.warning(
            "xyz_dim=%d (not 3) — positional encoding active. "
            "3 raw coordinates are mapped to %d engineered features "
            "via feat_eng() before decoder evaluation.",
            xyz_dim,
            xyz_dim,
        )

    # ---- Determine output directory --------------------------------------
    output_dir = args.output_dir or os.path.join(
        exp_dir, "SensitivityAnalysis", args.checkpoint
    )
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # ---- Build spatial bounds list ---------------------------------------
    sb = args.spatial_bounds
    spatial_bounds = [[sb[0], sb[1]], [sb[2], sb[3]], [sb[4], sb[5]]]

    # =====================================================================
    #  Sobol' spatial analysis
    # =====================================================================
    logger.info("=== Sobol' Spatial Sensitivity Analysis ===")
    sobol_analyzer = SpatialSobolAnalyzer(
        decoder=decoder,
        latent_codes=codes_array,
        latent_size=latent_size,
        xyz_dim=xyz_dim,
        device=args.device,
        max_batch=args.max_batch,
    )
    sobol_analyzer.define_problem(spatial_bounds)
    sobol_samples = sobol_analyzer.generate_samples(args.N)
    sobol_eval = sobol_analyzer.evaluate_samples(
        sobol_samples, output_metric=args.output_metric
    )

    sobol_metric_results = {}
    for metric_key, Y in sobol_eval.items():
        logger.info("Computing Sobol indices for metric: %s", metric_key)
        si = sobol_analyzer.compute_indices(Y)
        si["output_metric"] = metric_key
        n_unreliable = len(check_convergence(si))
        logger.info(
            "Sobol [%s]: %d dimension(s) flagged as unreliable",
            metric_key,
            n_unreliable,
        )
        sobol_metric_results[metric_key] = si

    # Use 'sdf' as primary for plotting (or first available)
    primary_sobol_key = (
        "sdf" if "sdf" in sobol_metric_results
        else next(iter(sobol_metric_results))
    )
    sobol_results = sobol_metric_results[primary_sobol_key]

    # =====================================================================
    #  PCA Sobol latent analysis
    # =====================================================================
    logger.info("=== PCA Sobol Latent Sensitivity Analysis ===")
    pca_analyzer = LatentPCASobolAnalyzer(
        decoder=decoder,
        latent_codes=codes_array,
        latent_size=latent_size,
        xyz_dim=xyz_dim,
        device=args.device,
        max_batch=args.max_batch,
        pca_variance=args.pca_variance,
    )
    pca_analyzer.define_problem()
    pca_samples = pca_analyzer.generate_samples(args.N)
    pca_eval = pca_analyzer.evaluate_samples(
        pca_samples, output_metric=args.output_metric
    )

    pca_metric_results = {}
    for metric_key, Y in pca_eval.items():
        logger.info("Computing Sobol indices for metric: %s", metric_key)
        pi = pca_analyzer.compute_indices(Y)
        pi["output_metric"] = metric_key
        n_unreliable = len(check_convergence(pi))
        logger.info(
            "PCA Sobol [%s]: %d PC(s) flagged as unreliable",
            metric_key, n_unreliable,
        )
        pca_metric_results[metric_key] = pi

    primary_pca_key = (
        "sdf" if "sdf" in pca_metric_results
        else next(iter(pca_metric_results))
    )
    pca_results = pca_metric_results[primary_pca_key]

    # =====================================================================
    #  Visualizations
    # =====================================================================
    logger.info("=== Generating Visualizations ===")
    plot_sobol_spatial_bar(
        sobol_results,
        os.path.join(plots_dir, "sobol_S1_ST_bar.png"),
    )
    plot_pca_sobol_bar(
        pca_results,
        os.path.join(plots_dir, "pca_sobol_bar.png"),
    )
    plot_pc_importance_ranking(
        pca_results,
        os.path.join(plots_dir, "pc_importance_ranking.png"),
    )
    plot_spatial_vs_pc_importance(
        sobol_results, pca_results,
        os.path.join(plots_dir, "spatial_vs_pc_importance.png"),
    )
    plot_convergence_diagnostic(sobol_results, pca_results, plots_dir)

    # =====================================================================
    #  Save results
    # =====================================================================
    save_results_csv(sobol_results, pca_results, output_dir)

    # =====================================================================
    #  Summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Experiment:       {exp_dir}")
    print(f"  Checkpoint:       {args.checkpoint}")
    print(f"  Latent codes:     {codes_array.shape[0]} shapes "
          f"× {code_length} dims")
    print(f"  PCA:              {pca_results['k']} PCs "
          f"({pca_results['explained_variance']:.1%} variance)")
    print(f"  Sobol N:          {args.N} "
          f"({sobol_samples.shape[0]:,} spatial + "
          f"{pca_samples.shape[0]:,} PC samples)")
    print(f"  Output metric:    {args.output_metric}")
    print(f"  Primary plot key: {primary_sobol_key} / {primary_pca_key}")
    print(f"  Output dir:       {output_dir}")

    # Spatial Sobol summary
    print(f"\n  Sobol' spatial [{primary_sobol_key}]:")
    for i, name in enumerate(sobol_results["names"]):
        print(f"    {name}: S₁={sobol_results['S1'][i]:.4f}, "
              f"S_T={sobol_results['ST'][i]:.4f}")

    # PCA Sobol summary — top-5 PCs by S_T
    pc_ST = np.array(pca_results["ST"])
    pc_names = pca_results["names"]
    top5_idx = np.argsort(pc_ST)[::-1][:5]
    print(f"\n  PCA Sobol [{primary_pca_key}] — top-5 PCs by S_T:")
    for idx in top5_idx:
        print(f"    {pc_names[idx]}: S_T={pc_ST[idx]:.4f}")

    print("")


if __name__ == "__main__":
    main()
