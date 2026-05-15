#!/bin/bash

# Simple migration script to copy evolved DeepSDF improvements
# Only uses rm and cp commands
# WARNING: This script modifies the current directory by removing original files
# Make sure you're running this from the correct directory!

echo "Starting DeepSDF migration..."
echo "WARNING: This will remove existing src/, third-party/ directories and overwrite several files."
echo "Press Ctrl+C to cancel or wait 3 seconds to continue..."
sleep 3

# Remove original C++ dependencies
echo "Removing original C++ dependencies..."
rm -rf src/ third-party/
rm -f CMakeLists.txt .gitattributes

# Copy evolved repo contents
echo "Copying evolved repository content..."

# Create directories first
mkdir -p bin scripts

# Copy preprocessing tools
echo "Copying preprocessing tools..."
cp -r /opt/data/DeepSDF_cloud/DeepSDF_apr_30/bin/ ./

# Copy enhanced data handling (from deep_sdf directory)
echo "Copying enhanced data handling..."
cp /opt/data/DeepSDF_cloud/DeepSDF_apr_30/deep_sdf/async_loader.py ./deep_sdf/
cp /opt/data/DeepSDF_cloud/DeepSDF_apr_30/deep_sdf/data.py ./deep_sdf/
cp /opt/data/DeepSDF_cloud/DeepSDF_apr_30/deep_sdf/watchdog.py ./deep_sdf/
cp /opt/data/DeepSDF_cloud/DeepSDF_apr_30/deep_sdf/mesh.py ./deep_sdf/
cp /opt/data/DeepSDF_cloud/DeepSDF_apr_30/deep_sdf/utils.py ./deep_sdf/

# Copy utility scripts
echo "Copying utility scripts..."
# Root level scripts
cp /opt/data/DeepSDF_cloud/DeepSDF_apr_30/create_experiment.py ./scripts/
cp /opt/data/DeepSDF_cloud/DeepSDF_apr_30/deep_sdf/experiment_config.py ./scripts/

# Script directory contents
cp /opt/data/DeepSDF_cloud/DeepSDF_apr_30/scripts/monitor_usage.py ./scripts/monitor.py
cp /opt/data/DeepSDF_cloud/DeepSDF_apr_30/scripts/monitor_usage_fast.py ./scripts/monitor_fast.py
cp /opt/data/DeepSDF_cloud/DeepSDF_apr_30/scripts/plot_training_logs.py ./scripts/
cp /opt/data/DeepSDF_cloud/DeepSDF_apr_30/scripts/stage6_interpolate_shapes.py ./scripts/
cp /opt/data/DeepSDF_cloud/DeepSDF_apr_30/scripts/stage6_analyze_latent_codes.py ./scripts/
cp /opt/data/DeepSDF_cloud/DeepSDF_apr_30/scripts/stage6_batch_chamfer_evaluation.py ./scripts/

# Copy main scripts
echo "Copying main scripts..."
cp /opt/data/DeepSDF_cloud/DeepSDF_apr_30/train_deep_sdf.py ./
cp /opt/data/DeepSDF_cloud/DeepSDF_apr_30/preprocess_data.py ./
cp /opt/data/DeepSDF_cloud/DeepSDF_apr_30/evaluate.py ./


# Copy additional useful files
echo "Copying additional files..."
# Create a basic requirements file based on observed files
echo "trimesh>=3.12.0" > requirements.txt
echo "pykdtree" >> requirements.txt
echo "scipy>=1.5.0" >> requirements.txt
echo "numpy" >> requirements.txt
echo "matplotlib" >> requirements.txt
echo "torch>=1.7.0" >> requirements.txt

echo "Migration complete!"
echo ""
echo "Next steps:"
echo "1. Install requirements: pip install -r requirements.txt"
echo "2. Test preprocessing: python preprocess_data.py -d data -s ShapeNet -n ShapeNetV2"
echo "3. Create experiment: python scripts/create_experiment.py -d data/ShapeNet -e experiments/test"
echo "4. Train: python train_deep_sdf.py -e experiments/test"
echo ""
echo "Note: Some files in scripts/ may have different names than expected."
