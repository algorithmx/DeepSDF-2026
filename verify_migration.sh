#!/bin/bash

# Verification script for DeepSDF migration
# Checks that all source files exist before migration

EVOLVED_REPO="/opt/data/DeepSDF_cloud/DeepSDF_apr_30"

echo "Verifying migration script..."
echo "Source repo: $EVOLVED_REPO"
echo ""

# Check if source repo exists
if [ ! -d "$EVOLVED_REPO" ]; then
    echo "❌ ERROR: Source repository does not exist at $EVOLVED_REPO"
    exit 1
fi

echo "✅ Source repository exists"
echo ""

# List of files that should exist
declare -a required_files=(
    "$EVOLVED_REPO/bin"
    "$EVOLVED_REPO/bin/PreprocessMesh.py"
    "$EVOLVED_REPO/bin/SampleMeshSurface.py"
    "$EVOLVED_REPO/create_experiment.py"
    "$EVOLVED_REPO/preprocess_data.py"
    "$EVOLVED_REPO/train_deep_sdf.py"
    "$EVOLVED_REPO/evaluate.py"
    "$EVOLVED_REPO/deep_sdf/async_loader.py"
    "$EVOLVED_REPO/deep_sdf/data.py"
    "$EVOLVED_REPO/deep_sdf/watchdog.py"
    "$EVOLVED_REPO/deep_sdf/mesh.py"
    "$EVOLVED_REPO/deep_sdf/utils.py"
    "$EVOLVED_REPO/deep_sdf/experiment_config.py"
    "$EVOLVED_REPO/scripts/monitor_usage.py"
    "$EVOLVED_REPO/scripts/plot_training_logs.py"
    "$EVOLVED_REPO/docs"
    "$EVOLVED_REPO/research"
)

# Check each file
all_good=true
for file in "${required_files[@]}"; do
    if [ -e "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ MISSING: $file"
        all_good=false
    fi
done

echo ""
if [ "$all_good" = true ]; then
    echo "✅ All required files exist! Migration should be safe."
else
    echo "❌ Some files are missing. Migration may fail."
    echo "Check the missing files above."
    exit 1
fi

echo ""
echo "=== Current Directory Contents ==="
ls -la

echo ""
echo "=== Migration Plan ==="
echo "The following will be REMOVED:"
echo "- src/ directory"
echo "- third-party/ directory"
echo "- CMakeLists.txt"
echo "- .gitattributes"
echo ""
echo "The following will be COPIED/overwritten:"
echo "- All files from $EVOLVED_REPO"
echo ""
echo "This script only runs on the target repository (DeepSDF-2026),"
echo "and does NOT modify the evolved repository in any way."
echo ""
echo "Ready to run migration with './migrate.sh'"