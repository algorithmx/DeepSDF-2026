#!/bin/bash
set -e
cd /home/vipuser/DeepSDF

iteration=0
max_iterations=100

while ! grep -q "DONE" COMPLETE 2>/dev/null; do
    if [ $iteration -ge $max_iterations ]; then
        echo "Error: Max iterations ($max_iterations) reached without completion"
        exit 1
    fi

    echo "=== Iteration $iteration ==="

    EXTRA_PROMPT=""
    if [ $iteration -gt 0 ]; then
        EXTRA_PROMPT="You have resumed from a previous run. Review the current state of the project and find actual stages to continue. "
    fi

    # Use -p for non-interactive print mode + skip permissions
    if ! claude -p --dangerously-skip-permissions \
        "Follow CLAUDE.md and execute PLAN.md until DONE appears in COMPLETE file. Clearly identify the finished and unfinished items in PLAN.md, update the progress marks whenever appropriate. Current iteration: $iteration. If DONE is found, exit immediately. ${EXTRA_PROMPT}"; then
        echo "Warning: claude exited with non-zero status on iteration $iteration, retrying..."
        sleep 10
    fi

    iteration=$((iteration + 1))
    sleep 1
done

echo "Plan complete!"
