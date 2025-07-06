#!/bin/bash

# ClawFoxyVision Pre-commit Hook
# This script runs quality checks before allowing a commit

set -e

echo "🔍 Running pre-commit quality checks..."
echo "======================================"

# Run the quality check script
if ./scripts/quality_check.sh; then
    echo ""
    echo "✅ Pre-commit checks passed! Proceeding with commit..."
    exit 0
else
    echo ""
    echo "❌ Pre-commit checks failed! Please fix the issues before committing."
    echo ""
    echo "Common fixes:"
    echo "  - Run 'cargo fmt' to fix formatting"
    echo "  - Run 'cargo clippy' to fix linting issues"
    echo "  - Add tests for new code"
    echo "  - Update documentation"
    echo "  - Add examples for new features"
    echo ""
    echo "Refer to .cursorrules for detailed requirements."
    exit 1
fi 