#!/bin/bash

# Fix all imports from gonum.org/v1/gonum to github.com/guda/guda
echo "Fixing import paths from gonum.org/v1/gonum to github.com/guda/guda..."

# Find all Go files and fix imports
find . -name "*.go" -type f ! -path "./.git/*" ! -path "./gonum/*" -exec sed -i 's|gonum\.org/v1/gonum|github.com/guda/guda|g' {} +

echo "Import paths updated!"