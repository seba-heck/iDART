#!/bin/bash

# List of required files/directories
required_paths=(
  "./mld_denoiser"
  "./policy_train"
  "./mvae"
  "./data"
  "./data/smplx_lockedhead_20230207"
  "./data/smplx_lockedhead_20230207/models_lockedhead/smplh"
  "./data/smplx_lockedhead_20230207/models_lockedhead/smplx"
)

echo "Checking required files and directories..."

all_exist=true
for path in "${required_paths[@]}"; do
  if [ -e "$path" ]; then
    echo "[OK] Found: $path"
  else
    echo "[MISSING] Not found: $path"
    all_exist=false
  fi
done

if [ "$all_exist" = true ]; then
  echo "✅ All required files and directories are present."
else
  echo "❌ Some required files or directories are missing."
  exit 1
fi

