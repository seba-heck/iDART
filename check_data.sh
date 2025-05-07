#!/bin/bash

# List of required files/directories
default_required_paths=(
  "./mld_denoiser"
  "./policy_train"
  "./mvae"
  "./data"
  "./data/smplx_lockedhead_20230207"
  "./data/smplx_lockedhead_20230207/models_lockedhead/smplh"
  "./data/smplx_lockedhead_20230207/models_lockedhead/smplx"
)

required_paths=("${default_required_paths[@]}" "$@")

echo "Checking required files and directories with permissions..."

all_exist=true
for path in "${required_paths[@]}"; do
  if [ -e "$path" ]; then
    echo "[OK] Found: $path"

    # Check read permission
    if [ -r "$path" ]; then
      echo "    [READ]  ✅ You have read access."
    else
      echo "    [READ]  ❌ No read access!"
      all_exist=false
    fi

    # Check write permission
    if [ -w "$path" ]; then
      echo "    [WRITE] ✅ You have write/overwrite access."
    else
      echo "    [WRITE] ❌ No write access!"
      all_exist=false
    fi

  else
    echo "[MISSING] ❌ Not found: $path"
    all_exist=false
  fi
done

if [ "$all_exist" = true ]; then
  echo -e "\n✅ All required files/directories are present and accessible."
else
  echo -e "\n❌ Some required paths are missing or lack sufficient permissions."
  exit 1
fi
