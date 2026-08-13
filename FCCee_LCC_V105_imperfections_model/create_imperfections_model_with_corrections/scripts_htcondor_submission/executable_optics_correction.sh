#!/bin/bash
# ============================
# HTCondor executable file
# ============================

# To be modified:
# name of the script to be copied from EOS
# name of the folder storing the results
# EOS_BASE path to fetch the script and store the results
# ensure that the file paths in the "Fetch script" and "Copy results" sections are correctly set

set -e

# ---------------- USER SETTINGS ----------------
script_name=Imperfections_model_optics_correction_FINAL.py
results_folder_name=output_lattices_imperfections_correction
EOS_BASE=root://eosproject.cern.ch//eos/user/l/lsievert/08_LCC_V105

echo "=== Starting job for seed $1 ==="

# ---------------- ENV ----------------
export PYTHONNOUSERSITE=1
unset PYTHONPATH CONDA_PYTHON_EXE LD_LIBRARY_PATH
export PATH=/usr/bin:/bin

# ---------------- UNPACK ENV ----------------
ENV_DIR="my_env"
ENV_ARCHIVE="fcc-2026-afs.tar.gz"

mkdir -p "$ENV_DIR"
tar -xzf "$ENV_ARCHIVE" -C "$ENV_DIR"
source "$ENV_DIR/bin/activate"

echo "Using Python: $(which python)"

# ---------------- FETCH SCRIPT ----------------
echo "Copying script from EOS"
xrdcp ${EOS_BASE}/${script_name} .

# ---------------- RUN ----------------
echo "Running Python script"
python ${script_name} "$1"

# ---------------- COPY RESULTS ----------------
echo "Copying results to EOS"

EOS_DIR="${EOS_BASE}/${results_folder_name}"

for f in ${results_folder_name}/*; do
    [ -e "$f" ] || continue
    echo "Uploading $f"
    xrdcp -f "$f" "$EOS_DIR/"
done

echo "=== Job finished successfully ==="
