#!/bin/bash
# ============================
# HTCondor executable file
# ============================

# To be modified:
# name of the script to be copied from EOS
# name of the folder storing the results
# EOS_BASE path to fetch the script and store the results
# check if EOS_SUBDIR is needed for copying results and if yes, check the name

set -e

# ---------------- USER SETTINGS ----------------
script_name=DA_MA_script_withBB.py
results_folder_name=Performance_scans_corrected_lattice_withBB
EOS_BASE=root://eosproject.cern.ch//eos/user/l/lsievert/12_LCC_V106
EOS_SUBDIR=OUTPUT_performance_scans

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

EOS_DIR="${EOS_BASE}/${EOS_SUBDIR}/${results_folder_name}"

for f in ${results_folder_name}/*; do
    [ -e "$f" ] || continue
    echo "Uploading $f"
    xrdcp -f "$f" "$EOS_DIR/"
done

echo "=== Job finished successfully ==="
