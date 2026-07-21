#!/bin/bash
set -ex
cat /etc/os-release
ls -l
pip uninstall -y deep-gemm || true
source PPU_SDK/envsetup.sh
pip install torch/*.whl
pip install ./*.whl
find -name "deepgemm*.whl" -exec python3 -m pip install {} +
pip install numpy==1.26.4 junit-xml
pip install --upgrade pytest
python ./ci_deepgemm_ut.py
