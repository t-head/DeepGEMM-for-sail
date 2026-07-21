#!/bin/bash
set -ex
cat /etc/os-release
ls -l
echo $(pwd)

pip uninstall -y deep-gemm || true
source PPU_SDK/envsetup.sh
pip install torch/*.whl
pip install ./*.whl
find -name "deepgemm*.whl" -exec python3 -m pip install {} +
pip install numpy==1.26.4 junit-xml pytest
pip install xtest xtest-ppu-uploader --index-url https://art-pub.eng.t-head.cn/artifactory/api/pypi/ptgai-pypi/simple

export XTEST_PLUGINS=core.plugins
export PPU_TEST_UPLOAD=${PPU_TEST_UPLOAD:-true}
export PPU_ES_INDEX_PREFIX=${PPU_ES_INDEX_PREFIX:-acompute_test}
export PPU_COMPONENT=deep_gemm
export PPU_HOME=/home/image
export XTEST_LOG_DIR=${PPU_HOME}/case_logs

cd /home/image/acompute_v2_test
pip install -e .

XTEST_LOG=/home/image/xtest_output.log
xtest plan test/plans/deepgemm_test/ci/dg_ci_plan.py 2>&1 | tee ${XTEST_LOG}
XTEST_EXIT=${PIPESTATUS[0]}

# 兜底：若 PlanSummaryCollector 未生成 XML，使用 gen_junit_xml.py
XML_OUTPUT=/home/image/test-results.xml
if [ ! -f ${XML_OUTPUT} ]; then
    python3 scripts/gen_junit_xml.py ${XTEST_LOG} --output ${XML_OUTPUT}
fi

exit ${XTEST_EXIT}
