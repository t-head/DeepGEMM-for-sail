#!/bin/bash
set -ex
echo 'Start deepgemm ut build'
cp -r $AONE_CI_WORKSPACE/opt/PPU_SDK  /usr/local/
export source_home=$AONE_CI_SOURCE
export build_output=$AONE_CI_SOURCE/package
echo "========= Setup PPU SDK ================"
source /usr/local/PPU_SDK/envsetup.sh
export PATH=${source_home}/bin:$PATH
export UMD_PLATFORM_TYPE=0
export ALIPPU_HW_TYPE=MODEL_BEHAVIOR
export HGGC_DRIVER_CANDIDATE=FAKE
echo "Backend=0" >> /usr/local/PPU_SDK/cfgs/fakedriver.config
export FAKEDRIVER_CONFIG_PATH=/usr/local/PPU_SDK/cfgs/fakedriver.config
export ALIPPU_QUIET_BUILD=TRUE
echo "========= Setup Torch  ================"
pip install $AONE_CI_WORKSPACE/opt/torch/*.whl
echo "========= Setup third party  ================"
rm -rf ${source_home}/DeepGemm/third-party/actlize_v0.5.0
rm -rf ${source_home}/DeepGemm/third-party/actlize_v1.0.0
rm -rf ${source_home}/DeepGemm/third-party/fmt
cp -rf ${source_home}/actlize_v0.5.0 ${source_home}/DeepGemm/third-party/actlize_v0.5.0
cp -rf ${source_home}/actlize_v1.0.0 ${source_home}/DeepGemm/third-party/actlize_v1.0.0
cp -rf ${source_home}/fmt ${source_home}/DeepGemm/third-party/fmt
echo "========= Convert actlize_v1.0.0 ================"
cd ${source_home}/DeepGemm/third-party/actlize_v1.0.0
python3 ${source_home}/cudafy-for-sail/cudafy.py actlize --version=1.0.0 include
echo "========= Convert actlize_v0.5.0 ================"
cd ${source_home}/DeepGemm/third-party/actlize_v0.5.0
python3 ${source_home}/cudafy-for-sail/cudafy.py actlize --version=0.5.0 include
echo "========= Start to build deepgemm  ================"
cd ${source_home}/DeepGemm
python3 ${source_home}/cudafy-for-sail/cudafy.py deepgemm ${source_home}/DeepGemm
# in case there exist cache from previous build
rm -rf ./dist || true
python3 setup.py bdist_wheel
find -name "deepgemm*.whl" -exec python3 -m pip install {} +

mkdir -p ${source_home}/output
mv ./dist/*.whl ${source_home}/output
cp .aoneci/scripts/* ${source_home}/output
cp -r ${source_home}/DeepGemm/* ${source_home}/output
cp -r ${source_home}/DeepGemm/.git* ${source_home}/output
cp -rf /usr/local/PPU_SDK ${source_home}/output
cp -rf $AONE_CI_WORKSPACE/opt/torch/ ${source_home}/output/
echo "BUILD INFO: COPY deepgemm succeed!"

# 复制 acompute_v2_test 测试项目到产物中（从 ACTest 仓库 checkout）
export ACTEST_HOME=${source_home}/ACTest
if [ -d "${ACTEST_HOME}/acompute_v2_test" ]; then
    cp -rf ${ACTEST_HOME}/acompute_v2_test ${source_home}/output/acompute_v2_test
    echo "========= acompute_v2_test copied from ACTest repo ========="
else
    echo "WARNING: ACTest/acompute_v2_test not found at ${ACTEST_HOME}/acompute_v2_test"
fi
