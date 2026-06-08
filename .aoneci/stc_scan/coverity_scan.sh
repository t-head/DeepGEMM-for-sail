#! /bin/bash
# coverity默认目录
COV_DIR=''
COV_IDIR='idir'
# 源码目录
CODE_DIR=''
# 编译脚本文件
BUILD_SCRIPT=''
# 编译日志文件
BUILD_LOG='cov_build.log'

# build config
COV_BUILD_CONFIG=''

# 设置coverity目录
function setCoverityHome() {
    covDir=$1
    echo ${covDir}
    # 字符串为空
    if [ -z ${covDir} ];then
	return 1;
    fi

    # 目录是否存在
    if [ ! -d ${covDir} ];then
	return 2;
    fi

    COV_DIR=${covDir}
    return 0
}

# 设置源码目录
function setSourceCodePath() {
    codeDir=$1
    echo ${codeDir}
    # 字符串为空
    if [ -z ${codeDir} ];then
	return 1;
    fi

    # 目录是否存在
    if [ ! -d ${codeDir} ];then
	return 2;
    fi

    CODE_DIR=${codeDir}
    return 0
}

# 编译脚本
function setBuildScript() {
    script=$1
    if [ -z ${script} ];then
	return 1
    fi

    # 文件是否存在
    if [ ! -f ${CODE_DIR}/${script} ];then
	return 2
    fi
    BUILD_SCRIPT=${script}
    return 0
}

# 初始化coverity
function initCoverity() {
    currentDir=`pwd`
    cd ${CODE_DIR}

    configure=${COV_DIR}/bin/cov-configure
    ${configure} --gcc
    ${configure} --java
    ${configure} --clang
    ${configure} --scala
    ${configure} --go
    ${configure} --python

    ${configure} --template --compiler qcc --comptype qnxcc
    ${configure} --comptype prefix --compiler ccache
    ${configure} --comptype prefix --compiler distcc

    ${configure} --template --comptype gcc --compiler cc
    ${configure} --template --comptype gcc --compiler c++
    ${configure} --comptype clangcc --template --compiler arm-linux-androideabi-clang
    ${configure} --comptype armcc --template --compiler armcc
    ${configure} --comptype gcc --template --compiler arm-openwrt-linux-gcc
    ${configure} --comptype gcc --template --compiler arm-linux-gnueabihf-gcc
    ${configure} --comptype gcc --template --compiler arm-none-linux-gnueabi-gcc
    ${configure} --comptype gcc --template --compiler arm-linux-gnueabi-gcc
    ${configure} --comptype gcc --template --compiler arm-eabi-gcc
    ${configure} --comptype gcc --template --compiler arm-none-eabi-gcc
    ${configure} --comptype gcc --template --compiler arm-linux-androideabi-gcc

    ${configure} --comptype gcc --template --compiler x86_64-linux-android-gcc
    ${configure} --comptype gcc --template --compiler i686-linux-gnu-gcc
    ${configure} --comptype gcc --template --compiler csky-elf-gcc
    ${configure} --comptype gcc --template --compiler csky-abiv2-elf-gcc

    ${configure} --comptype gcc --template --compiler aarch64-linux-android-gcc
    ${configure} --comptype gcc --template --compiler aarch64-linux-gnueabi-gcc
    ${configure} --comptype gcc --template --compiler aarch64-linux-gnu-gcc
    ${configure} --comptype gcc --template --compiler aarch64-poky-linux-gcc

    ${configure} --comptype gcc --template --compiler aarch64-openwrt-linux-musl-gcc
    ${configure} --comptype g++ --template --compiler aarch64-openwrt-linux-gnu-g++
    ${configure} --comptype g++ --template --compiler aarch64-openwrt-linux-gnu-gcc

    ${configure} --comptype gcc --template --compiler mipsel-openwrt-linux-uclibc-gcc
    ${configure} --comptype gcc --template --compiler xtensa-esp32-elf-gcc

    ${configure} --comptype g++ --template --compiler aarch64-gnu-linux-g++
    ${configure} --comptype gcc --template --compiler aarch64-oe-linux-gcc

    ${configure} --comptype g++ --template --compiler aarch64-linux-g++
    ${configure} --comptype gcc --template --compiler aarch64-linux-gcc

    ${configure} --comptype gcc --template --compiler arm-alios-eabi-gcc --xml-option "@:<prepend_arg>--enable_128bit_float</prepend_arg>"

    ${configure} --comptype gcc --template --compiler arm-ali-aoseabi-gcc

    cd ${currentDir}
}

# 编译代码
function build() {
    idirTar=${COV_IDIR}.tar.gz
    currentDir=`pwd`
    # 进入目录
    cd ${CODE_DIR}

    # 创建中间目录文件
    idir=${CODE_DIR}/${COV_IDIR}

    if [ -d ${idir} ];then
	rm -rf ${idir}
    fi
    mkdir -p ${idir}

    if [ -f ${idirTar} ];then
	rm -rf ${idirTar}
    fi

    # 创建编译文件
    log=${idir}/${BUILD_LOG}
    touch ${log}

    # coverity编译文件
    cov_build=${COV_DIR}/bin/cov-build

    # 判断是否设置了 coverity build config
    if [[ ${#str} -gt 2 ]]; then
        echo "add build config: ${COV_BUILD_CONFIG}"
        ${cov_build} --dir ${idir} ${COV_BUILD_CONFIG} bash ${BUILD_SCRIPT} | tee ${log}
    else
        ${cov_build} --dir ${idir} bash ${BUILD_SCRIPT} | tee ${log}
    fi

    if test ${PIPESTATUS[0]} -ne 0; then
        echo "==========  ERROR ================="
        echo "build failed"; exit 1;
    fi

    if grep -q "\[WARNING\] No files were emitted" ${log}; then
        echo ""
        echo "==========  ERROR ================="
        echo "no file emmited,forget make clean?"; exit 1;
    fi

    # 压缩文件
    tar czvf ${idirTar} ${COV_IDIR} > /dev/null
    if [ $? != 0 ]; then
        echo ""
        echo "==========  ERROR ================="
        echo "compress package failed"; exit 1;
    fi

    # 拷贝到当前目录
    cp ${idirTar} ${currentDir}

    cd ${currentDir}
}

# alios 5u7补丁
function patch_5u7(){

    currentDir=`pwd`
    # 进入coverity目录
    cd ${COV_DIR}

    patchName=patch_5u.zip
    if [ ! -f ${patchName} ];then
	    echo "${patchName} is not exist."
     	exit 1
    fi

    unzip -o patch_5u.zip

    GLIBC=${COV_DIR}/glibc/lib
    INTERPRETER=$GLIBC/ld-linux-x86-64.so.2
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-build
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-configure
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-translate
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-emit
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-capture

    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-emit-java
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-emit-project
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-emit-text

    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-inspect-project
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-instrument
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-internal-calc-xrefs
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-internal-clang
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-internal-dm
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-internal-dump-rjt
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-internal-dw
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-internal-emit-clang
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-internal-emit-java
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-internal-emit-java-bytecode
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-internal-emit-java-webapp
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-internal-emit-misc
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-internal-emit-recompile
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-internal-format-forcheck-errors
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-internal-import-emit

    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-internal-supervise
    bin/patchelf --set-interpreter $INTERPRETER --set-rpath $GLIBC bin/cov-link

    cd ${currentDir}
}

# 打补丁
function patch() {
    if grep -q ".*5\.7" /etc/issue ; then
        patch_5u7
    	if [ $? -ne 0 ];then
    	    echo "failed to patch."
    	    return 1;
    	fi
    fi
    return 0
}

# 处理流程
function process() {
    #1 检查参数个数
    if [ $# -ne 3 ];then
	echo "USAGE: $0 coverity目录路径 源码目录 build.sh"
	return 1;
    fi

    #5 检查依赖
    checkDependencies

    #2 配置coverity目录
    setCoverityHome $1
    if [ $? -ne 0 ];then
	echo "coverity's directory is not provide."
 	return 2;
    fi

    #3 配置代码路径
    setSourceCodePath $2
    if [ $? -ne 0 ];then
	echo "source code's directory is not provide."
 	return 3;
    fi

    #4 打补丁
    patch
    if [ $? -ne 0 ];then
	echo "failed to patch."
 	return 4;
    fi

    #5 设置编译脚本
    setBuildScript $3

    #6 初始化贤者石
    initCoverity

    #7 编译代码
    build

    return $?
}

# 检查依赖
function checkDependency() {
    command -v $1 >/dev/null 2>&1 || { echo >&2 "required command $1 not installed.  Aborting."; exit 1; }
}

function checkDependencies() {
    checkDependency curl
    checkDependency wget
    checkDependency tee
    checkDependency tar
    checkDependency awk
    checkDependency sed
    checkDependency uuidgen
}

process $*