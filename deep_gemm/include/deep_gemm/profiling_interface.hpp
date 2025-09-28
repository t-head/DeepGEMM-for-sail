#pragma once
#include <nvtx3/nvToolsExt.h>
#include <string>
#include <iostream>
#include "utils.cuh"


class DgProfParam {
public:

DgProfParam() {}

void initialize_args(GemmType gemm_type, int m, int group, int* grouped_layout, int device_id, cudaStream_t stream = 0) {
    op_name_ = GemmTypeS[static_cast<int>(gemm_type)];
    m_ = m;
    group_ = group;
    grouped_layout_ = grouped_layout;
    stream_ = stream;
    need_bincount_ = (gemm_type == GemmType::GroupedContiguous);
    is_normal_gemm_ = (gemm_type == GemmType::DenseGemm);
    device_id_ = device_id;
    add_argument("data_type");
    add_argument("groups");
    add_argument("m");
    add_argument("n");
    add_argument("k");
    add_argument("em");
}

template <typename T>
void add_params(const std::string& key, const T& val) {
    if (args_.find(key) == args_.end()) {
        args_.insert(std::make_pair(key, val_to_string(val)));
    }
    args_.at(key) = val_to_string(val);
    insertionOrder.push_back(key);
}

void set_params(GemmType gemm_type,
                          std::string data_type,
                          int group, int m, int n, int k, int em,
                          int* grouped_layout, cudaStream_t stream = 0) {
    int gpu = -1;
    cudaError_t result = cudaGetDevice(&gpu);
    if (result != cudaSuccess) {
        printf("get device id failed\n");
        return;
    }
    initialize_args(gemm_type, m, group, grouped_layout, gpu, stream);
    add_params("data_type", data_type);
    add_params("groups", group);
    add_params("m", m);
    add_params("n", n);
    add_params("k", k);
    add_params("em", em);
    add_params("gpu", gpu);
}

std::string format() {
    std::stringstream ss;

    ss << "[DeepGemm] --format=" << op_name_ << ",";

    for (auto& key : insertionOrder) {
        ss << key << ":" << args_[key];
        if (&key != &insertionOrder.back())
        ss << ',';
    }

    return ss.str();
}

std::string distribution() {
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaGetLastError());
    std::ostringstream outDist;
    outDist << ",distribution:[" ;
    int size = need_bincount_ ? m_ : group_;
    int* tmp = new int[size];
    CHECK_CUDA(cudaMemcpyAsync(tmp, grouped_layout_, sizeof(int) * size, cudaMemcpyDeviceToHost, stream_));
    if (need_bincount_) {
        int* counts = new int[group_];
        for (int i = 0; i < group_; ++i) {
            counts[i] = 0;
        }
        for (int i = 0; i < size; ++i) {
            if (tmp[i] < 0 || tmp[i] >= group_) {
                continue;
            }
            counts[tmp[i]]++;
        }
        for (int i = 0; i < group_ - 1; ++i) {
            outDist << counts[i] << ",";
        }
        outDist << counts[group_ - 1] << "].";
        delete[] counts;
    } else {
        for (int i = 0; i < size - 1; ++i) {
            outDist << tmp[i] << ",";
        }
        outDist << tmp[size - 1] << "].";
    }
    delete[] tmp;
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
    return outDist.str().c_str();
}

bool check_support_dump(){
    char *pEnv_dump_device = std::getenv("PPU_LIB_DUMP_DEVICE");
    static int target_device_id = pEnv_dump_device != nullptr ? std::stoi(pEnv_dump_device) : 0;
    if (target_device_id != device_id_) {
        return false;
    }
    if (is_normal_gemm_) {
        printf("\ndump_group_m not supported for normal gemm.\n");
        return false;
    }
    // check if cuda graph captured
    cudaStreamCaptureStatus captureStatus;
    cudaStreamIsCapturing(stream_, &captureStatus);
    // add cuda graph mode later
    if (captureStatus != cudaStreamCaptureStatusNone) {
        printf("\ndump_group_m not supported in cuda graph mode.\n");
        return false;
    }
    return true;
}

private:

std::string val_to_string(int val) {
    return std::to_string(val);
}

std::string val_to_string(float val) {
    std::stringstream float_str;
    float_str << std::fixed << std::setprecision(4) << val;
    return float_str.str();
}

std::string val_to_string(bool val) {
    return std::to_string(int(val));
}

std::string val_to_string(const std::string& val) {
    return val;
}

void add_argument(const std::string& name) {
    std::string init_val = "";
    if (args_.find(name) != args_.end()) {
        std::cout << "[" << name << "] already exists." << std::endl;
        throw std::runtime_error("Add argument fail.");
    }

    args_.insert(std::make_pair(name, init_val));
}

protected:
    std::string op_name_;
    std::unordered_map<std::string, std::string> args_;
    std::vector<std::string> insertionOrder;
    int m_;
    int group_;
    int* grouped_layout_;
    bool need_bincount_;
    bool is_normal_gemm_;
    int device_id_;
    cudaStream_t stream_;
};


class ProfilingInterface {
public:
    ProfilingInterface(ProfilingInterface const&) = delete;
    void operator=(ProfilingInterface const&) = delete;

    static ProfilingInterface& Instance() {
        static ProfilingInterface instance;
        return instance;
  }

    bool get_op_info() {
        return show_params_ || use_nvtx_;
    }

    void instrument(bool start, DgProfParam &params) {
        if (!get_op_info()){
        return;
        }

        if (start) {
        std::string op_name = params.format();
        if (show_params_) {
            std::cout << op_name;
            if (params.check_support_dump()) {
            std::string distribution = params.distribution();
            std::cout << distribution << std::endl;
            }
            std::cout << std::endl;
        }
        if (use_nvtx_) {
            nvtxEventAttributes_t eventAttrib = {0};
            eventAttrib.version = NVTX_VERSION;
            eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
            eventAttrib.message.ascii = op_name.c_str();
            nvtxDomainRangePushEx(domain_, &eventAttrib);
        }
        } else {
        if (use_nvtx_) {
            nvtxDomainRangePop(domain_);
        }
        } // if start

  }

private:
    ProfilingInterface() {
        // TODO: add print log
        domain_ = nvtxDomainCreateA("deepgemm");
        use_nvtx_ = false;
        show_params_ = false;

        char *pEnv_perf = std::getenv("PPU_LIB_PERF_INSTRUMENT");
        if (pEnv_perf && isdigit(*pEnv_perf)) {
        int value = std::stoi(std::string(pEnv_perf));
        if (value == 0) {
            use_nvtx_ = false;
        } else if (value == 1) {
            use_nvtx_ = true;
        } else {
            printf("Invalid value for PPU_LIB_PERF_INSTRUMENT : %d\n", value);
        }
        }

        char *pEnv_params = std::getenv("PPU_LIB_SHOW_PARAMS");
        if (pEnv_params && isdigit(*pEnv_params)) {
        int value = std::stoi(std::string(pEnv_params));
        if (value == 0) {
            show_params_ = false;
        } else if (value == 1) {
            show_params_ = true;
        } else {
            printf("Invalid value for PPU_LIB_SHOW_PARAMS : %d\n", value);
        }
        }
  }


    ~ProfilingInterface() {
        nvtxDomainDestroy(domain_);
    }

    bool use_nvtx_;
    bool show_params_;
    nvtxDomainHandle_t domain_;

};
