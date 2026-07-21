#pragma once
#ifdef DG_USE_HGTX
#include <hgtx3/hgToolsExt.h>
#endif
#include <hggc_runtime.h>
#include <string>
#include <iostream>
#include "utils.cuh"


class DgProfParam {
public:

DgProfParam() {}

void initialize_args(GemmType gemm_type, bool is_gemv, int m, int group, int* grouped_layout, int device_id, hggcStream_t stream = 0) {
    op_name_ = GemmTypeS[static_cast<int>(gemm_type)];
    m_ = m;
    group_ = group;
    grouped_layout_ = grouped_layout;
    stream_ = stream;
    device_id_ = device_id;
    gemm_type_ = gemm_type;
    is_gemv_ = is_gemv;
    add_argument("data_type");
    if (gemm_type_ != GemmType::DenseGemm){
        add_argument("groups");
    }
    add_argument("m");
    add_argument("n");
    add_argument("k");
    if (gemm_type_ == GemmType::GroupedMasked){
        add_argument("em");
    }
}

void set_mqa_logits_params(std::string data_type, int seq_len_q, int seq_len_kv, int num_heads, int head_dim, hggcStream_t stream = 0) {
    op_name_ = "MqaLogits";
    add_argument("data_type");
    add_argument("seq_len_q");
    add_argument("seq_len_kv");
    add_argument("num_heads");
    add_argument("head_dim");

    add_params("data_type", data_type);
    add_params("seq_len_q", seq_len_q);
    add_params("seq_len_kv", seq_len_kv);
    add_params("num_heads", num_heads);
    add_params("head_dim", head_dim);

    stream_ = stream;
    // Set default values (unused)
    m_ = seq_len_q;
    group_ = 1;
    grouped_layout_ = nullptr;
    gemm_type_ = GemmType::DenseGemm;
    is_gemv_ = false;
    device_id_ = -1; // avoid check_support_dump print
}

void set_paged_mqa_logits_params(std::string data_type, int batch_size, int next_n, int num_heads, int head_dim, int* context_lens, hggcStream_t stream = 0) {
    op_name_ = "PagedMqaLogits";
    add_argument("data_type");
    add_argument("batch_size");
    add_argument("next_n");
    add_argument("num_heads");
    add_argument("head_dim");

    add_params("data_type", data_type);
    add_params("batch_size", batch_size);
    add_params("next_n", next_n);
    add_params("num_heads", num_heads);
    add_params("head_dim", head_dim);

    stream_ = stream;
    // Set values for distribution
    m_ = batch_size;
    group_ = batch_size;
    grouped_layout_ = context_lens;
    next_n_ = next_n;
    gemm_type_ = GemmType::GroupedNoPad;
    is_gemv_ = false;
    int gpu = -1;
    hggcError_t result = hggcGetDevice(&gpu);
    if (result != hggcSuccess) {
        printf("get device id failed\n");
        return;
    }
    device_id_ = gpu;
}

template <typename T>
void add_params(const std::string& key, const T& val) {
    if (args_.find(key) == args_.end()) {
        args_.insert(std::make_pair(key, val_to_string(val)));
    }
    args_.at(key) = val_to_string(val);
    insertionOrder.push_back(key);
}

void set_params(GemmType gemm_type, bool is_gemv,
                          std::string data_type,
                          int group, int m, int n, int k, int em,
                          int* grouped_layout, hggcStream_t stream = 0) {
    int gpu = -1;
    hggcError_t result = hggcGetDevice(&gpu);
    if (result != hggcSuccess) {
        printf("get device id failed\n");
        return;
    }
    initialize_args(gemm_type, is_gemv, m, group, grouped_layout, gpu, stream);
    add_params("data_type", data_type);
    if (gemm_type_ != GemmType::DenseGemm){
        add_params("groups", group);
    }
    add_params("m", m);
    add_params("n", n);
    add_params("k", k);
    if (gemm_type_ == GemmType::GroupedMasked){
        add_params("em", em);
    }
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

bool is_paged_mqa_logits() {
    return op_name_ == "PagedMqaLogits";
}

void distribution() {
    hggcDeviceSynchronize();

    CHECK_HGGC(hggcGetLastError());
    std::ostringstream outDist;
    outDist << "[" ;

    // Check if this is for paged_mqa_logits (using batch_size and context_lens)
    if (is_paged_mqa_logits()) {
        // For paged_mqa_logits, batch_size is stored in group_ and context_lens in grouped_layout_
        int batch_size = group_;
        int sum_context_len = 0;
        int* context_lens = grouped_layout_;
        int total_elements = batch_size * next_n_;
        int* tmp = new int[total_elements];
        CHECK_HGGC(hggcMemcpyAsync(tmp, context_lens, sizeof(int) * total_elements, hggcMemcpyDeviceToHost, stream_));
        hggcStreamSynchronize(stream_);

        // Output the last context_len for each batch (i.e., tmp[i * next_n_ + next_n_ - 1])
        for (int i = 0; i < batch_size - 1; ++i) {
            int val = tmp[i * next_n_ + next_n_ - 1];
            outDist << val << ",";
            sum_context_len += val;
        }
        int last_val = tmp[(batch_size - 1) * next_n_ + next_n_ - 1];
        sum_context_len += last_val;
        outDist << last_val << "].";
        delete[] tmp;
        // Dump average instead of distribution when batch size is large
        if (batch_size > 64) {
            add_params("avg_context_len", sum_context_len / batch_size);
            return;
        }
    } else {
        // Original gemm distribution logic
        bool need_bincount = (gemm_type_ == GemmType::GroupedContiguous || is_gemv_ );
        int size = need_bincount ? m_ : group_;
        int* tmp = new int[size];
        CHECK_HGGC(hggcMemcpyAsync(tmp, grouped_layout_, sizeof(int) * size, hggcMemcpyDeviceToHost, stream_));
        hggcStreamSynchronize(stream_);

        if (need_bincount) {
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
    }

    hggcDeviceSynchronize();
    CHECK_HGGC(hggcGetLastError());
    add_params("distribution", outDist.str());
}

bool check_support_dump(){
    char *pEnv_dump_device = std::getenv("PPU_LIB_DUMP_DEVICE");
    static int target_device_id = pEnv_dump_device != nullptr ? std::stoi(pEnv_dump_device) : 0;
    // export PPU_LIB_DUMP_DEVICE=-1 to dump all devices
    if (target_device_id != -1 && target_device_id != device_id_ && !is_paged_mqa_logits()) {
        return false;
    }
    if (gemm_type_ == GemmType::DenseGemm || gemm_type_ == GemmType::BatchGemm) {
        printf("\ndump_group_m not supported for normal gemm.\n");
        return false;
    }
    // check if device graph captured
    hggcStreamCaptureStatus captureStatus;
    hggcStreamIsCapturing(stream_, &captureStatus);
    // add device graph mode later
    if (captureStatus != hggcStreamCaptureStatusNone) {
        printf("\ndump_group_m not supported in device graph mode.\n");
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
    int next_n_ = 1;
    int device_id_;
    GemmType gemm_type_;
    bool is_gemv_;
    hggcStream_t stream_;
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
        return show_params_ || use_hgtx_;
    }

    void instrument(bool start, DgProfParam &params) {
        if (!get_op_info()){
            return;
        }

        if (start) {
            if ((show_params_ || (params.is_paged_mqa_logits() && use_hgtx_)) && params.check_support_dump()) {
                params.distribution();
            }
            std::string op_name = params.format();
            if (show_params_) {
                std::cout << op_name << std::endl;
            }
#ifdef DG_USE_HGTX
            if (use_hgtx_) {
                hgtxEventAttributes_t eventAttrib = {0};
                eventAttrib.version = HGTX_VERSION;
                eventAttrib.size = HGTX_EVENT_ATTRIB_STRUCT_SIZE;
                eventAttrib.messageType = HGTX_MESSAGE_TYPE_ASCII;
                eventAttrib.message.ascii = op_name.c_str();
                hgtxDomainRangePushEx(domain_, &eventAttrib);
            }
#endif
        } else {
#ifdef DG_USE_HGTX
            if (use_hgtx_) {
                hgtxDomainRangePop(domain_);
            }
#endif
        } // if start

  }

private:
    ProfilingInterface() {
        // TODO: add print log
#ifdef DG_USE_HGTX
        domain_ = hgtxDomainCreateA("deepgemm");
#else
        domain_ = nullptr;
#endif
        use_hgtx_ = false;
        show_params_ = false;

        char *pEnv_perf = std::getenv("PPU_LIB_PERF_INSTRUMENT");
        if (pEnv_perf && isdigit(*pEnv_perf)) {
        int value = std::stoi(std::string(pEnv_perf));
        if (value == 0) {
            use_hgtx_ = false;
        } else if (value == 1) {
            use_hgtx_ = true;
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
#ifdef DG_USE_HGTX
        hgtxDomainDestroy(domain_);
#endif
    }

    bool use_hgtx_;
    bool show_params_;
#ifdef DG_USE_HGTX
    hgtxDomainHandle_t domain_;
#else
    void* domain_;
#endif

};
