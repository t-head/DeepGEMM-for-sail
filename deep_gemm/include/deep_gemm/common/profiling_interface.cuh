#pragma once
#include <hgtx3/hgToolsExt.h>
#include <hggc_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/types.cuh>


class DgProfParam {
public:
    using GemmType = deep_gemm::GemmType;

DgProfParam() {}

DgProfParam(const char* ut_name, const std::string& data_type): op_name_(ut_name) {
    add_params("data_type", data_type);
}

DgProfParam(const GemmType& gemm_type, const std::string& data_type):
    op_name_(deep_gemm::GemmTypeS[static_cast<int>(gemm_type)]) {
    add_params("data_type", data_type);
}

void initialize_args(GemmType gemm_type, bool is_gemv, int m, int group, int* grouped_layout, int device_id, hggcStream_t stream = 0) {
    op_name_ = deep_gemm::GemmTypeS[static_cast<int>(gemm_type)];
    m_ = m;
    group_ = group;
    grouped_layout_ = grouped_layout;
    stream_ = stream;
    device_id_ = device_id;
    gemm_type_ = gemm_type;
    is_gemv_ = is_gemv;
    add_argument("data_type");
    if (gemm_type_ != GemmType::Normal){
        add_argument("groups");
    }
    add_argument("m");
    add_argument("n");
    add_argument("k");
    if (gemm_type_ == GemmType::MGroupedMasked){
        add_argument("em");
    }
}

void set_mqa_logits_params(std::string data_type, int seq_len_q, int seq_len_kv, int num_heads, int head_dim, hggcStream_t stream = 0, bool is_avg = false) {
    op_name_ = is_avg ? "MqaAvgLogits" : "MqaLogits";
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
    gemm_type_ = GemmType::Normal;
    is_gemv_ = false;
    device_id_ = -1; // avoid check_support_dump print
}

void set_sparse_mqa_logits_params(std::string data_type, int seq_len_q, int seq_len_kv, int num_heads, int head_dim,
                                  int sparse_block_kv, int num_max_sparse_blocks, bool use_unaligned_ks,
                                  bool is_paged = false, hggcStream_t stream = 0) {
    op_name_ = is_paged ? "PagedSparseMqaLogits" : "SparseMqaLogits";
    add_argument("data_type");
    add_argument("seq_len_q");
    add_argument("seq_len_kv");
    add_argument("num_heads");
    add_argument("head_dim");
    add_argument("sparse_block_kv");
    add_argument("num_max_sparse_blocks");
    add_argument("use_unaligned_ks");

    add_params("data_type", data_type);
    add_params("seq_len_q", seq_len_q);
    add_params("seq_len_kv", seq_len_kv);
    add_params("num_heads", num_heads);
    add_params("head_dim", head_dim);
    add_params("sparse_block_kv", sparse_block_kv);
    add_params("num_max_sparse_blocks", num_max_sparse_blocks);
    add_params("use_unaligned_ks", use_unaligned_ks);

    stream_ = stream;
    // Set default values (unused)
    m_ = seq_len_q;
    group_ = 1;
    grouped_layout_ = nullptr;
    gemm_type_ = GemmType::Normal;
    is_gemv_ = false;
    device_id_ = -1; // avoid check_support_dump print
}

void set_paged_mqa_logits_params(std::string data_type, int batch_size, int next_n, int num_heads, int head_dim, int* context_lens, hggcStream_t stream = 0, bool is_avg = false) {
    op_name_ = is_avg ? "PagedMqaAvgLogits" : "PagedMqaLogits";
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
    gemm_type_ = GemmType::MGroupedNoPad;
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
DgProfParam& add_params(const std::string& key, const T& val) {
    if (std::find(insertionOrder.begin(), insertionOrder.end(), key) == insertionOrder.end())
        insertionOrder.push_back(key);
    args_[key] = val_to_string(val);
    return *this;
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
    if (gemm_type_ != GemmType::Normal){
        add_params("groups", group);
    }
    add_params("m", m);
    add_params("n", n);
    add_params("k", k);
    if (gemm_type_ == GemmType::MGroupedMasked){
        add_params("em", em);
    }
    add_params("gpu", gpu);
}

void set_hc_prenorm_params(int m, int n, int k, hggcStream_t stream = 0) {
    set_params(GemmType::Normal, false, "tf32", 1, m, n, k, 1, nullptr, stream);
    op_name_ = "HcPrenormGemm";
}

void set_mega_mhc_params(int m, int hidden, int sinkhorn_iters, hggcStream_t stream = 0) {
    op_name_ = "MegaMhc";
    add_argument("data_type");
    add_argument("m");
    add_argument("hidden");
    add_argument("mode");
    add_argument("sinkhorn_iters");

    add_params("data_type", "bf16");
    add_params("m", m);
    add_params("hidden", hidden);
    add_params("mode", "shifted");
    add_params("sinkhorn_iters", sinkhorn_iters);

    stream_ = stream;
    m_ = m;
    group_ = 1;
    grouped_layout_ = nullptr;
    gemm_type_ = GemmType::Normal;
    is_gemv_ = false;
    device_id_ = -1;
}

void set_fused_moe_params(const std::string& data_type, const std::string& quant_type,
                          int groups, int num_token, int topk,
                          int n, int k, int* grouped_layout, hggcStream_t stream = 0) {
    op_name_ = "GroupedFused";
    add_argument("data_type");
    add_argument("groups");
    add_argument("num_token");
    add_argument("topk");
    add_argument("n");
    add_argument("k");

    add_params("data_type", data_type);
    add_params("groups", groups);
    add_params("num_token", num_token);
    add_params("topk", topk);
    add_params("n", n);
    add_params("k", k);

    stream_ = stream;
    // Values consumed by distribution() for the per-expert token histogram
    m_ = num_token;
    group_ = groups;
    grouped_layout_ = grouped_layout;
    gemm_type_ = GemmType::MGroupedFused;
    is_gemv_ = false;
    int gpu = -1;
    hggcError_t result = hggcGetDevice(&gpu);
    if (result != hggcSuccess) {
        printf("get device id failed\n");
        return;
    }
    device_id_ = gpu;
    add_params("quant_type", quant_type);
    add_params("gpu", gpu);
}

DgProfParam& add_distribution(const int* ptr, const int count, const int& num_groups,
                             const bool& group_m_is_row_expert_id = false,
                             const int psum_alignment = 0) {
        auto values = std::vector<int>(num_groups, 0);
        if (group_m_is_row_expert_id) {
            // Out-of-range ids (the padding rows of a psum layout) are dropped, as in ppu1.5
            for (int i = 0; i < count; ++ i)
                if (0 <= ptr[i] and ptr[i] < num_groups)
                    ++ values[ptr[i]];
        } else {
            using namespace deep_gemm;
            DG_HOST_ASSERT(count == num_groups);
            if (psum_alignment > 0) {
                // Each group starts after the preceding group's alignment padding.
                for (int i = 0; i < count; ++ i)
                    values[i] = ptr[i] - (i == 0 ? 0 : ((ptr[i - 1] + psum_alignment - 1) / psum_alignment * psum_alignment));
            } else {
                values.assign(ptr, ptr + count);
            }
        }
        return add_params("distribution", render(values));
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
        bool need_bincount = (gemm_type_ == GemmType::MGroupedContiguous || is_gemv_ );
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
    if (gemm_type_ == GemmType::Normal || gemm_type_ == GemmType::Batched) {
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

static std::string render(const std::vector<int>& values) {
        if (values.empty())
            return "[].";
        std::string out = "[";
        for (size_t i = 0; i + 1 < values.size(); ++ i)
            out += std::to_string(values[i]) + ",";
        return out + std::to_string(values.back()) + "].";
    }

template <typename T>
std::string val_to_string(const T& value) {
    return std::to_string(value);
}

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

std::string val_to_string(const char* val) {
    return val == nullptr ? "" : val;
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

    static ProfilingInterface& instance() {
        static ProfilingInterface instance;
        return instance;
  }

    // `PPU_LIB_SHOW_PARAMS=1`: echo the UT of every instrumented launch
    bool show_params() const { return show_params_; }

    // `PPU_LIB_PERF_INSTRUMENT=1`: push/pop HGTX ranges around every instrumented launch
    bool use_hgtx() const { return use_hgtx_; }

    // `PPU_LIB_DUMP_DEVICE=<id>`: which device may pay for a distribution read-back (`-1` for all)
    bool support_dump(const int& device_id) const {
        return dump_device_id_ == -1 or dump_device_id_ == device_id;
    }

    // Push an HGTX domained range with `message` as the annotation. No-op when PERF_INSTRUMENT
    // is unset.
    void hgtx_range_push(const std::string& message) {
        if (not use_hgtx_)
            return;
        hgtxEventAttributes_t event_attrib = {0};
        event_attrib.version = HGTX_VERSION;
        event_attrib.size = HGTX_EVENT_ATTRIB_STRUCT_SIZE;
        event_attrib.messageType = HGTX_MESSAGE_TYPE_ASCII;
        event_attrib.message.ascii = message.c_str();
        hgtxDomainRangePushEx(domain_, &event_attrib);
    }

    // Pop the most recent HGTX domained range. No-op when PERF_INSTRUMENT is unset.
    void hgtx_range_pop() {
        if (not use_hgtx_)
            return;
        hgtxDomainRangePop(domain_);
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
            if (use_hgtx_) {
                hgtx_range_push(op_name);
            }
        } else {
            if (use_hgtx_) {
                hgtx_range_pop();
            }
        } // if start

  }

private:
    ProfilingInterface() {
        // TODO: add print log
        domain_ = hgtxDomainCreateA("deepgemm");
        const auto dump_device = std::getenv("PPU_LIB_DUMP_DEVICE");
        dump_device_id_ = dump_device == nullptr ? 0 : std::stoi(dump_device);
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
        hgtxDomainDestroy(domain_);
    }

    bool use_hgtx_;
    bool show_params_;
    hgtxDomainHandle_t domain_;
    int dump_device_id_;

};

namespace deep_gemm {

inline bool profiling_enabled() {
    return ProfilingInterface::instance().show_params() or
           ProfilingInterface::instance().use_hgtx();
}

class HgtxRangeGuard {
public:
    HgtxRangeGuard() = default;
    explicit HgtxRangeGuard(const std::string& message) {
        ProfilingInterface::instance().hgtx_range_push(message);
        active_ = true;
    }
    ~HgtxRangeGuard() {
        if (active_)
            ProfilingInterface::instance().hgtx_range_pop();
    }
    HgtxRangeGuard(const HgtxRangeGuard&) = delete;
    HgtxRangeGuard& operator=(const HgtxRangeGuard&) = delete;
    HgtxRangeGuard(HgtxRangeGuard&& other) noexcept : active_(other.active_) { other.active_ = false; }
    HgtxRangeGuard& operator=(HgtxRangeGuard&& other) noexcept {
        if (this != &other) {
            if (active_) ProfilingInterface::instance().hgtx_range_pop();
            active_ = other.active_;
            other.active_ = false;
        }
        return *this;
    }
private:
    bool active_ = false;
};

} // namespace deep_gemm
