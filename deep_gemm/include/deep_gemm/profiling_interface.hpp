#pragma once
#include <nvtx3/nvToolsExt.h>
#include <string>
#include <iostream>
#include "utils.cuh"


class DgProfParam {
public:

DgProfParam() {}

void initialize_args() {
  add_argument("case");
  add_argument("gemm_type");
  add_argument("data_type");
  add_argument("groups");
  add_argument("m");
  add_argument("n");
  add_argument("k");
  add_argument("em");
  add_argument("gpu");
  add_argument("pid");
}

template <typename T>
void add_mha_params(const std::string& key, const T& val) {
  if (args_.find(key) == args_.end()) {
    args_.insert(std::make_pair(key, val_to_string(val)));
  }
  args_.at(key) = val_to_string(val);
  insertionOrder.push_back(key);
}

void set_deep_gemm_params(std::string gemm_type,
                          std::string data_type,
                          int case_id, int group, int m, int n, int k, int em, int gpu, int pid) {

  initialize_args();
  add_mha_params("case", case_id);
  add_mha_params("gemm_type", gemm_type);
  add_mha_params("data_type", data_type);
  add_mha_params("groups", group);
  add_mha_params("m", m);
  add_mha_params("n", n);
  add_mha_params("k", k);
  add_mha_params("em", em);
  add_mha_params("gpu", gpu);
  add_mha_params("pid", pid);
}

std::string format() {
  std::stringstream ss;

  ss << "[DeepGemm] --format=";

  for (auto& key : insertionOrder) {
    ss << key << ":" << args_[key];
    if (&key != &insertionOrder.back())
      ss << ',';
  }
  ss << '.';
  return ss.str();
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
  std::unordered_map<std::string, std::string> args_;
  std::vector<std::string> insertionOrder;
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
        std::cout << op_name << std::endl;
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
      } else if (value == 1 || value == 2) {
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
