# UT Functional Test
Use `run_deep_gemm.py` with --format to specify input parameters.
## Usage
run_deep_gemm.py [-h] [--format FORMAT] [--caselist CASELIST] [--case_idx CASE_IDX] [--disable_acc] [--benchmark] [--ref_backend {host,device}]

options: \
  -h, --help            show this help message and exit \
  --format FORMAT       Case cmd to describe problem size. \
  --caselist CASELIST   the folder of DG cases \
  --case_idx CASE_IDX   the line index(1~line) of case in caselist file \
  --disable_acc         if or not open accuracy check \
  --benchmark           specify if run benchmark \
  --ref_backend {host,device}
                        specify the backend used to compute ref output \
## Example
### Accuracy test: compare deepgemm with pytorch cpu implementation
[recommended for cmodel environment] \
python ./run_deep_gemm.py --format=GroupedMasked,data_type:int8,groups:16,m:4096,n:4096,k:7168,em:12 --ref_backend=host

### Accuracy test: compare deepgemm with pytorch device kernel implementation
python ./run_deep_gemm.py --format=GroupedMasked,data_type:int8,groups:16,m:4096,n:4096,k:7168,em:12

### Run deepgemm kernel only without correctness check
python ./run_deep_gemm.py --format=GroupedMasked,data_type:int8,groups:16,m:4096,n:4096,k:7168,em:12 --disable_acc


# UT Performance Test
Use `run_deep_gemm_perf.py` with --format to specify input parameters and --mode to specify mode.
## Usage
run_deep_gemm_perf.py [-h] [--caselist CASELIST] [--case_idx CASE_IDX] [--format FORMAT] [--output OUTPUT] [--mode {metrics,full,show_log,umd_perf}]
                             [--device DEVICE] [--disable_acc] [--ref_backend {host,device}]

options: \
  -h, --help            show this help message and exit \
  --caselist CASELIST   the folder of DG cases \
  --case_idx CASE_IDX   the line index(1~line) of case in caselist file \
  --format FORMAT       Case cmd to describe problem size. \
  --output OUTPUT       the output storing cycles of DG cases \
  --mode {metrics,full,show_log,umd_perf}
                        run perf mode \
  --device DEVICE       devices index to run cases, 0 means gpu0. 0,3 means gpu0,1,2,3 \
  --disable_acc         if or not open accuracy check \
  --ref_backend {host,device}
                        specify the backend used to compute ref output \

## Example
### Run a single case with parameters specified by format
python ./run_deep_gemm_perf.py --format=GroupedMasked,data_type:int8,groups:16,m:4096,n:4096,k:7168,em:12,distribution:[45,200,30,18,13,21,31,3,5,17,17,16,18,15,18,13]

### Run a batch of cases (accuracy + performance by default)
python ./run_deep_gemm_perf.py --caselist=./deepgemm.caselist

# Checkin Test
## Test fp8
python ./test_fp8_core.py

## Test int8 and bf16
python ./test_core.py

test_core.py options:
  -h, --help            show this help message and exit
  --func {DenseGemm,GroupedContiguous,GroupedMasked,DenseGemm}  target test func