import os

input_casepath = ""
output_file = "generate_deepgemm.caselist"
if input_casepath != "":
    for root, dirs, files in os.walk(input_casepath):
        for file in files:
            full_path = os.path.join(root, file)
            dg_cases.append(full_path)
    with open(output_file, "a") as f:
        f.writelines("\n".join(cases))

supported_gemm_type = ["GroupedContiguous", "GroupedNoPad", "GroupedMasked", "DenseGemm"]
supported_data_type = ["bf16", "int8", "fp8"]
supported_distribution = ['normal', 'gaussian', 'zipf']
num_groups_list = [8, 16, 128]
m_list = [2048, 4096, 132, 256]
nk_dict ={
"dpsk-v3_tp8":
[(512,7168),
(576,7168),
(1536,7168),
(3072,1536),
(4096,512),
(4608,7168),
(7168,256),
(7168,2048),
(7168,2304)],

"qwen3_tp8":
[(1536,4096),
(4096,1024),
(4096,192),
(384,4096)],

"dpsk-v3_ep":
[(2112, 7168),
(576, 7168),
(24576, 1536),
(32768, 512),
(36864, 7168),
(4096, 7168),
(7168, 16384),
(7168, 18432),
(7168, 2048),]

}
index = 0
cases = list()
output_cases = list()
for gemm_type in supported_gemm_type:
    for data_type in supported_data_type:
        for num_groups in num_groups_list:
            for m in m_list:
                for model, nk_list in nk_dict.items():
                    for (n, k) in nk_list:
                        for distribution in supported_distribution:
                            expected_m_per_group = m // num_groups
                            if gemm_type == "DenseGemm":
                                expected_m_per_group=1
                                num_groups=1
                            case_format = f"[DeepGemm] --format={gemm_type},data_type:{data_type},groups:{num_groups},m:{m},n:{n},k:{k},em:{expected_m_per_group},distribution:{distribution}"
                            if case_format not in cases:
                                cases.append(case_format)
# cases = set(cases)
for item in cases:
    if item not in output_cases:
        output_cases.append(item)
with open(output_file, "w") as f:
    f.writelines("\n".join(output_cases))