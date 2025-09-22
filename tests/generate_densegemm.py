import os

input_casepath = ""
output_file = "dense_gemm.caselist"
if input_casepath != "":
    for root, dirs, files in os.walk(input_casepath):
        for file in files:
            full_path = os.path.join(root, file)
            dg_cases.append(full_path)
    with open(output_file, "a") as f:
        f.writelines("\n".join(cases))

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
[(1280,4096),
(4096,1024),
(4096,192),
(384,4096)],

"dpsk-v3_ep":
[(2112,7168),
(24576,1536),
(36864,7168),
(4096,7168),
(7168,16384),
(7168,18432),
(7168,2048)] 

}
index = 0
cases = list()
for m in m_list:
    for model, nk_list in nk_dict.items():
        for (n, k) in nk_list:
            filename = f"case{index}_{model}_m{m}_n{n}_k{k}_DenseGemm.dump"
            cases.append(filename)
            # with open(f"{path}/{filename}", "w") as f:
            #     f.writelines("")
            index += 1
with open(output_file, "a") as f:
    f.writelines("\n".join(cases))