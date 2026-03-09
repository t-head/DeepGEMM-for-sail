from functools import lru_cache
import os
import signal
from collections import defaultdict
import json
import functools
import sys
import logging
import torch
import threading

best_configs = None
MAX_DECODE_BS = 1025
MAX_FINEGRAINED_BS_POWER = 4
CANDIDATE_Ms = [2**i for i in range(MAX_FINEGRAINED_BS_POWER)]+list(range(2**MAX_FINEGRAINED_BS_POWER, MAX_DECODE_BS, 16))
SAIL_DEEPGEMM_TUNER_VERBOSE = os.getenv('SAIL_DEEPGEMM_TUNER_VERBOSE', None)
SAIL_DEEPGEMM_TUNER_CUSTOM_CONFIG_DIR = os.getenv('SAIL_DEEPGEMM_TUNER_CUSTOM_CONFIG_DIR', None)

logger = logging.getLogger(__name__)

def check_main_thread():
    current_thread = threading.current_thread()
    if current_thread.name == 'MainThread':
        return True
    else:
        return False
    
def get_device_name():
    return str(torch.cuda.get_device_name(0))

@lru_cache(maxsize=None)
def get_deep_gemm_best_configs():
    config_dir = SAIL_DEEPGEMM_TUNER_CUSTOM_CONFIG_DIR or os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )
    if not os.path.exists(config_dir):
        logger.info(f"DeepGemm Tuner: Not exist config dir {SAIL_DEEPGEMM_TUNER_CUSTOM_CONFIG_DIR}")
        return None
    configs = os.listdir(config_dir)

    config_dict_in_all = dict()

    device_name = get_device_name().replace(" ", "_")
    device_name_signature = f",device_name={device_name}-"

    logger.info(f"deepgemm start loading config......")

    for config in configs:

        if device_name_signature not in config:
            logger.info(f"skipping device_name_signature {config}")
            continue
        logger.info(f"loading device_name_signature {config}")

        config_file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "configs",
            config
        )
        config_dict = dict()

        try:
            with open(config_file_path,'r') as f:
                config_dict = json.load(f)
            config_dict = dict(
                [((x["M"], x["N"], x["K"], x["num_groups"],
                   x['nopad'] if 'nopad' in x else False,
                   x['dtype'] if 'dtype' in x else "int8"), x) for x in config_dict]
            )
        except :
            logger.warning(
                f"Empty config found in {config_file_path}."
            )

        if config_dict:
            config_dict_in_all.update(config_dict)

    def gen_NKG_M_map(best_config):
        nkg_m_map = defaultdict(list)
        for key in best_config:
            m, n, k, g, nopad, dtype = key
            nkg_m_map[(n, k, g, nopad, dtype)].append(m)
        return nkg_m_map

    return config_dict_in_all, gen_NKG_M_map(config_dict_in_all)

def find_closest(lst, x):
    # As a dividing line of P and D
    if x > MAX_DECODE_BS:
        return -1
    else:
        return min(lst, key=lambda y: abs(y - x))

def check_appropriate_of_candidate_M(M_candidate, M):
    M_theoretical = find_closest(CANDIDATE_Ms, M)
    should_use_default = False

    # The candidate we select is far from its candidate range,
    # for example M=255, it's candidate range is [255-16, 255+16], and the closest one is M_theoretical=256
    # but M_candidate=64, so abs(M-M_candidate) > abs(M_theoretical-M) --> abs(255-64) > abs(256-255)
    if abs(M-M_candidate) > abs(M_theoretical-M):
        should_use_default = True

    return should_use_default

def search_for_suitable_config(M, N, K, num_groups, nopad, dtype, best_configs, nkg_m_map):
    config = None
    M_candidate = M
    # find neariest config
    if (M, N, K, num_groups, nopad, dtype) in best_configs:
        config = eval(best_configs[(M, N, K, num_groups, nopad, dtype)]["config"])
    else:
        # There are tuned configs found under (N, K, num_groups)
        if len(nkg_m_map[(N, K, num_groups, nopad, dtype)]) > 0:
            existed_Ms = nkg_m_map[(N, K, num_groups, nopad, dtype)]
            M_candidate = find_closest(existed_Ms, M)
            if SAIL_DEEPGEMM_TUNER_VERBOSE:
                logger.info(
                    f"DeepGemm Tuner: found configs for {(M, N, K, num_groups, nopad, dtype)}, attached to M_candidate {M_candidate}"
                )
            if (M_candidate, N, K, num_groups, nopad, dtype) in best_configs:
                config = eval(best_configs[(M_candidate, N, K, num_groups, nopad, dtype)]["config"])
                # if check appropriate  failed, we only record this (M,N,K,NUM_GROUP)
                should_use_default = check_appropriate_of_candidate_M(
                    M_candidate, M
                )
                if should_use_default:
                    if SAIL_DEEPGEMM_TUNER_VERBOSE:
                        logger.info(f"DeepGemm Tuner: check appropriate  failed: {(M, N, K, num_groups, nopad, dtype)}")
                        logger.info(f"DeepGemm Tuner: abort attaching {M} to M_candidate{M_candidate}")
                    config = None
    return config, M_candidate

@lru_cache(maxsize=None)
def get_deep_gemm_luts(M,N,K,num_groups, nopad=False, dtype='int8', force_heruistic=False):
    if force_heruistic:
        return None
    best_configs, nkg_m_map = get_deep_gemm_best_configs()
    if len(best_configs)>0:
        # find neariest config
        config, _ = search_for_suitable_config(M, N, K, num_groups, nopad, dtype, best_configs, nkg_m_map)

        if config is None:
            if SAIL_DEEPGEMM_TUNER_VERBOSE:
                logger.info(f"DeepGemm Tuner: no config found, target configs are {(M, N, K, num_groups, nopad, dtype)}")
            return None

        num_sms = config["num_min_sms"]
        block_m = config["best_block_m"]
        block_n = config["best_block_n"]
        block_k = config["block_k"]
        warp_m = config["warp_m"]
        warp_n = config["warp_n"]
        num_stages = config["best_num_stages"]
        smem_config = config["best_smem_config"]
        if SAIL_DEEPGEMM_TUNER_VERBOSE:
            logger.info(f"DeepGemm Tuner: found configs for {(M, N, K, num_groups, nopad, dtype)},")
        return (num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config)
    else:
        if SAIL_DEEPGEMM_TUNER_VERBOSE:
            logger.info(f"DeepGemm Tuner: best configs are None")
        return None

