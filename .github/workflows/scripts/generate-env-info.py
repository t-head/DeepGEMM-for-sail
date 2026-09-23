#!/usr/bin/env python3
"""Collect runtime environment information and write it as JSON to stdout."""

import argparse
import json
import os
import shutil
import subprocess
import sys

def command_output(command):
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
    except FileNotFoundError:
        return "not found"
    return (result.stdout or result.stderr).strip()

def collect_env_info(docker_image):
    """Collect build container, SDK, compiler, and Python information."""
    info = {
        "docker_image": docker_image,
        "hostname": command_output(["hostname"]),
        "python_version": sys.version,
        "ppu_sdk": os.environ.get("PPU_SDK", ""),
        "ppu_home": os.environ.get("PPU_HOME", ""),
        "pytorch_sail_arch": os.environ.get("PYTORCH_SAIL_ARCH", ""),
        "hgcc_version": command_output(["hgcc", "--version"]),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
    except ImportError:
        info["torch_version"] = "not found"
    return info


def main():
    parser = argparse.ArgumentParser(
        description="Collect runtime environment information as JSON."
    )
    parser.add_argument(
        "--docker-image",
        required=True,
        help="Docker image used by the current CI job",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the generated JSON file",
    )
    args = parser.parse_args()
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(
            collect_env_info(args.docker_image),
            output_file,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")


if __name__ == "__main__":
    main()
