import subprocess
import os
import sys
import time
import argparse
import json
import re
import signal
from junit_xml import TestCase, TestSuite

OUTPUT_XML = "test-results.xml"
DEFAULT_TIMEOUT_SECONDS = 1200
# Ceilings are sized for the first run of a fresh CI workspace, where every
# JIT kernel must be compiled by hgcc on a cold kernel cache. Measured on
# ZW810E (PR #15, commit 35fb2c0): test_core.py alone needs ~2580s single
# process; tile_scan_with_test.py spends most of its wall time compiling
# per-tile kernels, which also exceeds the 1200s default on a cold cache.
# Keep these comfortably above the observed times so slow-but-passing runs
# are not killed.
CASE_TIMEOUT_OVERRIDES = {
    "tests/test_core.py": 3600,
    "tests/tile_scan_with_test.py": 3600,
    "tests/test_fp4_core.py": 3000,
    "tests/test_fp8_core.py": 2100,
}


def get_case_timeout(case_command):
    """Return a case-specific timeout, falling back to the default limit."""
    for case_path, timeout_seconds in CASE_TIMEOUT_OVERRIDES.items():
        if case_path in case_command:
            return timeout_seconds
    return DEFAULT_TIMEOUT_SECONDS


def derive_test_name(case_command):
    """Build a test name from the full shell command.

    Keep the whole command (interpreter included, e.g. python/pytest) so each
    case stays distinguishable, and collapse any run of whitespace into a
    single underscore so the name is a single token in the JUnit report.
    """
    return re.sub(r"\s+", "_", case_command.strip())


def run_one_case(i, case_command, total_count):
    """
    Run a single test case.
    case_command: full shell command string with arguments.
    """
    test_name = derive_test_name(case_command)
    timeout_seconds = get_case_timeout(case_command)

    junit_case = TestCase(name=test_name, classname="shTest")

    print(f"\n[{i}/{total_count}] Running: {test_name}")
    print(f"Command: {case_command}")
    print(f"Timeout: {timeout_seconds}s")
    print("-" * 50)

    start_time = time.time()
    failed_keyword_found = False
    stdout_output = []

    try:
        process = subprocess.Popen(
            case_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        output, _ = process.communicate(timeout=timeout_seconds)
        if output:
            print(output, end='')
            stdout_output.append(output)
            failed_keyword_found = bool(re.search(
                r"\bfail(?:ed|ure)?\b|Traceback \(most recent call last\):|\bAssertionError\b",
                output,
                re.IGNORECASE,
            ))
        duration = time.time() - start_time
        junit_case.elapsed_sec = duration

        if process.returncode == 0 and not failed_keyword_found:
            print(f">>> Result: [PASS]")
            return junit_case, True
        else:
            msg = f"Exit code: {process.returncode}, Failed keyword: {failed_keyword_found}"
            print(f">>> Result: [FAIL] ({msg})")
            junit_case.add_failure_info(message="Test failed", output="".join(stdout_output))
            return junit_case, False

    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        if output:
            print(output, end='')
            stdout_output.append(output)
        duration = time.time() - start_time
        junit_case.elapsed_sec = duration
        print(f"Timeout after {timeout_seconds}s: {test_name}")
        junit_case.add_failure_info(
            message=f"Timeout after {timeout_seconds}s",
            output="".join(stdout_output) or "Test timed out",
        )
        return junit_case, False

    except Exception as e:
        duration = time.time() - start_time
        junit_case.elapsed_sec = duration
        print(f"Exception: {test_name}, error: {e}")
        junit_case.add_error_info(message="Exception occurred", output=str(e))
        return junit_case, False


def parse_commands(filepath, is_sh):
    """
    Parse test commands from a file.
    - .sh file: only accept lines starting with './examples/'
    - Other files: accept all non-empty, non-comment lines.
    """
    test_commands = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if is_sh and not line.startswith("./examples/"):
                continue
            test_commands.append(line)
    return test_commands


def load_caselist(caselist_path, suite=""):
    """Load test commands from a caselist.

    A .json caselist (testcase.json) maps suite keys (board names) to command
    arrays; `suite` selects the key. Plain .list/.sh caselists keep the legacy
    line-based parsing.
    """
    if not caselist_path.endswith(".json"):
        is_sh = os.path.splitext(caselist_path)[1].lower() == ".sh"
        return parse_commands(caselist_path, is_sh)

    with open(caselist_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if not suite:
        sys.exit("--suite is required when the caselist is a .json file")
    if suite not in cases:
        sys.exit(
            f"Suite '{suite}' not found in {caselist_path}; "
            f"available suites: {sorted(cases)}"
        )
    commands = [str(raw).strip() for raw in cases[suite] if str(raw).strip()]
    if not commands:
        sys.exit(f"Suite '{suite}' in {caselist_path} is empty")
    return commands


def resolve_build_dir(build_dir):
    """Resolve the test working directory; default to the DeepGEMM repository root."""
    if build_dir:
        return os.path.abspath(build_dir)

    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def run_cases(caselist_path, build_dir, output_xml=OUTPUT_XML, suite=""):
    if not os.path.exists(caselist_path):
        print(f"Error: test list file not found: {caselist_path}")
        sys.exit(1)

    test_commands = load_caselist(caselist_path, suite)

    if not test_commands:
        print("No valid test commands found in list")
        sys.exit(1)

    try:
        os.chdir(build_dir)
        print(f"Working directory changed to: {os.getcwd()}")
    except Exception as e:
        print(f"Failed to change directory: {e}")
        sys.exit(1)

    all_test_objects = []
    global_passed_count = 0
    total_count = len(test_commands)

    for i, cmd in enumerate(test_commands, 1):
        try:
            junit_case_obj, is_success = run_one_case(i, cmd, total_count)
            all_test_objects.append(junit_case_obj)
            if is_success:
                global_passed_count += 1
        except Exception as e:
            print(f"Critical error: {e}")

    # Resolve after changing directory so relative paths use the test working directory.
    output_xml_path = os.path.abspath(output_xml)
    output_xml_dir = os.path.dirname(output_xml_path)
    os.makedirs(output_xml_dir, exist_ok=True)

    suite = TestSuite("DeepGEMM PR Tests", all_test_objects)
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        TestSuite.to_file(f, [suite], prettyprint=True)

    print(f"\nTests completed. Total: {total_count}, Passed: {global_passed_count}")
    print(f"Report generated: {output_xml_path}")

    if global_passed_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepGEMM PR tests from a caselist.")
    parser.add_argument(
        "--caselist",
        required=True,
        help="Path to a testcase.json caselist or a legacy .list/.sh caselist",
    )
    parser.add_argument(
        "--suite",
        default="",
        help="Suite key selecting the command group from a testcase.json caselist (board name, e.g. ZW-M890P).",
    )
    parser.add_argument(
        "--build-dir",
        help="Directory in which test commands run. Defaults to the DeepGEMM repository root.",
    )
    parser.add_argument(
        "--output-xml",
        default=OUTPUT_XML,
        help=(
            "JUnit XML output path. Absolute paths are used directly; relative "
            f"paths are resolved from the test working directory (default: {OUTPUT_XML})."
        ),
    )
    args = parser.parse_args()

    caselist_abs_path = os.path.abspath(args.caselist)
    build_dir = resolve_build_dir(args.build_dir)

    run_cases(caselist_abs_path, build_dir, args.output_xml, args.suite)
