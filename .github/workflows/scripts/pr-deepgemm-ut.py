import subprocess
import os
import sys
import time
import argparse
import re
from junit_xml import TestCase, TestSuite

# Configuration: output file name
OUTPUT_XML = "test-results.xml"


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

    junit_case = TestCase(name=test_name, classname="shTest")

    print(f"\n[{i}/{total_count}] Running: {test_name}")
    print(f"Command: {case_command}")
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
            bufsize=1
        )

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line, end='')
                stdout_output.append(line)
                if re.search(r"\bfail(?:ed|ure)?\b", line, re.IGNORECASE):
                    failed_keyword_found = True

        process.wait(timeout=1200)
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
        process.kill()
        duration = time.time() - start_time
        junit_case.elapsed_sec = duration
        print(f"Timeout: {test_name}")
        junit_case.add_failure_info(message="Timeout", output="Test timed out")
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


def resolve_build_dir(build_dir):
    """Resolve the test working directory; default to the DeepGEMM repository root."""
    if build_dir:
        return os.path.abspath(build_dir)

    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def run_cases(caselist_path, build_dir, output_xml=OUTPUT_XML):
    if not os.path.exists(caselist_path):
        print(f"Error: test list file not found: {caselist_path}")
        sys.exit(1)

    # Determine file type by extension
    is_sh = os.path.splitext(caselist_path)[1].lower() == ".sh"

    # Parse commands based on file type
    test_commands = parse_commands(caselist_path, is_sh)

    if not test_commands:
        print("No valid test commands found in list")
        return

    # Switch to build directory
    try:
        os.chdir(build_dir)
        print(f"Working directory changed to: {os.getcwd()}")
    except Exception as e:
        print(f"Failed to change directory: {e}")
        sys.exit(1)

    all_test_objects = []
    global_passed_count = 0
    total_count = len(test_commands)

    # Run all cases
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

    # Generate JUnit XML report
    suite = TestSuite("DeepGEMM PR Tests", all_test_objects)
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        TestSuite.to_file(f, [suite], prettyprint=True)

    print(f"\nTests completed. Total: {total_count}, Passed: {global_passed_count}")
    print(f"Report generated: {output_xml_path}")

    if global_passed_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepGEMM PR tests from a caselist.")
    parser.add_argument("--caselist", required=True, help="Path to the caselist.list or run.sh file")
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

    run_cases(caselist_abs_path, build_dir, args.output_xml)
