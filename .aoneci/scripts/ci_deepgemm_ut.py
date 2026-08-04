import os
import xml.etree.ElementTree as ET
import subprocess
import sys
from junit_xml import TestCase, TestSuite, to_xml_report_string
from datetime import datetime
import time

# Configuration: paths and output file
test_dir = "tests"
output_xml = "test-results.xml"

# Explicitly specify the test files to run (all must end with .py)
test_files_cmds = [
    ["python", os.path.join(test_dir, "test_core.py")],
    ["python", os.path.join(test_dir, "test_attention.py")],
    ["python", os.path.join(test_dir, "test_jit.py")],
    ["python", os.path.join(test_dir, "run_deep_gemm.py"), "--format=DenseGemm,data_type:tf32,m:8192,n:24,k:16384"],
    ["python", os.path.join(test_dir, "tile_scan_with_test.py"), "--format=GroupedNoPad,data_type:int8,groups:32,m:4096,n:4096,k:7168,em:128,distribution:uniform"],
    ["python", os.path.join(test_dir, "tile_scan_with_test.py"), "--format=GroupedNoPad,data_type:bf16,groups:16,m:4096,n:1024,k:2048,em:256,distribution:uniform"],
    ["python", os.path.join(test_dir, "run_deep_gemm.py"), "--format=GroupedFused,data_type:int8,groups:256,num_token:256,topk:8,n:256,k:6144"],
    ["python", os.path.join(test_dir, "run_deep_gemm.py"), "--format=GroupedFused,data_type:int8,groups:256,num_token:2048,topk:8,n:6144,k:128"],
    ["python", os.path.join(test_dir, "run_deep_gemm.py"), "--format=GroupedFused,data_type:bf16,groups:256,num_token:64,topk:8,n:256,k:6144"],
    ["python", os.path.join(test_dir, "run_deep_gemm.py"), "--format=GroupedFused,data_type:bf16,groups:256,num_token:256,topk:8,n:6144,k:128"],
    ["python", os.path.join(test_dir, "run_deep_gemm.py"), "--format=GroupedNoPad,data_type:w4a16,groups:384,m:3072,n:512,k:7168,quant_type:group,group_size:32"],
    ["python", os.path.join(test_dir, "run_deep_gemm.py"), "--format=GroupedMasked,data_type:w4a16,groups:96,m:3072,n:2048,k:7168,quant_type:group,group_size:32"],
    ["python", os.path.join(test_dir, "run_deep_gemm.py"), "--format=GroupedFused,data_type:w4a16,groups:384,num_token:8,topk:8,n:512,k:7168,quant_type:group,group_size:32"],
    #["pytest", os.path.join(test_dir, "test_deep_gemm_tuner.py"), "::test_deepgemm_tuning[dense] -v"],
]
temp_xml_files = []

def run_test_python(cmd, idx):
    """Run a single python test command and generate a JUnit XML report."""
    temp_xml = f"results_{idx}.xml"
    pass_flag = 1
    # Extract the script name as the testcase name
    script_path = cmd[1]
    script_name = os.path.basename(script_path)
    test_name = f"{script_name}_{idx}"
    if len(cmd) > 2:
        args_str = " ".join(cmd[2:])
        test_name += f" [{args_str}]"

    print(f"🚀 Running: {' '.join(cmd)}")
    print(f"File path: {script_path} | Exists: {os.path.exists(script_path)}")

    case = TestCase(name=test_name, classname="ManualPythonTest")
    start_time = time.time()  # Start timer (seconds, float)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30-minute timeout
        stderr_output = result.stderr.strip()
        stdout_output = result.stdout.strip()
        end_time = time.time()
        duration = end_time - start_time  # Elapsed time (seconds)
        
        # Format duration with 6 decimal places (ms/μs precision)
        case.elapsed_sec = round(duration, 6)
        if result.returncode == 0:
            print(f"✅ Test passed: {test_name} (elapsed: {duration*1000:.2f} ms)")
            print(stdout_output)
        else:
            pass_flag = 0
            print(f"❌ Test failed: {test_name}, return code: {result.returncode} (elapsed: {duration*1000:.2f} ms)")
            print(stderr_output)
            case.add_failure_info(
                message="Test exited with non-zero code",
                output=f"STDOUT:\n{stdout_output}\nSTDERR:\n{stderr_output}"
            )
    except subprocess.TimeoutExpired:
        pass_flag = 0
        error_msg = "Test timed out after 600 seconds"
        print(f"⏰ Timeout: {test_name}")
        end_time = time.time()
        duration = end_time - start_time
        case.elapsed_sec = round(duration, 6)
        case.add_failure_info(message="Timeout", output=error_msg)
    except Exception as e:
        pass_flag = 0
        error_msg = str(e)
        end_time = time.time()
        duration = end_time - start_time
        case.elapsed_sec = round(duration, 6)
        print(f"🚨 Exception: {test_name}, error: {e}")
        case.add_error_info(message="Exception", output=error_msg)

    # Create TestSuite and write to XML
    suite = TestSuite("PythonScriptTests", [case])
    xml_content = to_xml_report_string([suite])

    with open(temp_xml, 'w', encoding='utf-8') as f:
        f.write(xml_content)

    temp_xml_files.append(temp_xml)
    print(f"📄 Generated JUnit XML: {temp_xml}")
    if not pass_flag:
        exit(1)
    return temp_xml


def merge_xml(temp_xml_files, output_xml):
    """Merge all temporary XML files into the final result."""
    final_root = ET.Element("testsuite", name="pytest", tests="0", failures="0", errors="0", time="0.0")

    for temp_xml in temp_xml_files:
        if not os.path.exists(temp_xml) or os.path.getsize(temp_xml) == 0:
            print(f"⚠️ Skipping empty file: {temp_xml}")
            continue

        try:
            tree = ET.parse(temp_xml)
            root = tree.getroot()

            # Find <testsuite> elements under <testsuites>
            for testsuite in root.findall("testsuite"):
                # Extract and merge all testcase elements
                for testcase in testsuite.findall("testcase"):
                    final_root.append(testcase)

                # Update aggregate statistics
                final_root.set("tests", str(int(final_root.get("tests")) + int(testsuite.get("tests", "0"))))
                final_root.set("failures", str(int(final_root.get("failures")) + int(testsuite.get("failures", "0"))))
                final_root.set("errors", str(int(final_root.get("errors")) + int(testsuite.get("errors", "0"))))
                final_root.set("time", str(float(final_root.get("time")) + float(testsuite.get("time", "0.0"))))

                print(f"✅ Merged {testsuite.get('tests', '0')} test cases from {temp_xml}")
        except ET.ParseError as e:
            print(f"❌ Failed to parse {temp_xml}: {e}")

    # Write the final XML file
    final_tree = ET.ElementTree(final_root)
    final_tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    print(f"✅ Test results merged into {output_xml}")

def main():
    # 1. Run tests and generate temporary XML files
    for i, cmd in enumerate(test_files_cmds):
        try:
            run_test_python(cmd, i)
        except Exception as e:
            print(f"🔥 Critical error while running test command {cmd}: {e}")

    # 2. Merge XML files
    merge_xml(temp_xml_files, output_xml)

    # 3. Clean up temporary files
    for temp_xml in temp_xml_files:
        try:
            os.remove(temp_xml)
            print(f"🗑️ Deleted temporary file: {temp_xml}")
        except Exception as e:
            print(f"⚠️ Failed to delete {temp_xml}: {e}")

if __name__ == "__main__":
    main()