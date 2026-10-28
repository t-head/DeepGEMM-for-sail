"""CPU-only regression tests for check-in scheduling, parsing and exit status."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import run_checkin

ROOT = Path(__file__).resolve().parents[2]


def parser_namespace():
    # Execute the actual parser without importing Torch/the PPU extension.
    tree = ast.parse((ROOT / 'tests/utils.py').read_text())
    names = {'parse_deepgemm_string_re', 'convert_data_type_to_dtype', 'test_batch_gemm'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    torch = SimpleNamespace(**{name: name for name in
                              ('bfloat16', 'int8', 'float8_e4m3fn', 'uint8', 'float32')})
    namespace = {'ast': ast, 'torch': torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'utils.py', 'exec'), namespace)
    return namespace


class ParserTests(unittest.TestCase):
    def test_cli_lists_cases_from_any_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for cwd in (ROOT, Path(directory)):
                for suite, filename in run_checkin.SUITES.items():
                    with self.subTest(cwd=cwd, suite=suite):
                        output = subprocess.check_output(
                            [sys.executable, str(ROOT / 'tests/checkin/run_checkin.py'),
                             '--suite', suite, '--list'], cwd=cwd, text=True)
                        cases = run_checkin.load_cases(ROOT / 'tests/checkin' / filename)
                        self.assertEqual(output.splitlines(),
                                         [f'{i}: {spec}' for i, spec in enumerate(cases, 1)])

    def test_all_board_cases_have_dispatch_and_batch_arguments(self):
        namespace = parser_namespace()
        tree = ast.parse((ROOT / 'tests/run_deep_gemm.py').read_text())
        dispatch = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'call_test_func')
        formats = {key.value for key in dispatch.body[0].value.keys}
        for name in run_checkin.SUITES.values():
            cases = run_checkin.load_cases(ROOT / 'tests/checkin' / name)
            for spec in cases:
                with self.subTest(case=spec):
                    case = namespace['parse_deepgemm_string_re'](spec)
                    self.assertIn(case['gemm_type'], formats)
                    if case['gemm_type'] == 'BatchGemm':
                        captured = []
                        namespace['test_einsum'] = captured.append
                        namespace['test_batch_gemm'](case)
                        mapped = captured[0]
                        self.assertIn(mapped['expr'], ('bhr,hdr->bhd', 'bhr,hdr->hbd'))
                        self.assertEqual((mapped['h'], mapped['b'], mapped['d'], mapped['r']),
                                         (case['groups'], case['m'], case['n'], case['k']))

    def test_expression_and_list_can_precede_more_fields(self):
        parse = parser_namespace()['parse_deepgemm_string_re']
        case = parse('BatchGemm,data_type:fp8,expr:bhr,hdr->hbd,quant_type:channel,distribution:[0,1,7],m:8')
        self.assertEqual(case['expr'], 'bhr,hdr->hbd')
        self.assertEqual(case['distribution'], [0, 1, 7])
        self.assertEqual(case['m'], 8)

    def test_relative_includes_empty_and_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'part.list').write_text('DenseGemm,data_type:bf16,m:2,n:128,k:128\n')
            (root / 'root.list').write_text('# comment\npart.list\n')
            self.assertEqual(len(run_checkin.load_cases(root / 'root.list')), 1)
            (root / 'part.list').write_text('root.list\n')
            with self.assertRaisesRegex(ValueError, 'Recursive'):
                run_checkin.load_cases(root / 'root.list')
            (root / 'part.list').write_text('# no tests\n')
            with self.assertRaisesRegex(ValueError, 'Empty'):
                run_checkin.load_cases(root / 'part.list')



class ReusableWorkerTests(unittest.TestCase):
    def test_actual_worker_logging_failure_and_per_case_reset(self):
        stub = '''from types import SimpleNamespace as NS
import os
torch=NS(backends=NS(cuda=NS(matmul=NS()),cudnn=NS()),manual_seed=lambda n: print('seed',n),cuda=NS(synchronize=lambda:None,empty_cache=lambda:None,reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:0,max_memory_reserved=lambda:0))
deep_gemm=NS(__path__=['cpu-test'])
def set_benchmark(v): print('benchmark',v)
def set_ref_backend(v): print('reference',v)
def set_acc_check(v): print('accuracy',v)
def parse_deepgemm_string_re(spec): return dict(gemm_type=spec)
def device_version_supported(case): return case['gemm_type']!='unsupported','unsupported board'
def call_test_func(kind, args):
 os.write(1,b'native output\\n')
 if kind=='bad': raise AssertionError('injected accuracy failure')
 if kind=='legacy_exit': exit(1)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tests/checkin').mkdir(parents=True)
            (root / 'tests/checkin/checkin_worker.py').write_text((ROOT / 'tests/checkin/checkin_worker.py').read_text())
            (root / 'tests/run_deep_gemm.py').write_text(stub)
            with mock.patch.object(run_checkin, 'REPO_ROOT', root):
                worker = run_checkin.Worker('run', root)
                try:
                    results = [worker.run(i, spec, 'run', root, 5)
                               for i, spec in enumerate(('ok', 'bad', 'unsupported', 'legacy_exit', 'ok'), 1)]
                    self.assertEqual([bool(r.error) for r in results], [False, True, True, True, False])
                    self.assertIn('SystemExit: 1', results[3].error)
                    for index in (1, 2, 5):
                        output = (root / f'run-{index:04d}.log').read_text()
                        self.assertIn('native output', output)
                        self.assertIn('accuracy True', output)
                        self.assertEqual(output.count('seed 0'), 1)
                    self.assertIn('injected accuracy failure', (root / 'run-0002.log').read_text())
                finally:
                    worker.close()

    def test_reuse_phase_environment_and_timeout_restart(self):
        # Exercise real pipes/processes without requiring a Torch installation.
        script = '''import argparse, json, os, sys, time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--result-fd', type=int); a=p.parse_args()
for line in sys.stdin:
 r=json.loads(line)
 Path(r['log']).write_text(json.dumps(dict(pid=os.getpid(),warm=os.environ.get('HGGC_WARM_UP'))))
 if r['spec']=='hang': time.sleep(30)
 if r['spec']=='crash': os._exit(7)
 os.write(a.result_fd,(json.dumps(dict(index=r['index'],error='bad case' if r['spec']=='bad' else ''))+'\\n').encode())
'''
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tests/checkin').mkdir(parents=True)
            (root / 'tests/checkin/checkin_worker.py').write_text(script)
            with mock.patch.object(run_checkin, 'REPO_ROOT', root), mock.patch.dict(os.environ, {'HGGC_WARM_UP': '0'}):
                worker = run_checkin.Worker('run', root)
                try:
                    for i, spec in enumerate(('ok', 'bad', 'ok', 'hang', 'ok', 'crash', 'ok'), 1):
                        result = worker.run(i, spec, 'run', root, .3 if spec == 'hang' else 5)
                        self.assertEqual(bool(result.error), spec in ('bad', 'hang', 'crash'))
                        if spec == 'hang':
                            self.assertIn('Timeout', result.error)
                    records = [json.loads((root / f'run-{i:04d}.log').read_text().splitlines()[0]) for i in range(1, 8)]
                    self.assertEqual(len({r['pid'] for r in records[:4]}), 1)
                    self.assertEqual(records[4]['pid'], records[5]['pid'])
                    self.assertNotEqual(records[3]['pid'], records[4]['pid'])
                    self.assertNotEqual(records[5]['pid'], records[6]['pid'])
                    self.assertTrue(all(r['warm'] is None for r in records))
                finally:
                    worker.close()
                worker = run_checkin.Worker('compile', root)
                try:
                    self.assertFalse(worker.run(1, 'ok', 'compile', root, 5).error)
                    self.assertEqual(json.loads((root / 'compile-0001.log').read_text())['warm'], '1')
                finally:
                    worker.close()


class ScriptChecksTests(unittest.TestCase):
    def test_ci_uses_staged_runner_for_both_source_modes(self):
        workflow = (ROOT / '.github/workflows/deepgemm-ut-test-ppu.yml').read_text()
        start = workflow.index('            set +e', workflow.index('echo "START TO TEST CASES"'))
        end = workflow.index('            TEST_EXIT_CODE=$?', start)
        command = '\n'.join(line[12:] for line in workflow[start:end].splitlines())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            (bin_dir / 'python').symlink_to(sys.executable)
            runner = root / 'tests/checkin/run_checkin.py'
            runner.parent.mkdir(parents=True)
            runner.write_text("import json,sys; from pathlib import Path; "
                              "Path('runner-args.json').write_text(json.dumps(sys.argv[1:]))")
            for mode in ('deepgemm', 'actlize1.0.0'):
                for suite in run_checkin.SUITES:
                    with self.subTest(mode=mode, suite=suite):
                        env = dict(os.environ, SOURCE_MODE=mode,
                                   PATH=str(bin_dir) + os.pathsep + os.environ.get('PATH', ''))
                        subprocess.run(['bash', '-c', command.replace('${{ matrix.runner }}', suite)],
                                       cwd=root, env=env, check=True)
                        self.assertEqual(json.loads((root / 'runner-args.json').read_text()),
                                         ['--suite', suite, '--compile-workers', '16'])

    def test_daily_retains_original_script_and_tile_commands_verbatim(self):
        original = json.loads((ROOT / '.github/workflows/scripts/testcase.json').read_text())
        for suite, scripts in (
            ('OAM-810E', ['test_core.py', 'test_attention.py']),
            ('ZW-M890P', ['test_core.py', 'test_fp4_core.py', 'test_fp8_core.py', 'test_attention.py']),
        ):
            with self.subTest(suite=suite):
                selected = run_checkin.load_script_checks(suite)
                expected = [f'python ./tests/{script}' for script in scripts]
                expected += [cmd for cmd in original[suite] if 'tests/tile_scan_with_test.py' in cmd]
                self.assertEqual(selected, expected)
                self.assertEqual(len(selected), len(scripts) + 2)
                self.assertFalse(any('DG_USE_CUTE=0' in cmd for cmd in selected))

    def test_new_interfaces_are_in_the_matching_board_caselists(self):
        original = json.loads((ROOT / '.github/workflows/scripts/testcase.json').read_text())
        parse = parser_namespace()['parse_deepgemm_string_re']
        for suite, filename in run_checkin.SUITES.items():
            cases = run_checkin.load_cases(ROOT / 'tests/checkin' / filename)
            for cmd in original[suite]:
                if cmd.startswith('python ') and any(dtype in cmd for dtype in
                        ('data_type:bf16_bf16_fp32', 'data_type:w4a8_int8')):
                    self.assertIn(cmd.split('--format=', 1)[1], cases)
            parsed = [parse(case) for case in cases]
            if suite == 'OAM-810E':
                self.assertEqual(sum(case['data_type'] == 'w4a8_int8' for case in parsed), 6)
            else:
                avg = [case for case in parsed if case['gemm_type'] == 'MqaAvgLogits'
                       and case['data_type'] == 'uint8']
                self.assertEqual({case['q_scale'] for case in avg}, {0, 1, 2})
                self.assertEqual({case['logits_dtype'] for case in avg}, {'bfloat16', 'float32'})
                self.assertEqual({case.get('compressed_logits', 0) for case in avg}, {0, 1})
                self.assertTrue({258, 513, 1025, 8193}.issubset({case['seq_len_kv'] for case in avg}))

    def test_legacy_subprocess_environment_and_junit_failures(self):
        script = '''import argparse,json,os,sys,xml.etree.ElementTree as ET
p=argparse.ArgumentParser()
for arg in ('caselist','suite','build-dir','output-xml'): p.add_argument('--'+arg)
a=p.parse_args()
print('warmup_present='+str('HGGC_WARM_UP' in os.environ))
commands=json.load(open(a.caselist))[a.suite]
if commands==['missing']: sys.exit(3)
root=ET.Element('testsuites');suite=ET.SubElement(root,'testsuite')
for command in commands:
 case=ET.SubElement(suite,'testcase',name=command,time='1.25')
 if command in ('failure','error','skipped'): ET.SubElement(case,command)
ET.ElementTree(root).write(a.output_xml)
if commands==['exit-only']:sys.exit(2)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / '.github/workflows/scripts'
            scripts.mkdir(parents=True)
            (scripts / 'pr-deepgemm-ut.py').write_text(script)
            commands = ['ok', 'failure', 'error', 'skipped']
            (scripts / 'testcase.json').write_text(json.dumps({'OAM-810E': commands}))
            with mock.patch.object(run_checkin, 'REPO_ROOT', root), \
                    contextlib.redirect_stdout(io.StringIO()):
                for warm in ('1', '0'):
                    with mock.patch.dict(os.environ, HGGC_WARM_UP=warm):
                        results = run_checkin.run_script_checks('OAM-810E', root / 'logs')
                    self.assertEqual([bool(result.error) for result in results], [False, True, True, True])
                    self.assertTrue(all(result.elapsed == 1.25 for result in results))
                    self.assertIn('warmup_present=False', (root / 'logs/script-checks.log').read_text())
                    run_checkin.write_report(results, root / 'merged.xml')
                    xml = ET.parse(root / 'merged.xml').getroot()
                    self.assertEqual(xml.get('failures'), '3')
                    self.assertTrue(all(case.get('classname') == 'DeepGEMM.script'
                                        for case in xml.findall('testcase')))
                for command in ('missing', 'exit-only'):
                    (scripts / 'testcase.json').write_text(json.dumps({'OAM-810E': [command]}))
                    results = run_checkin.run_script_checks('OAM-810E', root / 'logs')
                    self.assertEqual(len(results), 1)
                    self.assertTrue(results[0].error)

    def test_only_daily_runs_scripts_after_accuracy_and_propagates_failures(self):
        for phase, error, include, expected in (
            ('all', '', False, False), ('all', '', True, True),
            ('all', 'compile failed', True, False),
            ('compile', '', True, False), ('run', '', True, True),
        ):
            with self.subTest(phase=phase, error=error, include=include):
                events = []
                def execute(*args):
                    events.append('caselist')
                    return [run_checkin.Result(1, 'case', 'compile', 0, error, Path('unused'))]
                def scripts(*args):
                    events.append('scripts')
                    return [run_checkin.Result(1, 'legacy', 'script', 0, 'failed', Path('unused'))]
                args = ['--suite', 'OAM-810E', '--phase', phase]
                if include:
                    args.append('--include-daily-scripts')
                with mock.patch.object(run_checkin, 'execute', side_effect=execute), \
                        mock.patch.object(run_checkin, 'run_script_checks', side_effect=scripts), \
                        mock.patch.object(run_checkin, 'write_report'), \
                        contextlib.redirect_stdout(io.StringIO()):
                    code = run_checkin.main(args)
                self.assertEqual(events, ['caselist', 'scripts'] if expected else ['caselist'])
                self.assertEqual(code, int(bool(error) or expected))

    def test_daily_entry_lists_sweeps_from_any_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for suite in run_checkin.SUITES:
                output = subprocess.check_output(
                    [sys.executable, str(ROOT / 'tests/daily/run_daily.py'),
                     '--suite', suite, '--list-daily-scripts'], cwd=directory, text=True)
                self.assertEqual(output.splitlines(), run_checkin.load_script_checks(suite))

    def test_fp4_default_epilogue_retains_legacy_accuracy_boundary(self):
        tree = ast.parse((ROOT / 'tests/utils.py').read_text())
        fused = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'is_act_and_quant_fused')
        for name in ('test_m_grouped_gemm_nopad', 'test_m_grouped_gemm_fused'):
            func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            threshold = next(n for n in func.body if isinstance(n, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == 'threshold' for t in n.targets))
            assertion = next(n for n in ast.walk(func) if isinstance(n, ast.Assert)
                             and ast.unparse(n.test) == 'diff < threshold')
            boundary = ast.Assert(test=assertion.test, msg=ast.Constant(value='accuracy boundary'))
            code = compile(ast.fix_missing_locations(ast.Module(
                body=[fused, threshold, boundary], type_ignores=[])), 'utils.py', 'exec')
            for dtype, epilogue, limit in (
                ('fp4', 'Default', .001), ('fp4', 'SiluAndMulPostQuantFp4', .0015),
                ('int8', 'Default', .0015), ('fp8', 'Default', .0015),
            ):
                for diff in (.0009, .0012, .0016):
                    with self.subTest(function=name, dtype=dtype, epilogue=epilogue, diff=diff):
                        namespace = dict(torch=SimpleNamespace(uint8='fp4'), d=dtype,
                                         epilogue_type=epilogue, diff=diff)
                        if diff >= limit:
                            with self.assertRaises(AssertionError):
                                exec(code, namespace)
                        else:
                            exec(code, namespace)


class SchedulingTests(unittest.TestCase):
    def test_cli_defaults_to_sixteen_workers_and_accepts_override(self):
        for options, expected in (([], 16), (['--compile-workers', '4'], 4)):
            with self.subTest(options=options), \
                    mock.patch.object(run_checkin, 'execute', return_value=[]) as execute, \
                    mock.patch.object(run_checkin, 'run_script_checks', return_value=[]), \
                    mock.patch.object(run_checkin, 'write_report'), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_checkin.main(['--suite', 'OAM-810E', *options]), 0)
                self.assertEqual(execute.call_args.args[1], expected)

    def test_compile_barrier_and_serial_accuracy(self):
        events, active = [], set()
        lock = threading.Lock()
        barrier = threading.Barrier(2)

        def runner(index, spec, phase, log_dir, timeout):
            with lock:
                if phase == 'run':
                    self.assertEqual(len([event for event in events if event[0] == 'compiled']), 2)
                    self.assertFalse(active)
                active.add(index)
            if phase == 'compile':
                barrier.wait(timeout=5)
            with lock:
                active.remove(index)
                events.append(('compiled' if phase == 'compile' else 'ran', index))
            return run_checkin.Result(index, spec, phase, 0., '', Path(log_dir) / 'unused')

        with contextlib.redirect_stdout(io.StringIO()):
            results = run_checkin.execute(['a', 'b'], 2, '.', 10, runner=runner)
        self.assertEqual(len(results), 4)
        self.assertEqual(events[2:], [('ran', 1), ('ran', 2)])

    def test_compile_failure_prevents_all_accuracy_runs(self):
        calls = []

        def runner(index, spec, phase, log_dir, timeout):
            calls.append(phase)
            return run_checkin.Result(index, spec, phase, 0., 'compiler error' if index == 1 else '', Path('unused'))

        with contextlib.redirect_stdout(io.StringIO()):
            results = run_checkin.execute(['a', 'b'], 2, '.', 10, runner=runner)
        self.assertEqual(calls, ['compile', 'compile'])
        self.assertEqual(len(results), 2)

    def test_child_environment_return_code_timeout_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tests').mkdir()
            # A real child process stands in for the device-dependent UT.
            (root / 'tests/run_deep_gemm.py').write_text('''import os, sys, time
assert '--require_supported' in sys.argv
assert '--disable_acc' not in sys.argv
spec = sys.argv[sys.argv.index('--format') + 1]
print('warm_up=' + repr(os.environ.get('HGGC_WARM_UP')), flush=True)
if spec == 'error':
    print('compiler <error> with control character: \\x01', flush=True)
    sys.exit(3)
if spec == 'timeout':
    time.sleep(30)
''')
            env = dict(os.environ, HGGC_WARM_UP='0')
            with mock.patch.object(run_checkin, 'REPO_ROOT', root):
                compiled = run_checkin.run_case(1, 'ok', 'compile', root, 10, env)
                checked = run_checkin.run_case(1, 'ok', 'run', root, 10, env)
                failed = run_checkin.run_case(2, 'error', 'compile', root, 10, env)
                timed = run_checkin.run_case(3, 'timeout', 'compile', root, 1, env)
            self.assertIn("warm_up='1'", compiled.log.read_text())
            self.assertIn('warm_up=None', checked.log.read_text())
            self.assertEqual(env['HGGC_WARM_UP'], '0')
            self.assertEqual(failed.error, 'Exit code 3')
            self.assertIn('Timeout', timed.error)
            report = root / 'results.xml'
            run_checkin.write_report([compiled, checked, failed, timed], report)
            xml = ET.parse(report).getroot()
            self.assertEqual(xml.get('failures'), '2')
            self.assertEqual(xml.get('tests'), '4')
            self.assertIn('compiler <error>', xml.find('testcase/failure').text if xml.find('testcase/failure') is not None
                          else ''.join(element.text or '' for element in xml.iter('failure')))


class ExitStatusTests(unittest.TestCase):
    def run_main(self, failure=None, warm_up=False, arch=89, spec='DenseGemm,data_type:bf16,m:2,n:128,k:128', modules=None):
        namespace = parser_namespace()
        torch = namespace['torch']
        torch.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                         cudnn=SimpleNamespace(allow_tf32=False))
        torch.manual_seed = lambda _: None
        torch.cuda = SimpleNamespace(is_available=lambda: False, get_device_name=lambda: 'M890P')
        state = {'acc': True, 'benchmark': False, 'called': 0}

        def case_function(_):
            state['called'] += 1
            if failure:
                raise failure

        helpers = SimpleNamespace(**{key: value for key, value in namespace.items() if callable(value)})
        tree = ast.parse((ROOT / 'tests/run_deep_gemm.py').read_text())
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == 'utils':
                for name in node.names:
                    if not hasattr(helpers, name.name):
                        setattr(helpers, name.name, case_function)
        helpers.get_arch_version = lambda: arch
        helpers.judge_device_type = lambda: None
        helpers.set_acc_check = lambda value: state.update(acc=bool(value))
        helpers.get_acc_check = lambda: state['acc']
        helpers.set_benchmark = lambda value: state.update(benchmark=value)
        helpers.set_ref_backend = lambda value: None
        argv = ['run_deep_gemm.py', '--format', spec, '--require_supported', '--benchmark']
        with mock.patch.dict(sys.modules, {'torch': torch, 'utils': helpers, 'deep_gemm': SimpleNamespace(__path__=[]), **(modules or {})}), \
                mock.patch.object(sys, 'argv', argv), \
                mock.patch.dict(os.environ, {'HGGC_WARM_UP': '1' if warm_up else ''}), \
                mock.patch('atexit.register'), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                runpy.run_path(str(ROOT / 'tests/run_deep_gemm.py'), run_name='__main__')
        return result.exception.code, state

    def test_accuracy_failure_and_legacy_system_exit_are_nonzero(self):
        for error in (AssertionError('precision mismatch'), SystemExit(1)):
            code, state = self.run_main(failure=error)
            self.assertEqual(code, 1)
            self.assertEqual(state['called'], 1)

    def test_warmup_disables_accuracy_and_benchmark(self):
        code, state = self.run_main(warm_up=True)
        self.assertEqual(code, 0)
        self.assertFalse(state['acc'])
        self.assertFalse(state['benchmark'])
        code, state = self.run_main()
        self.assertEqual(code, 0)
        self.assertTrue(state['acc'])

    def test_unsupported_board_case_cannot_pass(self):
        code, state = self.run_main(arch=80, spec='DenseGemm,data_type:fp4,m:64,n:512,k:512')
        self.assertEqual(code, 1)
        self.assertEqual(state['called'], 0)

    def test_functional_cases_call_test_modules_with_phase_accuracy(self):
        for warm_up in (False, True):
            for spec, module, function in (
                ('Activation,data_type:bf16,activation:silu,quant_mode:MXFP4',
                 'test_activation', 'test_checkin_activation'),
                ('Preprocess,data_type:fp4', 'test_fp4_core', 'test_mxfp4_preprocessing'),
                ('AttentionRegression,data_type:int8,case:ks_ke', 'test_attention', 'test_ks_ke'),
                ('AttentionRegression,data_type:int8,case:cache_view', 'test_attention', 'test_per_layer_cache_view'),
            ):
                with self.subTest(spec=spec, warm_up=warm_up):
                    functions = {name: mock.Mock() for name in
                                 ('test_checkin_activation', 'test_mxfp4_preprocessing',
                                  'test_ks_ke', 'test_per_layer_cache_view')}
                    code, _ = self.run_main(warm_up=warm_up, spec=spec,
                                            modules={module: SimpleNamespace(**functions)})
                    self.assertEqual(code, 0)
                    target = functions[function]
                    target.assert_called_once()
                    self.assertEqual(target.call_args.kwargs, {'check_accuracy': not warm_up})
                    if module == 'test_activation':
                        self.assertEqual(target.call_args.args[0]['quant_mode'], 'MXFP4')
                    else:
                        self.assertEqual(target.call_args.args, ())
                    for name, other in functions.items():
                        if name != function:
                            other.assert_not_called()


class ActivationComparisonTests(unittest.TestCase):
    def test_empty_experts_and_strided_scale_views(self):
        class ScaleView:
            dtype = 'uint16'

            def __getitem__(self, item):
                if item[0] == 0:
                    raise AssertionError('empty expert has no valid values to compare')
                return self

            def view(self, dtype):
                raise AssertionError('strided scales must not be reinterpreted as bytes')

        tree = ast.parse((ROOT / 'tests/test_activation.py').read_text())
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'assert_masked_rows_equal')
        namespace = {'torch': SimpleNamespace(float8_e4m3fn='float8', uint8='uint8', equal=lambda x, y: True)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'test_activation.py', 'exec'), namespace)
        namespace['assert_masked_rows_equal'](ScaleView(), ScaleView(), (0, 1, 7))


class FP4WarmupContractTests(unittest.TestCase):
    def test_fused_outputs_keep_quantized_contract_without_accuracy(self):
        # Shape-only tensors let the real constructors run on a CPU-only host.
        # No reference arithmetic should be requested when accuracy is disabled.
        class Tensor:
            def __init__(self, shape, dtype='bfloat16'):
                self.shape, self.dtype = tuple(shape), dtype

            def to(self, kind):
                return self if kind in ('cuda', 'cpu') else Tensor(self.shape, kind)

            def view(self, *shape):
                return Tensor(self.shape, shape[0])

            def clone(self):
                return Tensor(self.shape, self.dtype)

            def int(self):
                return self.to('int32')

            def amax(self):
                return self

            def item(self):
                return 7

            def __getitem__(self, key):
                return Tensor(self.shape[1:] if isinstance(key, int) else self.shape, self.dtype)

            def __setitem__(self, key, value):
                pass

        def tensor(*shape, dtype='bfloat16', **kwargs):
            return Tensor(shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape, dtype)

        torch = SimpleNamespace(**{name: name for name in ('bfloat16', 'uint8', 'uint16', 'int8',
                                 'float8_e4m3fn', 'float', 'int', 'int32')})
        torch.Tensor, torch.dtype = Tensor, str
        torch.randn = torch.empty = torch.zeros = tensor
        torch.tensor = lambda values, **kwargs: tensor(len(values), **kwargs)
        torch.randperm = lambda n: Tensor((n,), 'int32')
        torch.empty_like = lambda x: Tensor(x.shape, x.dtype)
        torch.stack = lambda items, dim=0: Tensor((len(items), *items[0].shape), items[0].dtype)

        def quantize(x):
            return (Tensor((*x.shape[:-1], x.shape[-1] // 2), 'uint8'),
                    Tensor((*x.shape[:-1], x.shape[-1] // 32), 'uint8'))

        def preprocess(mode, fused, weight=None, weight_scale=None):
            return weight, weight_scale.to('uint16')

        tree = ast.parse((ROOT / 'tests/utils.py').read_text())
        names = {'construct_non_permute_grouped', 'construct_contiguous_grouped', 'construct_grouped_masked'}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {'torch': torch, 'Tuple': __import__('typing').Tuple, '_acc_check': False,
                     'get_ref_backend': lambda: 'device', 'ceil_div': lambda a, b: (a + b - 1) // b,
                     'construct_group_m_list': lambda *a, **kw: [0, 1, 7],
                     'get_masked_padding_m': lambda *a: 128, 'is_act_and_quant_fused': lambda _: True,
                     'quantize_fp4_weight_torch': lambda weight: quantize(weight),
                     'deep_gemm': SimpleNamespace(PreprocessMode=SimpleNamespace(MXFP4=0),
                                                 preprocess_weights_and_scales=preprocess)}
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'utils.py', 'exec'), namespace)
        quant_module = SimpleNamespace(quantize_fp4_torch=quantize, dequantize_fp4_torch=lambda *a: None)
        with mock.patch.dict(sys.modules, {'test_fp4_core': quant_module}):
            for epilogue in ('SiluAndMulPostQuantFp4', 'SiluAndMulStepPostQuantFp4'):
                kwargs = dict(epilogue_type=epilogue)
                outputs = [
                    namespace['construct_non_permute_grouped'](3, 8, 128, 128, 2, 'uint8', 'group', **kwargs),
                    namespace['construct_contiguous_grouped'](3, 8, 128, 128, 'uint8', 'uniform', 1, **kwargs),
                    namespace['construct_grouped_masked'](3, 8, 3, 128, 128, 'uint8', 'uniform', **kwargs),
                ]
                for constructed in outputs:
                    # Each constructor puts its output/ref immediately after counts/ids.
                    offset = 4 if isinstance(constructed[0], int) else 3
                    out, ref = constructed[offset:offset + 2]
                    self.assertIsInstance(out, tuple)
                    self.assertIsInstance(ref, tuple)
                    self.assertEqual([part.dtype for part in out], ['uint8', 'uint16'])
                    self.assertEqual([part.shape for part in out], [part.shape for part in ref])


if __name__ == '__main__':
    unittest.main()
