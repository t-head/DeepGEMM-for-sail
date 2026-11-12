# DeepGEMM Development Guidelines

These guidelines apply to contributions to this repository, including changes made by coding agents. Read the existing implementation and the relevant API, JIT, scheduler, and test code before making changes. Follow established contracts unless an API change is explicitly reviewed.

## Scope and architecture

- Keep changes focused on the requested operator, platform, and behavior. Separate unrelated cleanup, environment patches, and new features from kernel optimizations.
- Use the C++ JIT path under `csrc/` and device implementations under `deep_gemm/include/deep_gemm/`. The former Python JIT implementation has been removed; do not restore it or use it as an implementation dependency. Continue maintaining the Python APIs, utilities, and tests used by the current implementation.
- Keep public API validation and registration in the existing API layer. Put configuration selection, tile rules, and resource calculations in the corresponding heuristics layer; keep execution orchestration in the runtime implementation.
- Refer to upstream open-source DeepGEMM when designing APIs and argument checks. Align signatures, parameter meanings, validation behavior, and output semantics with upstream wherever practical. PPU architectural differences may justify API adjustments or new PPU-specific APIs; document the differences and their architectural rationale.
- Reuse existing copy primitives, schedulers, preprocessing, and helpers when they provide the required semantics and code generation. If a custom primitive is necessary, explain the concrete difference and provide generated-code or measurement evidence.
- Share common computation and isolate meaningful differences with small policies or helpers. Avoid cloning complete mainloops, adding many public wrappers, or coupling unrelated optimizations behind one strategy flag.
- Keep dtype-specific configuration logic independent. Move genuinely shared argument types to an appropriate common location rather than including another operator's implementation for incidental dependencies.
- Do not change shared compiler, cache, or runtime interfaces to accommodate one operator when the requirement can be handled in that operator's runtime. Cross-cutting changes require a clear justification and validation of affected callers.

## Optimization structure and tradeoffs

AI contributors are encouraged to explore any plausible optimization path, including specialized configurations, duplicated code, and temporary strategies. The generalization and structural requirements below apply when preparing changes for formal submission, not during exploration. Attribute the performance impact of each optimization separately through controlled comparisons, and record its measured gain or regression rather than reporting only the combined result. Before submission, select and consolidate the retained changes according to the following requirements.

- Generalization should consolidate experimental branches into a small set of strategies with distinct data flows or resource requirements. Do not carry every historical flag, buffer count, or fallback into the maintained implementation.
- Express shared computation once and place differences where they arise, such as route indexing, operand loading, or fragment lifetime. Use small policies or helpers for those differences rather than copying complete mainloops for each interface or tile.
- Use one resolved plan for configuration selection, resource calculations, routing preparation, and execution. Heuristics select the plan, the runtime orchestrates it, and kernels consume it; avoid independently reconstructing or implicitly reselecting the strategy in each layer.
- Decide whether to retain an optimization from its measured benefit, supported workload coverage, and maintenance cost. When performance is comparable, prefer fewer branches, clearer constraints, and lower resource requirements. A modest loss on some cases can be acceptable in exchange for a simpler general implementation.
- Retain multiple implementations only when each has a demonstrated, worthwhile domain that the others do not serve adequately. Explain the selection boundary and the added code and validation cost. Merge or remove strategies whose selection or measured benefits no longer meaningfully differ.
- Generalization does not require every case to improve. Report the largest regression, affected workloads, and overall benefit for review. Evaluate specialized configurations together with their maintenance cost and the quality of the general strategy.
- Remove rejected experiments completely, including unused switches, configuration fields, helpers, and their dedicated tests and documentation. Keep experimental evidence in local records rather than preserving inactive implementations without a supported use.
- Keep maintained tests focused on contracts and representative strategy, tail, and resource boundaries. Keep broad performance exploration in offline tools, and consolidate redundant tests when implementations merge or disappear without dropping distinct behavioral coverage.

## Configuration and dispatch

- Configuration selection must be deterministic and based on host-known shapes, documented hints, configured compute resources, and device capabilities. Performance lookup tables for tuned workloads are allowed, but must coexist with a maintained general strategy that also covers workloads absent from the table. On representative workloads, compare the general strategy with exact tuned configurations and keep the performance gap reasonably small; report typical and worst-case gaps for review rather than relying on table entries to hide a weak general path. Apply kernel and heuristic changes to the general path as well, and validate it independently with lookup overrides disabled.
- Do not benchmark candidate kernels on the device during production dispatch. Avoid timing calibration, candidate trial launches, device-to-host synchronization for tuning, and cached measured cost coefficients. Offline tuning tools are separate from production dispatch.
- Follow the existing caller-selected device and stream convention. Do not add automatic device switching to an individual API without a reviewed repository-wide need. Check device-dependent resources consistently with that convention.

## Graph replay and concurrency

- Graph replay must correctly use updated inputs. Do not incorrectly reuse cached results that depend on previous input contents.
- When using shared workspace or internal streams, ensure concurrency safety, correct execution dependencies, and valid resource lifetimes.

## Numerical and functional correctness

- APIs and execution paths may support only a subset of cases, but must not silently produce incorrect results. Clearly document support boundaries and explicitly reject unsupported inputs or combinations.
- State input formats, scale semantics, layouts, output order, alignment, supported dimensions, and masking behavior. For masked inputs, distinguish allocated capacity from valid row counts and expected-work hints.
- Preserve numerical ordering and rounding unless a numerical change is explicitly reviewed. Do not relax tolerances to make an optimization pass. Use references derived from the actual quantized inputs and declared scale semantics.
- Validate expensive static properties during preprocessing when possible; do not rescan immutable weights on every GEMM call. Preserve the established preprocessing contract and reject unsupported values clearly.
- Cover nonuniform and empty experts, partial tiles, repeated token use, routing changes, and invalid-output regions as applicable. Use basis or structured-input checks for layout changes; random aggregate error alone can hide permutations.
- New execution paths must account for existing scalar options and output semantics. Document and review unsupported combinations rather than silently ignoring parameters or dropping an existing API capability.

## Validation and performance evidence

- Add corresponding UTs to the checkin list when adding or changing APIs. Keep the tests few and focused, covering key behavior without redundant cases.
- New APIs must integrate with this repository's UT dump/replay tooling and cover common execution paths in replay. Use `DgProfParam` and `ProfilingInterface::instrument` in [profiling_interface.cuh](../deep_gemm/include/deep_gemm/common/profiling_interface.cuh), enable dump output with `PPU_LIB_SHOW_PARAMS=1`, and replay emitted cases with [tests/run_deep_gemm.py](../tests/run_deep_gemm.py) using `--format`. Full path coverage is not required; some device-side parameters may not be dumpable. Document such limitations and the supported replay scope.
- When `show_log` is enabled, new APIs must print kernel information using the repository's existing logging conventions, including the selected execution path, tile configuration, and relevant resource information where applicable.
- Base optimization targets and specialization predicates on realistic application behavior and supported calling contracts. Scenario-specific optimizations are acceptable when the scenario can reasonably occur and is supported by application evidence or representative traces. Do not add specializations for conditions that are practically absent from real workloads merely because a synthetic test constructs them. In particular, balanced expert counts do not imply regularly spaced token IDs in a fused gather; cyclic or uniformly arranged test routing must not be treated as a production invariant.
- Rebuild the host extension after changing `csrc/`. For device-only changes, verify that the JIT uses the intended source and configuration. Check the loaded package and synchronized source when validating remotely.
- Verify functionality and numerical accuracy before measuring performance, and use idle devices without disrupting other workloads.
- Distinguish useful GEMM utilization, the tensor work actually issued, and profiler counters.

## Documentation and commits

- Every submitted pull request must link to a corresponding issue.
- Update README and CHANGELOG when APIs change; complex APIs may have detailed documentation under `docs/api/`. Update relevant existing tests for API or behavior changes. Document constraints and performance claims precisely. Use public product names such as M890 in formal README and release-note text; existing internal architecture identifiers may remain in code.
- Add release notes only for changes in the corresponding commit. Explain why an optimization is needed and how it changes execution; keep commit messages focused on the change and its validation.
- Keep local experiments, profiler reports, build products, machine-specific settings, `tmp/` artifacts, and unrequested examples out of commits. Do not include machine names or device identifiers in commit messages; retain that provenance in validation logs.
- Before submission, inspect the complete diff, remove stale experimental interfaces, and confirm that the documented configuration actually selects the stated plan. Do not publish unmeasured performance claims or claim a full regression suite from a small sample.

## Applying these guidelines

Contributors and coding agents should load this document before implementation or review and resolve conflicts with explicit task instructions and approved API contracts. This document is guidance, not an automated enforcement mechanism. Changes to public configuration formats, execution topology, numerical acceptance, or shared infrastructure should be reviewed before implementation.
