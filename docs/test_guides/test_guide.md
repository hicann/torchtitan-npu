# Test Guide

## Core Commands
### Unit Tests
```bash
# Run all unit tests and generate reports
bash .ci/unit_test.sh --generate-report

# Run only local `torchtitan-npu` unit tests
RUN_TORCHTITAN_UT=false bash .ci/unit_test.sh --generate-report
```

### Smoke Tests
```bash
# Run the default smoke suite (core + extended)
bash .ci/smoke_test.sh --generate-report

# Run only core smoke
ONLY_CORE_SMOKE=true bash .ci/smoke_test.sh --generate-report

# Run only extended smoke
ONLY_EXTENDED_SMOKE=true bash .ci/smoke_test.sh --generate-report

# Run only upstream smoke
ONLY_UPSTREAM_SMOKE=true bash .ci/smoke_test.sh --generate-report
```

### Integration Test

`tests/smoke_tests/integration_test.py` is the entry point for end-to-end integration tests, used to validate:
- New model functionality support
- Feature compatibility
- Parallelism strategy compatibility

#### Running

```bash
# Via .ci/smoke_test.sh (runs core + extended smoke by default)
ONLY_CORE_SMOKE=true bash .ci/smoke_test.sh --generate-report

# Run integration_test.py directly
python tests/smoke_tests/integration_test.py output_dir \
    --test_name all \
    --ngpu 2
```

#### Command-line Arguments

| Argument | Default | Description |
|------|--------|------|
| `output_dir` | None (required) | Output directory for test results |
| `--test_name` | `all` | Specific test case name |
| `--ngpu` | `2` | Maximum GPU count |

#### OverrideDefinitions Usage

`OverrideDefinitions` is the configuration class for defining integration test cases:

```python
OverrideDefinitions(
    override_args=[[...]],  # Required: command-line argument list
    test_descr="...",        # Required: test description
    test_name="...",         # Required: test name
    ngpu=2,                  # Optional: required GPU count
    disabled=False,          # Optional: whether disabled
)
```

#### Steps to Add a New Test Case

1. Open `tests/smoke_tests/integration_test.py`
2. Add a new configuration to the `smoke_cases` list in `generate_smoke_tests()`:
```python
OverrideDefinitions(
    [
        [
            "--module your_model",
            "--config your_config",
            "--parallelism.tensor_parallel_degree 2",
        ],
    ],
    "Your Model TP Test",
    "your_model_tp",
    ngpu=2,
)
```
3. Run tests to verify:
```bash
python tests/smoke_tests/integration_test.py ./outputs --test_name your_model_tp
```

#### Config Registry Configuration

Integration tests no longer use `tests/smoke_tests/base_test.toml`. Each test passes
`--module` and `--config` to `scripts/run_train.sh`, then appends nested tyro overrides
such as `--training.steps 2` or `--parallelism.tensor_parallel_degree 2`.

### Model Parallel Commands
```bash
# Basic model-parallel smoke
python3 -m pytest -v tests/smoke_tests/model_parallel/

# Multi-rank model-parallel smoke
RUN_MODEL_PARALLEL_MULTI_RANK=true torchrun --nproc_per_node=4 -m pytest -v tests/smoke_tests/model_parallel/
```

## When to Use Which Command
| Command | Use It When |
|---|---|
| `.ci/unit_test.sh` | You changed hardware-independent logic such as converters, config, helpers, or patches |
| `.ci/smoke_test.sh` | You changed real NPU execution paths or wrapper behavior and want the default core + extended smoke set |
| `ONLY_CORE_SMOKE=true` | You changed the minimal training path (i.e., end-to-end integration tests defined in integration_test) |
| `ONLY_EXTENDED_SMOKE=true` | You changed local feature or model-parallel behavior |
| `ONLY_UPSTREAM_SMOKE=true` | You changed logic that depends on reused torchtitan upstream integration, or want to run the heavier upstream smoke path separately |

## Quick Decision Rule
- Changed only hardware-independent logic: start with `.ci/unit_test.sh`
- Changed NPU feature paths or wrappers: run `.ci/smoke_test.sh`
- Changed training-path wiring: at least run `ONLY_CORE_SMOKE=true .ci/smoke_test.sh`
- Changed model-parallel behavior: run `ONLY_EXTENDED_SMOKE=true .ci/smoke_test.sh`
- Upstream integration compatibility needs a separate check: run `ONLY_UPSTREAM_SMOKE=true .ci/smoke_test.sh`

## Test Reports
- Output directory: `test_reports/`
- Common artifacts:
  - `*.xml`: JUnit results
  - `*.html`: HTML reports when `--generate-report` is enabled
  - `coverage/`: UT coverage reports
  - `README.md`: generated index of report artifacts

## Quick Tips
1. Start with the smallest command that matches your change.
2. Prefer `.ci/unit_test.sh` when NPU is not required.
3. Use targeted smoke variants instead of full smoke when possible.
4. Update docs when test layout or execution changes.
