---
tags: [testing]
---

# Test-Exclusion Matrices

- **Status**: valid
- **Authors**: Edoardo Paone (@edopao), Enrique G. Paredes (@egparedes)
- **Created**: 2023-09-21
- **Updated**: 2024-01-25

In the context of Field View testing, lacking support for specific ITIR features while a certain backend
is being developed, we decided to use `pytest` fixtures to exclude unsupported tests.

## Context

It should be possible to run Field View tests on different backends. However, specific tests could be unsupported
on a certain backend, or the backend implementation could be only partially ready.
Therefore, we need a mechanism to specify the features required by each test and selectively enable
the supported backends, while keeping the test code clean.

## Decision

It was decided to apply fixtures and markers from `pytest` module. The fixture is the same used to execute the test
on different backends (`exec_alloc_descriptor` and `program_processor`), but it is extended with a check on the available feature markers.
If a test is annotated with a feature marker, the fixture will check if this feature is supported on the selected backend.
If no marker is specified, the test is supposed to run on all backends.

In the example below, `test_offset_field` requires the backend to support dynamic offsets in the translation from ITIR:

```python
@pytest.mark.uses_dynamic_offsets
def test_offset_field(cartesian_case):
```

In order to selectively enable the backends, the dictionary `next_tests.definitions.BACKEND_SKIP_TEST_MATRIX`
lists for each backend the features that are not supported.
The fixture will check if the annotated feature is present in the exclusion-matrix for the selected backend.
If so, the exclusion matrix will also specify the action `pytest` should take (e.g. `SKIP` or `XFAIL`).

The test-exclusion matrix is a dictionary, where `key` is the backend name and each entry is a tuple with the following fields:

`(<marker[str]>, <skip_definition[SKIP,XFAIL]>, <skip_message(format keys: 'marker', 'backend')>)`

The backend string, used both as dictionary key and as string formatter in the skip message, is retrieved
by calling `next_tests.get_processor_id()`, which returns the so-called processor name.
The following backend processors are defined:

```python
DACE_CPU = "gt4py.next.program_processors.runners.dace.run_dace_cpu"
DACE_GPU = "gt4py.next.program_processors.runners.dace.run_dace_gpu"
GTFN_CPU = "gt4py.next.program_processors.runners.gtfn.run_gtfn"
GTFN_CPU_IMPERATIVE = "gt4py.next.program_processors.runners.gtfn.run_gtfn_imperative"
GTFN_CPU_WITH_TEMPORARIES = "gt4py.next.program_processors.runners.gtfn.run_gtfn_with_temporaries"
GTFN_GPU = "gt4py.next.program_processors.runners.gtfn.run_gtfn_gpu"
```

Following the previous example, the GTFN backend with temporaries does not support yet dynamic offsets in ITIR:

```python
BACKEND_SKIP_TEST_MATRIX = {
    GTFN_CPU_WITH_TEMPORARIES: [
        ("uses_dynamic_offsets", pytest.XFAIL, "'{marker}' tests not supported by '{backend}' backend"),
    ]
}
```

## Consequences

Positive outcomes of this decision:

- The solution provides a central place to specify test exclusion.
- The test code remains clean from if-statements for backend exclusion.
- The exclusion matrix gives an overview of the feature-readiness of different backends.

Negative outcomes:

- There is not (yet) any code-style check to enforce this solution, so code reviews should be aware of the ADR.

## References <!-- optional -->

- [pytest - Using markers to pass data to fixtures](https://docs.pytest.org/en/6.2.x/fixture.html#using-markers-to-pass-data-to-fixtures)
