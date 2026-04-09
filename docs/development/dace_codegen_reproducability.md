# Debugging indeterministic behavior of dace transformations

- Enable printing each transformation step, e.g. using
  ```
  dace.Config.set("progress", value=True)
  ```
  TODO: introduce new config var that prints the hash instead of hard-coding it.
- Execute the program in question twice and compare the output.
- Set a conditonal breakpoint in beginning of the `apply` method of the first pass where the SDFG hash changes with condition `sdfg.hash_sdfg() == <last equal hash>`.
  Note: In case running the previous passes takes a long time it makes sense to serialize the SDFG to
  json (`sdfg.to_json("sdfg(1|2).json")`) and loading it again (see debug script below).
  In rare cases the serializing and deserializing the sdfg changes the hash. In such cases this
  trick doesn't work and the first location where the hash changes might not be the exact location where the indeterministic behavior is. It helps to use a different hash, e.g. `content_hash`, but this should be solved in general.
  Note: It makes sense to also place a breakpoint after `DaceTranslator.generate_sdfg` to recognize
  when all executions finished.
- When the location is found it is usually easy to spot the origin of the indeterminism. Often 
  there is a set operation or a symbol is named in an indeterministic way. Use ordered sets and
  deterministic symbol names.

## Appendix

__Debugging sdfg autooptimize__

Usage `python debug_auto_optimize_sdfg.py sdfg1.json`

```python
import pickle
import sys

import dace
import json

from dace import SDFG

from gt4py.next.program_processors.runners.dace import (
    lowering as gtx_dace_lowering,
    sdfg_args as gtx_dace_args,
    transformations as gtx_transformations,
)
from dace.utils import print_sdfg_hash

file = sys.argv[1]

with open(file) as f:
    data = json.load(f)
    sdfg = dace.SDFG.from_json(data)
    print_sdfg_hash(sdfg)

    gtx_transformations.gt_auto_optimize(
        sdfg,
        gpu=False,
        constant_symbols={},
        unit_strides_kind=None,
    )
```

__Debugging single sdfg transform__

Usage `python debug_single_sdfg_transform.py sdfg1.json`

```python
import pickle
import sys

import dace
import json

from dace import SDFG

from gt4py.next.program_processors.runners.dace import (
    lowering as gtx_dace_lowering,
    sdfg_args as gtx_dace_args,
    transformations as gtx_transformations,
)
from dace.utils import print_sdfg_hash

transformation = gtx_transformations.MoveDataflowIntoIfBody
file = sys.argv[1]

with open(file) as f:
    data = json.load(f)
    sdfg = dace.SDFG.from_json(data)
    print_sdfg_hash(sdfg)

    sdfg.apply_transformations_repeated(
        transformation(
            ignore_upstream_blocks=False,
        ),
        validate=False,
        validate_all=True,
    )
```
