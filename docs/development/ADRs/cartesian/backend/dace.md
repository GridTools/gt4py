# DaCe backends

In the context of performance optimization, facing the fragmentedness of Numerical Weather Prediction (NWP) codes, we decided to implement a backend based on DaCe to unlock full-program optimization. We accept the downside of having to maintain that (additional) performance backend.

## Context

NWP codes aren't like your typical optimization problem homework where 80% of runtime is spent within a single stencil which you can then optimize to oblivion. Instead, computations in NWP codes are fragmented and scattered all over the place with parts in-between that move memory around. Stencil-only optimizations don't cut through this. DaCe allows us to do (data-flow) optimization on the full program, not only inside stencils. As a nice side-effect, DaCe offers code generation to CPU and GPU targets.

## Decision

We chose to add DaCe backends,`dace:cpu` and `dace:gpu`, for CPU and GPU targets because we need full-program optimization to get the best possible performance.

## Consequences

We will need to maintain the `dace:*` backends. If we keep adding more and more backends, maintainability will be a question down the road. We thus decided to remove [the cuda backend](../archived/backend-cuda.md) after a deprecation period, focussing on `dace:*` backends instead.

Compared to the [`cuda` backend](../archived/backend-cuda.md), which only targets NVIDIA cards, we get support for both, NVIDIA and AMD cards, with the `dace:gpu` backends.

## References

[DaCe Promo Website](http://dace.is/fast) | [DaCe GitHub](https://github.com/spcl/dace) | [DaCe Documentation](https://spcldace.readthedocs.io/en/latest/)
