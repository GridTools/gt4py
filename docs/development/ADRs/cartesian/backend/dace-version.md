# DaCe backends: DaCe version

In the context of the [DaCe backend](./dace.md) and the [schedule tree](./dace-schedule-tree.md), facing time pressure, decided to implement the first version of schedule trees on top of `v1/maintenance` branch to minimize up-front cost and deliver CPU performance as fast as possible. Ever since, we are slowly moving back to mainline DaCe.

## Context

Originally, the schedule tree features was implemented on a branch based the maintenance branch of DaCe v1. DaCe v1 and what will be known as DaCe v2 have breaking chances, most notably the complete transition to control flow graphs (CFGs). Since GT4Py v1.1.5, we are on a branch that is based mainline DaCe with the goal to eventually merge the schedule tree feature into DaCe mainline. Until this happens, we stay on a branch that lives on the [GridTools fork](https://github.com/GridTools/dace) of DaCe.

## Decision

We decided to merge the schedule tree feature into mainline DaCe. A first step was to rebase our prototype on current mainline DaCe, which happened with GT4Py v1.1.5. The next step is to fully merge down to `main`.

## Consequences

Once the schedule tree feature is merge into mainline DaCe, we'll be able to simplify the dependencies again and use the same version of DaCe for `gt4py.cartesian` and `gt4py.next`.

## Alternatives considered

No need for alternatives - we can't stay on a branch of DaCe forever.
