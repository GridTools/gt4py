# DaCe version

In the context of the [DaCe backend](./backend-dace.md) and the [schedule tree](./backend-dace-schedule-tree.md), facing time pressure, we decided to stay at the `v1.x` branch of DaCe to minimize up-front cost and deliver CPU performance as fast as possible. We considered updating to the mainline version of DaCe and accept follow-up cost of partial rewrites once DaCe `v2` releases.

## Context

The [schedule tree](./backend-dace-schedule-tree.md) feature will need changes in DaCe to go from schedule tree to SDFG. Current released version of DaCe is on the `v1.x` branch. The mainline branch moved on (with breaking changes) to what is supposed to be DaCe `v2`. All feature development on the DaCe side has to be merged against mainline. Only bug fixes are allowed on the `v1.x` branch.

## Decision

We decided to build a first version of the schedule tree feature against the `v1.x` version of DaCe.

## Consequences

- We'll be able to code against familiar API (e.g. same as the previous GT4Py-DaCe bridge).
- In DaCe, we won't be able to merge changes into `v1.x`. We'll work on a branch and later refactor the schedule tree -> SDFG transformation to code flow regions in DaCe `v2`.

## Alternatives considered

### Update to DaCe mainline first

- Good because mainline DaCe is accepting new features while `v1.x` is closed for new feature development.
- Bad because it incurs an up-front cost, which we are trying to minimize to get results fast.
- Bad because we aren't trained to use the new control flow regions.
