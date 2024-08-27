---
tags: [backend, dace, optimization]
---

# Canonical Form of an SDFG in GT4Py (Especially for Optimizations)

- **Status**: valid
- **Authors**: Philip Müller (@philip-paul-mueller)
- **Created**: 2024-08-27

In the context of the implementation of the new DaCe fieldview we decided about a particular form of the SDFG.
Their main intent is to reduce the complexity of the GT4Py specific transformations.

## Context

The canonical form that is outlined in this document was mainly designed from the perspective of the optimization pipeline.
Thus it emphasizes a form that is can be handled in a simple and efficient way by a transformation.
In the pipeline we distinguishes between

- Intrastate optimization: The optimization of the data flow within states.
- Interstate optimization: The optimization between states, this are transformations that are _intended_ to _reduce_ the number of states.

The current (GT4Py) pipeline mainly focus on intrastate optimization and relays on DaCe, especially its simplify pass, for interstate optimizations.

## Decision

The canonical form is defined by several rules that affect different aspects of an SDFG and what a transformation can assume.
This allows to simplify the implementation of certain transformations.

#### General Aspects

The following rules, especially affects transformations and how they operate:

1. Intrastate transformation and interstate transformations must run separately and can not be mixed in the same (DaCe) pipeline.

   - [Rational]: As a consequence the number of "interstate transients" (transients that are used in multiple states) remains constant during intrastate transformations.
   - [Note 1]: It is allowed to run them after one another, as long as they are strictly separated.
   - [Note 2]: It is allowed that _intrastate_ transformation act in a way to allow state fusion by later intrastate transformations.
   - [Note 3]: The DaCe simplification pass violates this rule, for that reason this pass must always be called on its own, see also rule 2.

2. It is invalid to call the simplification pass directly, i.e. the usage of `SDFG.simplify()` is not allowed, the only valid way to call simplify is to call the `gt_simplify()` function provided by GT4Py.
   - [Rational]: It was observed that some sub passes in simplify have a negative impact and that additional passes might be needed in the future.
     By only using a single function later modifications to simplify are easy.
   - [Note]: One issue is that the remove redundant array transformation is not able to handle all cases.

#### Global Memory

The only restriction we impose on global memory is:

3. The same global memory is allowed to be used as input and output at the same time, iff the output depends _elementwise_ on the input.
   - [Rational 1]: Allows to remove double buffering, that DaCe may not remove, see also rule 2.
   - [Rational 2]: This formulation allows to write expressions such as `a += 1`, with only memory for `a`.
     Phrased more technically using global memory for input and output is allowed iff the two computations `tmp = computation(global_memory); global_memory = tmp;` and `global_memory = computation(global_memory);` are equivalent.
   - [Note]: In the long term this rule will be changed to: Global memory (an array) is either used as input (only read from) or as output (only written to) but never for both.

#### State Machine

For the SDFG state machine we assume that:

4. An interstate edge can only access scalars, i.e. use them in their assignment or condition expressions, but not arrays, even if they have shape `(1,)`.

   - [Rational]: If an array is also used in interstate edges it became very tedious to verify if the array could be removed or not.
   - [Note]: Running simplify might actually result in the violation of this rule, see note of rule 9.

5. The state graph does not contain any cycles, i.e. the implementation of a for/while loop using states is not allowed, the new loop construct or serial maps must be used in that case.
   - [Rational]: This is a simplification that makes it much simpler to define "later in the computation" means as we will never have a cycle.
   - [Note]: Currently the code generator does not support the `LoopRegion` construct and it is transformed to a state machine.

#### Transients

The rules we impose on transients are a bit more complicated, however, while sounding restrictive, they are very permissive.
It is important that these rules only have to be met after after simplify was called once on the SDFG:

6. Downstream of a write access, i.e. in all states that follows the state the access node is located in, there are no other access nodes that are used to write to the same array.

   - [Rational 1]: This rule together with rule 7 and 8 essentially boils down to ensure that the assignment in the SDFG follows SSA style, while allowing for expressions such as:

   ```python
   if cond:
       a = true_branch()
   else:
       a = false_branch()
   ```

   (**NOTE:** This could also be done with references, however, they are strongly discouraged.)

   - [Rational 2]: This still allows reductions with WCR as they write to the same access node and loops, whose body modifies a transient that outlives the loop body, as they use the same access node.

7. It is _recommended_ that a write access node should only have one incoming edge.

   - [Rational]: This case is handled poorly by some DaCe transformations, thus we should avoid them as much as possible.

8. No two access nodes in a state can refer to the same array.

   - [Rational]: Together with rule 5 this guarantees SSA style.
   - [Note]: An SDFG can still be constructed using different access node for the same underlying data; simplify will combine them.

9. Every access node that reads from an array (having an outgoing edge) that was not written to in the same state must be a source node.

   - [Rational]: Together with rule 1, 4, 5, 6, 7 and 8 this simplifies the check if a transient can be safely removed or if it is used somewhere else.
     These rules guarantee that the number of "interstate transients" remains constant and these set is given by the _set of source nodes and all access nodes that have an outgoing degree larger than one_.
   - [Note]: To prevent some issues caused by the violation of rule 4 by simplify, this set is extended with the transient sink nodes and all scalars.
     Excess interstate transients, that will be kept alive that way, will be removed by later calls to simplify.

10. Every AccessNode within a map scope must refer to a data descriptor whose lifetime must be `dace.dtypes.AllocationLifetime.Scope` and its storage class should be _preferable_ `dace.dtypes.StorageType.Register`.
    - [Rational 1]: Makes optimizations operating inside a maps/kernels simpler, as it guarantees that the AccessNode does not propagate outside.
    - [Rational 2]: The storage type avoids the need to dynamically allocate memory inside a kernel.

#### Maps

For maps we assume the following:

11. The names of map variables (iteration variable) follow the following pattern.

    - 11.1: All map variables iterating over the same dimension (disregarding the actual range), have the same deterministic name, that includes the `gtx.Dimension.value` string.
    - 11.2: The name of horizontal dimensions (`kind` attribute) always end in `__gtx_horizontal`.
    - 11.3: The name of vertical dimensions (`kind` attribute) always end in `__gtx_vertical`.
    - 11.4: The name of local dimensions always ends in `__gtx_localdim`.
    - 11.5: No transformation is allowed to modify the name of an iteration variable that follows rules 11.2, 11.3 or 11.4.
    - [Rational]: Without this rule it is very hard to tell which map variable does what, this way we can transmit information from GT4Py to DaCe, see also rule 12.

12. Two map ranges, i.e. the pair map/iteration variable and range, can only be fused if they have the same name _and_ cover the same range.
    - [Rational 1]: Because of rule 11 we will only fuse maps that actually makes sense to fuse.
    - [Rational 2]: This allows to fuse maps without performing a renaming on the map variables.
    - [Note]: This rule might be dropped in the future.

## Consequences

The rules outlined above impose a certain form of an SDFG.
Most of these rules are designed to ensure that the SDFG follows SSA style and to simplify transformations, especially making validation checks simple, while imposing a minimal number of restrictions.
