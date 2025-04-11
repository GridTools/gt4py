---
tags: [frontend]
---

# Field View Frontend Design

- **Status**: valid
- **Authors**: Rico Häuselmann (@dropd), Enrique G. Paredes (@egparedes), Till Ehrengruber (@tehrengruber)
- **Created**: 2022-01-31
- **Updated**: 2022-01-31

This document loosely tracks how the pieces of the Field View implementation fit together and which design decisions were deliberate and why.

## Guiding principles

Overall we tried to avoid:

1. having more IRs / ASTs than absolutely necessary goal
2. putting more logic than absolutely necessary into IR nodes (and their validators)
3. complex visitor methods (defining many subcases and deep `if-else` trees)

And we tried to achieve:

4. Make all error messages that relate to user code maximally useful from the start
5. Separate parsing (Python -> PAST/FOAST) from lowering (PAST/FOAST -> Iterator IR)

In order to achieve that we had to compromise in some aspects like, for example, when it comes to AST / IR nodes having or not having optional attributes.

We tried to avoid (1) because we have seen in previous projects that a long chain of IRs often leads to bad coupling problems, where a change on one end has to ripple through every IR and lowering along the way.

We tried to avoid (2) partially to avoid merge conflicts, by keeping the logic in passes we could work on them separately and mostly orthogonally. Specifically we wanted to avoid metadata logic (such as type detection rules) in validators because it decreases the visibility of said logic and increases the learning curve for new developers.

(3) just makes for really bad readability and maintainability.

In the past (4) has always been treated as an afterthought and never made it into production. Here we passed all source locations all the way and refer to them in error messages. We test for the presence of correct source locations in tests on purpose.

In order to help with maintainability, shallow learning curve and user experience, we decided to catch all compile time errors before lowering. That means the AST and dialect passes, as well as the parser in between, emit user-level warnings and errors, whenever possible with source location info. This also means if the lowering encounters an invalid dialect node it is not a user error. The only exception to this rule are errors in the lowering due to things not being implemented yet. These errors should be clearly marked as such.

## The different dialects

There exist two dialects right now. The FOAST and PAST. We will in the following refer to dialect AST as the AST of the respective dialect.

#### What to keep

##### Type hierarchy and type info in expressions

Symbols as well as expressions must have a type in all dialects. This deviates from the Python AST because all dialects are statically typed. The types have their own hierarchy of `eve` nodes. It also allows for the type info to be associated with source locations easily -> better debugging experience.

##### Source locations

All nodes that represent code must hold the location of that code.

##### Correspondence Python AST <-> dialect AST

In some cases there are Python constructs which would directly map to a function call in the iterator model (such as binary operators). It was chosen to still represent them as binary operators to keep the analogy to Python AST and clearly separate parsing and lowering. The correspondence between Python and dialect AST also means it would be relatively easy to "unparse" the dialect AST into valid Python code for debugging.

##### Symbol types are not dialect nodes

The symbol types in all dialects are implemented as simple dataclasses to ensure compatibility and reusability between different IRs. They are just _leaf_ nodes (not recursed into during tree traversals) describing which kind of symbol is defined in an IR node.

#### What could be changed

##### Ad-hoc pass management

The `DialectParser` has ended up being the main entry point for running all required passes to create a dialect AST that can be lowered to Iterator IR. This is not by design but simply because it was convenient. This is expected to change if a general pass manager is ever implemented. The two abstract class methods `._preprocess_definition_ast` and `._postprocess_definition_ast` encode the required order of AST passes.

### Field Operator AST (FOAST)

Essentially this is a Field Operator flavoured Python AST dialect. The tree structure follows the supported subset of Python AST. It uses the symbol table concept from `eve` because it was at the time a ready to use way of keeping track of type information, and names of temporaries.

#### What could be changed

##### Symbol table

The symbol table concept is not essential to it, replacing it would touch:

- The handling of type information from externals and parameters
- The inlining of assign statements into the return statement before lowering to Iterator IR

### Field Operator Parser

Parses from Python AST into FOAST.

#### What to keep

##### Assumptions on simplified AST

Some assumptions are made on the input AST which allow the visitor methods to stay quite simple. There are passes to be run on the AST before parsing which make it meet these assumptions. This pattern should be kept and extended. If Python AST allows for many cases which cause trouble in parsing, eliminate unneeded cases on the AST before parsing into FOAST.

### Field Operator Lowering

Lowers from FOAST into IteratorIR. FOAST expressions returning fields are transformed into ITIR expressions returning iterators with an additional `deref` at the very end.

This step makes use of the type information on expressions, for example to decide which names are fields and therefore have to be dereferenced. As of yet, there is no way to pass typing or source location info on to Iterator IR though.

#### What to keep

##### Assume FOAST is correct

As per guiding principle (5), the lowering should not worry if the FOAST is incorrect or invalid. It is the responsibility of the previous parser and passes to sanitize the user code.

### Program AST (PAST)

The Program AST is a stateful dialect whose main purpose is to apply operators to fields and storing the result on another field. In the future it could be extended to also support control flow (loops, conditionals).

#### What could be changed

The syntax to call field operators feels unnatural as their signature suggests they return a Field, but instead after decorating they return nothing and have an additional argument `out`.

#### Similarities to FOAST

The PAST is an independent dialect and (code-wise) only shares symbol types with the FOAST. This was a deliberate choice as there is currently no good way to share nodes and we wanted to allow changing each of the dialects independently until they become stable. Nonetheless nodes that exist in both dialects should stay in sync so that they could be generalized in the future.

#### PAST parser

Parses from Python AST into PAST.

#### PAST Lowering

Lowers from PAST to Iterator IR.

**Field operator out argument slicing**
The lowering contains some complex logic to emulate slicing of fields in order to restrict the output domain of a field operator call. This contradicts guiding principle (3) and should be removed in the future. The concept of specifying the domain for Field operators was not investigated during the frontend design as we concentrated on the beautified iterator dialect where the output domain is explicitly given when lifting a local operator to a field operator. The alternative to also explicitly specify the domain for calls to field operators was rejected as it:

- would require introducing significantly more syntax in order to specify domains, which was not only infeasible to implement in time, but there also didn't exist an accepted syntax to do so.
- no intuitive syntax was found to do this
  - `field_operator(in_field, out=out_field, domain=output_domain)`: Possible, but rather verbose.
  - `field_operator[output_domain](in_field, out=out_field)` This syntax is already used to create a field operator from a local operator.

### Decorators (@program, @field_operator)

#### What to keep

The decorators are designed to be constructed from the root node of their respective dialect contrary to a Python function object. This decision has been made to allow programmatically generating instances thereof.

The Program decorator is not aware of the concept of a FieldOperator but instead uses a generic mechanism (`GTCallable`) to inject functions into the resulting ITIR. This allows to keep the the various decorators independent from each other and avoids coupling the typing system (which is conceptually on a lower level), to them (e.g. `make_symbol_type_from_value` uses the `__gt_type__` method).

#### What could be changed

The GTCallable interface is not stable and was born as an ad-hoc concept to fullfil the requirements noted above. In particular the fact that it mixes types, i.e. a concept only found in the frontend, with the untyped iterator IR is not ideal. This leads to the strange situation, where the result of executing the ITIR, i.e. a stencil call, of a Field operator is a value, while the type suggests a Field return value and a call to the decorated function returns nothing.
