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
5. Separate parsing (Python -> FOAST) from lowering (FOAST -> Iterator IR)

In order to achieve that we had to compromise in some aspects like, for example, when it comes to AST / IR nodes having or not having optional attributes.

We tried to avoid (1) because we have seen in previous projects that a long chain of IRs often leads to bad coupling problems, where a change on one end has to ripple through every IR and lowering along the way. 

We tried to avoid (2) partially to avoid merge conflicts, by keeping the logic in passes we could work on them separately and mostly orthogonally. Specifically we wanted to avoid metadata logic (such as type detection rules) in validators because it decreases the visibility of said logic and increases the learning curve for new developers.

(3) just makes for really bad readability and maintainability.

In the past (4) has always been treated as an afterthought and never made it into production. Here we passed all source locations all the way and refer to them in error messages. We test for the presence of correct source locations in tests on purpose.

In order to help with maintainability, shallow learning curve and user experience, we decided to catch all compile time errors between the function definition and FOAST. That means the AST and FOAST passes, as well as the parser in between, emit user-level warnings and errors, whenever possible with source location info. This also means if the lowering encounters an invalid FOAST it is not a user error.


## The pieces

### Field Operator AST (FOAST)
Essentially this is a Field Operator flavoured Python AST dialect. The tree structure follows the supported subset of Python AST. It uses the symbol table concept from `eve` because it was at the time a ready to use way of keeping track of type information, and names of temporaries.

#### What to keep
##### Type hierarchy and type info in expressions
Symbols as well as expressions have a type in FOAST. This deviates from the Python AST because the Field View language is typed. The types have their own hierarchy of `eve` nodes. It also allows for the type info to be associated with source locations easily -> better debugging experience.

##### Source locations
All nodes that represent code must hold the location of that code.

##### Correspondence AST <-> FOAST
In some cases there are Python constructs which would directly map to a function call in the iterator model (such as binary operators). It was chosen to still represent them as binary operators in FOAST to keep the analogy to Python AST and clearly separate parsing and lowering. The correspondence between FOAST and AST nodes also means it would be relatively easy to "unparse" FOAST into valid Python code for debugging.

##### Symbol types are not FOAST nodes
The symbol types in FOAST are implemented as simple dataclasses to help reusability between different IRs, since they are just _leaf_ nodes (not recursed into during tree traverslas) describing which kind of symbol is defined in an IR node.

#### What could be changed
##### Symbol table
The symbol table concept is not essential to it, replacing it would touch:
* The handling of type information from externals and parameters
* The inlining of assign statements into the return statement before lowering to Iterator IR

### Field Operator Parser
Parses from Python AST into FOAST.

#### What to keep
##### Assumptions on simplified AST
Some assumptions are made on the input AST which allow the visitor methods to stay quite simple. There are passes to be run on the AST before parsing which make it meet these assumptions. This pattern should be kept and extended. If Python AST allows for many cases which cause trouble in parsing, eliminate unneeded cases on the AST before parsing into FOAST.

#### What could be changed
##### Ad-hoc pass management
This has ended up being the main entry point for running all required passes to create a FOAST that can be lowered to Iterator IR. This is not by design but simply because it was convenient. This is expected to change if a general pass manager is ever implemented.

The `.apply` classmethod encodes the required order of AST passes to simplify the Python AST before parsing into FOAST. These simplifications allow the visitor methods to be less complex than otherwise possible.

##### Applying to function / source code might be reused
The `.apply_to_function` classmethod contains also the code to go from a Python function to the required data to call `.apply`. That includes getting the source and source locations as well as the closure data (externals / globals, their types and values). That code could be split off for reuse in other parsers.

### Field Operator Lowering
Lowers from FOAST into IteratorIR.

This step makes use of the type information on expressions, for example to decide which names are fields and therefore have to be dereferenced. As of yet, there is no way to pass typing or source location info on to Iterator IR though.

#### What to keep
##### Assume FOAST is correct
As per guiding principle (5), the lowering should not worry if the FOAST is incorrect or invalid. It is the responsibility of the previous parser and passes to sanitize the user code.

#### What could be changed
##### Ad-hoc pass management
Since this lowering goes from a statement based AST to a functional IR, all the statements in the AST have to be inlined. This happens in one of the visitor methods currently which calls a FOAST pass, which is perhaps not ideal.

