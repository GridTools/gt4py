# Architecture Decision Records Index

This document contains links to all _Architecture Decision Record_ (ADR) documents written in this project. Existing ADRs should convey the rationale for the design decisions in the current codebase and their associated trade-offs.


## Writing ADRs
New ADRs should be written at the time when significant decisions for the project are taken, and they should capture both the final decision and the thought process that led to it. Basically, they should explain why a feature is built the current way and not some other way. Note that even if the reasons might look simple for the author, it could be complicated for others to understand the chosen option without the proper context.

Using ADRs to document the context and the design process of the technical decisions gives other team members more information about how the framework components work and how they fit within the whole design.  In the long term, it also helps the decision maker to recall the forces driving his decisions.

Writing a new ADR is simple:

1. Use the existing [Template](Template.md) as an ice-breaker to start a new ADR file, but modify it and simplify it as much as possible to fit the type of decision being documented. If extra files (e.g. images) are needed for whatever reason, add them to the `_static/` folder.
2. Add a link to the new ADR file to the fitting topic in the index section below.
3. Open a PR to merge the changes into the main branch and let the team know about the new ADR.


## Index by Topic

### General architecture #general
- [0005 - Extending Iterator IR](0005-Extending_Iterator_IR.md)
 
### Frontend and Parsing #frontend 
- [0001 - Field View Frontend Design](0001-Field_View_Frontend_Design.md)
- [0002 - Field View Lowering](0002-Field_View_Lowering.md)

### Iterator IR #iterator
- [0003 - Iterator View Tuple Support for Fields](0003-Iterator_View_Tuple_Support_for_Fields.md)
- [0004 - Lifted Stencils with Tuple Return](0004-Lifted_Stencils_with_Tuple_Return.md)

### Embedded Execution

### Transformations
_None_

### Backends and Code Generation
_None_

### Miscellanea
_None_


## Other References

- [GitHub Blog - Why Write ADRs. How architecture decision records can help your team](https://github.blog/2020-08-13-why-write-adrs/)
- [Thoughtworks Technology Radar - Lightweight Architecture Decision Records](https://www.thoughtworks.com/radar/techniques/lightweight-architecture-decision-records)
- [Michael NygADR - Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR on GitHub - Architectural Decision Records](https://adr.github.io/)
- [Joel Parker Henderson - Collection of Architecture Decision Record (ADR)](https://github.com/joelparkerhenderson/architecture-decision-record)
