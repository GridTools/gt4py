# Architecture Decision Records Index (`gt4py.next`)

This document contains links to all _Architecture Decision Record_ (ADR) documents written in the `gt4py.next` project. The [top-level README](../README.md) explains when and why we write ADRs.

## How to write ADRs

See [top-level README](../README.md) on when and why we write ADRs.

Writing a new ADR is simple:

1. Use the existing [Template](../Template.md) as an ice-breaker to start a new ADR file, but modify it and simplify it as much as possible to fit the type of decision being documented. If extra files (e.g. images) are needed for whatever reason, add them to the `_static/` folder.
2. Add a link to the new ADR file to the fitting topic in the index section below.
3. Open a PR to merge the changes into the main branch and let the team know about the new ADR.

## Index by Topic

### General architecture #general

- [0005 - Extending Iterator IR](0005-Extending_Iterator_IR.md)

### Frontend and Parsing #frontend

- [0001 - Field View Frontend Design](0001-Field_View_Frontend_Design.md)
- [0002 - Field View Lowering](0002-Field_View_Lowering.md)
- [0010 - Domain in Field View](0010-Domain_in_Field_View.md)
- [0013 - Scalar vs 0d-Fields](0013-Scalar_vs_0d_Fields.md)

### Iterator IR #iterator

- [0003 - Iterator View Tuple Support for Fields](0003-Iterator_View_Tuple_Support_for_Fields.md)
- [0004 - Lifted Stencils with Tuple Return](0004-Lifted_Stencils_with_Tuple_Return.md)

### Embedded Execution

_None_

### Transformations

- [0018 - Canonical Form of an SDFG in GT4Py (Especially for Optimizations)](0018-Canonical_SDFG_in_GT4Py_Transformations.md)

### Backends and Code Generation

- [0006 - C++ Backend](0006-Cpp-Backend.md)
- [0007 - Fencil Processors](0007-Fencil-Processors.md)
- [0008 - Mapping Domain to Cpp Backend](0008-Mapping_Domain_to_Cpp-Backend.md)
- [0016 - Multiple Backends and Build Systems](0016-Multiple-Backends-and-Build-Systems.md)
- [0017 - Toolchain Configuration](0017-Toolchain-Configuration.md)
- [0018 - Canonical Form of an SDFG in GT4Py (Especially for Optimizations)](0018-Canonical_SDFG_in_GT4Py_Transformations.md)

### Python Integration

- [0011 - On The Fly Compilation](0011-On_The_Fly_Compilation.md)
- [0012 - GridTools C++ OTF](0011-_GridTools_Cpp_OTF.md)

### Testing

- [0015 - Exclusion Matrices](0015-Test_Exclusion_Matrices.md)

### Superseded

- [0009 - Compiled Backend Integration](0009-Compiled-Backend-Integration.md)

## Other References

- [GitHub Blog - Why Write ADRs. How architecture decision records can help your team](https://github.blog/2020-08-13-why-write-adrs/)
- [Thoughtworks Technology Radar - Lightweight Architecture Decision Records](https://www.thoughtworks.com/radar/techniques/lightweight-architecture-decision-records)
- [Michael NygADR - Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR on GitHub - Architectural Decision Records](https://adr.github.io/)
- [Joel Parker Henderson - Collection of Architecture Decision Record (ADR)](https://github.com/joelparkerhenderson/architecture-decision-record)
