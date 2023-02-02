## The Code Structure and Dependencies:

  - [common](common). Contains definitions that are used pretty much everywhere in the directory.
  In user facing code, in core and in backends.
  - [backend](backend). Contains stencil composition backends definitions.
    The code here depends on [common](common) and [be_api.hpp](be_api.hpp) but should not explicitly
    depend on [core](core) and should not depend explicitly or implicitly on [frontend](frontend).
  - [frontend](frontend). Contains user facing definitions. Depends on [common](common)
    and [core](core) but should not depend on [backend](backend) explicitly.
  - [core](core). Everything that is not user facing and doesn't belong to the concrete backend.
    Should only depend on [common](common).
  - [cartesian.hpp](cartesian.hpp). All user facing definitions that are needed to do stencil computations
    on cartesian grids.
  - [icosahedral.hpp](icosahedral.hpp). All user facing definitions that are needed to do stencil computations
    on icosahedral grids.
  - [frontend.hpp](frontend.hpp). User facing definitions that are shared between [cartesian.hpp](cartesian.hpp)
    and [icosahedral.hpp](icosahedral.hpp).
  - [be_api.hpp](be_api.hpp) - Contains an API for backend creators.
  - [global_parameter.hpp](global_parameter.hpp) and [positional.hpp](positional.hpp). Models of the SID concept that
    are used both in [backend](backend) and user code. They are not included into [frontend.hpp](frontend.hpp) set
    because they are kind of optional. On the other hand they are not in [common](common) to avoid extra nesting
    for the user facing header.

The user code should only include the headers from that directory and from [backend](backend)
and not from any other subdirectory.

## Namespaces

  - `gridtools::stencil`. Grid independent user facing entities are defined here.
  - `gridtools::stencil::dim`, `gridtools::stencil::cache_io_policy`, `gridtools::stencil::cache_type` sub namespaces are of the same kind.
  - `gridtools::stencil::cartesian`. All user facing entities that are specific to cartesian grid.
  - `gridtools::stencil::cartesian::expressions`. Is used to enable constructs like `eval(in1() + in2())`  within stencil
    `apply` functions.
  - `gridtools::stencil::icosahedral`. All user facing entities that a specific to icosahedral grid.
  - `gridtools::stencil::core`. Internal stuff. Not user facing.
  
