# Unstructured C++ utils

## Interface for generated code (unstructured)

###  _Allocation_ of a computation object is provided by free function with the following properties

- takes one argument _Mesh_
- returns a Callable _Computation_

### _Mesh_

- `mesh::connectivity<NeighborChain>(Mesh)` and `mesh::connectivity<LocationType>(Mesh)` return an object modeling the _Connectivity_ concept
<!-- - `mesh::connectivity<LocationType>(Mesh)` returns an object modeling the _Location_ concept \[TODO: not a good name...\] -->
<!-- - `mesh::get_size<LocationType>(Mesh)` returns the number of elements of this LocationType (as `std::size_t`) \[Consider compile time sizes as well.\] -->
Notes:

- Only Neighbor Chains which are used in the computation need to be available.
- If a `NeighborChain` is present, we can provide a default implementation for the first element of the neighbor chain, e.g. if  `mesh::connectivity<std::tuple<edge, vertex>>(mesh)` is available, we can provide `mesh::connectivity<edge>>(mesh)` for free
- Mesh has a default implementation using hymap in unstructured_helper.hpp
- TODO: probably split the 2 concepts of real connectivity and primary location.

### _NeighborChain_ is

- a _tuple-like_ (see GridTools) object with _LocationTypes_, e.g. `std::tuple<edge, vertex>` (edge to vertex). \[Potentially, we could model more complex neighbor chains with this approach, e.g. vertex->cell->edge\]
- \[TODO: tuple-like or `std::tuple`?\]

### _LocationTypes_ are the following tag types

- `struct vertex;`
- `struct edge;`
- `struct cell;`

### _Connectivity_

Connectivity needs to be copyable to device, the access functions need to be callable on device (except `neighbor_table()`) all return values need to be valid on device (especially the SID neighbor table)

The following functions are defined
- `connectivity::size(Connectivity)` returns the number of elements of the primary location (as `std::size_t`).  \[Consider compile time sizes as well. TODO this information is also encoded in the connectivities, e.g. the upper_bound of the primary dimension of a neighbor table\]

For _NeighborChain_ _Connectivities_ (i.e. length > 1, not pure _LocationTypes_), additionally the following functions are defined
- `connectivity::max_neighbors(Connectivity)` returns a `std::integral_constant<std::size_t, N>` with `N` representing the maximal number of neighbors.
- `connectivity::skip_value(Connectivity)` returns the element signaling a non-existent value in a regular neighbor table
- `connectivity::neighbor_table(Connectivity)` returns a two dimensional SID with dimensions _LocationType_ and `neighbor` (TODO better name)

### _Computation_ has

- one argument for each input or output variable modelling _uSID_

### _unstructured SID (uSID)_
is a SID with the following dimensions:
- the unstructured dimension is identified with key _LocationType_
- a possible sparse dimension is identified with the `neighbor` tag
  \[TODO maybe something like `local<LocationType>` or `neighbor<LocationType>`, but then the same for the neighbor table to be able to iterate both with the same tag.\]
- the vertical dimension is identified with `namespace dim {struct k;}` (TODO better ideas?)
- TODO what to do with extra dimensions?


## TODO

- If elements are not indexed contiguously, we need to abstract the mechanism for iterating the primary location loop. E.g. loop over pole edges.
- k size?

## Build instructions

```
export BASEPATH=`pwd`

export ECBUILD_VERSION=3.3.2
wget https://github.com/ecmwf/ecbuild/archive/${ECBUILD_VERSION}.tar.gz
tar xzf ${ECBUILD_VERSION}.tar.gz
export "PATH=$PATH:$(pwd)/ecbuild-${ECBUILD_VERSION}/bin"

export ECKIT_VERSION=1.10.1
wget https://github.com/ecmwf/eckit/archive/${ECKIT_VERSION}.tar.gz
tar xzf ${ECKIT_VERSION}.tar.gz
pushd eckit-${ECKIT_VERSION}
mkdir build
pushd build
ecbuild ..
popd
popd

export ATLAS_VERSION=0.20.1
wget https://github.com/ecmwf/atlas/archive/${ATLAS_VERSION}.tar.gz
if [ ! -f "atlas-${ATLAS_VERSION}" ]; then
  tar xzf ${ATLAS_VERSION}.tar.gz
fi
cd atlas-${ATLAS_VERSION}
mkdir build
pushd build
ecbuild -DECKIT_PATH="$BASEPATH/eckit-${ECKIT_VERSION}/build" ..
make -j4
popd
```
