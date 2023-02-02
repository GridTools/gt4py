Storage Library provides the way to represent multidimensional typed contiguous memory allocations with arbitrary
layout and alignment. The library is headers only. All entities are defined in the `gridtools::storage` namespace.

## Data Store

The key entity of the library. Represents N-d array.
  
  - Data store has no user facing constructors. To create it one should use Builder API.
  - The access to the actual data is indirect. Data store has methods to request a view. The view provides data access
    via overloaded call operator.
  - Data store is aware of memory spaces. It distinguish between `target` and `host` data access. Views are
    requested with `target_view()`/`target_const_view()`/`host_view()`/`host_const_view()` methods. If `target` and
    `host` spaces are not the same and the data store holds non constant data, data store performs automatic memory
    synchronization if it needed. It is assumed that the target memory space access is used for doing computations
    and host access is used the filling, dumping and verifying the data.
 
 ### Data Store Synopsis
 
 ```C++
template</* Implementation defined parameters */>
class data_store {
  public:
    static constexpr size_t ndims; /* Dimensionality */
    using layout_t = /* Instantiation of gridtools::layout_map. */;
    using data_t = /* Type of the element. */;
    // The following invariant is held: for any data_store instancies that have the same kind_t
    // the strides are the same. 
    using kind_t = /* A type that identifies the strides set. */;
    
    // Data store arbitrary label. Mainly for debugging.
    std::string const &name() const;
    // The sizes of the data store in each dimension.
    array<unsigned, ndims> lengths() const;
    // The strides of the data store in each dimension.
    array<unsigned, ndims> strides() const;
    
    // lengths and strides in the form of tuples.
    // If the length along some dimension is known in compile time (N),
    // it is represented as an intergral_constant<int, N>,
    // otherwise as int.
    auto const& native_lengths() const;
    auto const& native_strides() const;
    
    // 1D length of the data store expressed in number of elements.
    // Namely it is a pointer difference between the last and the first element minus one.
    unsigned length() const;
    
    // Supplementary object that holds lengths and strides. 
    auto const& info() const;

    // Request the target view.
    // If the target and host spaces are different necessary synchronization is performed
    // and the host counterpart is marked as dirty.  
    auto target_view();
    // Const version doesn't mark host counterpart as dirty. Synchronization takes place. 
    auto const_target_view();

    // Raw ptr alternatives for target_view/const_target_view.
    // Synchronization behaviour is the same.
    data_t *get_target_ptr();
    data_t const *get_const_target_ptr();

    // Host access methods variations. They only exist if !std::is_const_v<data_t>.  
    auto host_view();
    auto host_const_view();
    data_t *get_host_ptr();
    data_t const *get_const_host_ptr();
};
 ```

### Data View Synopsis

Data view is a supplemental struct that is returned form data store access methods. The distinctive property:
data view is a POD. Hence it can be passed to the target device by copying the memory. For the gpu data stores
all data view methods are declared as device only.

```C++
template <class T, size_t N>
struct some_view {
    // POD members here

    // The meta info methods are the same as for data_store. 
    array<unsigned, N> lengths() const;
    array<unsigned, N> strides() const;
    auto const& native_lengths() const;
    auto const& native_strides() const&
    unsigned length() const;
    auto const& info() const;

    // raw access
    T *data() const;
    // multi dimensional indexed access
    T &operator()(int... /*number of arguments equals to N*/ ) const;
    // variation with array as an argument
    T& operator()(array<int, N> const &) const;
};
```

## Builder API

The builder design pattern is used for data store construction. The API is defined in [builder.hpp](builder.hpp).
Here a single user facing symbol is defined -- `storage::builder`.
It is a value template parametrized by `Traits` (see below).
The idea is that the user takes a builder with the desired traits, customize it with requested properties and finally
calls `build()` method (or alternatively overloaded call operator) to produce `std::shared_ptr` to a data store.
For example:
```C++
auto ds = storage::builder<storage::gpu>
        .type<double>()
        .name("my special data")
        .dimensions(132, 132, 80)
        .halo(2, 2, 0)
        .selector<1, 1, 0>()
        .value(42)
        .build();

assert(ds->const_host_view()(1, 2, 3) == 42);

```
One can also use partially specified builder to produce several data stores:
```C++
auto const my_builder = storage::builder<storage::gpu>.dimensions(10, 10, 10);
auto foo = my_builder.type<int>().name("foo")();
auto bar = my_builder.type<tuple<int, double>>()();
auto baz = my_builder.type<double const>.initialize([](int i, int j, int k){ return i + j + k; })();
```
This API implements advanced variation of the builder design pattern. Unlike classic builder, the setters don't
return `*this` but the new instance of potentially different class is returned. Because of that the improper usage
of builder is caught in compile time:
```C++
auto bad0 = builder<cpu_ifirst>.type<double>().build() // compilation failure: dimensions should be set.
auto bad1 = builder<cpu_ifirst>.type<int>().dimensions(10).value(42).initialize([](int i) { return i;})();
// compilation failure: value and initialize setters are mutually exclusive
```

### Builder Synopsis
```C++
template </* Implementation defined parameters. */>
class buider_type {
  public:
    template <class>
    auto type() const;
    template <int>
    auto id() const;
    auto unknown_id() const;
    template <int...>
    auto layout() const;
    template <bool...>
    auto selector() const;
    auto name(std::string) const;
    auto dimensions(...) const;
    auto halos(unsigned...) const;
    template <class Fun>
    auto initializer(Fun) const;
    template <class T>
    auto value(T) const;
    auto build() const;
    auto operator()() const { return build(); }
};
template <class Traits>
constexpr builder_type</* Implementation defined parameters. */> builder = {};
```

### Constrains on Builder Setters
  
  - `type` and `dimensions` should be set before calling `build`
  - any property could be set at most once
  - `layout` and `selector` properties are mutually exclusive
  - `value` and `initializer` properties are mutually exclusive
  - the template arity of `layout`/`selector` equals `dimension` arity
  - `halos` arity equals `dimension` arity
  - `initializer` argument is callable with `int`'s, has `dimention` arity,
     and its return type is convertible to `type` argument
  - `value` argument type is convertible to `type` argument.
  - if `type` argument is `const`, `value` or `initializer` should be set

### Notes on Builder Setters Semantics

  - `id`. The use case of setting `id` is to ensure the invariant for `data_store::kind_t`. It should identify
    the unique set of dimension sizes. Note the difference -- `data_store::kind_t` represents unique `strides`
    set but `id` represents unique size set. Example:
    ```C++
    // We have two different sizes that we use in our computation.
    // Hence we prepare two partially specified builders.  
    auto const builder_a = builder<gpu>.id<0>.dimensions(3, 4, 5);
    auto const builder_b = builder<gpu>.id<1>.dimensions(5, 6, 7);
    
    // We use our builders to make some data_stores.
    auto a_0 = builder_a.type<double>().build();
    auto a_1 = builder_a.type<double>().build();
    auto a_2 = builder_a.type<float>().halo(1, 1, 0).build();
    auto b_0 = builder_a.type<double>().build();

    // kind_t aliases of a_0 and a_1 are the same.
    // kind_t aliases of a_0 and b_0 are different.
    //   Because id property is different.
    // kind_t aliases of a_0 and a_2 are different.
    //   Even though id property is the same.
    //   This is because types are different.
    ```
    At a moment `id`/`kind_t` matters if data stores are used in the context of gridtools stencil computation.
    Otherwise there is no need to set `id`. Note also that setting `id` can be skipped if only one set
    of dimension sizes is used even in gridtools stencil computation context.
  - `unknown_id`  If `unknown_id` is set for the builder, the resulting `data_store::kind_t` will be equal to
    `sid::unknown_kind`. This will opt out this data store from the optimizations that are used in the gridtools
    stencil computation. it makes sense to set `unknown_id` if the same builder is used to create the data stores with
    different dimension set and those fields are participating in the same stencil computation.
  - `dimensions`. Allows to specify the dimensions of the array. Arguments are either
    of integral type or derived from the `std::integral_constant` instantiation. Examples:
    ```C++
    using gridtools::literals;
    auto const my_builder = builder<cpu_ifirst>.type<int>();
    auto dynamic_ds = my_builder.dimensions(2, 3)();
    auto static_ds = my_builder.dimensions(2_c, 3_c)();
    auto mixed_ds = my_builder.dimensions(2, 3_c)();
    ```
    In this example all data stores act almost the same way.
    But `static_ds` (unlike `dynamic_ds`) does not hold its dimensions in runtime, it
    is encoded in the type instead. I.e. meta information (`lengths`/`strides`) takes less space
    and also indexing/iterating code can be more aggressively optimized by compiler. 
  - `halos`. The memory alignment is controlled by specifying `Traits` (the template parameter of the builder).
    By default each first element of the innermost dimension is aligned. `halos` allows to explicitly specify
    the index of element that should be aligned. Together with chosen element, all elements that share it's
    innermost index will be aligned as well.
  - `selector` allows to mask out any dimension or several. Example:
    ```C++
    auto ds = builder<cpu_ifirst>.type<int>().selector<1,0>().dimensions(10, 10).value(-1)();
    auto view = ds->host_view();
    // even though the second dimension is masked out we can used indices in the defined range;
    assert(ds->lengths()[1], 10);  
    assert(view(0, 0) == -1);
    assert(view(0, 9) == -1);
    // but elements that differs only by the masked out index refer to the same data 
    assert(&view(0, 1) == &view(0, 9));
    ```
  - `layout`. By default the data layout is controlled by `Traits`. However it is overridable with
     the `layout` setter. Example:
     ```C++
     auto ds0 = builder<gpu>
         .type<int>()
         .layout<0, 2, 4, 1, 3>()
         .dimensions(10, 10, 10, 10, 10)
         .name("my tuned ds for specific use case")
         .build(); 
     ```
 
## Traits
 
 Builder API needs a traits type to instantiate the `builder` object. In order to be used in this context
 this type should model `Storage Traits Concept`. The library comes with three predefined traits:
   - [cpu_kfirst](cpu_kfirst.hpp). Layout is chosen to benefit from data locality while doing 3D loop.
     `malloc` allocation. No alignment. `target` and `host` spaces are same. 
   - [cpu_ifirst](cpu_ifirst.hpp).  Huge page allocation. `64 bytes` alignment. Layout is tailored to utilize vectorization while
     3D looping. `target` and `host` spaces are same.
   - [gpu](gpu.hpp). Tailored for GPU. `target` and `host` spaces are different.
   
 Each traits resides in its own header. Note that the [builder.hpp](builder.hpp) doesn't include specific
 traits headers.  To use a particular trait the user should include the correspondent header.
 
 ### Defining Custom Traits
 
 To use their own traits users should provide a type that models `Storage Traits Concept`. There is no need
 to place a custom traits within `gridtools` source tree. The concept is `ADL` based. The easyest way to go
 is to copy any predefined traits and to modify it. Skipping some details the concept is defined as follows:
   - traits must specify if the `target` and `host` memory spaces are the same by providing
   `storage_is_host_referenceable` ADL based overload function.
   - traits must specify alignment in bytes by defining `storage_alignment` function.
   - `storage_allocate` function must be defined to say the library how to target memory is allocated.
   - `storage_layout` function is needed to define meta function form the number of dimensions to layout_map.
   - if `target` and `host` memory spaces are different:
        - `storage_update_target` function is needed to define how to move the data from `host` to `target`.
        - `storage_update_host` function is needed to define how to move the data from `target` to `host`.
        - `storage_make_target_view` function is needed to define a target view.
        
 ## SID Concept Adaptation
 
 [Stencil Composition Library](../stencil) doesn't use `Storage Library` directly.
 Instead [SID Concept](../sid) is used to specify the requirements on input/output fields.
 `Data store` models `SID` if [sid.hpp](sid.hpp) header is included.
 
