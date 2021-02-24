from devtools import debug  # noqa: F401

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from gtc_unstructured.irs.gtir import Computation, SparseField, UField


class IconBindingsCodegen(codegen.TemplatedGenerator):
    @classmethod
    def apply(cls, root, stencil_code: str, **kwargs) -> str:
        generated_code = cls().visit(root, stencil_code=stencil_code)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code

    def visit_UField(self, node: UField, **kwargs):
        if node.name in kwargs["dimensionality"]:
            return self.generic_visit(node, **kwargs)
        else:
            return ""

    UField = as_fmt(
        "gridtools::fortran_array_view<T, {len(dimensionality[name])}, field_kind<{','.join(str(i) for i in dimensionality[name])}>> {name}"
    )

    def visit_SparseField(self, node: SparseField, **kwargs):
        if node.name in kwargs["dimensionality"]:
            return self.generic_visit(node, **kwargs)
        else:
            return ""

    SparseField = as_fmt(
        "gridtools::fortran_array_view<T,{len(dimensionality[name])}, field_kind<{','.join(str(i) for i in dimensionality[name])}>> {name}"
    )

    Connectivity = as_fmt("neigh_tbl_t {name}")

    def visit_Computation(self, node: Computation, **kwargs):
        dimensionality = {}
        for p in node.params:
            dimensionality[p.name] = [0, 1, 2]
            if not p.dimensions.horizontal:
                dimensionality[p.name].remove(0)
            if not p.dimensions.vertical:
                dimensionality[p.name].remove(1)
            if not isinstance(p, SparseField):
                dimensionality[p.name].remove(2)

        param_names = []
        for name, dims in dimensionality.items():
            renames = {}
            for index in range(0, 3):
                if index < len(dims):
                    if index != dims[index]:
                        renames[index] = dims[index]

            if renames:
                param_names.append(
                    "gridtools::sid::rename_dimensions<"
                    + ",".join(
                        f"gridtools::integral_constant<int,{old}>, gridtools::integral_constant<int,{new}>>"
                        for old, new in renames.items()
                    )
                    + f"({name})"
                )
            else:
                param_names.append(name)
        return self.generic_visit(
            node, param_names=param_names, dimensionality=dimensionality, **kwargs
        )

    Computation = as_mako(
        """
    # include <cpp_bindgen/export.hpp>
    # include <gridtools/storage/adapter/fortran_array_view.hpp>
    # include <gridtools/storage/sid.hpp>
    # include <gridtools/usid/icon.hpp>

    ${stencil_code}

    namespace icon_bindings_${name}_impl{

    struct default_tag{};

    template<int...>
    struct field_kind{};

    // template<class Tag>
    using neigh_tbl_t = gridtools::fortran_array_view<int, 2, default_tag, false>;

    auto alloc_${name}_impl(${','.join(['int n_edges', 'int n_k'] + connectivities)}) {
        return ${name}({-1, n_edges, -1, n_k}, ${','.join(f"icon::make_connectivity_producer<{c.max_neighbors}>({c.name})" for c in _this_node.connectivities)});
    }

    BINDGEN_EXPORT_BINDING_WRAPPED(${2+len(connectivities)}, alloc_${name}, alloc_${name}_impl);

    // template<class Tag>
    using ${name}_t = decltype(alloc_${name}_impl(0,0, neigh_tbl_t/*<Tag>*/{{}}));

    template <class T>
    void ${name}_impl(${name}_t ${name}, ${','.join(params)}){
        ${name}(${','.join(param_names)});
    }

    BINDGEN_EXPORT_GENERIC_BINDING_WRAPPED(${1+len(params)}, ${name}, ${name}_impl,
                                            (double));
    }
    """
    )
