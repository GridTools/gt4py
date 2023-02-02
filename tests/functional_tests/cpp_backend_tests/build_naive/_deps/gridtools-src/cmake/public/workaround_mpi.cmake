# This function is a workaround for https://gitlab.kitware.com/cmake/cmake/issues/18558
# For CUDA, the mpi compile options are non-empty, but the flags are invalid with nvcc, so we need
# to pass them using -Xcompiler

function(_fix_mpi_flags)
    cmake_policy(PUSH)
    cmake_policy(SET CMP0057 NEW) # Allow "IN_LIST" inside if() statement
    get_property(_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    if("CUDA" IN_LIST _languages)
        foreach (_LANG IN ITEMS C CXX Fortran)
            if( TARGET MPI::MPI_${_LANG})
                get_property(_mpi_compile_options TARGET MPI::MPI_${_LANG} PROPERTY INTERFACE_COMPILE_OPTIONS)
                set(_new_mpi_options)
                foreach(_mpi_compile_option IN LISTS _mpi_compile_options)
                    if((${_mpi_compile_option} MATCHES ".*-Xcompiler.*")) # already properly prefixed
                        list(APPEND _new_mpi_options ${_mpi_compile_option})
                    else()
                        list(APPEND _new_mpi_options
                            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_mpi_compile_option}>
                            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${_mpi_compile_option}>)
                    endif()
                endforeach()
                set_property(TARGET MPI::MPI_${_LANG} PROPERTY INTERFACE_COMPILE_OPTIONS ${_new_mpi_options})
            endif()
        endforeach()
    endif()
    cmake_policy(POP)
endfunction()
