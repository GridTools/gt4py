#!/bin/bash

# no_dependency "<of_module>" "<on_module>"
# checks that <of_module> does not depend on <on_module>
# i.e. <of_module> does not include any file from <on_module>
function no_dependency() {
    last_result=`grep -r "#include .*$2/.*hpp" include/gridtools/$1 | wc -l`
    if [ "$last_result" -gt 0 ]; then
        echo "ERROR Modularization violated: found dependency of $1 on $2"
        echo "`grep -r "#include .*$2/.*hpp" include/gridtools/$1`"
    fi
    modularization_result=$(( modularization_result || last_result ))
}

function are_independent() {
    no_dependency "$1" "$2"
    no_dependency "$2" "$1"
}
modularization_result=0

no_dependency "meta" "common"
no_dependency "meta" "sid"
no_dependency "meta" "thread_pool"
no_dependency "meta" "gcl"
no_dependency "meta" "layout_transformation"
no_dependency "meta" "storage"
no_dependency "meta" "boundaries"
no_dependency "meta" "stencil"

no_dependency "common" "sid"
no_dependency "common" "thread_pool"
no_dependency "common" "gcl"
no_dependency "common" "layout_transformation"
no_dependency "common" "storage"
no_dependency "common" "boundaries"
no_dependency "common" "stencil"

are_independent "sid" "thread_pool"
are_independent "sid" "gcl"
are_independent "sid" "layout_transformation"
no_dependency "sid" "storage"
are_independent "sid" "boundaries"
no_dependency "sid" "stencil"

are_independent "thread_pool" "gcl"
are_independent "thread_pool" "layout_transformation"
are_independent "thread_pool" "storage"
are_independent "thread_pool" "boundaries"
no_dependency "thread_pool" "stencil"

are_independent "gcl" "layout_transformation"
are_independent "gcl" "storage"
no_dependency "gcl" "boundaries"
are_independent "gcl" "stencil"

are_independent "layout_transformation" "storage"
are_independent "layout_transformation" "boundaries"
are_independent "layout_transformation" "stencil"

no_dependency "storage" "boundaries"
are_independent "storage" "stencil"

are_independent "boundaries" "stencil"

# we cannot use an exit code here because the git hook will terminate immediately
