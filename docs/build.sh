#!/bin/sh

DEFAULT_TARGETS='html latexpdf'

REBUILD=""
SPHINX_OPTS=""

while getopts "rh" opt; do
  case $opt in
    h)
      echo "./build.sh [-r] [target_1] [target_2] [...]"
      echo ""
      echo "Options:"
      echo "	-r : rebuild from scratch"
      echo ""
      exit 0
      ;;
    r)
      REBUILD=True
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

TARGETS=${@:-'html latexpdf'}

mkdir -p source
if [ -n "$REBUILD" ]; then
	SPHINX_OPTS="-f"
	rm -Rf source/*
	rm -Rf _build/*
fi

echo "sphinx-apidoc ${SPHINX_OPTS} -o source ../src/gt4py"
sphinx-apidoc ${SPHINX_OPTS} -o source ../src/gt4py
echo "make ${TARGETS}"
make ${TARGETS}
