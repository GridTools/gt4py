#!/bin/bash

set -e
set -x

# Refactor 'gt4py' imports according to the following rules:
#
#     - "from gt4py..." -> "from gt4py.cartesian..."
#     - "import gt4py" is translated in two steps:
#         + "import gt4py" -> "from gt4py import cartesian as gt4pyc"
#         + "gt4py.SOME_NAME" -> "gt4pyc.SOME_NAME"
#     - "import gt4py.SOME_NAME as ALIAS_NAME" -> "import gt4py.cartesian.SOME_NAME as ALIAS_NAME"
#     - "import gt4py.SOME_NAME" is translated in two steps:
#         + "import gt4py.SOME_NAME" -> "from gt4py import cartesian as gt4pyc"
#         + "gt4py.SOME_NAME" -> "gt4pyc.SOME_NAME"
#
# Refactor 'gtc' imports according to the following rules:
#
#     - "from gtc..." -> "from gt4py.cartesian.gtc..."
#     - "import gtc" -> "from gt4py.cartesian import gtc"
#     - "import gtc.SOME_NAME" is translated in two steps:
#         "import gtc.SOME_NAME" -> "import gt4py.cartesian.gtc.SOME_NAME"
#         "gtc.SOME_NAME" -> "gt4py.cartesian.gtc.SOME_NAME"
#
# Refactor 'eve' imports according to the following rules:
#
#     - "from eve..." -> "from gt4py.eve..."
#     - "import eve" -> "from gt4py import eve"
#     - "import eve.SOME_NAME" is translated in two steps:
#         "import eve.SOME_NAME" -> "import gt4py.eve.SOME_NAME"
#         "eve.SOME_NAME" -> "gt4py.eve.SOME_NAME"


git grep -l 'from gt4py' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/from gt4py/from gt4py.cartesian/' 
git grep -l 'import gt4py' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/^\([ ]*\)import gt4py$/\1from gt4py import cartesian as gt4pyc\n\1import gt4py/g'
git grep -l 'import gt4py as' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/^\([ ]*\)import gt4py as /\1from gt4py import cartesian as /'
git grep -l 'import gt4py.' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/^\([ ]*\)import gt4py\.\([\._A-Za-z0-9]*\) as /\1import gt4py.cartesian.\2 as /g'
git grep -l 'import gt4py.' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/^\([ ]*\)import gt4py\.\([\._A-Za-z0-9]*\)$/\1from gt4py import cartesian as gt4pyc/'
#git grep -l 'import gt4py.' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/^\([ ]*\)import gt4py\.\([\._A-Za-z0-9]*\)$/\1from gt4py import cartesian as gt4pyc  # [ORIGINAL IMPORT]: import gt4py.\2\nimport gt4py/'
#git grep -l 'as gt4pyc' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/gt4py\.\([^\"]\)/gt4pyc.\1/' -e 's/^\([ ]*\)from gt4pyc\.cartesian/\1from gt4py\.cartesian/'  -e 's/\[ORIGINAL IMPORT\]: import gt4pyc\./[ORIGINAL IMPORT]: import gt4py./'
git grep -l 'as gt4pyc' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/gt4py\.\([^\"]\)/gt4pyc.\1/g'
git grep -l 'gt4pyc.cartesian' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/gt4pyc\.cartesian/gt4py.cartesian/'

git grep -l 'import storage' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/from gt4py\.cartesian import storage/from gt4py import storage/g' 
git grep -l 'cartesian.storage' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/cartesian\.storage/storage/g'
git grep -l 'gt4pyc.storage' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/gt4pyc\.storage/gt4py.storage/g'

git grep -l 'from gtc' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/^\([ ]*\)from gtc/\1from gt4py.cartesian.gtc/' 
git grep -l 'import gtc.' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/^\([ ]*\)import gtc\./\1import gt4py.cartesian.gtc./'
git grep -l 'import gtc' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/^\([ ]*\)import gtc/\1from gt4py.cartesian import gtc/'
git grep -l 'import gt4py.cartesian.gtc.' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/\([^.A-Za-z0-9]\)gtc\./\1gt4py.cartesian.gtc./'

git grep -l 'from eve' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/^\([ ]*\)from eve/\1from gt4py.eve/' 
git grep -l 'import eve.' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/^\([ ]*\)import eve\./\1import gt4py.eve./'
git grep -l 'import eve' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/^\([ ]*\)import eve/\1from gt4py import eve/'
git grep -l 'import gt4py.eve.' | grep -v _local_refactor_imports.sh | xargs sed -i -e 's/\([^.A-Za-z0-9]\)eve\./\1gt4py.eve./'

git status
