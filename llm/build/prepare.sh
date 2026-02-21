
#!/bin/bash

set -e
set -x

pre_env()
{
    CUR_PATH=`pwd`
    ROOT_PATH=${CUR_PATH}/..
    KERNEL_PATH=${ROOT_PATH}/src/kernels/cuda
    OUTPUT_PATH=${ROOT_PATH}/output
}



main()
{
    pre_env
    rm -rf ${OUTPUT_PATH}/

    cd ${KERNEL_PATH}

    python setup.py build_ext  --build-lib $OUTPUT_PATH

}

main