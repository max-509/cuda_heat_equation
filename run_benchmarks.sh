#!/bin/bash

ETOL=${1:-1e-6}
GRID_SIZE=${2:-512}
MAX_ITER=${3:-1000000}

BENCHMARKS_TABLE='benchmarks.csv'

if [ ! -f $BENCHMARKS_TABLE ]; then
    echo "Target device;Algo ver;Grid size;Number of iters;Elapsed Time;Last error" > $BENCHMARKS_TABLE
fi

VERSIONS="openacc_gpu cuda_naive cuda_without_sync cuda_once_mem_alloc cuda_cub_one_block cuda_cub_partial_errors"

make clean
make

for version in $VERSIONS; do
    ./${version}.out $ETOL $GRID_SIZE $MAX_ITER >> $BENCHMARKS_TABLE
done
