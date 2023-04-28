#!bin/bash
for k in poisson_outer poisson_inner poisson_collapse
do
for m in 1 2 4 6 12 16
do
echo -n $k $m >> output_parallel
export OMP_NUM_THREADS=$m
./$k 256 >> output_parallel
done
done
