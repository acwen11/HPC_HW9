
PROGS=add_managed add_unmanaged prop wave vector_sum wave_managed poisson poisson_sync poisson_cooperative
all: $(PROGS)

clean:
	-rm $(PROGS)

%: %.cu
	nvcc -O3 $< -o $@
