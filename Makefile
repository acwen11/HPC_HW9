
PROGS=poisson 
all: $(PROGS)

clean:
	-rm $(PROGS)

%: %.cu
	nvcc -O3 $< -o $@
