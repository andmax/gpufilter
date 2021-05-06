
SM = sm_61

BIN = plr1 plr2 plr3 plr4 plr5 plr1_fwd_rev plrps

ALL = $(foreach obj,$(BIN),bin/$(obj))

all: $(ALL)

bin/:
	mkdir bin/

bin/%: %.cu | bin/
	nvcc -O3 -arch=$(SM) $< -o $@

clean:
	rm -f $(ALL)
