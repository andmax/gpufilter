
# make <target> [sm=<XXX,...>] [cdp=<0|1>] [force32=<0|1>] [abi=<0|1>] [open64=<0|1>] [verbose=<0|1>] [keep=<0|1>]
include ../../cub/common.mk

INC += -I$(CUB_DIR) -I$(CUB_DIR)test

GPUFILTER_INCDIR = ../../gpufilter/lib
GPUFILTER_LIBDIR = ../../gpufilter/build/lib/util

ALL = simple7s simple8s simple1Gis cubiir cubps cubrf

clean :
	rm -f bin/*
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o

all : $(ALL)

run : 
	for i in $(ALL); do ./bin/$${i} --device=$(device) || exit 1; done

simple7s: bin/simple7s

bin/simple7s : simple7s.cu
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/simple7s simple7s.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

simple8s: bin/simple8s

bin/simple8s : simple8s.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/simple8s simple8s.cu \
		$(NVCCFLAGS) $(CPU_ARCH) $(INC) \
		-I$(GPUFILTER_INCDIR) -L$(GPUFILTER_LIBDIR) -lutil $(LIBS) -O3

simple1Gis: bin/simple1Gis

bin/simple1Gis : simple1Gis.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/simple1Gis simple1Gis.cu \
		$(NVCCFLAGS) $(CPU_ARCH) $(INC) \
		-I$(GPUFILTER_INCDIR) -L$(GPUFILTER_LIBDIR) -lutil $(LIBS) -O3

cubiir: bin/cubiir

bin/cubiir : cubiir.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/cubiir cubiir.cu \
		$(NVCCFLAGS) $(CPU_ARCH) $(INC) \
		-I$(GPUFILTER_INCDIR) -L$(GPUFILTER_LIBDIR) -lutil $(LIBS) -O3

cubps: bin/cubps

bin/cubps : cubps.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/cubps cubps.cu \
		$(NVCCFLAGS) $(CPU_ARCH) $(INC) \
		-I$(GPUFILTER_INCDIR) -L$(GPUFILTER_LIBDIR) -lutil $(LIBS) -O3

cubrf: bin/cubrf

bin/cubrf : cub_highLinRec_v7_int.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/cubrf cub_highLinRec_v7_int.cu \
		$(NVCCFLAGS) $(CPU_ARCH) $(INC) \
		-I$(GPUFILTER_INCDIR) -L$(GPUFILTER_LIBDIR) -lutil $(LIBS) -O3
