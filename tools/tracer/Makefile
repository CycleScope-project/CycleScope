NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS=ptxas

# CYCLESCOPE_NVBIT_DIR points at the NVBit release root (core/ holds nvbit.h
# and libnvbit.a); NVBit binaries are not vendored into the repo.
NVBIT_ROOT?=$(CYCLESCOPE_NVBIT_DIR)
NVBIT_PATH?=$(NVBIT_ROOT)/core
NVBIT_LIB_DIR?=$(NVBIT_ROOT)/core
INCLUDES=-I$(NVBIT_PATH)

LIBS=-L$(NVBIT_LIB_DIR) -lnvbit
CUDA_DRIVER_LIB?=-lcuda
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

SOURCES=tracer.cu inject.cu
HEADERS=record_format.h arch_table.h arch_hopper.h arch_blackwell.h \
	tensor_map_operand.h uniform_register.h

OBJDIR?=$(CYCLESCOPE_ARTIFACT_ROOT)/build/tracer/$(ARCH)
OBJECTS=$(patsubst %.cu,$(OBJDIR)/%.o,$(SOURCES))
ARCH?=sm_103a
CYCLESCOPE_NVBIT_HAS_SASS_BINARY?=1
# Extra -D flags for diagnostic builds (e.g. -DCYCLESCOPE_TRACER_HOST_CHANNEL to
# put the NVBit channel buffer back in host memory, -DCHANNEL_SIZE=...);
# recorded in build-manifest.json by build.sh.
EXTRA_DEFINES?=

# The injected device functions need the arch-specific tool patch;
# CYCLESCOPE_BUILD_HOPPER/CYCLESCOPE_BUILD_BLACKWELL mark the build's default
# arch table (the runtime CC probe still picks the real table).
ifeq ($(ARCH),sm_103a)
ARCH_DEFINE=-DCYCLESCOPE_BUILD_BLACKWELL
else
ARCH_DEFINE=-DCYCLESCOPE_BUILD_HOPPER
endif

NVBIT_TOOL=$(OBJDIR)/tracer.so

all: $(NVBIT_TOOL)

tracer.so: $(NVBIT_TOOL)

# Keep the driver dependency explicit when linking with the CUDA toolkit
# stub: only the real libcuda.so.1 is resolved by the loader at runtime.
$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_LIB_DIR)/libnvbit.a Makefile
	@mkdir -p "$(OBJDIR)"
	$(NVCC) -arch=$(ARCH) -O3 $(OBJECTS) $(LIBS) $(NVCC_PATH) \
	  -Xlinker --no-as-needed -Xlinker $(CUDA_DRIVER_LIB) -Xlinker --as-needed \
	  -lcudart_static -shared -o $@

$(OBJDIR)/%.o: %.cu $(HEADERS)
	@mkdir -p "$(OBJDIR)"
	$(NVCC) -dc -c -std=c++17 $(INCLUDES) $(ARCH_DEFINE) $(EXTRA_DEFINES) \
	  -DCYCLESCOPE_NVBIT_HAS_SASS_BINARY=$(CYCLESCOPE_NVBIT_HAS_SASS_BINARY) \
	  -Xptxas -cloning=no -Xcompiler -Wall -arch=$(ARCH) -O3 \
	  -Xcompiler -fPIC $< -o $@

$(OBJDIR)/inject.o: inject.cu $(HEADERS)
	@mkdir -p "$(OBJDIR)"
	$(NVCC) $(INCLUDES) $(ARCH_DEFINE) $(EXTRA_DEFINES) \
	  -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH) \
	  -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f $(OBJECTS) $(NVBIT_TOOL)
