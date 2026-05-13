# ── Makefile for CS259 Project 2: Conv with parametric TILE_NI ────────────────
#
# Usage:
#   make            – build ./conv with default TILE_NI
#   make tile_ni    – build multiple binaries (conv_tni_2, conv_tni_4, ..., conv_tni_32)
#   make clean      – remove binaries
#
# For tile size sweeps, this Makefile creates binaries with different SMWI_TILE_NI
# values to measure performance impact of tile size on K3 (smem_wi).
# ---------------------------------------------------------------------------

NVCC    := nvcc
TARGET  := conv
SRC     := conv.cu
HEADERS := conv_ref.h

ARCH     := sm_70
CFLAGS   := -O3 -arch=$(ARCH) -std=c++14 \
             -Xcompiler -O3 \
             -Xcompiler -fopenmp \
             -lineinfo

LDFLAGS  := -lm

# ── Targets ─────────────────────────────────────────────────────────────────

.PHONY: all clean info tile_ni

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(NVCC) $(CFLAGS) $(LDFLAGS) -o $@ $<
	@echo "Built $@ (arch=$(ARCH), default TILE_NI)"

# Build binaries for tile size sweep
TILE_NI_VALUES := 2 4 8 16 32
tile_ni: $(addprefix conv_tni_,$(TILE_NI_VALUES))

$(addprefix conv_tni_,$(TILE_NI_VALUES)): conv_tni_%: $(SRC) $(HEADERS)
	$(NVCC) $(CFLAGS) -DSMWI_TILE_NI=$* $(LDFLAGS) -o $@ $<
	@echo "Built $@ (TILE_NI=$*)"

clean:
	rm -f $(TARGET) $(addprefix conv_tni_,$(TILE_NI_VALUES))

info:
	@echo "NVCC    : $$($(NVCC) --version | head -1)"
	@echo "ARCH    : $(ARCH)"
	@echo "CFLAGS  : $(CFLAGS)"
	@echo "TILE_NI_VALUES: $(TILE_NI_VALUES)"
