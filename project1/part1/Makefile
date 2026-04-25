# ── Makefile for CS259 Part 1: Convolution ────────────────────────────────
#
# Usage:
#   make            – build ./conv
#   make clean      – remove binary
#   make info       – print build flags
#
# Architecture: sm_70 = V100 (CC 7.0).  Change to sm_80 for A100, etc.
# ---------------------------------------------------------------------------

NVCC    := nvcc
TARGET  := conv
SRC     := conv.cu
HEADERS := conv_ref.h

# sm_70  = V100
# -std=c++14       : needed for generic lambdas in time_kernel_ms
# -Xcompiler ...   : flags passed through to the host (gcc) compiler
# -lineinfo        : keep source-line info for ncu without full debug overhead
ARCH     := sm_70
CFLAGS   := -O3 -arch=$(ARCH) -std=c++14 \
             -Xcompiler -O3 \
             -Xcompiler -fopenmp \
             -lineinfo

LDFLAGS  := -lm

# ── Targets ─────────────────────────────────────────────────────────────────

.PHONY: all clean info

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(NVCC) $(CFLAGS) $(LDFLAGS) -o $@ $<
	@echo "Built $@ (arch=$(ARCH))"

clean:
	rm -f $(TARGET)

info:
	@echo "NVCC    : $$($(NVCC) --version | head -1)"
	@echo "ARCH    : $(ARCH)"
	@echo "CFLAGS  : $(CFLAGS)"
