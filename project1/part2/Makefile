NVCC     := nvcc
CXXFLAGS := -O3 -std=c++17 -arch=sm_70
NVFLAGS  := -O3 -std=c++17 -arch=sm_70 \
            -Xptxas -v \
            -lineinfo \
            --generate-line-info

TARGET   := attention
SRC      := attention.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) attention_ref.h timing.h
	$(NVCC) $(NVFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
	rm -f results/*.ncu-rep results/*.txt results/*.png results/summary.txt

results:
	mkdir -p results
