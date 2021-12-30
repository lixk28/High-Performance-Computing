CXX := nvcc
SRC := ./cudnn_conv2d.cu
TARGET := cudnn_conv2d

CUDNN_INC_PATH := /opt/conda/include
CUDNN_LIB_PATH := /opt/conda/lib
CUDNN_LD_FLAG := -lcudnn

.PHONY: all
all: $(TARGET)

$(TARGET): $(SRC)
	export LD_LIBRARY_PATH=$(CUDNN_LIB_PATH) && $(CXX) -I$(CUDNN_INC_PATH) -L$(CUDNN_LIB_PATH) -o $@ $^ $(CUDNN_LD_FLAG)

.PHONY: test
test:
	for size in 256 512 1024 2048 4096 ; do \
		for stride in 1 2 3 ; do \
			./$(TARGET) $$size $$size $$stride 32 ; \
		done ; \
	done

.PHONY: clean
clean:
	rm -rf ./$(TARGET)

