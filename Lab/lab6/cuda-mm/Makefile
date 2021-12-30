CC := nvcc
SRC := ./cuda_mm.cu
TARGET := cuda_mm

.PHONY: all
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $^ -o $@

.PHONY: test
test:
	for size in 512 1024 2048 4096 8192 ; do \
		for block_size in 32 64 128 256 512 ; do \
			./$(TARGET) $$size $$size $$size $$block_size ; \
		done ; \
	done

.PHONY: clean
clean:
	rm -rf ./$(TARGET)
