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
		./$(TARGET) $$size $$size $$size 32 ; \
	done

.PHONY: clean
clean:
	rm -rf ./$(TARGET)
