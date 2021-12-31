CC := nvcc
SRC := ./cublas_mm.cu
TARGET := cublas_mm

.PHONY: all
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $^ -o $@ -lcublas

.PHONY: test
test:
	for size in 512 1024 2048 4096 8192 ; do \
		./$(TARGET) $$size $$size $$size 32 ; \
	done

.PHONY: clean
clean:
	rm -rf ./$(TARGET)

