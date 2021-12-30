CC := nvcc
SRC := ./im2col_conv2d.cu
TARGET := im2col_conv2d

.PHONY: all
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $^ -o $@

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

