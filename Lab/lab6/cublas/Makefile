CC := nvcc
SRC := ./cublasmm.cu
TARGET := cublasmm

.PHONY: all
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $^ -o $@ -lcublas

.PHONY: test
test:
	./cublasmm 1024 1024 1024 32

.PHONY: clean
clean:
	rm -rf ./cublasmm

