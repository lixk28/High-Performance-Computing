CC := gcc
CFLAGS := -g -Wall

AST_DIR := ./asset
BIN_DIR := ./bin

INC := $(wildcard *.h)
SRC := $(wildcard *.c)

TARGET := $(BIN_DIR)/fft_serial $(BIN_DIR)/fft_pf $(BIN_DIR)/fft_openmp

.PHONY: all
all: $(TARGET)

$(BIN_DIR)/fft_serial: ./fft-serial/fft_serial.c
	$(CC) $(CFLAGS) ./fft-serial/fft_serial.c -o $@ -lm

$(BIN_DIR)/fft_pf: ./fft-pf/fft_pf.c ./fft-pf/parallel_for.c ./fft-pf/parallel_for.h
	$(CC) $(CFLAGS) ./fft-pf/fft_pf.c ./fft-pf/parallel_for.c -o $@ -lm -lpthread -fopenmp

$(BIN_DIR)/fft_openmp: ./fft-openmp/fft_openmp.c
	$(CC) $(CFLAGS) ./fft-openmp/fft_openmp.c -o $@ -lm -fopenmp

.PHONY: test
test:
	$(BIN_DIR)/fft_serial > $(AST_DIR)/fft_serial_test.txt
	$(BIN_DIR)/fft_pf $(shell grep -c ^processor /proc/cpuinfo) > $(AST_DIR)/fft_pf_test.txt
	$(BIN_DIR)/fft_openmp > $(AST_DIR)/fft_openmp_test.txt

.PHONY: memory-test
memory-test:
	valgrind --tool=massif --time-unit=B --stacks=yes --massif-out-file=$(AST_DIR)/massif.out.pf $(BIN_DIR)/fft_pf $(shell grep -c ^processor /proc/cpuinfo)
	valgrind --tool=massif --time-unit=B --stacks=yes --massif-out-file=$(AST_DIR)/massif.out.omp $(BIN_DIR)/fft_openmp

.PHONY: print
print:
	ms_print ./asset/massif.out.pf
	ms_print ./asset/massif.out.omp

.PHONY: clean
clean:
	-rm -f $(BIN_DIR)/*