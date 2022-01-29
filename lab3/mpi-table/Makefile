CXX 			:= mpicc
CXXFLAGS 	:= -g -Wall -std=c99 -msse3
EXEC 			:= mpirun
# CFLAGS 	+= $(shell mpicc -showme:compile)
# LDFLAGS := $(shell mpicc -showme:link)

INC_DIR 	:= ./include
SRC_DIR 	:= ./src
OBJ_DIR 	:= ./build
BIN_DIR 	:= ./bin
TEST_DIR 	:= ./test

# MPI_PATH := /lib/x86_64-linux-gnu/mpich/include

INC 	:= $(wildcard $(INC_DIR)/*.h)
SRC 	:= $(wildcard $(SRC_DIR)/*.c)
OBJ 	:= $(SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

TEST_SRC	:= $(wildcard $(TEST_DIR)/*.c)
TEST_OBJ	:= $(TEST_SRC:$(TEST_DIR)/%.c=$(OBJ_DIR)/%.o)
EXES			:= $(TEST_SRC:$(TEST_DIR)/%.c=$(BIN_DIR)/%)

.PHONY: all
all: $(EXES)

$(EXES): $(BIN_DIR)/%: $(OBJ) $(OBJ_DIR)/%.o
	$(CXX) $^ -o $@

$(OBJ): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(INC_DIR)/%.h
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -c $< -o $@

$(TEST_OBJ): $(OBJ_DIR)/%.o: $(TEST_DIR)/%.c
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -c $^ -o $@ 

.PHONY: serial
serial:
	echo General Matrix Multiplication ; \
	for size in 128 256 512 1024 2048 ; do \
		$(BIN_DIR)/gemm $$size $$size $$size ; \
	done

.PHONY: p2p
p2p:
	echo P2P MPI Matrix Multiplication ; \
	for size in 128 256 512 1024 2048 ; do \
		for proc in 1 2 4 8 16 ; do \
			$(EXEC) -n $$proc $(BIN_DIR)/mpi_p2p $$size $$size $$size ; \
		done ; \
	done

.PHONY: collect
collect:
	echo Collective MPI Matrix Multiplication ; \
	for size in 128 256 512 1024 2048 ; do \
		for proc in 1 2 4 8 16 ; do \
			$(EXEC) -n $$proc $(BIN_DIR)/mpi_collect $$size $$size $$size ; \
		done ; \
	done

.PHONY: optmm
optmm:
	echo My Optimized Matrix Multiplication ; \
	for size in 128 256 512 1024 2048 ; do \
		$(BIN_DIR)/optmm $$size $$size $$size ; \
	done 

.PHONY: clean
clean:
	-rm -f $(OBJ_DIR)/* $(BIN_DIR)/*
