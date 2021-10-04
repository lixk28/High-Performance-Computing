CXX 			:= mpicc
CXXFLAGS 	:= -g -Wall -std=c99
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
	$(CXX) -I$(INC_DIR) -c $< -o $@

$(TEST_OBJ): $(OBJ_DIR)/%.o: $(TEST_DIR)/%.c
	$(CXX) -I$(INC_DIR) -c $^ -o $@ 

.PHONY: run
run:
	@echo General Matrix Multiplication
	@echo -------------------------------
	@$(BIN_DIR)/gemm
	@echo 
	@echo Matrix Multiplication MPI P2P Communication
	@echo ---------------------------------------------
	@$(EXEC) -n 4 $(BIN_DIR)/gemm_mpi_p2p
	@echo 
	@echo Matrix Multiplication MPI Set Communication
	@echo ---------------------------------------------
	@$(EXEC) -n 4 $(BIN_DIR)/gemm_mpi_set
	@echo 

# TODO: benchmark and performance analysis
.PHONY: test
test:

.PHONY: plot
plot:
	python3 ./script/plot.py

.PHONY: clean
clean:
	-rm -f $(OBJ_DIR)/* $(BIN_DIR)/*
