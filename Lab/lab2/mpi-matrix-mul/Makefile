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
	@echo Matrix Multiplication MPI P2P Communication
	@echo proc = 4, scale = 512
	@echo ---------------------------------------------
	@$(EXEC) -n 4 $(BIN_DIR)/mpi_p2p 512 512 512
	@echo 
	@echo Matrix Multiplication MPI Set Communication
	@echo proc = 4, scale = 512
	@echo ---------------------------------------------
	@$(EXEC) -n 4 $(BIN_DIR)/mpi_collect 512 512 512
	@echo 

.PHONY: test
test:
	for size in 512 768 1024 1280 1536 1792 2048; do \
		for proc in 1 2 4 ; do \
			for time in 1 2 3 ; do \
				echo Matrix Multiplication MPI P2P Communication ; \
				echo proc = $$proc, size = $$size ; \
				echo --------------------------------------------- ; \
				$(EXEC) -n $$proc $(BIN_DIR)/mpi_p2p $$size $$size $$size ; \
				echo ; \
				echo Matrix Multiplication MPI Set Communication ; \
				echo proc = $$proc, size = $$size ; \
				echo --------------------------------------------- ; \
				$(EXEC) -n $$proc $(BIN_DIR)/mpi_collect $$size $$size $$size ; \
				echo ; \
			done ; \
		done ; \
	done

.PHONY: plot
plot:
	python3 ./script/plot.py

.PHONY: clean
clean:
	-rm -f $(OBJ_DIR)/* $(BIN_DIR)/*
