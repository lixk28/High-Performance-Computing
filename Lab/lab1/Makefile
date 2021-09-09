INC_DIR = ./include
SRC_DIR = ./src
OBJ_DIR = ./build
RUN_DIR = ./bin

CXX = g++
CXXFLAGS = -g -Wall -DMKL_ILP64 -m64 -I /usr/include/mkl
MKLFLAGS = -Wl,--start-group /lib/x86_64-linux-gnu/libmkl_intel_ilp64.a /lib/x86_64-linux-gnu/libmkl_sequential.a /lib/x86_64-linux-gnu/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl

INC = $(wildcard $(INC_DIR)/*.h)
SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

.PHONY: all
all: $(OBJ)
	$(CXX) $(OBJ) $(MKLFLAGS) -o $(RUN_DIR)/test

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# TODO: test cases
.PHONY: test
test:
	for size in 64 128 256 512 1024 2048 ; do \
		for time in 1 2 3 4 5 ; do \
			$(RUN_DIR)/test $$size $$size $$size ; \
		done ; \
	done

.PHONY: clean
clean:
	-rm -f $(RUN_DIR)/* $(OBJ_DIR)/*.o