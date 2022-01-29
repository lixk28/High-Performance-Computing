INC_DIR = ./include
SRC_DIR = ./src
OBJ_DIR = ./build
RUN_DIR = ./bin

CXX = g++
CXXFLAGS = -g -Wall -DMKL_ILP64 -m64 -msse3 -I /usr/include/mkl
MKLFLAGS = -Wl,--start-group /lib/x86_64-linux-gnu/libmkl_intel_ilp64.a /lib/x86_64-linux-gnu/libmkl_sequential.a /lib/x86_64-linux-gnu/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl

INC = $(wildcard $(INC_DIR)/*.h)
SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

.PHONY: all
all: $(OBJ)
	$(CXX) $(OBJ) $(MKLFLAGS) -o $(RUN_DIR)/test

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: smalltest
smalltest:
	./bin/test 512 512 512

.PHONY: test
test:
	for size in 100 200 300 400 500 600 700 800 1000 1200 1500 1600 1800 2000; do \
		for time in 1 2 3 4 5 ; do \
			$(RUN_DIR)/test $$size $$size $$size ; \
		done ; \
	done

.PHONY: clean
clean:
	-rm -f $(RUN_DIR)/* $(OBJ_DIR)/*.o