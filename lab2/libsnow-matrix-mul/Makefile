CXX 			:= g++
CXXFLAGS 	:= -g -Wall -fPIC -msse3
LDFLAGS 	:= -shared

INC_DIR		:= ./include
SRC_DIR		:= ./src
OBJ_DIR		:= ./build
BIN_DIR		:= ./bin
LIB_DIR		:= ./lib

INC 			:= $(wildcard $(INC_DIR)/*.h)
SRC 			:= $(wildcard $(SRC_DIR)/*.cpp)
OBJ 			:= $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

TARGET 		:= $(LIB_DIR)/libsnow-matrix-mul.so

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(LDFLAGS) $^ -o $@

$(OBJ): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) -I$(INC_DIR) $(CXXFLAGS) $^ -c -o $@

.PHONY: test
test: test.cpp
	$(CXX) test.cpp -o $(BIN_DIR)/test -I$(INC_DIR) -L$(shell pwd)/lib -Wl,-rpath=$(shell pwd)/lib -lsnow-matrix-mul

.PHONY: run
run:
	$(BIN_DIR)/test

.PHONY: clean
clean:
	-rm -f $(OBJ_DIR)/*.o $(LIB_DIR)/*.so $(BIN_DIR)/*
