CXX 			:= g++
CXXFLAGS 	:= -g -Wall -fPIC -lpthread
LDFLAGS		:= -shared

INC_DIR 	:= ./include
SRC_DIR 	:= ./src
OBJ_DIR 	:= ./build
LIB_DIR		:= ./lib

INC 			:= $(wildcard $(INC_DIR)/*.h)
SRC 			:= $(wildcard $(SRC_DIR)/*.cpp)
OBJ 			:= $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

TARGET 		:= $(LIB_DIR)/libparallel-for.so

.PHONY: lib
lib: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(LDFLAGS) $^ -o $@

$(OBJ): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $^ -c -o $@

.PHONY: test
test:
	$(CXX) -L$(shell pwd)/lib -Wl,-rpath=$(shell pwd)/lib test.cpp -o test -lparallel-for -lpthread

.PHONY: run
run:
	./test 1024 1024 1024 8

.PHONY: clean
clean:
	-rm -f ./test $(OBJ_DIR)/*.o $(LIB_DIR)/*.so