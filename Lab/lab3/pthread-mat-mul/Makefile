CXX 			:= g++
CXXFLAGS 	:= -g -Wall
LDFLAGS 	:= -lpthread

INC_DIR := ./include
SRC_DIR := ./src
OBJ_DIR := ./build
BIN_DIR := ./bin

INC := $(wildcard $(INC_DIR)/*.h)
SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

TARGET := $(BIN_DIR)/test

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(OBJ): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@ 

.PHONY: test
test:
	chmod +x ./script/test.sh && ./script/test.sh

.PHONY: plot
plot:
	python3 ./script/plot.py

.PHONY: clean
clean:
	-rm -f $(OBJ_DIR)/*.o $(BIN_DIR)/*
