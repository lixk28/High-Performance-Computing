CXX	:= g++
CFLAGS := -g -Wall
LDFLAGS := -lpthread -lm

AST_DIR := ./asset
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
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

.PHONY: test
test:
	$(BIN_DIR)/test > $(AST_DIR)/fft_pf_test.txt

.PHONY: clean
clean:
	-rm -f $(BIN_DIR)/* $(OBJ_DIR)/*.o