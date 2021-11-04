CXX 			:= g++
CXXFLAGS 	:= -g -Wall -fopenmp

AST_DIR 	:= ./asset
SPT_DIR		:= ./script
INC_DIR 	:= ./include
SRC_DIR 	:= ./src
OBJ_DIR 	:= ./build
BIN_DIR 	:= ./bin

INC 			:= $(wildcard $(INC_DIR)/*.h)
SRC 			:= $(wildcard $(SRC_DIR)/*.cpp)
OBJ 			:= $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

TARGET 		:= $(BIN_DIR)/test

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(OBJ): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $^ -c -o $@

.PHONY: test
test:
	for thread in 1 2 3 4 5 6 7 8 ; do \
		for size in 512 1024 1536 2048 ; do \
			echo thread = $$thread, size = $$size: ; \
			$(BIN_DIR)/test $$size $$size $$size $$thread ; \
			echo ; \
		done ; \
	done

.PHONY: plot
plot:
	python3 $(SPT_DIR)/plot.py

.PHONY: clean
clean:
	-rm -f $(BIN_DIR)/* $(OBJ_DIR)/*.o
	-rm -f $(AST_DIR)/*