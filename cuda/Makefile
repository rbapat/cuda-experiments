# CUDA_ROOT_DIR=/usr/local/cuda
# CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA_LINK_LIBS= -lcudart

# NVCC=nvcc
# NVCC_FLAGS=
# NVCC_LIBS=

# $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
# 	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)



CXX = g++
CXXFLAGS = -Wall -Wextra -pedantic -std=c++17

SRCDIR = src
INCDIR = include
BINDIR = bin

SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(patsubst $(SRCDIR)/%.cpp,$(BINDIR)/%.o,$(SRCS))
INC = -I $(INCDIR)

EXE = out
TARGET = $(BINDIR)/${EXE}

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@

$(BINDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)