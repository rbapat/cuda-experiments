NVCC=nvcc
NVCC_FLAGS= -std=c++17 # -Wall -Wextra -pedantic -std=c++17 -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas
NVCC_LIBS=

CUDA_ROOT_DIR=/usr/local/cuda
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS= -lcudart

SRCDIR = src
INCDIR = include
BINDIR = bin

SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(patsubst $(SRCDIR)/%.cpp,$(BINDIR)/%.o,$(SRCS))

CU_SRCS = $(wildcard $(SRCDIR)/*.cu)
CU_OBJS = $(patsubst $(SRCDIR)/%.cu,$(BINDIR)/%.o,$(CU_SRCS))

INC = -I $(INCDIR)

EXE = out
TARGET = $(BINDIR)/${EXE}

all: $(TARGET)

$(TARGET): $(OBJS) $(CU_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INC) $^ -o $@

$(BINDIR)/%.o: $(SRCDIR)/%.cpp
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

$(BINDIR)/%.o: $(SRCDIR)/%.cu # $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@ $(NVCC_LIBS)	

clean:
	rm -f $(OBJS) $(CU_OBJS) $(TARGET)