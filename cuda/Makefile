CUDA_HOME = /usr/local/cuda

NVCC=nvcc
NVCC_FLAGS= -std=c++17 # -Wall -Wextra -pedantic -std=c++17 -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas
NVCC_LIBS= -L$(CUDA_HOME)/lib64 -lcublas -lcudnn

SRCDIR = src
INCDIR = include 
BINDIR = bin

SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(patsubst $(SRCDIR)/%.cpp,$(BINDIR)/%.o,$(SRCS))

CU_SRCS = $(wildcard $(SRCDIR)/*.cu)
CU_OBJS = $(patsubst $(SRCDIR)/%.cu,$(BINDIR)/%.o,$(CU_SRCS))

INC = -I $(INCDIR) -I /usr/local/cuda/include

EXE = out
TARGET = $(BINDIR)/${EXE}

all: $(TARGET)

$(TARGET): $(OBJS) $(CU_OBJS) 
	$(NVCC) $(NVCC_FLAGS) $(INC) $^ -o $@ $(NVCC_LIBS)	

$(BINDIR)/%.o: $(SRCDIR)/%.cpp # $(INC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@ $(NVCC_LIBS)	

$(BINDIR)/%.o: $(SRCDIR)/%.cu #$(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@ $(NVCC_LIBS)	

clean:
	rm -f $(OBJS) $(CU_OBJS) $(TARGET)