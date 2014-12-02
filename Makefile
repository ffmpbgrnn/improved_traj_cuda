# set the binaries that have to be built
TARGETS := DenseTrackStab Video FarnebackOpticalFlow

# set the build configuration set 
BUILD := release
#BUILD := debug

# set bin and build dirs
BUILDDIR := .build_$(BUILD)
BINDIR := $(BUILD)

# libraries 
LDLIBS = $(addprefix -l, $(LIBS) $(LIBS_$(notdir $*)))
LIBS := \
	opencv_core opencv_highgui opencv_video opencv_imgproc opencv_calib3d opencv_features2d opencv_nonfree \
	avformat avdevice avutil avcodec swscale

# set some flags and compiler/linker specific commands
CXXFLAGS = -pipe -D __STDC_CONSTANT_MACROS -D STD=std -Wall $(CXXFLAGS_$(BUILD)) -I. -I/opt/cuda/include -I/usr/local/include -I/usr/local/cuda-6.5/include
CXXFLAGS_debug := -ggdb
CXXFLAGS_release := -O3 -DNDEBUG -ggdb
LDFLAGS = -L/opt/lib -L/usr/local/lib -L/opt/cuda/lib64 -L/usr/local/cuda-6.5/lib64 -lcuda -lcudart -lopencv_gpu -lpthread -pipe -Wall $(LDFLAGS_$(BUILD))
LDFLAGS_debug := -ggdb
LDFLAGS_release := -O3 -ggdb

# include make/generic.mk

CC := g++
NVCC := nvcc
EXEC = DenseTrackStab
CXXSOURCES = $(wildcard *.cpp)
CUSOURCES = $(wildcard *.cu)

CXXOBJECTS = $(CXXSOURCES:.cpp=.o)
CUOBJECTS = $(CUSOURCES:.cu=.o)

$(EXEC): $(CXXOBJECTS) $(CUOBJECTS)
	$(CC) $(CXXOBJECTS) $(CUOBJECTS) -o $(EXEC) $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CC) -c $(CXXFLAGS) $(CXXFLAGS_debug) $< -o $@

%.o: %.cu
	$(NVCC) -c $< -o $@

clean:
	rm -rf $(CXXOBJECTS)
