CXX :=g++
CC :=gcc

SFLAG := -fPIC -shared

IMAGE_CLASSIFIER_INC = -I./include
IMAGE_CLASSIFIER_SRC = ./src/image_classifier.cpp

CAFFE_INC = -I/home/xiaotian/caffe/include
CAFFE_LIB = -L/home/xiaotian/caffe/build/lib

INCLUDE = -I/usr/local/include -I/usr/local/cuda/include -I/usr/local/atlas/include
INCLUDE += $(IMAGE_CLASSIFIER_INC) $(CAFFE_INC)

LIBRARY = -L/usr/local/lib -L/usr/lib -L/usr/local/cuda/lib64 -L/usr/local/atlas/lib
LIBRARY += $(CAFFE_LIB)

CXXFLAGS = $(SFLAG) -Wall -w -O3 -msse2 -std=c++11
CXXFLAGS += $(INCLUDE)
CFLAGS = $(CXXFLAGS) -std=c99 -pedantic -c -O3 -fPIC -Wall -W -Wstrict-prototypes -Wwrite-strings -g -ggdb

LDFLAGS = -lpthread -lglog
LDFLAGS += -lcaffe -lcblas
LDFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc


SRCS_CPP = $(IMAGE_CLASSIFIER_SRC)
OBJS_CPP = $(SRCS_CPP:%.cpp=%.o)
OBJECTS := $(OBJS_CPP)

LIB = ./lib

.c.o:
	@echo ----------------------------------------------
	$(CC) $(CFLAGS) -c $< -o $@
	@echo ----------------------------------------------

.cpp.o:
	@echo ----------------------------------------------
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo ----------------------------------------------

all:lib

lib: $(OBJECTS)
	mkdir -p $(LIB)
	@echo ----------------------------------------------
	$(CXX) $(CXXFLAGS) -o $(LIB)/libimageclassifier.so $(OBJECTS) $(LIBRARY) $(LDFLAGS)
	@echo ----------------------------------------------

clean:
	rm -r $(LIB)
	rm -f $(OBJECTS)
