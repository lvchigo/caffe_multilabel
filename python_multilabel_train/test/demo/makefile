CXX :=g++
CC :=gcc


CV_INC := -I/usr/local/include/opencv

CAFFE_INC = -I/home/xiaotian/caffe/include
CAFFE_LIB = -L/home/xiaotian/caffe/build/lib


INCLUDE = -I/usr/local/include -I/usr/local/cuda/include -I/usr/local/atlas/include
INCLUDE += $(CV_INC) $(CAFFE_INC)
INCLUDE += -I../include

LIBRARY = -L/usr/local/lib -L/usr/lib -L/usr/local/cuda/lib64 -L/usr/local/atlas/lib
LIBRARY += $(CAFFE_LIB)
LIBRARY += -L../lib


CXXFLAGS = -w -O3 -std=c++11
CXXFLAGS += $(INCLUDE)
CFLAGS = $(CXXFLAGS) -std=c99 -pedantic -c -O3 -fPIC -Wall -W -Wstrict-prototypes -Wwrite-strings -g -ggdb

LDFLAGS = -lpthread -lglog
LDFLAGS += -lcaffe -lcblas
LDFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc
LDFLAGS += -limageclassifier


.c.o:
	@echo ----------------------------------------------
	$(CC) $(CFLAGS) -c $< -o $@
	@echo ----------------------------------------------

.cpp.o:
	@echo ----------------------------------------------
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo ----------------------------------------------


all:demo demo_mean

demo: demo.cpp
	@echo ----------------------------------------------
	$(CXX) $(CXXFLAGS) demo.cpp -o demo $(LIBRARY) $(LDFLAGS)
	@echo ----------------------------------------------

demo_mean: demo_mean.cpp
	@echo ----------------------------------------------
	$(CXX) $(CXXFLAGS) demo_mean.cpp -o demo_mean $(LIBRARY) $(LDFLAGS)
	@echo ----------------------------------------------
