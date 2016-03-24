UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
    MATLABROOT ?= /Applications/MATLAB_R2015a.app
endif
ifeq ($(UNAME), Linux)
    MATLABROOT ?= /usr/local/MATLAB/R2014b/
endif

MEX = $(MATLABROOT)/bin/mex -O CFLAGS='-std=c99 -fPIC' 

#MEX ?= /usr/local/MATLAB/R2014b/bin/mex -O CFLAGS='-std=c99 -fPIC' 

# Makefile for SVM-struct, 03.10.06

#Use the following to compile under unix or cygwin
CC = gcc
LD = gcc

#Call 'make' using the following line to make CYGWIN produce stand-alone Windows executables
#		make 'SFLAGS=-mno-cygwin'

CFLAGS =    $(SFLAGS) -O3 -fomit-frame-pointer -ffast-math -Wall 
LDFLAGS =   $(SFLAGS) -O3 -lm -Wall
#CFLAGS =   $(SFLAGS) -pg -Wall
#LDFLAGS =  $(SFLAGS) -pg -Wall

all: svm_struct_noexe

svm_struct_noexe: svm_struct_learn.o svm_struct_classify.o svm_struct_common.o svm_struct_main.o 

.PHONY: clean
clean:
	rm -f *.o *.tcov *.d core gmon.out *.stackdump


#----------------------#
#----  STRUCT SVM  ----#
#----------------------#

svm_struct_common.o: svm_struct_common.c svm_struct_common.h ../svm_struct_api_types.h
	$(MEX) -c svm_struct_common.c  svm_struct_common.o

svm_struct_learn.o: svm_struct_learn.c ../svm_light/svm_learn.h svm_struct_common.h ../svm_struct_api.h ../svm_struct_api_types.h
	$(MEX) -c  svm_struct_learn.c svm_struct_learn.o

svm_struct_main.o: svm_struct_main.c ../svm_light/svm_common.h ../svm_light/svm_learn.h svm_struct_learn.h svm_struct_common.h ../svm_struct_api.h ../svm_struct_api_types.h
	$(MEX) -c  svm_struct_main.c svm_struct_main.o

svm_struct_classify.o: svm_struct_classify.c svm_struct_common.h ../svm_struct_api_types.h ../svm_struct_api.h ../svm_light/svm_common.h 
	$(MEX)  -c  svm_struct_classify.c  svm_struct_classify.o
