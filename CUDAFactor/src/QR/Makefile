###############################################################################
##################   Makefile for QR Factorization routine ####################
###############################################################################

# Uncomment the -DDEBUG flag to print the time spent in each individual
# function

CC = nvcc
ROUTINE = qr
FLAGS = -lcublas -lm -I.

default: *.cu
	$(CC) -o $(ROUTINE) $^ $(FLAGS) #-DDEBUG 

clean:
	rm -rf qr

