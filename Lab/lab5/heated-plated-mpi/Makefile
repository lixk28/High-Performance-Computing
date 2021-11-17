
.PHONY: all
all:
	gcc -g -Wall heated_plate_openmp.c -o heated_plate_openmp -fopenmp -lm
	mpicc -g -Wall heated_plate_mpi.c -o heated_plate_mpi -lm

.PHONY: test
test:
	./heated_plate_openmp > heated_plate_openmp_test.txt
	./heated_plate_mpi > heated_plate_mpi_test.txt

.PHONY: clean
clean:
	-rm -f heated_plate_openmp heated_plate_mpi