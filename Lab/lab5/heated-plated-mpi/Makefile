
.PHONY: all
all:
	gcc -g -Wall heated_plate_openmp.c -o ./bin/heated_plate_openmp -fopenmp -lm
	mpicc -g -Wall heated_plate_mpi.c -o ./bin/heated_plate_mpi -lm -fopenmp

.PHONY: test
test:
	./bin/heated_plate_openmp > heated_plate_openmp_test.txt
	mpirun -n 6 ./bin/heated_plate_mpi > heated_plate_mpi_test.txt

.PHONY: clean
clean:
	-rm -f ./bin/*