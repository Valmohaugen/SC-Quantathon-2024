#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#define QUANTUM_RNG_FILE "quantum_bits.txt"


FILE *qrng_file = NULL;
// Init methods
__attribute__((constructor)) void getFile (void) {
    qrng_file = fopen(QUANTUM_RNG_FILE, "r");
}

__attribute__((destructor)) void closeFile (void) {
    fclose(qrng_file);
}



int quantumRandInt() {

    int number;

    size_t bytesRead = fread(&number, sizeof(int), 1, qrng_file);
    
    // Check if the end of the file is reached
    if (bytesRead != 1) {
        // Rewind the file and try again
        rewind(qrng_file);
        bytesRead = fread(&number, sizeof(int), 1, qrng_file);
    }
    
    // printf("%d\n",number);
    // printf("who called?\n");
    return number;
}

float quantumRandFloat() {

    float number;

    size_t bytesRead = fread(&number, sizeof(float), 1, qrng_file);
    // printf("who called?\n");
    return number;
}

double quantumRandDouble() {

    double number;

    size_t bytesRead = fread(&number, sizeof(double), 1, qrng_file);
    // printf("who called?\n");
    return number;
}

long quantumRandLong() {

    long number;

    size_t bytesRead = fread(&number, sizeof(long), 1, qrng_file);
    // printf("who called?\n");
    return number;
}



// Shim for rand() that intercepts calls
int rand(void) {
    int myInt = quantumRandInt();
    if (myInt < 0) {
        myInt = myInt * -1;
    }
    // printf("yoink\n");
    return (int) myInt;
}



// compile: gcc -shared -fPIC quantum_shim.c -o quantum_shim.so -ldl