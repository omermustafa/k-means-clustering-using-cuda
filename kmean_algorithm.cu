#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <sys/time.h>

__global__ void initializeRandomStates(curandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void generateRandomNumbers(curandState *states, float *randomNumbers, int numData, int numCentroids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numData) {
        curandState state = states[idx];
        randomNumbers[idx] = curand_uniform(&state) * numCentroids;
    }
}

__global__ void assignCentroids(float *data, float *centroids, int *assignments, int numData, int numFeatures, int numCentroids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numData) {
        float minDist = INFINITY;
        int minIdx = 0;
        for (int i = 0; i < numCentroids; i++) {
            float dist = 0.0f;
            for (int j = 0; j < numFeatures; j++) {
                float diff = data[idx * numFeatures + j] - centroids[i * numFeatures + j];
                dist += diff * diff;
            }
            if (dist < minDist) {
                minDist = dist;
                minIdx = i;
            }
        }
        assignments[idx] = minIdx;
    }
}

__global__ void updateCentroids(float *data, float *centroids, int *assignments, int numData, int numFeatures, int numCentroids) {
    int centroidIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (centroidIdx < numCentroids) {
        int count = 0;
        float *sumFeatures = new float[numFeatures]();
        for (int i = 0; i < numData; i++) {
            if (assignments[i] == centroidIdx) {
                count++;
                for (int j = 0; j < numFeatures; j++) {
                    sumFeatures[j] += data[i * numFeatures + j];
                }
            }
        }
        if (count > 0) {
            for (int j = 0; j < numFeatures; j++) {
                centroids[centroidIdx * numFeatures + j] = sumFeatures[j] / count;
            }
        }
        delete[] sumFeatures;
    }
}

int main() {
    FILE *file = fopen("input.txt", "r");
    if (file == NULL) {
        printf("Error opening input file.\n");
        return 1;
    }

    int numData, numFeatures, numCentroids, numIterations;

    if (fscanf(file, "%d %d %d %d", &numData, &numFeatures, &numCentroids, &numIterations) != 4) {
        printf("Error reading input parameters from file.\n");
        fclose(file);
        return 1;
    }

    printf("Number of Data Points: %d\n", numData);
    printf("Number of Features: %d\n", numFeatures);
    printf("Number of Centroids: %d\n", numCentroids);
    printf("Number of Iterations: %d\n", numIterations);

    // Allocate memory for data, centroids, and assignments on host
    float *h_data = (float *)malloc(numData * numFeatures * sizeof(float));
    float *h_centroids = (float *)malloc(numCentroids * numFeatures * sizeof(float));
    int *h_assignments = (int *)malloc(numData * sizeof(int));

    // Read data points
    printf("Data Points:\n");
    for (int i = 0; i < numData; i++) {
        for (int j = 0; j < numFeatures; j++) {
            if (fscanf(file, "%f", &h_data[i * numFeatures + j]) != 1) {
                printf("Error reading input data at data point %d, feature %d.\n", i, j);
                fclose(file);
                return 1;
            }
            printf("%.2f ", h_data[i * numFeatures + j]);
        }
        printf("\n");
    }

    // Read centroids
    printf("Initial Centroids:\n");
    for (int i = 0; i < numCentroids; i++) {
        for (int j = 0; j < numFeatures; j++) {
            if (fscanf(file, "%f", &h_centroids[i * numFeatures + j]) != 1) {
                printf("Error reading input data at centroid %d, feature %d.\n", i, j);
                fclose(file);
                return 1;
            }
            printf("%.2f ", h_centroids[i * numFeatures + j]);
        }
        printf("\n");
    }

    fclose(file);

    // Allocate memory for data, centroids, and assignments on device
    float *d_data, *d_centroids;
    int *d_assignments;
    cudaMalloc((void **)&d_data, numData * numFeatures * sizeof(float));
    cudaMalloc((void **)&d_centroids, numCentroids * numFeatures * sizeof(float));
    cudaMalloc((void **)&d_assignments, numData * sizeof(int));

    // Transfer data and centroids from host to device
    cudaMemcpy(d_data, h_data, numData * numFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, numCentroids * numFeatures * sizeof(float), cudaMemcpyHostToDevice);

    // Set up CURAND states and generate random numbers on device
    curandState *d_states;
    cudaMalloc((void **)&d_states, numData * sizeof(curandState));
    unsigned long seed = time(NULL);
    initializeRandomStates<<<(numData + 255) / 256, 256>>>(d_states, seed);
    generateRandomNumbers<<<(numData + 255) / 256, 256>>>(d_states, (float *)d_assignments, numData, numCentroids);

    // Initialize assignments
    int block_size, grid_size;
    cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, assignCentroids, 0, numData);
    assignCentroids<<<grid_size, block_size>>>(d_data, d_centroids, d_assignments, numData, numFeatures, numCentroids);
    cudaDeviceSynchronize();

    // Update centroids
    cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, updateCentroids, 0, numCentroids);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int iter = 0; iter < numIterations; iter++) {
        updateCentroids<<<grid_size, block_size>>>(d_data, d_centroids, d_assignments, numData, numFeatures, numCentroids);
        cudaDeviceSynchronize();
        assignCentroids<<<grid_size, block_size>>>(d_data, d_centroids, d_assignments, numData, numFeatures, numCentroids);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Transfer assignments from device to host
    cudaMemcpy(h_assignments, d_assignments, numData * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the generated assignments
    printf("Final Assignments:\n");
    for (int i = 0; i < numData; i++) {
        printf("Data Point %d: Assigned to Centroid %d\n", i, h_assignments[i]);
    }

    // Print execution time
    printf("Execution Time: %.6f milliseconds\n", milliseconds);

    // Free memory on device
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    cudaFree(d_states);

    // Free memory on host
    free(h_data);
    free(h_centroids);
    free(h_assignments);

    return 0;
}
