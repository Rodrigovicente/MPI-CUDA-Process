#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <mpi.h>

#define NUM_THREADS 512

#define min(a, b) (((a) < (b)) ? (a) : (b))

__device__ void run_mdf(float *univ, int w, int size, int id, float *new_univ) {

	// Neighbor positions
	unsigned int x = id % w;
	unsigned int y = id - x;
	unsigned int x_l = x - 1;
	unsigned int x_r = x + 1;
	unsigned int y_u = y - w;
	unsigned int y_d = y + w;

    new_univ[x + y] = (0.25 * (univ[x_r + y] + univ[x_l + y] + univ[x + y_u] + univ[x + y_d] - (4 * univ[y + x]))) + univ[y + x];

}

__global__ void middle_kernel(float *univ, int h, int w, int p_id, float *new_univ) {
	int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int size = h * w;
	//printf("%d %d %d -> %d\n", blockIdx.x, threadIdx.x, blockDim.x, id);

	if (p_id == 0) {
		if (id < (size / 2) - w) { // Caso não seja borda compartilhada
			run_mdf(univ, w, size, id, new_univ);

		}
		else {
			new_univ[id] = 100;
		}
	} else if (p_id == 1) {
		if ((id >= (size / 2) + w) && (id <= size)) { // Caso não seja borda compartilhada
			run_mdf(univ, w, size, id, new_univ);

		}
		else {
			new_univ[id] = 100;
		}
	}
}

__global__ void border_kernel(float *univ, int h, int w, int p_id, float *new_univ) {
	int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int size = h * w;
	//printf("%d %d %d -> %d\n", blockIdx.x, threadIdx.x, blockDim.x, id);
	
	if (p_id == 0) {
		if ((id >= (size / 2) - w) && (id < size / 2)) { // Caso SEJA borda compartilhada
			run_mdf(univ, w, size, id, new_univ);

		} 
		else {
			new_univ[id] = 100;
		}
	} else if (p_id == 1) {
		if ((id >= size / 2) && (id < (size) / 2 + w)) { // Caso SEJA borda compartilhada
			run_mdf(univ, w, size, id, new_univ);

		}
		else {
			new_univ[id] = 100;
		}
	}
}

void print_array(int arr[], int w, int size) {
	printf("\n");

	for (int i = 0; i < size; i++)
	{
		printf("%s", (arr[i] == 1 ? "0" : " "));
		//printf("%d", arr[i]);

		if ((i + 1) % w == 0) {
			printf("\n");
		}
	}

	printf("\n");
}

void create_universe(float *univ, int w, int h) {
	for(int i = 0; i < h; i++){
		for(int j = 0; j < w; j++){
			int k = (i * w) + j;
			if(i == 0 || j == 0 || i == h-1 || j == w-1){
				univ[k] = 100;
			} else{
				univ[k] = 0;
			}
		}
	}
}

int main(int argc, char **argv)
{
	int g, h, w;

	printf("Enter desired number of generations:\n");
	scanf("%d", &g);

	printf("Enter desired height of universe:\n");
	scanf("%d", &h);

	printf("Enter desired width of universe:\n");
	scanf("%d", &w);

	cudaStream_t border_p1_stream;
	cudaStream_t middle_p1_stream;
	cudaStream_t border_p2_stream;
	cudaStream_t middle_p2_stream;
	cudaStreamCreate(&border_p1_stream);
	cudaStreamCreate(&middle_p1_stream);
	cudaStreamCreate(&border_p2_stream);
	cudaStreamCreate(&middle_p2_stream);


	MPI_Status status;
	int p_id, p_group, p_name;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p_group);
	MPI_Comm_rank(MPI_COMM_WORLD, &p_id);
	MPI_Get_processor_name(processor_name, &p_name);

	// Number of cells in universe
	int size = h * w;

	// Host(CPU) arrays
	float *h_univ = (float*)malloc(size * sizeof(float));
	float *h_new_univ = (float*)malloc(size * sizeof(float));

	// Devide(GPU) arrays
	float *d_univ;
	float *d_new_univ;
	cudaMalloc((void**)&d_univ, size * sizeof(float));
	cudaMalloc((void**)&d_new_univ, size * sizeof(float));

	create_universe(h_univ, size, 0.15);

	size_t n_threads = size > NUM_THREADS ? NUM_THREADS : size;
	unsigned n_blocks = size > NUM_THREADS ? (unsigned)size / NUM_THREADS : (unsigned)1;
	//printf("size: %d - blocks: %d - threads: %d\n", size, n_blocks, t);

	int my_part;
	int iter_count = g;

	if (p_id == 0) {
		
		while (iter_count > 0) {

			my_part = (h * w) / 2;

			cudaMemcpyAsync(d_univ, h_univ, size * sizeof(int), cudaMemcpyHostToDevice, middle_p1_stream); // passa matriz para a GPU
			middle_kernel <<<n_blocks, n_threads, 0, middle_p1_stream >>> (d_univ, h, w, p_id, d_new_univ); // processa a matriz

			//std::swap(d_univ, d_new_univ);

			if (iter_count < g) {
				for (int i = 1; i < w - 1; i++) {
					MPI_Recv(&h_univ[my_part + i], 1, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &status); // recebe borda do outro processo
				}

				cudaMemcpyAsync(d_univ, h_univ, size * sizeof(int), cudaMemcpyHostToDevice, border_p1_stream); // passa a matriz para a GPU em outra stream (para processamento paralelo)
			}
			
			border_kernel <<<n_blocks, n_threads, 0, border_p1_stream >>>(d_univ, h, w, p_id, d_new_univ);
			cudaDeviceSynchronize();

			cudaMemcpyAsync(h_univ, d_univ, size * sizeof(int), cudaMemcpyDeviceToHost, border_p1_stream);

			// Envia a borda para o próximo processo
			my_part = my_part - w;
			for (int i = 1; i < w - 1; i++) {
				MPI_Send(&h_univ[my_part + i], 1, MPI_FLOAT, 1, 1, MPI_COMM_WORLD);
			}

			// print_array(h_univ, w, size);
			iter_count--;
		}
	}
	else {
		while (iter_count > 0) {

			my_part = (h * w) / 2;

			cudaMemcpyAsync(d_univ, h_univ, size * sizeof(int), cudaMemcpyHostToDevice, middle_p2_stream); // passa matriz para a GPU
			middle_kernel << <n_blocks, n_threads, 0, middle_p2_stream >> > (d_univ, h, w, p_id, d_new_univ); // processa a matriz

																											  //std::swap(d_univ, d_new_univ);

			if (iter_count < g) {
				for (int i = 1; i < w - 1; i++) {
					MPI_Recv(&h_univ[my_part + i], 1, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &status); // recebe borda do outro processo
				}
			}

			cudaMemcpyAsync(d_univ, h_univ, size * sizeof(int), cudaMemcpyHostToDevice, border_p2_stream); // passa a matriz para a GPU em outra stream (para processamento paralelo)

			border_kernel << <n_blocks, n_threads, 0, border_p2_stream >> >(d_univ, h, w, p_id, d_new_univ);
			cudaDeviceSynchronize();

			cudaMemcpyAsync(h_univ, d_univ, size * sizeof(int), cudaMemcpyDeviceToHost, border_p2_stream);

			// Envia a borda para o próximo processo
			my_part = my_part - w;
			for (int i = 1; i < w - 1; i++) {
				MPI_Send(&h_univ[my_part + i], 1, MPI_FLOAT, 1, 1, MPI_COMM_WORLD);
			}

			// print_array(h_univ, w, size);
			iter_count--;
		}

	}
	// Release memory? 
	free(h_univ);
	free(h_new_univ);
	cudaFree(d_univ);
	cudaFree(d_new_univ);
	cudaStreamDestroy(border_p1_stream);
	cudaStreamDestroy(middle_p1_stream);
	cudaStreamDestroy(border_p2_stream);
	cudaStreamDestroy(middle_p2_stream);


	MPI_Finalize();

	return 0;
}