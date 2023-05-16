import cupy as cp
from cupyx.profiler import benchmark

NUM_REPS = 20
MAT_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]


def matmul(mat1, mat2):
    return cp.matmul(mat1, mat2)


def main():
    for size in MAT_SIZES:
        mat1 = cp.random.random((size, size))
        mat2 = cp.random.random((size, size))
        avg_time = benchmark(matmul, (mat1, mat2), n_repeat=NUM_REPS)

        print(
            f"CuPy takes {avg_time.gpu_times.mean() * 1000} ms on average to multiply two {size} by {size} matrices"
        )


if __name__ == "__main__":
    main()
