import cupy as cp
from cupyx.profiler import benchmark

NUM_REPS = 20
NUM_SAMPLES = [
    128 * 64 * 16,
    128 * 128 * 16,
    128 * 128 * 32,
    128 * 128 * 64,
    128 * 128 * 128,
    128 * 256 * 128,
    128 * 256 * 256,
    256 * 256 * 256,
]


def compute_pi(num_points):
    points = 2 * cp.random.random((2, num_points)) - 1
    distances = cp.sum(cp.square(points), axis=0)
    num_in_circle = cp.sum(cp.where(distances <= 1)[0])

    return 4 * num_in_circle / num_points


def main():
    for num_samples in NUM_SAMPLES:
        avg_time = benchmark(compute_pi, (num_samples,), n_repeat=NUM_REPS)
        print(
            f"CuPy takes {avg_time.gpu_times.mean() * 1000} ms on average to run monte carlo estimations of pi with {num_samples} points"
        )


if __name__ == "__main__":
    main()
