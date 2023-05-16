import torch
import torch.utils.benchmark as benchmark

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


def compute_pi(num_points, device):
    points = 2 * torch.rand(2, num_points, device=device) - 1
    distances = torch.sum(torch.square(points), dim=0)
    num_in_circle = torch.sum(distances <= 1)
    return 4 * num_in_circle / num_points


def main():
    for num_samples in NUM_SAMPLES:
        device = torch.device("cuda")

        t0 = benchmark.Timer(
            stmt="compute_pi(num_points, device)",
            setup="from __main__ import compute_pi",
            globals={"num_points": num_samples, "device": device},
        )

        print(
            f"PyTorch takes {t0.timeit(NUM_REPS).mean * 1e3} ms on average to run monte carlo estimations of pi with {num_samples} points"
        )


if __name__ == "__main__":
    main()
