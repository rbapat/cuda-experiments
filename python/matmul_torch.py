import torch
import torch.utils.benchmark as benchmark

NUM_REPS = 20
MAT_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]


def main():
    for size in MAT_SIZES:
        device = torch.device("cuda")
        mat1 = torch.rand(size, size, device=device)
        mat2 = torch.rand(size, size, device=device)
        out = torch.zeros(size, size, device=device)

        t0 = benchmark.Timer(
            stmt="torch.matmul(mat1, mat2, out = out)",
            globals={"mat1": mat1, "mat2": mat2, "out": out},
        )

        print(
            f"PyTorch takes {t0.timeit(NUM_REPS).mean * 1e3} ms on average to multiply two {size} by {size} matrices"
        )


if __name__ == "__main__":
    main()
