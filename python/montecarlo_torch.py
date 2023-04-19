import torch
import torch.utils.benchmark as benchmark

def compute_pi(num_points, device):
    points = 2 * torch.rand(2, num_points, device = device) - 1
    distances = torch.sum(torch.square(points), dim = 0)
    num_in_circle = torch.sum(distances <= 1)
    return 4 * num_in_circle / num_points

def main():
    N = 100000
    device = torch.device('cuda')

    t0 = benchmark.Timer(
        stmt='compute_pi(num_points, device)',
        setup='from __main__ import compute_pi',
        globals={'num_points': N, 'device': device}
    )

    print(t0.timeit(10))

if __name__ == '__main__':
    main()