import torch
import torch.utils.benchmark as benchmark

def main():
    N = 512
    device = torch.device('cuda')
    mat1 = torch.rand(N, N, device = device)
    mat2 = torch.rand(N, N, device = device)
    out = torch.zeros(N, N, device = device)
    
    t0 = benchmark.Timer(
        stmt='torch.matmul(mat1, mat2, out = out)',
        globals={'mat1': mat1, 'mat2': mat2, 'out': out}
    )

    print(t0.timeit(100))

if __name__ == '__main__':
    main()