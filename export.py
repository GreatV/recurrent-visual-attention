import torch

from model import RecurrentAttention


if __name__ == "__main__":
    model = RecurrentAttention(8, 1, 1, 1, 128, 126, 0.05, 256, 10)
    x = torch.randn(4, 3, 320, 320)
    l_t_prev = torch.randn(4, 2)
    h_t_prev = torch.randn(4, 256)

    try:
        torch.export.export(f=model, args=(x, l_t_prev, h_t_prev))
        print ("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print ("[JIT] torch.export failed.")
        raise e