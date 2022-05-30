from typing import Any

import torch


def detach_tensors(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    else:
        return x
