import torch
from .matmul import _matmul_2d


def triton_linear(
    x: torch.Tensor, 
    weight: torch.Tensor, 
    bias: torch.Tensor = None
) -> torch.Tensor:

    assert x.is_cuda and weight.is_cuda
    assert x.shape[-1] == weight.shape[1], f"Feature mismatch: {x.shape[-1]} vs {weight.shape[1]}"
    
            
    original_shape = x.shape
    
                                                            
    x_2d = x.reshape(-1, x.shape[-1])
    
                                                                     
    weight_t = weight.T
    
                                                                    
    output_2d = _matmul_2d(x_2d, weight_t)
    
            
    if bias is not None:
        output_2d = output_2d + bias
    
                                                            
    output_shape = list(original_shape[:-1]) + [weight.shape[0]]
    output = output_2d.reshape(output_shape)
    
    return output
