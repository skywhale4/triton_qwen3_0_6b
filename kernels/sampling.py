import torch


def sample_with_temperature(
    logits: torch.Tensor, 
    temperature: float = 0.7,
    top_p: float = 1.0
):
    """
    Temperature + top-p sampling using PyTorch
    
    Args:
        logits: [M, N] tensor of logits (M=batch_size, N=vocab_size)
        temperature: temperature value (default 0.7)
        top_p: top-p value (default 1.0 = no filtering)
        
    Returns:
        [M] tensor of sampled token indices
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    if top_p < 1.0:
        # Apply top-p filtering
        probs = torch.softmax(scaled_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Compute cumulative probabilities
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create mask for tokens to keep
        mask = cumsum_probs <= top_p
        mask[:, 0] = True  # Always keep at least one token
        
        # Create filtered logits
        filtered_logits = scaled_logits.clone()
        
        # Create mask in original order
        sorted_mask = torch.zeros_like(mask, dtype=torch.bool)
        sorted_mask.scatter_(1, sorted_indices, mask)
        filtered_logits.masked_fill_(~sorted_mask, float('-inf'))
        
        # Sample from filtered distribution
        probs = torch.softmax(filtered_logits, dim=-1)
    else:
        # Just apply temperature
        probs = torch.softmax(scaled_logits, dim=-1)
    
    # Sample from probability distribution
    sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return sampled_indices