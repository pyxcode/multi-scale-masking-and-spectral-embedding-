import torch

def vicreg_loss(embeddings: torch.Tensor, lambda_reg: float = 0.04) -> torch.Tensor:
    z = embeddings.reshape(-1, embeddings.shape[-1])  # [B*N, D]
    n, d = z.shape

    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (n - 1)

    # pénalise si std != 1
    var_loss = (torch.diagonal(cov) - 1).pow(2).mean()

    # pénalise les off-diagonal
    off_diag = cov - torch.diag(torch.diagonal(cov))
    cov_loss = off_diag.pow(2).sum() / d

    return lambda_reg * (var_loss + cov_loss)
