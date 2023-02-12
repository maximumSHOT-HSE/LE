def calculate_f_beta_score(precision, recall, beta, eps: float = 1e-9):
    return (1 + (beta ** 2)) * precision * recall / ((beta ** 2) * precision + recall + eps)
