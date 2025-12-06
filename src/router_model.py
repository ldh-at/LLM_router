import math
import torch
import torch.nn as nn


class RouterQNetwork(nn.Module):
    """
    선형 Q-router.

    Q(h, a) = W_a^T h + b_a

    - h: SAE latent (이미 top-k 마스킹된 sparse state)  shape = (B, H)
    - n_models: 라우팅 대상 LLM 개수 (예: 11)
    - latent_dim: SAE latent 차원 (예: 4096)

    선택 옵션:
      use_mlp_head=True 로 두면,
        Q_total = Q_linear + Q_mlp  형태의 hybrid head 로 쓸 수 있음.
      (기본은 해석 가능한 linear head만 사용)
    """

    def __init__(
        self,
        latent_dim: int = 4096,
        n_models: int = 3,
        use_mlp_head: bool = False,
        hidden_dim: int = 1024,
    ):
        super().__init__()

        # 각 모델별 weight vector (n_models, latent_dim)
        self.W = nn.Parameter(torch.empty(n_models, latent_dim))
        self.b = nn.Parameter(torch.empty(n_models))

        # 선택: 추가 MLP head (표현력 보강용)
        self.use_mlp_head = use_mlp_head
        if use_mlp_head:
            self.mlp = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_models),
            )
        else:
            self.mlp = None

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming 초기화로 선형 head 안정화
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.zeros_(self.b)

        if self.mlp is not None:
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, H)
        return: Q(h, ·) shape = (B, n_models)
        """
        # 선형 Q-head
        q_linear = h @ self.W.t() + self.b  # (B, n_models)

        if self.mlp is not None:
            q_mlp = self.mlp(h)             # (B, n_models)
            return q_linear + q_mlp

        return q_linear
