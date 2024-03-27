import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.dropout(F.relu(layer(x)), p=self.dropout_rate) if i < self.num_layers - 1 else layer(x)
        return x


EXIT_TYPE_CONFIGS = [
    {
        'name': 'mlp1x1000',
        'neurons': 1000,
        'mlp_layers': 1
    }, {
        'name': 'mlp3x500',
        'neurons': 500,
        'mlp_layers': 3
    }, {
        'name': 'mlp3x1000',
        'neurons': 1000,
        'mlp_layers': 3
    }, {
        'name': 'mlp5x500',
        'neurons': 500,
        'mlp_layers': 5
    }, {
        'name': 'mlp5x1000',
        'neurons': 1000,
        'mlp_layers': 5
    }
]


class Exit(nn.Module):

    def __init__(self, ch_in, dim, num_classes, mlp_layers) -> None:
        super().__init__()
        self.mlp = MLP(ch_in, dim, num_classes, mlp_layers)

    def forward(self, x):
        feature = x.mean([-2, -1])
        return self.mlp(feature)


class MultiExitNetowrk(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        input_size: list[int] | tuple[int] = (3, 224, 224),
        num_classes: int = 10,
        exit_type_configs: dict = EXIT_TYPE_CONFIGS,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        self.model = model
        self.freeze_backbone = freeze_backbone
        device = next(model.parameters()).device
        self.max_ee_length, self.interm_feat_shape = self._get_interm_feat_info(model, input_size, device)
        self.exit_type_configs = exit_type_configs
        self.exits = nn.ModuleList(
            [
                self._build_exits(self.interm_feat_shape, config['neurons'], num_classes, config['mlp_layers'], device)
                for config in self.exit_type_configs
            ]
        )

        if self.freeze_backbone:
            self._freeze_backbone()
        else:
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        self.apply(self._init_weights)
        self.to(device)

    def _get_interm_feat_info(self, model: nn.Module, input_size: list[int] | tuple[int], device: int | str):
        model.eval()
        inp = torch.rand(input_size).unsqueeze(0).to(device)
        _, interm_feats = model(inp)
        max_ee_length = len(interm_feats) - 1
        model.train()
        return max_ee_length, [f.shape for f in interm_feats[1:]]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if m.weight.requires_grad:
                trunc_normal_(m.weight, std=.02)
            if m.bias is not None and m.bias.requires_grad:
                nn.init.constant_(m.bias, 0)

    def _freeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def _build_exits(
        self, interm_feat_shape: list[torch.Tensor], dim: int, num_classes: int, mlp_layers: int, device: int | str
    ):
        exits = torch.nn.ModuleList()
        for shape in interm_feat_shape:
            exits.append(Exit(shape[1], dim, num_classes, mlp_layers).to(device))
        return exits

    def get_exit_type_config(self):
        return self.exit_type_configs

    def get_max_ee_length(self):
        return self.max_ee_length

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()

    def forward(self, x: torch.Tensor):
        if self.freeze_backbone:
            with torch.no_grad():
                out, interm_feats = self.model(x)
        else:
            out, interm_feats = self.model(x)
        ee_out = []

        for i in range(self.max_ee_length):
            for exits in self.exits:
                ee_out.append(exits[i](interm_feats[i + 1].detach()))
        ee_out = torch.stack(ee_out, dim=1)  # (B, N, C)
        return out, ee_out

    @torch.no_grad()
    def forward_profiling_backbone(self, x: torch.Tensor):
        out, interm_feats = self.model(x)

        return out, interm_feats  #, mask

    @torch.no_grad()
    def forward_profiling_head(self, pos: int, exit_typei: int, interm_feats: torch.Tensor):
        return self.exits[exit_typei][pos](interm_feats[pos + 1].detach())
