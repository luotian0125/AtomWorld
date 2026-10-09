import torch.nn as nn
import torch.nn.init as init

class ElegantMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_size,
                 num_layers,
                 activation='relu',
                 use_layernorm=True,
                 use_orthogonal=True):
        super().__init__()

        act_cls = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'gelu': nn.GELU,
            'leaky_relu': nn.LeakyReLU
        }[activation]
        gain = init.calculate_gain(activation)
        init_fn = init.orthogonal_ if use_orthogonal else init.xavier_uniform_

        def linear_block(in_dim, out_dim, add_act=True):
            layers = []
            linear = nn.Linear(in_dim, out_dim)
            init_fn(linear.weight, gain=gain)
            nn.init.constant_(linear.bias, 0.)
            layers.append(linear)
            if use_layernorm:
                layers.append(nn.LayerNorm(out_dim))
            if add_act:
                layers.append(act_cls())
            return layers

        dims = [input_dim] + [hidden_size] * (num_layers - 1) + [output_dim]
        blocks = [linear_block(dims[i], dims[i+1], i < len(dims)-2) for i in range(num_layers)]
        self.model = nn.Sequential(*[layer for block in blocks for layer in block])
    def forward(self, x):
        return self.model(x)
