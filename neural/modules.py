import torch
from torch import nn

from pyro.distributions import Delta


def rmv(A, b):
    """Tensorized matrix vector multiplication of rightmost dimensions."""
    return torch.matmul(A, b.unsqueeze(-1)).squeeze(-1)


class LazyDelta(Delta):
    def __init__(self, fn, prototype, log_density=0.0, event_dim=0, validate_args=None):
        self.fn = fn
        super().__init__(
            prototype,
            log_density=log_density,
            event_dim=event_dim,
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LazyDelta, _instance)
        new.fn = self.fn
        batch_shape = torch.Size(batch_shape)
        new.v = self.v.expand(batch_shape + self.event_shape)
        new.log_density = self.log_density.expand(batch_shape)
        super(Delta, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        # The shape of self.v will have expanded along with any .expand calls
        shape = sample_shape + self.v.shape
        output = self.fn()
        if len(output.shape) > len(shape):
            shape = torch.Size([1]) + shape
        return output.expand(shape) # It won't take effect if the output has the shape already.

    @property
    def variance(self):
        return torch.zeros_like(self.v)

    def log_prob(self, x):
        return self.log_density


class LazyFn:
    def __init__(self, f, prototype):
        self.f = f
        self.prototype = prototype.clone()

    def lazy(self, *design_obs_pairs):
        def delayed_function():
            return self.f(*design_obs_pairs)

        lazy_delta = LazyDelta(
            delayed_function, self.prototype, event_dim=self.prototype.dim()
        )
        return lazy_delta


class SetEquivariantDesignNetwork(nn.Module):
    def __init__(
        self,
        encoder_network,
        emission_network,
        empty_value,
    ):
        super().__init__()
        self.encoder = encoder_network
        self.emitter = emission_network
        self.register_buffer("prototype", empty_value.clone())
        self.register_parameter("empty_value", nn.Parameter(empty_value))
        self.emitter_hidden_state = None

    def lazy(self, *design_obs_pairs):
        def delayed_function():
            return self.forward(*design_obs_pairs)

        lazy_delta = LazyDelta(
            delayed_function, self.prototype, event_dim=self.prototype.dim()
        )
        return lazy_delta

    def forward(self, *design_obs_pairs):
        if len(design_obs_pairs) == 0:
            sum_encoding = self.empty_value.new_zeros(self.encoder.encoding_dim)

        else:
            sum_encoding = sum(
                self.encoder(xi=design, y=obs, t=[idx + 1])
                for idx, (design, obs) in enumerate(design_obs_pairs)
            )
        output = self.emitter(sum_encoding)

        return output


class SetEquivariantDesignRNN(nn.Module):
    def __init__(
        self,
        encoder_network,
        emission_network,
        empty_value,
    ):
        super().__init__()
        self.encoder = encoder_network
        self.emitter = emission_network
        self.register_buffer("prototype", empty_value.clone())
        self.register_parameter("empty_value", nn.Parameter(empty_value))
        # The following variables are for storing hidden states of the encoder 
        # and emitter which are GRU units. The network must be exclusive to each
        # complete run of T steps of experiments.
        self.emitter_hidden_state = None # Store the hidden state of the emitter
        self.encoder_hidden_state = None # Store the hidden state of the encoder

    def lazy(self, design, obs):
        def delayed_function():
            return self.forward(design, obs)

        lazy_delta = LazyDelta(
            delayed_function, self.prototype, event_dim=self.prototype.dim()
        )
        return lazy_delta

    def forward(self, design, obs):
        if design is None:
            encoding = self.empty_value.new_zeros(1, self.encoder.encoding_dim)
            self.encoder_hidden_state = None
            self.emitter_hidden_state = None
        else:
            encoding, encoder_hidden_state = self.encoder(design, obs, self.encoder_hidden_state)
            self.encoder_hidden_state = encoder_hidden_state
        if self.emitter_hidden_state is not None:
            self.emitter_hidden_state = \
                self.emitter_hidden_state.expand([self.emitter_hidden_state.shape[0], 
                                                  encoding.shape[0],
                                                  *self.emitter_hidden_state.shape[2:]]).contiguous()
        output, emitter_hidden_state = self.emitter(encoding, self.emitter_hidden_state)
        self.emitter_hidden_state = emitter_hidden_state
        # output = output.squeeze(1)
        return output
    
    
class BatchDesignBaseline(SetEquivariantDesignNetwork):
    def __init__(self, T, design_dim):
        nn.Module.__init__(self)
        self.register_buffer("prototype", torch.zeros(design_dim))
        self.designs = nn.ParameterList(
            [nn.Parameter(torch.zeros(design_dim)) for i in range(T)]
        )

    def forward(self, *design_obs_pairs):
        j = len(design_obs_pairs)
        return self.designs[j]


class RandomDesignBaseline(SetEquivariantDesignNetwork):
    def __init__(self, T, design_dim, random_designs_dist=None):
        nn.Module.__init__(self)
        self.register_buffer("prototype", torch.zeros(design_dim))
        if random_designs_dist is None:
            random_designs_dist = torch.distributions.Normal(
                torch.zeros(design_dim), torch.ones(design_dim)
            )
        self.random_designs_dist = random_designs_dist

    def forward(self, *design_obs_pairs):
        return self.random_designs_dist.sample()
