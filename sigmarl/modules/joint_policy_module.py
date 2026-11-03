import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torch.distributions import Distribution

from sigmarl.modules.motion_policy_module import MotionPolicyModule
from sigmarl.modules.cbf_lambda_policy_module import CBFLambdaPolicyModule


class JointActionLambdaDistribution(Distribution):
    has_rsample = True

    def __init__(self, action_dist, lambda_dist, action_key, lambda_key):
        super().__init__(validate_args=False)
        self.action_dist = action_dist
        self.lambda_dist = lambda_dist
        self.action_key = action_key
        self.lambda_key = lambda_key

    @staticmethod
    def _reduce_log_prob(log_prob, sample):
        if log_prob.shape == sample.shape:
            log_prob = log_prob.sum(dim=-1)
        return log_prob

    @staticmethod
    def _reduce_entropy(entropy):
        if entropy.ndim > 0 and entropy.shape[-1] > 1:
            entropy = entropy.sum(dim=-1)
        elif entropy.ndim > 0 and entropy.shape[-1] == 1:
            entropy = entropy.squeeze(-1)
        return entropy

    def sample(self, sample_shape=torch.Size()):
        action = self.action_dist.sample(sample_shape)
        lambda_value = self.lambda_dist.sample(sample_shape)
        return TensorDict(
            {
                self.action_key: action,
                self.lambda_key: lambda_value,
            },
            batch_size=action.shape[:-1],
            device=action.device,
        )

    def rsample(self, sample_shape=torch.Size()):
        action = self.action_dist.rsample(sample_shape)
        lambda_value = self.lambda_dist.rsample(sample_shape)
        return TensorDict(
            {
                self.action_key: action,
                self.lambda_key: lambda_value,
            },
            batch_size=action.shape[:-1],
            device=action.device,
        )

    def log_prob(self, td):
        if not isinstance(td, TensorDictBase):
            raise TypeError("Expected TensorDictBase in joint log_prob.")

        action = td.get(self.action_key)
        lambda_value = td.get(self.lambda_key)

        action_log_prob = self.action_dist.log_prob(action)
        action_log_prob = self._reduce_log_prob(action_log_prob, action)

        lambda_log_prob = self.lambda_dist.log_prob(lambda_value)
        lambda_log_prob = self._reduce_log_prob(lambda_log_prob, lambda_value)

        joint_log_prob = action_log_prob + lambda_log_prob

        return joint_log_prob

    def entropy(self):
        action_entropy = self._reduce_entropy(self.action_dist.entropy())
        lambda_entropy = self._reduce_entropy(self.lambda_dist.entropy())
        return action_entropy + lambda_entropy


class JointPolicyModule(TensorDictModuleBase):
    def __init__(self, env, parameters, observation_key):
        super().__init__()
        self.env = env

        self.motion_policy = MotionPolicyModule(env, parameters, observation_key)
        self.lambda_policy = CBFLambdaPolicyModule(env, parameters, observation_key)

    @property
    def in_keys(self):
        return self.motion_policy.in_keys

    @property
    def out_keys(self):
        return (
            self.motion_policy.out_keys
            + self.lambda_policy.out_keys
            + [("agents", "sample_log_prob")]
        )

    @property
    def dist_sample_keys(self):
        return [self.env.action_key, ("agents", "cbf", "lambda")]

    @property
    def log_prob_keys(self):
        return [("agents", "action_log_prob"), ("agents", "cbf", "lambda_log_prob")]

    def forward(self, tensordict):
        tensordict = self.motion_policy(tensordict)
        tensordict = self.lambda_policy(tensordict)

        joint_log_prob = tensordict.get(("agents", "action_log_prob")) + tensordict.get(
            ("agents", "cbf", "lambda_log_prob")
        )
        tensordict.set(("agents", "sample_log_prob"), joint_log_prob)
        return tensordict

    def get_dist(self, tensordict):
        action_dist = self.motion_policy.build_dist(tensordict)
        lambda_dist = self.lambda_policy.build_dist(tensordict)
        return JointActionLambdaDistribution(
            action_dist=action_dist,
            lambda_dist=lambda_dist,
            action_key=self.env.action_key,
            lambda_key=("agents", "cbf", "lambda"),
        )

    def set_train_mode(
        self,
        train_action: bool,
        train_lambda: bool,
        allow_action: bool = True,
        allow_lambda: bool = True,
    ):
        final_train_action = train_action and allow_action
        final_train_lambda = train_lambda and allow_lambda

        for p in self.motion_policy.parameters():
            p.requires_grad = final_train_action

        for p in self.lambda_policy.parameters():
            p.requires_grad = final_train_lambda

    def load_motion_policy(self, path: str, map_location=None):
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        self.motion_policy.load_state_dict(state_dict)

    def load_lambda_policy(self, path: str, map_location=None):
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        self.lambda_policy.load_state_dict(state_dict)

    def motion_policy_state_dict(self):
        return self.motion_policy.state_dict()

    def lambda_policy_state_dict(self):
        return self.lambda_policy.state_dict()
