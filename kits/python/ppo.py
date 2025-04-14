# %%
from enum import IntEnum

from luxai_s3.wrappers import LuxAIS3GymEnv

# taken from https://www.kaggle.com/code/yizhewang3/ppo-stable-baselines3

SPACE_SIZE = 24
NUM_TEAMS = 2
MAX_UNITS = 16
RELIC_REWARD_RANGE = 2
MAX_STEPS_IN_MATCH = 100
MAX_ENERGY_PER_TILE = 20
MAX_RELIC_NODES = 6
LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR = 50
LAST_MATCH_WHEN_RELIC_CAN_APPEAR = 2

# We will find the exact value of these constants during the game
UNIT_MOVE_COST = 1  # OPTIONS: list(range(1, 6))
UNIT_SAP_COST = 30  # OPTIONS: list(range(30, 51))
UNIT_SAP_RANGE = 3  # OPTIONS: list(range(3, 8))
UNIT_SENSOR_RANGE = 2  # OPTIONS: [1, 2, 3, 4]
OBSTACLE_MOVEMENT_PERIOD = 20  # OPTIONS: 6.67, 10, 20, 40
OBSTACLE_MOVEMENT_DIRECTION = (0, 0)  # OPTIONS: [(1, -1), (-1, 1)]

# We will NOT find the exact value of these constants during the game
NEBULA_ENERGY_REDUCTION = 5  # OPTIONS: [0, 1, 2, 3, 5, 25]


class NodeType(IntEnum):
    unknown = -1
    empty = 0
    nebula = 1
    asteroid = 2

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


_DIRECTIONS = [
    (0, 0),  # center
    (0, -1),  # up
    (1, 0),  # right
    (0, 1),  #  down
    (-1, 0),  # left
    (0, 0),  # sap
]


class ActionType(IntEnum):
    center = 0
    up = 1
    right = 2
    down = 3
    left = 4
    sap = 5

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @classmethod
    def from_coordinates(cls, current_position, next_position):
        dx = next_position[0] - current_position[0]
        dy = next_position[1] - current_position[1]

        if dx < 0:
            return ActionType.left
        elif dx > 0:
            return ActionType.right
        elif dy < 0:
            return ActionType.up
        elif dy > 0:
            return ActionType.down
        else:
            return ActionType.center

    def to_direction(self):
        return _DIRECTIONS[self]


def get_match_step(step: int) -> int:
    return step % (MAX_STEPS_IN_MATCH + 1)


def get_match_number(step: int) -> int:
    return step // (MAX_STEPS_IN_MATCH + 1)


# def warp_int(x):
#     if x >= SPACE_SIZE:
#         x -= SPACE_SIZE
#     elif x < 0:
#         x += SPACE_SIZE
#     return x


# def warp_point(x, y) -> tuple:
#     return warp_int(x), warp_int(y)


def get_opposite(x, y) -> tuple:
    # Returns the mirrored point across the diagonal
    return SPACE_SIZE - y - 1, SPACE_SIZE - x - 1


def is_upper_sector(x, y) -> bool:
    return SPACE_SIZE - x - 1 >= y


def is_lower_sector(x, y) -> bool:
    return SPACE_SIZE - x - 1 <= y


def is_team_sector(team_id, x, y) -> bool:
    return is_upper_sector(x, y) if team_id == 0 else is_lower_sector(x, y)


# %%
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ParallelEnv

# modified from https://www.kaggle.com/code/yizhewang3/ppo-stable-baselines3


def flatten_obs(base_obs, env_cfg):
    """
    Convert the multi-agent observation dictionary for the *current* player
    into the flattened dict structure you had.
    """

    # Sometimes there's an extra "obs" key; adapt as needed
    if "obs" in base_obs:
        base_obs = base_obs["obs"]

    flat_obs = {}
    # 处理 units 数据
    if "units" in base_obs:
        flat_obs["units_position"] = np.array(
            base_obs["units"]["position"], dtype=np.int32
        )
        flat_obs["units_energy"] = np.array(base_obs["units"]["energy"], dtype=np.int32)
        # 如果 units_energy 的 shape 为 (NUM_TEAMS, MAX_UNITS) 则扩展一个维度
        if flat_obs["units_energy"].ndim == 2:
            flat_obs["units_energy"] = np.expand_dims(flat_obs["units_energy"], axis=-1)
    else:
        flat_obs["units_position"] = np.array(
            base_obs["units_position"], dtype=np.int32
        )
        flat_obs["units_energy"] = np.array(base_obs["units_energy"], dtype=np.int32)
        if flat_obs["units_energy"].ndim == 2:
            flat_obs["units_energy"] = np.expand_dims(flat_obs["units_energy"], axis=-1)

    # 处理 units_mask
    if "units_mask" in base_obs:
        flat_obs["units_mask"] = np.array(base_obs["units_mask"], dtype=np.int8)
    else:
        flat_obs["units_mask"] = np.zeros(
            flat_obs["units_position"].shape[:2], dtype=np.int8
        )

    # 处理 sensor_mask：若返回的是 3D 数组，则取逻辑 or 得到全局 mask
    sensor_mask_arr = np.array(base_obs["sensor_mask"], dtype=np.int8)
    if sensor_mask_arr.ndim == 3:
        sensor_mask = np.any(sensor_mask_arr, axis=0).astype(np.int8)
    else:
        sensor_mask = sensor_mask_arr
    flat_obs["sensor_mask"] = sensor_mask

    # 处理 map_features（tile_type 与 energy）
    if "map_features" in base_obs:
        mf = base_obs["map_features"]
        flat_obs["map_features_tile_type"] = np.array(mf["tile_type"], dtype=np.int8)
        flat_obs["map_features_energy"] = np.array(mf["energy"], dtype=np.int8)
    else:
        flat_obs["map_features_tile_type"] = np.array(
            base_obs["map_features_tile_type"], dtype=np.int8
        )
        flat_obs["map_features_energy"] = np.array(
            base_obs["map_features_energy"], dtype=np.int8
        )

    # 处理 relic 节点信息
    if "relic_nodes_mask" in base_obs:
        flat_obs["relic_nodes_mask"] = np.array(
            base_obs["relic_nodes_mask"], dtype=np.int8
        )
    else:
        max_relic = env_cfg.get("max_relic_nodes", 6) if env_cfg is not None else 6
        flat_obs["relic_nodes_mask"] = np.zeros((max_relic,), dtype=np.int8)
    if "relic_nodes" in base_obs:
        flat_obs["relic_nodes"] = np.array(base_obs["relic_nodes"], dtype=np.int32)
    else:
        max_relic = env_cfg.get("max_relic_nodes", 6) if env_cfg is not None else 6
        flat_obs["relic_nodes"] = np.full((max_relic, 2), -1, dtype=np.int32)

    # 处理团队得分与胜局
    if "team_points" in base_obs:
        flat_obs["team_points"] = np.array(base_obs["team_points"], dtype=np.int32)
    else:
        flat_obs["team_points"] = np.zeros(2, dtype=np.int32)
    if "team_wins" in base_obs:
        flat_obs["team_wins"] = np.array(base_obs["team_wins"], dtype=np.int32)
    else:
        flat_obs["team_wins"] = np.zeros(2, dtype=np.int32)

    # 处理步数信息
    if "steps" in base_obs:
        flat_obs["steps"] = np.array([base_obs["steps"]], dtype=np.int32)
    else:
        flat_obs["steps"] = np.array([0], dtype=np.int32)
    if "match_steps" in base_obs:
        flat_obs["match_steps"] = np.array([base_obs["match_steps"]], dtype=np.int32)
    else:
        flat_obs["match_steps"] = np.array([0], dtype=np.int32)

    # 注意：不在此处处理 remainingOverageTime，
    # 将在 Agent.act 中利用传入的参数添加

    # 补全环境配置信息
    if env_cfg is not None:
        flat_obs["env_cfg_map_width"] = np.array([env_cfg["map_width"]], dtype=np.int32)
        flat_obs["env_cfg_map_height"] = np.array(
            [env_cfg["map_height"]], dtype=np.int32
        )
        flat_obs["env_cfg_max_steps_in_match"] = np.array(
            [env_cfg["max_steps_in_match"]], dtype=np.int32
        )
        flat_obs["env_cfg_unit_move_cost"] = np.array(
            [env_cfg["unit_move_cost"]], dtype=np.int32
        )
        flat_obs["env_cfg_unit_sap_cost"] = np.array(
            [env_cfg["unit_sap_cost"]], dtype=np.int32
        )
        flat_obs["env_cfg_unit_sap_range"] = np.array(
            [env_cfg["unit_sap_range"]], dtype=np.int32
        )
    else:
        flat_obs["env_cfg_map_width"] = np.array([0], dtype=np.int32)
        flat_obs["env_cfg_map_height"] = np.array([0], dtype=np.int32)
        flat_obs["env_cfg_max_steps_in_match"] = np.array([0], dtype=np.int32)
        flat_obs["env_cfg_unit_move_cost"] = np.array([0], dtype=np.int32)
        flat_obs["env_cfg_unit_sap_cost"] = np.array([0], dtype=np.int32)
        flat_obs["env_cfg_unit_sap_range"] = np.array([0], dtype=np.int32)

    return flat_obs


class LuxAICAECEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "LuxAI-AEC"}

    def __init__(self, base_env):
        """
        :param base_env: Your two-agent environment, e.g. LuxAIS3GymEnv, that returns:
                         obs: {"player_0": {...}, "player_1": {...}}
                         reward: {"player_0": float, "player_1": float}
                         done/trunc: {"player_0": bool, "player_1": bool}, etc.
        """
        super().__init__()
        self.base_env = base_env

        # In AEC, we specify the set of agents:
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]

        # We'll manage done/trunc states in dictionaries
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self.infos = dict.fromkeys(self.agents, {})
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)

        # We store pending actions until we have both, if we want truly simultaneous stepping
        # or we can do "turn-by-turn" in the sense that each step in the base env is a single move.
        self._actions = {agent: None for agent in self.agents}

        # Agent selector to cycle between player_0 -> player_1 -> player_0 ...
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # Construct observation_spaces and action_spaces for each agent
        # from your flatten logic. The user code has a dictionary observation_space, so define that:
        self.observation_spaces = {}
        self.action_spaces = {}

        # Example: replicate your dictionary-based observation space for each agent:
        obs_space_dict = {
            "units_position": spaces.Box(
                low=0,
                high=SPACE_SIZE - 1,
                shape=(NUM_TEAMS, MAX_UNITS, 2),
                dtype=np.int32,
            ),
            "units_energy": spaces.Box(
                low=0, high=400, shape=(NUM_TEAMS, MAX_UNITS, 1), dtype=np.int32
            ),
            "units_mask": spaces.Box(
                low=0, high=1, shape=(NUM_TEAMS, MAX_UNITS), dtype=np.int8
            ),
            "sensor_mask": spaces.Box(
                low=0, high=1, shape=(SPACE_SIZE, SPACE_SIZE), dtype=np.int8
            ),
            "map_features_tile_type": spaces.Box(
                low=-1, high=2, shape=(SPACE_SIZE, SPACE_SIZE), dtype=np.int8
            ),
            "map_features_energy": spaces.Box(
                low=-1,
                high=MAX_ENERGY_PER_TILE,
                shape=(SPACE_SIZE, SPACE_SIZE),
                dtype=np.int8,
            ),
            "relic_nodes_mask": spaces.Box(
                low=0, high=1, shape=(MAX_RELIC_NODES,), dtype=np.int8
            ),
            "relic_nodes": spaces.Box(
                low=-1, high=SPACE_SIZE - 1, shape=(MAX_RELIC_NODES, 2), dtype=np.int32
            ),
            "team_points": spaces.Box(
                low=0, high=1000, shape=(NUM_TEAMS,), dtype=np.int32
            ),
            "team_wins": spaces.Box(
                low=0, high=1000, shape=(NUM_TEAMS,), dtype=np.int32
            ),
            "steps": spaces.Box(
                low=0, high=MAX_STEPS_IN_MATCH, shape=(1,), dtype=np.int32
            ),
            "match_steps": spaces.Box(
                low=0, high=MAX_STEPS_IN_MATCH, shape=(1,), dtype=np.int32
            ),
            "env_cfg_map_width": spaces.Box(
                low=0, high=SPACE_SIZE, shape=(1,), dtype=np.int32
            ),
            "env_cfg_map_height": spaces.Box(
                low=0, high=SPACE_SIZE, shape=(1,), dtype=np.int32
            ),
            "env_cfg_max_steps_in_match": spaces.Box(
                low=0, high=MAX_STEPS_IN_MATCH, shape=(1,), dtype=np.int32
            ),
            "env_cfg_unit_move_cost": spaces.Box(
                low=0, high=100, shape=(1,), dtype=np.int32
            ),
            "env_cfg_unit_sap_cost": spaces.Box(
                low=0, high=100, shape=(1,), dtype=np.int32
            ),
            "env_cfg_unit_sap_range": spaces.Box(
                low=0, high=100, shape=(1,), dtype=np.int32
            ),
        }
        # Create a Dict space from that
        self.obs_space_single = spaces.Dict(obs_space_dict)

        # Example action space: MultiDiscrete for each possible unit, as in your code
        self.act_space_single = spaces.MultiDiscrete([len(ActionType)] * MAX_UNITS)

        for agent in self.agents:
            self.observation_spaces[agent] = self.obs_space_single
            self.action_spaces[agent] = self.act_space_single

        self.env_cfg = None
        self.has_reset = False

    def reset(self, seed=None, return_info=True, options=None):
        obs_dict, info = self.base_env.reset(seed=seed, options=options)
        self.env_cfg = info.get("params", {})
        self.has_reset = True
        self.agents = self.possible_agents[:]
        # Clear terminations/truncations
        ret = {}
        infos = {}
        for ag in self.agents:
            self.terminations[ag] = False
            self.truncations[ag] = False
            self.rewards[ag] = 0.0
            self.infos[ag] = {}
            self._cumulative_rewards[ag] = 0.0
            ret[ag] = flatten_obs(obs_dict[ag], self.env_cfg)
            infos[ag] = self.env_cfg

        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()

        return ret, infos

    def step(self, simplified_actions_dict):
        if not self.has_reset:
            raise AssertionError("Environment must be reset before step().")
        action_dict = {}
        for ag in self.agents:
            actions = np.zeros((MAX_UNITS, 3), dtype=np.int16)

            for i, action_type in enumerate(simplified_actions_dict[ag]):
                dir_x, dir_y = _DIRECTIONS[action_type]
                actions[i, 0] = action_type  # ActionType (0 to 5)
                actions[i, 1] = dir_x  # delta_x
                actions[i, 2] = dir_y  # delta_y
            action_dict[ag] = np.array(actions, dtype=np.int16)
        obs, rew, terminated_dict, truncated_dict, info = self.base_env.step(
            action_dict
        )
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # Some or all agents might be done. We keep them in self.agents until the end of the game
        # if your logic calls for that. Or remove them once done.
        # Typically in a 2-player zero-sum environment, if one agent is done => both are done.
        # But you can do your own logic. For example:
        new_active_agents = []

        for ag in self.possible_agents:
            observations[ag] = flatten_obs(obs[ag], self.env_cfg)
            rewards[ag] = float(rew[ag])  # ensure it's float
            terminations[ag] = bool(terminated_dict[ag])
            truncations[ag] = bool(truncated_dict[ag])
            infos[ag] = info.get(ag, {})

            # If not done, we keep them in the list
            if not (terminations[ag] or truncations[ag]):
                new_active_agents.append(ag)

        self.agents = new_active_agents

        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.base_env.render()

    def close(self):
        pass


# %%
# from pettingzoo.utils import parallel_to_aec

base_env = LuxAIS3GymEnv(numpy_output=True)
wrapped_env = LuxAICAECEnv(base_env)
obs, _ = wrapped_env.reset()
print(obs["player_0"].keys())
action_dict = {}
for player, act_space in wrapped_env.action_spaces.items():
    action_dict[player] = act_space.sample()
obs, _, _, _, _ = wrapped_env.step(action_dict)


# %%
import torch
import torch.nn as nn
import torch.optim as optim

env = LuxAICAECEnv(LuxAIS3GymEnv(numpy_output=True))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        # We assume env.observation_spaces["player_0"] and env.action_spaces["player_0"] exist
        self.net = MultiInputActorCriticPolicy(
            observation_space=env.observation_spaces["player_0"],
            action_space=env.action_spaces["player_0"],
            lr_schedule=MultiInputActorCriticPolicy._dummy_schedule,
        )

    def get_value(self, obs):
        features_vf = self.net.vf_features_extractor(obs).float()
        latent_vf = self.net.mlp_extractor.value_net(features_vf)
        value = self.net.value_net(latent_vf)
        return value.squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        features_pi = self.net.pi_features_extractor(obs).float()
        latent_pi = self.net.mlp_extractor.policy_net(features_pi)
        dist = self.net._get_action_dist_from_latent(latent_pi)

        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        if log_prob.ndim > 1:
            # MultiDiscrete: sum log_probs across action dimensions
            log_prob = log_prob.sum(-1)
        entropy = dist.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(-1)

        # Compute value using separate extractor and head
        features_vf = self.net.vf_features_extractor(obs).float()
        latent_vf = self.net.mlp_extractor.value_net(features_vf)
        value = self.net.value_net(latent_vf).squeeze(-1)
        return action, log_prob, entropy, value


# -------------------------------------------------------
# 4. Batchify/unbatchify for your dictionary-based obs
# -------------------------------------------------------
def batchify_obs(obs):
    """
    obs is like:
      {
        'player_0': { 'units_position': ..., 'units_energy': ..., ... },
        'player_1': { 'units_position': ..., 'units_energy': ..., ... }
      }
    Returns a dict of Tensors with shape [2, ...].
    """
    batched_obs = {}
    for key in obs["player_0"]:
        # For each key, convert each player's observation to a tensor and stack
        stacked = []
        for agent in ("player_0", "player_1"):
            stacked.append(torch.tensor(obs[agent][key], device=device))
        batched_obs[key] = torch.stack(stacked, dim=0)
    return batched_obs


def unbatchify_actions(action_tensor):
    """
    Converts a [2, ...] torch tensor of actions into:
      { 'player_0': action_for_player_0, 'player_1': action_for_player_1 }
    For a MultiDiscrete action of shape [16], each row is the 16-dim action.
    """
    action_np = action_tensor.cpu().numpy()
    return {
        "player_0": action_np[0],
        "player_1": action_np[1],
    }


# %%
# modified from https://pettingzoo.farama.org/tutorials/cleanrl/implementing_PPO/
import wandb


def train_ppo(env, total_episodes=2, rollout_num_steps=200):
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    train_epochs = 4
    gae_lambda = 0.95
    max_grad_norm = 0.5
    lr = 2.5e-4
    anneal_lr = True
    seed = 2025
    run = wandb.init(
        entity="ay2425s2-cs3263-group-13",
        project="lux-ppo",
        config={
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "clip_coef": clip_coef,
            "gamma": gamma,
            "learning_rate": lr,
            "batch_size": batch_size,
            "train_epochs": train_epochs,
            "total_episodes": total_episodes,
            "max_steps_per_episode": rollout_num_steps,
        },
        save_code=True,
    )
    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    for episode in range(1, total_episodes + 1):
        next_obs, info = env.reset(seed=seed)
        total_episodic_return = np.zeros(2, dtype=np.float32)

        rb_obs = []
        rb_actions = []
        rb_logprobs = []
        rb_rewards = []
        rb_dones = []
        rb_values = []

        end_step = 0  # track how many steps actually took place

        if anneal_lr:
            frac = 1.0 - (episode - 1.0) / total_episodes
            lrnow = frac * lr
            optimizer.param_groups[0]["lr"] = lrnow

        # 1. Collect experience
        for step in range(rollout_num_steps):
            obs_tensor = batchify_obs(next_obs)
            with torch.no_grad():
                actions, logprobs, entropy, values = agent.get_action_and_value(
                    obs_tensor
                )

            action_dict = unbatchify_actions(actions)
            next_obs, rewards, terms, truncs, infos = env.step(action_dict)

            rb_obs.append(obs_tensor)
            rb_actions.append(actions)
            rb_logprobs.append(logprobs)
            rb_values.append(values)

            r0, r1 = rewards["player_0"], rewards["player_1"]
            trunc0, trunc1 = truncs["player_0"], truncs["player_1"]
            term0, term1 = terms["player_0"], terms["player_1"]
            next_done = torch.tensor([
                np.logical_or(trunc0, term0),
                np.logical_or(trunc1, term1),
            ])
            rb_rewards.append(torch.tensor([r0, r1], device=device))
            rb_dones.append(next_done)

            total_episodic_return += np.array([r0, r1])
            end_step = step + 1

            if all(terms.values()) or all(truncs.values()):
                break

        # 2. Bootstrapping if not done
        with torch.no_grad():
            if not all(terms.values()):
                final_obs_tensor = batchify_obs(next_obs)
                _, _, _, next_values = agent.get_action_and_value(final_obs_tensor)
            else:
                next_values = torch.zeros(2, device=device)

        # 3. Convert lists -> stacked Tensors

        num_steps = len(rb_obs)
        stacked_obs = {}
        for key in rb_obs[0].keys():
            cat_list = [step_dict[key] for step_dict in rb_obs]
            stacked_obs[key] = torch.stack(cat_list, dim=0)

        rb_actions = torch.stack(rb_actions, dim=0)  # [num_steps, 2, (action_dim)]
        rb_logprobs = torch.stack(rb_logprobs, dim=0)  # [num_steps, 2]
        rb_values = torch.stack(rb_values, dim=0)  # [num_steps, 2]
        rb_rewards = torch.stack(rb_rewards, dim=0)  # [num_steps, 2]
        rb_dones = torch.stack(rb_dones, dim=0)  # [num_steps, 2]

        # 4. GAE or simple advantage
        rb_advantages = torch.zeros_like(rb_rewards)
        rb_returns = torch.zeros_like(rb_rewards)
        gae = torch.zeros(2, device=device)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_val = next_values
                done_mask = 1.0 - rb_dones[t].float()
            else:
                next_val = rb_values[t + 1]
                done_mask = 1.0 - rb_dones[t + 1].float()

            delta = rb_rewards[t] + gamma * next_val * done_mask - rb_values[t]
            gae = delta + gamma * gae_lambda * gae * done_mask
            rb_advantages[t] = gae
            rb_returns[t] = gae + rb_values[t]

        # 5. Flatten time & agent
        b_obs = {}
        for key, val in stacked_obs.items():
            b_obs[key] = val.view(num_steps * 2, *val.shape[2:])

        b_actions = rb_actions.view(num_steps * 2, -1)
        b_logprobs = rb_logprobs.view(num_steps * 2)
        b_values = rb_values.view(num_steps * 2)
        b_advantages = rb_advantages.view(num_steps * 2)
        b_returns = rb_returns.view(num_steps * 2)

        # We'll track these for logging outside the minibatch loop
        clip_fracs = []
        old_approx_kls = []
        approx_kls = []
        last_v_loss = 0.0
        last_pg_loss = 0.0

        # 6. PPO update
        total_batch = num_steps * 2
        indices = np.arange(total_batch)
        for _ in range(train_epochs):
            np.random.shuffle(indices)
            for start in range(0, total_batch, batch_size):
                end = start + batch_size
                batch_inds = indices[start:end]

                mb_obs = {k: v[batch_inds] for k, v in b_obs.items()}
                mb_actions = b_actions[batch_inds]
                mb_old_logprob = b_logprobs[batch_inds]
                mb_adv = b_advantages[batch_inds]
                mb_returns = b_returns[batch_inds]
                mb_values = b_values[batch_inds]
                # Evaluate new logprob
                _, new_logprob, entropy, value = agent.get_action_and_value(
                    mb_obs, action=mb_actions
                )
                logratio = new_logprob - mb_old_logprob
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()

                # Record them
                old_approx_kls.append(old_approx_kl.item())
                approx_kls.append(approx_kl.item())

                # Compute clip fraction
                clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                clip_fracs.append(clip_fraction)

                # Normalize advantages
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                value = value.view(-1)
                v_loss_unclipped = (value - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(
                    value - mb_values, -clip_coef, clip_coef
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy
                entropy_loss = entropy.mean()

                loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                # We'll keep track of the last minibatch losses for logging
                last_v_loss = v_loss.item()
                last_pg_loss = pg_loss.item()

        # 7. Explained Variance
        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        run.log({
            "episode": episode,
            "episode_length": end_step,
            "player0_return": np.mean(total_episodic_return[0]),
            "player1_return": np.mean(total_episodic_return[1]),
            "policy_loss": -last_pg_loss,
            "value_loss": last_v_loss,
            "old_approx_kl": np.mean(old_approx_kls),
            "approx_kl": np.mean(approx_kls),
            "clip_fraction": np.mean(clip_fracs),
            "explained_variance": explained_var,
        })

        # 8. Logging (similar structure to your snippet)
        print(f"Training episode {episode}")
        print(
            f"Episodic Return: {np.mean(total_episodic_return[0])}"
        )  # zero sum game, so log only 0th agent returns
        print(f"Episode Length: {end_step}")
        print("")
        print(f"Value Loss: {last_v_loss}")
        print(f"Policy Loss: {last_pg_loss}")
        print(f"Old Approx KL: {np.mean(old_approx_kls)}")
        print(f"Approx KL: {np.mean(approx_kls)}")
        print(f"Clip Fraction: {np.mean(clip_fracs)}")
        print(f"Explained Variance: {explained_var}")
        print("\n-------------------------------------------\n")

    torch.save(agent.state_dict(), f"./checkpoints/model_ep{episode}.pt")
    # Upload to wandb
    wandb.save(f"model_ep{episode}.pt")
    run.finish()
    print("Training complete.")


train_ppo(
    LuxAICAECEnv(LuxAIS3GymEnv(numpy_output=True)),
    total_episodes=1000,
    rollout_num_steps=512,
)
