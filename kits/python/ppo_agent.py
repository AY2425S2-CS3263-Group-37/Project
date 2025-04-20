import sys
from enum import IntEnum
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from agent import Agent
from gymnasium import spaces

# from lux.config import EnvConfig
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = (
    dict()
)  # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs = dict()

ckpt_path = "/Users/jayanth/Desktop/AY2425S2/CS3263/Lux-Design-S3/kits/python/checkpoints/league/main_snapshot_ep2000_04-20_18-01.pt"

# modified from https://www.kaggle.com/code/yizhewang3/ppo-stable-baselines3
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

device = torch.device("cpu")


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
    for key in obs:
        # For each key, convert each player's observation to a tensor and stack
        stacked = []

        stacked.append(torch.tensor(obs[key], device=device))
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


def flatten_obs(base_obs, env_cfg, id: int):
    """
    Convert the multi-agent observation dictionary for the *current* player
    into the flattened dict structure you had.
    """

    # Sometimes there's an extra "obs" key; adapt as needed
    if "obs" in base_obs:
        base_obs = base_obs["obs"]

    flat_obs = {}
    flat_obs["pid"] = np.array([id], dtype=np.int8)
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


obs_space_dict = {
    "pid": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
    "units_position": spaces.Box(
        low=0, high=SPACE_SIZE - 1, shape=(NUM_TEAMS, MAX_UNITS, 2), dtype=np.int32
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
        low=-1, high=MAX_ENERGY_PER_TILE, shape=(SPACE_SIZE, SPACE_SIZE), dtype=np.int8
    ),
    "relic_nodes_mask": spaces.Box(
        low=0, high=1, shape=(MAX_RELIC_NODES,), dtype=np.int8
    ),
    "relic_nodes": spaces.Box(
        low=-1, high=SPACE_SIZE - 1, shape=(MAX_RELIC_NODES, 2), dtype=np.int32
    ),
    "team_points": spaces.Box(low=0, high=1000, shape=(NUM_TEAMS,), dtype=np.int32),
    "team_wins": spaces.Box(low=0, high=1000, shape=(NUM_TEAMS,), dtype=np.int32),
    "steps": spaces.Box(low=0, high=MAX_STEPS_IN_MATCH, shape=(1,), dtype=np.int32),
    "match_steps": spaces.Box(
        low=0, high=MAX_STEPS_IN_MATCH, shape=(1,), dtype=np.int32
    ),
    "env_cfg_map_width": spaces.Box(low=0, high=SPACE_SIZE, shape=(1,), dtype=np.int32),
    "env_cfg_map_height": spaces.Box(
        low=0, high=SPACE_SIZE, shape=(1,), dtype=np.int32
    ),
    "env_cfg_max_steps_in_match": spaces.Box(
        low=0, high=MAX_STEPS_IN_MATCH, shape=(1,), dtype=np.int32
    ),
    "env_cfg_unit_move_cost": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
    "env_cfg_unit_sap_cost": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
    "env_cfg_unit_sap_range": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
}
# Create a Dict space from that
obs_space_single = spaces.Dict(obs_space_dict)

# Example action space: MultiDiscrete for each possible unit, as in your code
act_space_single = spaces.MultiDiscrete(
    np.array([[len(ActionType), SPACE_SIZE, SPACE_SIZE]] * MAX_UNITS).flatten()
)


class Agent(nn.Module):
    def __init__(self, player: str, env_config: dict):
        super().__init__()
        self.player = player
        self.pid = 0 if self.player == "player_0" else 1
        self.game_count = 0
        self.sustain_count = 1
        self.previous_action = None
        # We assume env.observation_spaces["player_0"] and env.action_spaces["player_0"] exist
        self.net = MultiInputActorCriticPolicy(
            observation_space=obs_space_single,
            action_space=act_space_single,
            lr_schedule=MultiInputActorCriticPolicy._dummy_schedule,
        )
        self.env_config = env_config
        self.load_state_dict(torch.load(ckpt_path))

    @torch.inference_mode
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

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        self.game_count += 1
        if self.previous_action is None or self.game_count % self.sustain_count == 0:
            flat_obs = flatten_obs(obs, self.env_config, self.pid)

            # print(flat_obs, file=sys.stderr)
            obs = batchify_obs(flat_obs)
            # print(obs, file=sys.stderr)
            action, *_ = self.get_action_and_value(obs)
            # print(action, file=sys.stderr)
            # action = unbatchify_actions(action)
            action = np.array(action[0]).reshape(-1, 3)
            self.previous_action = action
            print(action, file=sys.stderr)
        return self.previous_action
