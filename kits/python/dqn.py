import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from collections import deque
from enum import IntEnum
from scipy.signal import convolve2d
import wandb
from luxai_s3.wrappers import LuxAIS3GymEnv
from lux.utils import direction_to

# Global constants
SPACE_SIZE   = 24
FRAME_FACTOR = None  # will be set in Space.__init__

# ——— GPU/CPU device selection ———
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
else:
    torch.manual_seed(42)

class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity    = capacity
        self.buffer      = deque(maxlen=capacity)
        self.priorities  = deque(maxlen=capacity)
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self.frame       = 1

    def push(self, state, action, reward, next_state, done):
        max_p = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(float(max_p))

    def sample(self, batch_size):
        beta = min(
            1.0,
            self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames
        )
        self.frame += 1

        prio_arr = np.array(self.priorities, dtype=np.float32)
        probs    = prio_arr ** self.alpha
        probs   /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        # put weights on the same device as the sample tensors
        device   = samples[0][0].device
        weights  = torch.as_tensor(weights, dtype=torch.float32, device=device)
        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            p_val = (abs(err) + 1e-5) ** self.alpha
            self.priorities[idx] = float(p_val)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, input_channels, action_size, local_view_size, hidden_size=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        # figure out conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, local_view_size, local_view_size)
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            conv_output_size = int(x.numel())

        self.fc_shared = nn.Linear(conv_output_size, hidden_size)

        self.value_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.value_fc2 = nn.Linear(hidden_size // 2, 1)
        self.adv_fc1   = nn.Linear(hidden_size, hidden_size // 2)
        self.adv_fc2   = nn.Linear(hidden_size // 2, action_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(obs))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))

        v = F.relu(self.value_fc1(x))
        v = self.value_fc2(v)

        a = F.relu(self.adv_fc1(x))
        a = self.adv_fc2(a)

        a_mean = a.mean(dim=1, keepdim=True)
        q      = v + (a - a_mean)
        return q


class Global:
    MAX_UNITS                           = 16
    RELIC_REWARD_RANGE                 = 5
    ALL_RELICS_FOUND                   = False
    ALL_REWARDS_FOUND                  = False
    LAST_MATCH_WHEN_RELIC_CAN_APPEAR   = 10
    LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR = 0
    REWARD_RESULTS                     = []


class NodeType(IntEnum):
    unknown  = -1
    empty    = 0
    nebula   = 1
    asteroid = 2


_xs, _ys = np.meshgrid(np.arange(SPACE_SIZE),
                       np.arange(SPACE_SIZE),
                       indexing="ij")
OPP_X = SPACE_SIZE - 1 - _xs
OPP_Y = SPACE_SIZE - 1 - _ys

NEIGHBOR_OFFSETS = {
    r: [(dx,dy) for dx in range(-r, r+1) for dy in range(-r, r+1)]
    for r in range(Global.RELIC_REWARD_RANGE + 1)
}


def warp_point(x, y):         return (x % SPACE_SIZE, y % SPACE_SIZE)
def get_opposite(x, y):       return (SPACE_SIZE - 1 - x, SPACE_SIZE - 1 - y)
def manhattan_distance(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])


ACTION_DIRS = [(0,0),(0,-1),(1,0),(0,1),(-1,0),(0,0)]
class ActionType(IntEnum):
    center = 0; up = 1; right = 2; down = 3; left = 4; sap = 5
    def to_direction(self): return ACTION_DIRS[self.value]
    @classmethod
    def from_coordinates(cls, cur, nxt):
        dx, dy = nxt[0]-cur[0], nxt[1]-cur[1]
        for v, (mx, my) in enumerate(ACTION_DIRS[:5]):
            if (dx,dy) == (mx,my):
                return cls(v)
        return cls.center

MIRROR_ACTION = {
    ActionType.center.value: ActionType.center.value,
    ActionType.up.value:     ActionType.down.value,
    ActionType.down.value:   ActionType.up.value,
    ActionType.left.value:   ActionType.right.value,
    ActionType.right.value:  ActionType.left.value,
}


class Node:
    __slots__ = ("x","y","type","energy","is_visible",
                 "_relic","_reward","_explored_for_relic","_explored_for_reward")
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.type      = NodeType.unknown
        self.energy    = None
        self.is_visible= False
        self._relic               = False
        self._reward              = False
        self._explored_for_relic  = False
        self._explored_for_reward = False

    @property
    def coordinates(self): return (self.x, self.y)
    @property
    def relic(self):       return self._relic
    @property
    def reward(self):      return self._reward
    @property
    def explored_for_relic(self):  return self._explored_for_relic
    @property
    def explored_for_reward(self): return self._explored_for_reward

    def update_relic_status(self, status):
        if status is None:
            self._explored_for_relic = False
            return
        if self._explored_for_relic and self._relic and not status:
            raise ValueError(f"Cannot flip relic at {self.coordinates}")
        self._relic              = status
        self._explored_for_relic = True

    def update_reward_status(self, status):
        if status is None:
            self._explored_for_reward = False
            return
        if self._explored_for_reward and self._reward and not status:
            return
        self._reward               = status
        self._explored_for_reward  = True


class Ship:
    def __init__(self, uid):
        self.unit_id = uid
        self.clean()
    def clean(self):
        self.energy = None
        self.node   = None
        self.task   = None
        self.target = None
        self.action = None
    @property
    def coordinates(self):
        return self.node.coordinates if self.node else None


class Fleet:
    def __init__(self, team_id):
        self.team_id = team_id
        self.points  = 0
        self.ships   = [Ship(i) for i in range(Global.MAX_UNITS)]

    def clear(self):
        self.points = 0
        for s in self.ships:
            s.clean()

    def update(self, obs, space):
        self.points = int(obs["team_points"][self.team_id])
        for ship, active, pos, energy in zip(
            self.ships,
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        ):
            if active:
                ship.node   = space.get_node(*pos)
                ship.energy = int(energy)
                ship.action = None
            else:
                ship.clean()

class Space:
    def __init__(self, max_steps):
        global FRAME_FACTOR
        FRAME_FACTOR = max_steps + 1

        # map arrays
        self.tile_type       = np.full((SPACE_SIZE,SPACE_SIZE),
                                       NodeType.unknown.value,
                                       dtype=np.int8)
        self.visible         = np.zeros((SPACE_SIZE,SPACE_SIZE), dtype=bool)
        self.energy          = np.full((SPACE_SIZE,SPACE_SIZE),
                                       -1, dtype=np.int32)
        # relic & reward
        self.relic_mask      = np.zeros((SPACE_SIZE,SPACE_SIZE), dtype=bool)
        self.reward_mask     = np.zeros((SPACE_SIZE,SPACE_SIZE), dtype=bool)
        self.explored_relic  = np.zeros((SPACE_SIZE,SPACE_SIZE), dtype=bool)
        self.explored_reward = np.zeros((SPACE_SIZE,SPACE_SIZE), dtype=bool)

        # Node grid
        self._nodes = [
            [Node(x, y) for x in range(SPACE_SIZE)]
            for y in range(SPACE_SIZE)
        ]

        # obstacle drift
        self.obstacle_map     = np.zeros((SPACE_SIZE,SPACE_SIZE), dtype=bool)
        self.obstacle_history = deque(maxlen=100)
        self.best_direction   = (0, 0)
        self.best_period      = 40
        self._movement_counter= 0

    def get_node(self, x, y):
        return self._nodes[y][x]

    def update(self, step, obs, team_id, team_reward):
        self._update_map(step, obs)
        self._update_relic_map(step, obs, team_id, team_reward)

    def _update_map(self, step, obs):
        sensor = obs["sensor_mask"]
        tiles  = obs["map_features"]["tile_type"]
        ener   = obs["map_features"]["energy"]

        self.visible[:]            = sensor
        self.tile_type[sensor]     = tiles[sensor]
        self.energy   [sensor]     = ener[sensor]
        self.energy   [~sensor]    = -1
        self.tile_type[OPP_X,OPP_Y]= self.tile_type
        self.energy   [OPP_X,OPP_Y]= self.energy
        self.visible  [OPP_X,OPP_Y]= self.visible

        ys,xs = np.nonzero(sensor)
        for x,y in zip(xs,ys):
            n = self.get_node(x,y)
            n.is_visible = True
            n.type       = NodeType(int(self.tile_type[x,y]))
            n.energy     = int(self.energy[x,y])
            ox,oy = get_opposite(x,y)
            m = self.get_node(ox,oy)
            m.is_visible = True
            m.type       = n.type
            m.energy     = n.energy

        ys,xs = np.nonzero(~sensor)
        for x,y in zip(xs,ys):
            n = self.get_node(x,y)
            n.is_visible = False
            n.energy     = None

        self.obstacle_history.append(self.obstacle_map.copy())
        self.obstacle_map = (self.tile_type == NodeType.asteroid.value)

        if self._movement_counter >= self.best_period:
            self._learn_obstacle_drift()
            self._movement_counter = 0

        self._apply_obstacle_drift(step)

    def _learn_obstacle_drift(self):
        best_score=-1.0
        for per in (10,20,40):
            if len(self.obstacle_history) < per: continue
            past = self.obstacle_history[-per]
            for d in ((1,0),(-1,0),(0,1),(0,-1)):
                rolled = np.roll(past, shift=d, axis=(0,1))
                score  = float((rolled == self.obstacle_map).mean())
                if score > best_score:
                    best_score,best_dir,best_per = score,d,per
        if best_score > 0.5:
            self.best_direction = best_dir
            self.best_period    = best_per

    def _apply_obstacle_drift(self, step):
        if self.best_period <= 0: return
        phn = (step / self.best_period) % 1.0
        pho = ((step-1) / self.best_period) % 1.0
        if pho > phn:
            dx,dy = self.best_direction
            self.obstacle_map = np.roll(
                self.obstacle_map, shift=(dx,dy), axis=(0,1)
            )

    def _update_relic_map(self, step, obs, team_id, team_reward):
        for mask, (x,y) in zip(obs["relic_nodes_mask"], obs["relic_nodes"]):
            if mask:
                ox,oy = get_opposite(x,y)
                for xx,yy in ((x,y),(ox,oy)):
                    self.relic_mask[xx,yy]     = True
                    self.explored_relic[xx,yy] = True
                for nx,ny in self.get_neighbors(x,y, Global.RELIC_REWARD_RANGE):
                    self.explored_reward[nx,ny] = False
                    self.reward_mask[nx,ny]     = False

        newly_seen = self.visible & (~self.explored_relic)
        ys,xs = np.nonzero(newly_seen)
        for x,y in zip(xs,ys):
            ox,oy = get_opposite(x,y)
            for xx,yy in ((x,y),(ox,oy)):
                self.relic_mask[xx,yy]     = False
                self.explored_relic[xx,yy] = True

        Global.ALL_RELICS_FOUND  = self.explored_relic.all()
        Global.ALL_REWARDS_FOUND = self.explored_reward.all()

        match      = step // FRAME_FACTOR
        match_step = step % FRAME_FACTOR
        num_th     = 2 * min(match, Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR) + 1
        if (not Global.ALL_RELICS_FOUND and self.relic_mask.sum() >= num_th):
            Global.ALL_RELICS_FOUND = True
            unseen = ~self.explored_relic
            ys,xs = np.nonzero(unseen)
            for x,y in zip(xs,ys):
                ox,oy = get_opposite(x,y)
                for xx,yy in ((x,y),(ox,oy)):
                    self.relic_mask[xx,yy]     = False
                    self.explored_relic[xx,yy] = True

        if (not Global.ALL_REWARDS_FOUND and
            (match_step > Global.LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR
             or self.relic_mask.sum() >= num_th)):
            self._infer_from_relic_distribution()
            self._record_reward_result(obs, team_id, team_reward)
            self._infer_reward_status()

        for y in range(SPACE_SIZE):
            for x in range(SPACE_SIZE):
                n = self.get_node(x,y)
                n._relic               = bool(self.relic_mask[x,y])
                n._explored_for_relic  = bool(self.explored_relic[x,y])
                n._reward              = bool(self.reward_mask[x,y])
                n._explored_for_reward = bool(self.explored_reward[x,y])

    def _infer_from_relic_distribution(self):
        mask = (self.relic_mask | ~self.explored_relic).astype(np.int32)
        k    = 2 * Global.RELIC_REWARD_RANGE + 1
        conv = convolve2d(mask,
                          np.ones((k,k),dtype=np.int32),
                          mode="same", boundary="fill", fillvalue=0)
        ys,xs = np.nonzero(conv == 0)
        for x,y in zip(xs,ys):
            ox,oy = get_opposite(x,y)
            for xx,yy in ((x,y),(ox,oy)):
                self.reward_mask[xx,yy]     = False
                self.explored_reward[xx,yy] = True

    def _record_reward_result(self, obs, team_id, team_reward):
        ship_positions = set()
        for active, energy, pos in zip(
            obs["units_mask"][team_id],
            obs["units"]["energy"][team_id],
            obs["units"]["position"][team_id],
        ):
            if active and energy >= 0:
                ship_positions.add(tuple(pos))
        Global.REWARD_RESULTS.append({
            "nodes":  ship_positions,
            "reward": int(team_reward)
        })

    def _infer_reward_status(self):
        for result in Global.REWARD_RESULTS:
            known = 0
            unknown = set()
            for (x,y) in result["nodes"]:
                if self.explored_reward[x,y] and not self.reward_mask[x,y]:
                    continue
                if self.reward_mask[x,y]:
                    known += 1
                else:
                    unknown.add((x,y))
            if not unknown:
                continue
            rem = result["reward"] - known
            if rem == 0:
                for x,y in unknown:
                    ox,oy = get_opposite(x,y)
                    for xx,yy in ((x,y),(ox,oy)):
                        self.reward_mask[xx,yy]     = False
                        self.explored_reward[xx,yy] = True
            elif rem == len(unknown):
                for x,y in unknown:
                    ox,oy = get_opposite(x,y)
                    for xx,yy in ((x,y),(ox,oy)):
                        self.reward_mask[xx,yy]     = True
                        self.explored_reward[xx,yy] = True

    def get_neighbors(self, x, y, r):
        for dx,dy in NEIGHBOR_OFFSETS[r]:
            yield warp_point(x+dx, y+dy)

TARGET_UPDATE = 4000
TAU_SOFT      = 0.005

class Agent:
    def __init__(self, player: str, env_cfg: dict, training: bool = True) -> None:
        self.player       = player
        self.team_id      = 0 if player == "player_0" else 1
        self.opp_team_id  = 1 - self.team_id
        self.env_cfg      = env_cfg
        self.training     = training
    
        self.map_width    = env_cfg["map_width"]
        self.map_height   = env_cfg["map_height"]
        self.max_steps    = env_cfg["max_steps_in_match"]
        self.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
    
        self.num_matches = 5
    
        self.space       = Space(self.max_steps)
        self.fleet       = Fleet(self.team_id)
        self.opp_fleet   = Fleet(self.opp_team_id)
    
        # DQN hyperparameters
        self.local_view_size = 11
        self.input_channels  = 9 + self.num_matches
        self.action_size     = 5
        self.hidden_size     = 512
        self.batch_size      = 128
        self.gamma           = 0.99
        self.epsilon_start   = 1.0
        self.epsilon_min     = 0.15
        self.epsilon_decay   = 0.9998
        self.learning_rate   = 1e-4
        self.replay_capacity = 50_000
        self.beta_frames     = 200_000
        self.frame_count     = 1
    
        # store device on agent and move nets there
        self.device     = device
        self.policy_net = DQN(
            self.input_channels,
            self.action_size,
            self.local_view_size,
            self.hidden_size
        ).to(self.device)
        self.target_net = DQN(
            self.input_channels,
            self.action_size,
            self.local_view_size,
            self.hidden_size
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
    
        # optimizer, scheduler, replay buffer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        self.scheduler = StepLR(self.optimizer, step_size=10_000, gamma=0.1)
        self.memory    = ReplayBuffer(self.replay_capacity,
                                      beta_frames=self.beta_frames)
        self.epsilon   = self.epsilon_start if training else 0.0
    
        # count‑based exploration map
        self.visit_counts = np.zeros((self.map_width, self.map_height), dtype=int)

        if not training:
            self.load_model()

    def _state_representation(self, unit_pos, unit_energy, step, obs):
        L = self.local_view_size
        R = L // 2
        xs = (np.arange(-R, R+1) + unit_pos[0]) % self.map_width
        ys = (np.arange(-R, R+1) + unit_pos[1]) % self.map_height
    
        tile_v   = self.space.tile_type[np.ix_(xs, ys)].astype(np.float32) / 3.0
        en_v     = self.space.energy   [np.ix_(xs, ys)].astype(np.float32) / 100.0
        relic_v  = self.space.relic_mask   [np.ix_(xs, ys)].astype(np.float32)
        reward_v = self.space.reward_mask  [np.ix_(xs, ys)].astype(np.float32)
        explored_rel = self.space.explored_relic[np.ix_(xs, ys)]
    
        units_v = np.zeros((L, L), dtype=np.float32)
        for ship in self.fleet.ships:
            if ship.node:
                dx = (ship.node.x - unit_pos[0]) % self.map_width
                dy = (ship.node.y - unit_pos[1]) % self.map_height
                if dx <= R or dx >= self.map_width - R - 1:
                    if dy <= R or dy >= self.map_height - R - 1:
                        xi, yi = (dx + R) % L, (dy + R) % L
                        units_v[xi, yi] = ship.energy / 100.0
        for opp in self.opp_fleet.ships:
            if opp.node:
                dx = (opp.node.x - unit_pos[0]) % self.map_width
                dy = (opp.node.y - unit_pos[1]) % self.map_height
                if dx <= R or dx >= self.map_width - R - 1:
                    if dy <= R or dy >= self.map_height - R - 1:
                        xi, yi = (dx + R) % L, (dy + R) % L
                        units_v[xi, yi] = -opp.energy / 100.0
    
        dist_k    = np.fromfunction(lambda i,j: np.abs(i-R)+np.abs(j-R), (L,L), dtype=np.float32)
        mask_unexp= (~explored_rel).astype(np.float32)
        dr        = convolve2d(mask_unexp, np.ones((1,1)), mode="same", boundary="wrap")
        dist_relic = np.where(dr == 0, 1.0, dist_k / (self.map_width + self.map_height))
    
        mask_rw    = reward_v.astype(bool)
        dist_reward= np.ones((L,L), dtype=np.float32)
        denom      = self.map_width + self.map_height
        I,J        = np.indices((L,L))
        rows, cols = np.where(mask_rw)
        for r,c in zip(rows, cols):
            d = (np.abs(I-r) + np.abs(J-c)) / denom
            dist_reward = np.minimum(dist_reward, d)
    
        ue = np.full((L,L), unit_energy/100.0, dtype=np.float32)
        sp = np.full((L,L), step/FRAME_FACTOR,    dtype=np.float32)
    
        base = np.stack([
            tile_v, en_v, relic_v, reward_v,
            units_v, ue, sp,
            dist_relic, dist_reward
        ], axis=0)
    
        match = step // FRAME_FACTOR
        phase_oh = np.zeros((self.num_matches, L, L), dtype=np.float32)
        if match < self.num_matches:
            phase_oh[match, :, :] = 1.0
    
        state_np = np.concatenate([base, phase_oh], axis=0)
        return torch.from_numpy(state_np).float().to(self.device)

    def _get_action_mask(self, unit_pos):
        mask = np.ones(self.action_size, dtype=np.int8)
        x,y = unit_pos
        if x == 0:                mask[ActionType.left]  = 0
        if x == self.map_width-1: mask[ActionType.right] = 0
        if y == 0:                mask[ActionType.up]    = 0
        if y == self.map_height-1:mask[ActionType.down]  = 0
        return mask

    def try_sap_on_enemies(self, ship):
        targets = []
        for opp in self.opp_fleet.ships:
            if opp.node:
                d = manhattan_distance(ship.coordinates, opp.coordinates)
                if d <= self.UNIT_SAP_RANGE:
                    targets.append((opp.energy - d, opp))
        if not targets:
            return False
        _, best = max(targets, key=lambda x: x[0])
        dx = best.coordinates[0] - ship.coordinates[0]
        dy = best.coordinates[1] - ship.coordinates[1]
        ship.action = (ActionType.sap, dx, dy)
        return True

    def act(self, step, obs, remainingOverageTime=60):
        pts = int(obs["team_points"][self.team_id])
        rew = max(0, pts - self.fleet.points)

        self.space.update(step, obs, self.team_id, rew)
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)

        for ship in self.fleet.ships:
            if ship.node:
                x,y = ship.coordinates
                self.visit_counts[x,y] += 1

        acts = []
        for ship in self.fleet.ships:
            if ship.node is None:
                acts.append((0,0,0))
                continue
            if self.try_sap_on_enemies(ship):
                acts.append(ship.action)
                continue

            state = self._state_representation(
                ship.coordinates, ship.energy, step, obs
            )
            mask  = self._get_action_mask(ship.coordinates)

            if random.random() < self.epsilon and self.training:
                valid = np.where(mask==1)[0]
                a     = int(random.choice(valid)) if valid.size else 0
            else:
                with torch.no_grad():
                    q = self.policy_net(state.unsqueeze(0))
                    mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device)
                    inv    = (1.0 - mask_t) * -1e9
                    a      = int((q + inv).argmax().item())

            acts.append((a,0,0))

        return np.array(acts)

    def learn(self):
        if not self.training or len(self.memory) < self.batch_size:
            return None

        batch, idxs, wts = self.memory.sample(self.batch_size)
        self.frame_count += 1

        states, actions, rewards, next_states, dones = zip(*batch)
        # move all tensors to device
        states      = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions     = torch.as_tensor(actions, dtype=torch.long, device=self.device)\
                          .clamp(0, self.action_size-1)
        rewards     = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones       = torch.tensor([float(d) for d in dones],
                                  dtype=torch.float32, device=self.device)
        weights     = wts.unsqueeze(1)  # already on device

        q_vals = self.policy_net(states)
        q_sa   = q_vals.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_a = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_a)
            target = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * self.gamma * next_q

        td_error = target - q_sa
        self.memory.update_priorities(idxs, td_error.abs().detach().cpu().numpy())

        loss = (weights * F.smooth_l1_loss(q_sa, target, reduction="none")).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        if TAU_SOFT is not None:
            for tp, pp in zip(self.target_net.parameters(),
                              self.policy_net.parameters()):
                tp.data.mul_(1-TAU_SOFT).add_(TAU_SOFT * pp.data)
        elif self.frame_count % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

        return {
            "loss":    loss.item(),
            "td_mean": td_error.abs().mean().item(),
            "lr":      self.optimizer.param_groups[0]["lr"],
            "eps":     self.epsilon,
            "frame":   self.frame_count,
        }

    def save_model(self):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "frame_count":self.frame_count,
            "epsilon":    self.epsilon,
        }, f"dqn_{self.player}.pth")

    def load_model(self):
        try:
            ckpt = torch.load(f"dqn_{self.player}.pth",
                              map_location="cpu", weights_only=True)
            self.policy_net.load_state_dict(ckpt["policy_net"])
            self.target_net.load_state_dict(ckpt["target_net"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.frame_count = ckpt.get("frame_count", self.frame_count)
            self.epsilon     = ckpt.get("epsilon", self.epsilon)
            self.target_net.eval()
        except FileNotFoundError:
            pass

class SimpleAgent:
    def __init__(self, player: str, env_cfg: dict, training: bool = False):
        self.player  = player
        self.team_id = 0 if player=="player_0" else 1
        self.env_cfg = env_cfg

    def act(self, step: int, obs: dict, remainingOverageTime: int = 60):
        actions = []
        for uid in range(self.env_cfg["max_units"]):
            if obs["units_mask"][self.team_id][uid]:
                x,y = obs["units"]["position"][self.team_id][uid]
                moves=[]
                if x>0: moves.append(ActionType.left)
                if x<self.env_cfg["map_width"]-1: moves.append(ActionType.right)
                if y>0: moves.append(ActionType.up)
                if y<self.env_cfg["map_height"]-1: moves.append(ActionType.down)
                moves.append(ActionType.center)
                a = random.choice(moves).value
                actions.append((a,0,0))
            else:
                actions.append((0,0,0))
        return np.array(actions)

class BaseAgent:
    def __init__(self, player: str, env_cfg: dict, training: bool = False):
        self.player  = player
        self.team_id = 0 if player=="player_0" else 1
        self.env_cfg = env_cfg
        self._seen_relic_ids  = set()
        self._relic_positions = []
        self._explore_targets = {}

    def act(self, step: int, obs: dict, remainingOverageTime: int = 60):
        max_u = self.env_cfg["max_units"]
        actions = np.zeros((max_u,3), dtype=int)
        unit_mask = obs["units_mask"][self.team_id]
        unit_pos  = obs["units"]["position"][self.team_id]
        relic_mask= obs["relic_nodes_mask"]
        relics    = obs["relic_nodes"]

        for ridx, seen in enumerate(relic_mask):
            if seen and ridx not in self._seen_relic_ids:
                self._seen_relic_ids.add(ridx)
                self._relic_positions.append(tuple(relics[ridx]))

        for u in range(max_u):
            if not unit_mask[u]:
                continue
            src = tuple(unit_pos[u])
            if self._relic_positions:
                dest = self._relic_positions[0]
                dx,dy = dest[0]-src[0], dest[1]-src[1]
                dist  = abs(dx)+abs(dy)
                if dist<=4:
                    move = np.random.randint(0,5)
                else:
                    if abs(dx)>abs(dy):
                        move = ActionType.right.value if dx>0 else ActionType.left.value
                    else:
                        move = ActionType.down.value  if dy>0 else ActionType.up.value
            else:
                if (step%20==0) or (u not in self._explore_targets):
                    tx = np.random.randint(0,self.env_cfg["map_width"])
                    ty = np.random.randint(0,self.env_cfg["map_height"])
                    self._explore_targets[u] = (tx,ty)
                dest = self._explore_targets[u]
                dx,dy = dest[0]-src[0], dest[1]-src[1]
                if (dx,dy)==(0,0):
                    move = ActionType.center.value
                else:
                    if abs(dx)>abs(dy):
                        move = ActionType.right.value if dx>0 else ActionType.left.value
                    else:
                        move = ActionType.down.value  if dy>0 else ActionType.up.value

            actions[u] = (move,0,0)

        return actions


class CollectorAgent:
    def __init__(self, player: str, env_cfg: dict, training: bool = False):
        self.player    = player
        self.team_id   = 0 if player=="player_0" else 1
        self.env_cfg   = env_cfg
        self.training  = training
        self.neighbors = [] 

    def act(self, step: int, obs: dict, remainingOverageTime: int = 60):
        for mask, (rx, ry) in zip(obs["relic_nodes_mask"], obs["relic_nodes"]):
            if mask:
                self.neighbors = [
                    warp_point(rx + dx, ry + dy)
                    for dx, dy in NEIGHBOR_OFFSETS[2]
                ]
                break

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        for uid in range(self.env_cfg["max_units"]):
            if not obs["units_mask"][self.team_id][uid]:
                continue
            x, y = obs["units"]["position"][self.team_id][uid]
            if self.neighbors:
                src = np.array((x, y))
                tgt = np.array(random.choice(self.neighbors))
                dir_idx = direction_to(src, tgt)
                actions[uid] = (int(dir_idx), 0, 0)
            else:
                actions[uid] = (random.randint(0, 4), 0, 0)

        return actions

import json
from IPython.display import display, Javascript
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

def render_episode(episode: RecordEpisode) -> None:
    data = json.dumps(episode.serialize_episode_data(), separators=(",", ":"))
    display(Javascript(f"""
var iframe = document.createElement('iframe');
iframe.src = 'https://s3vis.lux-ai.org/#/kaggle';
iframe.width = '100%';
iframe.scrolling = 'no';

iframe.addEventListener('load', event => {{
    event.target.contentWindow.postMessage({data}, 'https://s3vis.lux-ai.org');
}});

new ResizeObserver(entries => {{
    for (const entry of entries) {{
        entry.target.height = `${{Math.round(320 + 0.3 * entry.contentRect.width)}}px`;
    }}
}}).observe(iframe);

element.append(iframe);
    """))

def validate(agent_0, agent_1, env_cfg, num_games=3):
    env = LuxAIS3GymEnv(numpy_output=True)
    total_r0 = 0.0
    total_r1 = 0.0

    for _ in range(num_games):
        obs, _ = env.reset(seed=random.randint(0, 100_000))
        done = False
        step = 0
        r0 = 0.0
        r1 = 0.0

        while not done:
            actions = {}
            for ag in (agent_0, agent_1):
                if obs[ag.player] is not None:
                    actions[ag.player] = ag.act(step, obs[ag.player])
                else:
                    actions[ag.player] = np.zeros((env_cfg["max_units"], 3), dtype=int)

            obs, rewards, terminated, truncated, _ = env.step(actions)

            r0 = max(r0, rewards.get("player_0", 0.0))
            r1 = max(r1, rewards.get("player_1", 0.0))
                          
            done = (all(terminated.values()) or all(truncated.values()))
            step += 1
            
        total_r0 += r0
        total_r1 += r1

    env.close()
    return total_r0, total_r1



def evaluate_agents(agent_cls_p0, agent_cls_p1,
                    seed=42, training=True,
                    games_to_play=500, validate_every=10):
    wandb.init(
        project="dqn_luxai_s3_gpu",
        config={
            "seed": seed,
            "games_to_play": games_to_play,
            "validate_every": validate_every,
            "batch_size": 128,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.9995,
            "learning_rate": 1e-4,
            "replay_buffer_capacity": 50_000,
            "beta_frames": 200_000,
        }
    )
    cfg = wandb.config

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg.seed)

    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset(seed=cfg.seed)
    env_cfg = info["params"]

    p0 = agent_cls_p0("player_0", env_cfg, training=training)
    p1 = agent_cls_p1("player_1", env_cfg, training=training)
    # ensure nets on correct device
    p0.policy_net.to(device); p0.target_net.to(device)
    p1.policy_net.to(device); p1.target_net.to(device)

    for ep in range(games_to_play):
        obs, _ = env.reset(seed=seed + ep)
        done     = False
        step     = 0
        last_obs = None
        last_act = None
        ep_r0 = ep_r1 = 0.0

        p0_losses    = []
        p0_td_means  = []
        p1_losses    = []
        p1_td_means  = []

        while not done:
            if training:
                last_obs = {
                    "player_0": obs["player_0"].copy() if obs["player_0"] else None,
                    "player_1": obs["player_1"].copy() if obs["player_1"] else None,
                }

            actions = {}
            for ag in (p0, p1):
                if obs[ag.player] is not None:
                    actions[ag.player] = ag.act(step, obs[ag.player])
                else:
                    actions[ag.player] = np.zeros((env_cfg["max_units"],3), dtype=int)
            if training:
                last_act = actions.copy()

            obs, rewards, term, trunc, _ = env.step(actions)
            dones = {
                "player_0": term["player_0"] or trunc["player_0"],
                "player_1": term["player_1"] or trunc["player_1"],
            }
            done = all(term.values()) or all(trunc.values())

            ep_r0 = max(ep_r0, rewards.get("player_0", 0.0))
            ep_r1 = max(ep_r1, rewards.get("player_1", 0.0))

            if training and last_obs is not None:
                match = step // FRAME_FACTOR

                for ag in (p0, p1):
                    tid = ag.team_id
                    for uid in range(env_cfg["max_units"]):
                        if not (last_obs[ag.player]
                                and last_obs[ag.player]["units_mask"][tid][uid]):
                            continue

                        s = ag._state_representation(
                            last_obs[ag.player]["units"]["position"][tid][uid],
                            last_obs[ag.player]["units"]["energy"][tid][uid],
                            step,
                            last_obs[ag.player]
                        )
                        if obs[ag.player] and obs[ag.player]["units_mask"][tid][uid]:
                            s_next = ag._state_representation(
                                obs[ag.player]["units"]["position"][tid][uid],
                                obs[ag.player]["units"]["energy"][tid][uid],
                                step + 1,
                                obs[ag.player]
                            )
                        else:
                            s_next = torch.zeros_like(s)

                        base_r = 0.0
                        a      = last_act[ag.player][uid][0]

                        x,y = last_obs[ag.player]["units"]["position"][tid][uid]
                        node = ag.space.get_node(x, y)
                        explore_bonus  = 0.0
                        if not node.explored_for_relic:
                            explore_bonus += 5.0
                        if not node.explored_for_reward:
                            explore_bonus += 2.5
                        vis = ag.visit_counts[x, y]
                        explore_bonus += (10.0 / np.sqrt(vis)) if vis > 0 else 0.5

                        exploit_bonus = 6.0 if node.reward else 0.0

                        if match < 2:
                            intrinsic = explore_bonus * 1.5 + exploit_bonus * 0.5
                        else:
                            intrinsic = exploit_bonus * 1.5 + explore_bonus * 0.125

                        if (step + 1) % FRAME_FACTOR == 0:
                            intrinsic += 1.0 

                        r_shaped = (base_r + intrinsic)

                        ag.memory.push(s, a, r_shaped, s_next, dones[ag.player])

                        s_np    = s.cpu().numpy()
                        s_flip  = s_np[:, ::-1, ::-1].copy()
                        s_flip_t= torch.from_numpy(s_flip).to(ag.device)
                        ns_np   = s_next.cpu().numpy()
                        ns_flip = ns_np[:, ::-1, ::-1].copy()
                        ns_flip_t= torch.from_numpy(ns_flip).to(ag.device)
                        a_flip  = MIRROR_ACTION.get(a, a)
                        ag.memory.push(s_flip_t, a_flip, r_shaped, ns_flip_t, dones[ag.player])

                stats0 = p0.learn()
                stats1 = p1.learn()
                if stats0:
                    p0_losses.append(stats0["loss"])
                    p0_td_means.append(stats0["td_mean"])
                if stats1:
                    p1_losses.append(stats1["loss"])
                    p1_td_means.append(stats1["td_mean"])

            step += 1

        log_data = {
            "episode":    ep,
            "ep_reward_p0": ep_r0,
            "ep_reward_p1": ep_r1,
        }
        if p0_losses:
            log_data.update({
                "avg_loss_p0":   np.mean(p0_losses),
                "avg_td_p0":     np.mean(p0_td_means),
                "final_eps_p0":  p0.epsilon,
            })
        if p1_losses:
            log_data.update({
                "avg_loss_p1":   np.mean(p1_losses),
                "avg_td_p1":     np.mean(p1_td_means),
                "final_eps_p1":  p1.epsilon,
            })

        for ag in (p0, p1):
            if ag.epsilon == ag.epsilon_min:
                ag.epsilon = 0.2
            else:
                ag.epsilon = min(
                    ag.epsilon_start,
                    ag.epsilon * (ag.epsilon_decay ** (-(FRAME_FACTOR // 2)))
                )

        wandb.log(log_data)

        if training and (ep + 1) % validate_every == 0:
            baseAgent1 = BaseAgent("player_1", env_cfg, training=False)
            val_p0_vsBase, _ = validate(p0, baseAgent1, env_cfg)
            baseAgent0 = BaseAgent("player_0", env_cfg, training=False)
            _, val_p1_vsBase = validate(baseAgent0, p1, env_cfg)
            wandb.log({
                "val_p0_vsBase": val_p0_vsBase,
                "val_p1_vsBase": val_p1_vsBase,
            })

        if training and (ep + 1) % validate_every == 0:
            p0.save_model()
            p1.save_model()

        if training and (ep + 1) % (50) == 0:
            #--- record one match
            rec_env = RecordEpisode(
                LuxAIS3GymEnv(numpy_output=True),
                save_on_close=False,
                save_on_reset=False,
                save_dir=None
            )
            obs_r, _ = rec_env.reset(seed=cfg.seed + ep)
            done_r, step_r = False, 0
            baseAgent = BaseAgent("player_1", env_cfg, training=False)
            while not done_r:
                acts = {}
                for ag in (p0, baseAgent):
                    if obs_r[ag.player] is not None:
                        acts[ag.player] = ag.act(step_r, obs_r[ag.player])
                    else:
                        acts[ag.player] = np.zeros((env_cfg["max_units"],3), dtype=int)
                obs_r, _, term_r, trunc_r, _ = rec_env.step(acts)
                done_r = term_r["player_0"] or trunc_r["player_0"] or term_r["player_1"] or trunc_r["player_1"]
                step_r += 1

            render_episode(rec_env)
            rec_env.close()

    env.close()
    if training:
        p0.save_model()
        p1.save_model()
    wandb.finish()


if __name__ == "__main__":
    evaluate_agents(
        agent_cls_p0=Agent,
        agent_cls_p1=Agent,
        seed=42,
        training=True,
        games_to_play=50,
        validate_every=10,
    )