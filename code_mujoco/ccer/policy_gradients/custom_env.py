import os
import numpy as np
from PIL import Image
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box as Continuous
import gym
import random
from .torch_utils import RunningStat, ZFilter, Identity, StateWithTime, RewardFilter

class Env:
    '''
    A wrapper around the OpenAI gym environment that adds support for the following:
    - Rewards normalization
    - State normalization
    - Adding timestep as a feature with a particular horizon T
    Also provides utility functions/properties for:
    - Whether the env is discrete or continuous
    - Size of feature space
    - Size of action space
    Provides the same API (init, step, reset) as the OpenAI gym
    '''
    def __init__(self, game, norm_states, norm_rewards, params, add_t_with_horizon=None, clip_obs=None, clip_rew=None, 
            show_env=False, save_frames=False, save_frames_path=""):
        self.env = gym.make(game)
        clip_obs = None if clip_obs < 0 else clip_obs
        clip_rew = None if clip_rew < 0 else clip_rew

        # Environment type
        self.is_discrete = type(self.env.action_space) == Discrete
        assert self.is_discrete or type(self.env.action_space) == Continuous

        # Number of actions
        action_shape = self.env.action_space.shape
        assert len(action_shape) <= 1 # scalar or vector actions
        self.num_actions = self.env.action_space.n if self.is_discrete else 0 \
                            if len(action_shape) == 0 else action_shape[0]
        
        # Number of features
        assert len(self.env.observation_space.shape) == 1
        self.num_features = self.env.reset().shape[0]
        self.state_shape = self.env.reset().shape[0:3]

        # Support for state normalization or using time as a feature
        self.state_filter = Identity()
        if norm_states:
            self.state_filter = ZFilter(self.state_filter, shape=[*self.state_shape], \
                                            clip=clip_obs)
        if add_t_with_horizon is not None:
            self.state_filter = StateWithTime(self.state_filter, horizon=add_t_with_horizon)
        
        # Support for rewards normalization
        self.reward_filter = Identity()
        if norm_rewards == "rewards":
            self.reward_filter = ZFilter(self.reward_filter, shape=(), center=False, clip=clip_rew)
        elif norm_rewards == "returns":
            self.reward_filter = RewardFilter(self.reward_filter, shape=(), gamma=params.GAMMA, clip=clip_rew)
        # Running total reward (set to 0.0 at resets)
        self.total_true_reward = 0.0

        # Set normalizers to read-write mode by default.
        self._read_only = False

        self.setup_visualization(show_env, save_frames, save_frames_path)

    # For environments that are created from a picked object.
    def setup_visualization(self, show_env, save_frames, save_frames_path):
        self.save_frames = save_frames
        self.show_env = show_env
        self.save_frames_path = save_frames_path
        self.episode_counter = 0
        self.frame_counter = 0
        self.frames=[]

        if self.save_frames:
            print(f'We will save frames to {self.save_frames_path}!')
            os.makedirs(os.path.join(self.save_frames_path, "000"), exist_ok=True)
    
    @property
    def normalizer_read_only(self):
        return self._read_only

    @normalizer_read_only.setter
    def normalizer_read_only(self, value):
        self._read_only = bool(value)
        if isinstance(self.state_filter, ZFilter):
            if not hasattr(self.state_filter, 'read_only') and value:
                print('Warning: requested to set state_filter.read_only=True but the underlying ZFilter does not support it.')
            elif hasattr(self.state_filter, 'read_only'):
                self.state_filter.read_only = self._read_only
        if isinstance(self.reward_filter, ZFilter) or isinstance(self.reward_filter, RewardFilter):
            if not hasattr(self.reward_filter, 'read_only') and value:
                print('Warning: requested to set reward_filter.read_only=True but the underlying ZFilter does not support it.')
            elif hasattr(self.reward_filter, 'read_only'):
                self.reward_filter.read_only = self._read_only
    

    def reset(self):
        # Set a deterministic random seed for reproduicability
        self.env.seed(random.getrandbits(31))
        # Reset the state, and the running total reward
        start_state = self.env.reset()
        self.total_true_reward = 0.0
        self.counter = 0.0
        self.episode_counter += 1
        if self.save_frames:
            os.makedirs(os.path.join(self.save_frames_path, f"{self.episode_counter:03d}"), exist_ok=True)
            self.frame_counter = 0
        self.state_filter.reset()
        self.reward_filter.reset()
        return self.state_filter(start_state, reset=True)

    def step(self, action):
        state, reward, is_done, info = self.env.step(action)
        if self.show_env:
            self.env.render()
        # Frameskip (every 6 frames, will be rendered at 25 fps)
        if self.save_frames and int(self.counter) % 6 == 0:
            image = self.env.render(mode='rgb_array')
            self.ep_img_path = os.path.join(self.save_frames_path, f"{self.episode_counter:03d}.gif")
            image = Image.fromarray(image)
            self.frames.append(image)
            self.frame_counter += 1
        state = self.state_filter(state)
        self.total_true_reward += reward
        self.counter += 1
        _reward = self.reward_filter(reward)
        if is_done:
            if isinstance(info, bool):
                info = {}
            info['done'] = (self.counter, self.total_true_reward)
            if self.save_frames:
                self.frames[0].save(self.ep_img_path, save_all=True, append_images=self.frames, duration=100, loop=0)
        return state, _reward, is_done, info
    

class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, dtype=np.float32):
        super(FlattenObsWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.state_len = old_space.shape[0] * old_space.shape[1]
        self.observation_space = gym.spaces.Box(old_space.low[0][0].repeat(self.state_len,
                 axis=0),old_space.high[0][0].repeat(self.state_len, axis=0),
                 dtype=dtype)
    def reset(self, seed=0):
        self.buffer = np.zeros_like(self.observation_space.low,
        dtype=self.dtype)
        return self.observation(self.env.reset())
    def observation(self, observation):
        self.buffer = np.array([elem for row in observation for elem in row])
        return self.buffer

skip_rate = 4 
    
def atari_env(env_id):
    env = gym.make(env_id)
    if 'NoFrameskip' in env_id:
        assert 'NoFrameskip' in env.spec.id
        env._max_episode_steps = 10000 * skip_rate
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=skip_rate)
    else:
        env._max_episode_steps = 10000
    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env._max_episode_steps = 10000
    env = AtariRescale(env)
    return env


def process_frame(frame):
    frame = frame[34:34 + 160, :160]
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = resize(frame, (80, 80))
    frame = resize(frame, (80, 80))
    frame = np.reshape(frame, [1, 80, 80])
    return frame


class AtariRescale(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Continuous(0.0, 1.0, [1, 80, 80], dtype=np.uint8)

    def observation(self, observation):
        return process_frame(observation)



class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, self.was_real_done

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=3)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs

    
