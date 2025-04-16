import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import cv2
import random

# Disable OpenCL for OpenCV (optional, but can avoid some issues).
cv2.ocl.setUseOpenCL(False)


class RandomNoopWrapper(gym.Wrapper):
    """
    RandomNoopWrapper performs a random number of no-op (do nothing) actions on reset.
    It randomizes the starting state by executing between 1 and `noop_max` no-op actions.
    Assumes that action 0 is the no-op.
    """
    def __init__(self, env, noop_max=30):
        super(RandomNoopWrapper, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        # Check that the first action is 'NOOP'
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        # Gymnasium reset returns (obs, info)
        obs, info = self.env.reset(**kwargs)
        # Determine number of no-ops to perform.
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        for _ in range(noops):
            # Step returns 5 values: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


class FrameSkipMaxPoolWrapper(gym.Wrapper):
    """
    FrameSkipMaxPoolWrapper repeats an action for a fixed number of frames (skip)
    and returns the max-pooled observation from the last two frames.
    This reduces computational load and deals with flickering.
    """
    def __init__(self, env, skip=4):
        super(FrameSkipMaxPoolWrapper, self).__init__(env)
        self._skip = skip
        # Allocate a buffer for the last two observations.
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            if term or trunc:
                break
        # Max-pool over the last two frames.
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class RewardClipperWrapper(gym.RewardWrapper):
    """
    RewardClipperWrapper clips rewards to {-1, 0, +1} based on their sign.
    This helps stabilize training.
    """
    def __init__(self, env):
        super(RewardClipperWrapper, self).__init__(env)

    def reward(self, reward):
        return np.sign(reward)


class FrameWarper(gym.ObservationWrapper):
    """
    FrameWarper converts raw RGB frames to grayscale and resizes them to 84x84 pixels,
    as done in the Nature DQN paper.
    This wrapper now returns observations in channel-first format.
    """
    def __init__(self, env):
        super(FrameWarper, self).__init__(env)
        self.width = 84
        self.height = 84
        # Define observation space as (channels, height, width)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(1, self.height, self.width),
                                            dtype=np.uint8)

    def observation(self, frame):
        # Convert to grayscale.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize frame.
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # Add channel dimension at the beginning.
        return frame[None, :, :]


class FrameStacker(gym.Wrapper):
    """
    FrameStacker stacks the last k frames to provide temporal context.
    Returns a LazyFrames object to optimize memory usage.
    Assumes that the underlying observations are in channel-first format.
    """
    def __init__(self, env, k):
        super(FrameStacker, self).__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape  # Expected shape is (1, 84, 84)
        # New observation space: (k, height, width).
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(k, shp[1], shp[2]),
                                            dtype=np.uint8)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), info

    def step(self, action):
        ob, reward, term, trunc, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, term, trunc, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        # Concatenate along the channel dimension.
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    """
    LazyFrames optimize memory usage when stacking frames by concatenating
    them only when needed.
    """
    def __init__(self, frames):
        self._frames = frames

    def __array__(self, dtype=None, copy=True):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        if copy:
            out = out.copy()
        return out

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._frames[i]


class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until a FIRE action is taken.
    Typically used for Breakout.
    """
    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Take a FIRE action.
        obs, _, term, trunc, info = self.env.step(1)  # Typically action 1 is FIRE.
        if term or trunc:
            obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)

class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Helps the agent learn value of life loss, as in DeepMind DQN paper.
    """
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.truncated_done = True

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.was_real_done = done
        self.truncated_done = truncated
        # Check current number of lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # Treat as end of episode but don't reset env
            done = True
        self.lives = lives
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done or self.truncated_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Skip reset, do a no-op to advance from terminal state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info
