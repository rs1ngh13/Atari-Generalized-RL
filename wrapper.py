import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import cv2
cv2.ocl.setUseOpenCL(False)

#Useful wrappers taken from OpenAI (https://github.com/openai/baselines)
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super(NoopResetEnv, self).__init__(env) 
        self.noop_max = noop_max                                    #max number of no-ops when reset
        self.override_num_noops = None                              #override
        self.noop_action = 0                                        #action index for "NOOP"
        #assert env.unwrapped.get_action_meanings()[0] == 'NOOP'     #TRY TO SEE IF THIS LINE IS REDUNDANT

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)                        #reset the env
        if self.override_num_noops is not None:                     #determine how many no-ops to do
            noops = self.override_num_noops                         #none
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)     #do anywhere from 1 - to max
        assert noops > 0
        for _ in range(noops):                                      #go through # of no-ops
            obs, reward, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info                                            #output final observation 

    def step(self, action):
        return self.env.step(action)                                #send through to the env


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._skip = skip                                           #how many frames will be skipped
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)     #buffer for the last 2 frames

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action): 
        total_reward = 0.0          #reward over the skips
        terminated = False
        truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            if i == self._skip - 2:             #keep the last two frames to be used in max-pookng 
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            if term or trunc:
                break
        max_frame = self._obs_buffer.max(axis=0)        #max-pool over the last two frames.
        return max_frame, total_reward, terminated, truncated, info


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super(ClipRewardEnv, self).__init__(env)

    def reward(self, reward):
        return np.sign(reward)          #clip the reward to be either -1, 0 or 1


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super(WarpFrame, self).__init__(env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, self.height, self.width), dtype=np.uint8)    #new observation space

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)     #rgb -> grayscale
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)  #frame size to 84 x 84
        return frame[None, :, :]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super(FrameStack, self).__init__(env)
        self.k = k                          #number of frames that are goig to be stacked
        self.frames = deque(maxlen=k)       #store the last k observations
        shp = env.observation_space.shape   #expected shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(k, shp[1], shp[2]), dtype=np.uint8)     #updated observation space

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), info

    def step(self, action):
        ob, reward, term, trunc, info = self.env.step(action)
        self.frames.append(ob)              #add the latest frame
        return self._get_ob(), reward, term, trunc, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class LazyFrames(object):       #called in framestack to concatenate list of frames when converted to an array
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
    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.truncated_done = True

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.was_real_done = done
        self.truncated_done = truncated
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done or self.truncated_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info
