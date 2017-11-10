import csv
import json
import os
import time

import gym
from gym.core import Wrapper


class Monitor(Wrapper):

    EXT = 'monitor.csv'
    f = None

    def __init__(self, env, filename, allow_early_resets=False,
                 reset_keywords=()):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if os.path.isdir(filename):
                    filename = os.path.join(filename, Monitor.EXT)
                else:
                    filename = filename + '.' + Monitor.EXT
            self.f = open(filename, 'wt')
            self.f.write('#{}\n'.format(json.dumps({
                't_start': self.tstart,
                'gym_version': gym.__version__,
                'env_id': env.spec.id if env.spec else 'Unknown',
            })))
            self.logger = csv.DictWriter(
                self.f, fieldnames=('r', 'l', 't') + reset_keywords)
            self.logger.writeheader()

        self.reset_keywords = reset_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_steps = 0
        self.current_reset_info = {}

    def _reset(self, **kwargs):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError('Tried to reset an environment before done.')

        self.rewards = []
        self.needs_reset = False
        for k in self.reset_keywords:
            v = kwargs.get(k)
            if v is None:
                raise ValueError('Expected to pass kwarg %s into reset' % k)
            self.current_reset_info[k] = v
        return self.env.reset(**kwargs)

    def _step(self, action):
        if self.needs_reset:
            raise RuntimeError('Tried to step environment that needs reset')

        ob, rew, done, info = self.env.step(action)
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {'r': round(eprew, 6), 'l': eplen,
                      't': round(time.time() - self.tstart, 6)}
            epinfo.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(epinfo)
                self.f.flush()
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            info['episode'] = epinfo
        self.total_steps += 1
        return (ob, rew, done, info)

    def close(self):
        if self.f is not None:
            self.f.close()

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths
