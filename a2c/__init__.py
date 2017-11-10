import logging
import os

import gym

from a2c.bench import Monitor
from a2c.model import learn
from a2c.policies import CnnPolicy, LnLstmPolicy, LstmPolicy
from a2c.util import set_global_seeds
from a2c.vecenv import SubprocVecEnv
from a2c.wrapper import make_atari, wrap_deepmind

logger = logging.getLogger(__name__)


def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu):
    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(
                env,
                logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
            )
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind(env)
        return _thunk

    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    learn(policy_fn, env, seed, total_timesteps=int(
        num_timesteps * 1.1), lrschedule=lrschedule)
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID',
                        default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture',
                        choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule',
                        choices=['constant', 'linear'], default='constant')
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy, lrschedule=args.lrschedule, num_cpu=16)


if __name__ == '__main__':
    main()
