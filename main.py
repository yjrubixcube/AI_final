import os

import gym.wrappers
import gym.wrappers.frame_stack
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

import torch.multiprocessing as mp
import torch

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame
from neural import MarioNet

NUM_PROCESS = 4

def train(index, globalMario: Mario, globalOptim: torch.optim.Adam, save_dir=None):
    torch.manual_seed(index)
    

    # Initialize Super Mario environment
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    env = JoypadSpace(
        env,
        [['right'],
        ['right', 'A']]
    )

    # Apply Wrappers to environment
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)

    env.reset()

    # TODO: turn off save for others
    localMario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, optimizer=globalOptim)
    localMario.net.train()
    if save_dir:
        logger = MetricLogger(save_dir)
    episodes = 5000

    ### for Loop that train the model num_episodes times by playing the game
    for e in range(episodes):

        state = env.reset().cuda()
        
        localMario.net.load_state_dict(globalMario.net.state_dict())

        # Play the game!
        while True:

            # 3. Show environment (the visual) [WIP]
            # env.render()

            # 4. Run agent on the state
            action = localMario.act(state)

            # 5. Agent performs action
            next_state, reward, done, info = env.step(action)

            # 6. Remember
            localMario.cache(state, next_state, action, reward, done)

            # 7. Learn
            q, loss = localMario.learn(globalMario)

            # 8. Logging
            logger.log_step(reward, loss, q)

            # 9. Update state
            state = next_state

            # 10. Check if end of game
            if done or info['flag_get']:
                break

        logger.log_episode()

        if e % 20 == 0 and save_dir:
            logger.record(
                episode=e,
                epsilon=localMario.exploration_rate,
                step=localMario.curr_step
            )


if __name__ == "__main__":

    

    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    # checkpoint = None # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
    # mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

    # Initialize Super Mario environment
    GlobalEnv = gym_super_mario_bros.make('SuperMarioBros-1-1-v3')

    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    GlobalEnv = JoypadSpace(
        GlobalEnv,
        [['right'],
        ['right', 'A']]
    )

    # Apply Wrappers to environment
    GlobalEnv = SkipFrame(GlobalEnv, skip=4)
    GlobalEnv = GrayScaleObservation(GlobalEnv, keep_dim=False)
    GlobalEnv = ResizeObservation(GlobalEnv, shape=84)
    GlobalEnv = TransformObservation(GlobalEnv, f=lambda x: x / 255.)
    GlobalEnv = FrameStack(GlobalEnv, num_stack=4)


    # logger = MetricLogger(save_dir)
    GlobalOptimizer = torch.optim.Adam(MarioNet((4, 84, 84), GlobalEnv.action_space.n).float().parameters(), lr=0.00025)
    
    GlobalMario = Mario(state_dim=(4, 84, 84), action_dim=GlobalEnv.action_space.n, save_dir=save_dir, optimizer=GlobalOptimizer)
    GlobalMario.share_memory()
    
    # episodes = 40000

    processes = []
    for i in range(NUM_PROCESS):
        if i == 0:
            process = mp.Process(target=train, args=(i, GlobalMario, GlobalOptimizer, save_dir))
        else:
            process = mp.Process(target=train, args=(i, GlobalMario, GlobalOptimizer))
        process.start()
        processes.append(processes)
    
    for p in processes:
        p.join()

   
    