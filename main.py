import os
import time

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

from argparse import ArgumentParser

# NUM_PROCESS = 4

movement_sets = {
        "default": [
            ['right'],
            ['right', 'A']
        ],
        "right": [
            ['NOOP'],
            ['right'],
            ['right', 'A'],
            ['right', 'B'],
            ['right', 'A', 'B'],
        ],
        "simple": [
            ['NOOP'],
            ['right'],
            ['right', 'A'],
            ['right', 'B'],
            ['right', 'A', 'B'],
            ['A'],
            ['left'],
        ],
        "complex": [
            ['NOOP'],
            ['right'],
            ['right', 'A'],
            ['right', 'B'],
            ['right', 'A', 'B'],
            ['A'],
            ['left'],
            ['left', 'A'],
            ['left', 'B'],
            ['left', 'A', 'B'],
            ['down'],
            ['up'],
        ]
    }

# episodes = 

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--num_process", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_episodes", type=int, default=20000)
    parser.add_argument("--eps_per_log", type=int, default=20)
    parser.add_argument("--mem_len", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--save_every", type=int, default=250000)
    
    parser.add_argument("--moves", type=str, default="simple", help="right, simple, complex")
    parser.add_argument("--version", type=int, default=3)

    args = parser.parse_args()
    return args

def train(opt, index, globalMario: Mario, globalOptim: torch.optim.Adam, localMario: Mario, save_dir=None):
    print("before init", index)
    torch.manual_seed(index)

    # Initialize Super Mario environment
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    
    print("env", index, index)

    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    env = JoypadSpace(
        env,
        movement_sets[opt.moves]
    )

    # Apply Wrappers to environment
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    
    print("env before reset", index)

    env.reset()
    
    print("env reset", index)

    # TODO: turn off save for others
    print(env.action_space.n, index)
    # localMario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, optimizer=globalOptim)
    
    print("done local mario", index)
    localMario.net.train()
    if save_dir:
        logger = MetricLogger(save_dir)
        print("create logger")
    episodes = opt.total_episodes // opt.num_process
    # print(localMario.net.state_dict())

    print("start training", index)
    ### for Loop that train the model num_episodes times by playing the game
    for e in range(episodes):

        state = env.reset()

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
            if save_dir:
                logger.log_step(reward, loss, q)

            # 9. Update state
            state = next_state

            # 10. Check if end of game
            if done or info['flag_get']:
                break

        if save_dir:
            logger.log_episode()

        if e % opt.eps_per_log == 0 and save_dir:
            logger.record(
                episode=e,
                epsilon=localMario.exploration_rate,
                step=localMario.curr_step
            )
            localMario.save()


if __name__ == "__main__":

    opt = get_args()

    mp.set_start_method("spawn")

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
        movement_sets[opt.moves]
    )

    # Apply Wrappers to environment
    GlobalEnv = SkipFrame(GlobalEnv, skip=4)
    GlobalEnv = GrayScaleObservation(GlobalEnv, keep_dim=False)
    GlobalEnv = ResizeObservation(GlobalEnv, shape=84)
    GlobalEnv = TransformObservation(GlobalEnv, f=lambda x: x / 255.)
    GlobalEnv = FrameStack(GlobalEnv, num_stack=4)


    # logger = MetricLogger(save_dir)
    # GlobalOptimizer = torch.optim.Adam(MarioNet((4, 84, 84), GlobalEnv.action_space.n).float().parameters(), lr=0.00025)
    
    GlobalMario = Mario(opt, state_dim=(4, 84, 84), action_dim=GlobalEnv.action_space.n, save_dir=save_dir, optimizer=None)
    GlobalMario.net.share_memory()
    GlobalOptimizer = GlobalMario.optimizer
    # print("done 1")
    # GlobalMario2 = Mario(state_dim=(4, 84, 84), action_dim=GlobalEnv.action_space.n, save_dir=save_dir, optimizer=GlobalOptimizer)
    # GlobalMario2.net.share_memory()
    # print("done 2")
    # GlobalMario3 = Mario(state_dim=(4, 84, 84), action_dim=GlobalEnv.action_space.n, save_dir=save_dir, optimizer=GlobalOptimizer)
    # GlobalMario3.net.share_memory()
    # print("done 3")
    # exit()
    # episodes = 40000
    # train(0, GlobalMario, GlobalOptimizer)#, save_dir)
    # train(1, GlobalMario, GlobalOptimizer)#, save_dir)
    # exit()
    
    processes = []
    local_marios = []
    for i in range(opt.num_process):
        if i == 0:
            local_mario = Mario(opt, state_dim=(4, 84, 84), action_dim=GlobalEnv.action_space.n, save_dir=save_dir, optimizer=GlobalOptimizer)
        else:
            local_mario = Mario(opt, state_dim=(4, 84, 84), action_dim=GlobalEnv.action_space.n, save_dir=None, optimizer=GlobalOptimizer)
        local_marios.append(local_mario)
    # exit()
    for i in range(opt.num_process):
        if i == 0:
            process = mp.Process(target=train, args=(opt, i, GlobalMario, GlobalOptimizer, local_marios[i], save_dir))
        else:
            process = mp.Process(target=train, args=(opt, i, GlobalMario, GlobalOptimizer, local_marios[i]))
        process.daemon = True
        process.start()
        processes.append(process)
        # time.sleep(3)
    
    for p in processes:
        p.join()
    