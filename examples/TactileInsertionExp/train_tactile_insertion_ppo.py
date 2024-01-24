import sys, os

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(base_dir)

import yaml
from arguments import *
from utils.common import *

import algorithms.ppo_rnn as ppo_rnn

if __name__ == '__main__':
    #pyTorch张量的默认数据类型设置为float64
    torch.set_default_dtype(torch.float64)
    args_list = ['--cfg', './cfg/tactile_insertion_trans_and_rot.yaml',
                 '--logdir', './trained_models/',
                 '--log-interval', '1',
                 '--save-interval', '50',
                 '--render-interval', '0',
                 '--seed', '0']
    #自定义函数来处理命令行参数的任何冲突
    solve_argv_conflict(args_list)
    #创建一个用于强化学习（RL）的参数解析器
    parser = get_rl_parser()
    #解析命令行参数
    args = parser.parse_args(args_list + sys.argv[1:])

    # load config
    with open(args.cfg, 'r') as f:
        #使用SafeLoader加载YAML配置，以避免潜在的安全问题
        cfg = yaml.load(f, Loader = yaml.SafeLoader)
    
    if not args.no_time_stamp:
        args.logdir = os.path.join(args.logdir, get_time_stamp())

    args.train = not args.play
    #解析的参数转换为字典
    vargs = vars(args)
    
    cfg["params"]["general"] = {}
    for key in vargs.keys():
        cfg["params"]["general"][key] = vargs[key]

    algo = ppo_rnn.PPO(cfg)

    if args.train:
        algo.train()
    else:
        algo.play(cfg)
