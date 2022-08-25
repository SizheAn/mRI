import argparse
import os
import json
import yaml
import copy
from pprint import pprint
import itertools

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v

def parse_exp_cfg(default_cfg, cfg_dict):
    """
    default_cfg: a reference config (template)
    cfg_dict: a dictionary of parameters that will be varied during experiments

    This function will enumerate all combinations of the parameters.
    """
    all_param_names = []
    all_param_cfgs = []
    for _, param in cfg_dict.items():
        param_names = []
        param_cfgs = []
        for name, cfg in param.items():
            param_names.append(name)
            param_cfgs.append(cfg)
        all_param_names.append(param_names)
        all_param_cfgs.append(param_cfgs)

    all_exp_cfgs = {}
    for exp_name_list, exp_cfg_list in zip(
        itertools.product(*all_param_names),
        itertools.product(*all_param_cfgs)
    ):
        exp_name = '_'.join(exp_name_list)
        exp_cfg = copy.deepcopy(exp_cfg_list[0])
        for cfg in exp_cfg_list:
            _merge(cfg, exp_cfg)
        _merge(default_cfg, exp_cfg)
        all_exp_cfgs[exp_name] = exp_cfg
    return all_exp_cfgs

def main(args):
    # prepare output folder
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    output_folder = args.output

    # Load a reference config file.
    # This config is further instantiated into different experiments
    with open(args.config, "r") as fd:
        default_cfg = yaml.load(fd, Loader=yaml.FullLoader)

    # Load all configs for experiments and unfold them
    with open(args.expcfg, "r") as fd:
        exp_cfg = yaml.load(fd, Loader=yaml.FullLoader)
        exp_cfgs = parse_exp_cfg(default_cfg, exp_cfg)

    all_cmds = []
    for exp_name, exp_cfg in exp_cfgs.items():
        # output to yaml file used for training / evaluation
        output_cfg_file = os.path.join(output_folder, exp_name + '.yaml')
        with open(output_cfg_file, "w") as fd:
            yaml.dump(exp_cfg, fd)

        # prepare for bash script for all training / evaluation
        train_cmd_str = [
            "python",
            "./train.py",
            output_cfg_file,
            "--output",
            "exp",
            "-c 40",
            "-p 2",
        ]
        train_cmd = ' '.join(train_cmd_str)
        eval_cmd_str = [
            "python",
            "./eval.py",
            output_cfg_file,
            os.path.join('./ckpt', exp_name + '_exp'),
            "2>&1 | tee",
            os.path.join('./ckpt', exp_name + '.txt'),
        ]
        eval_cmd = ' '.join(eval_cmd_str)

        all_cmds.append(train_cmd)
        all_cmds.append(eval_cmd)
        all_cmds.append('sleep 5')

    # write to bash scripts
    output_sh_file = os.path.join(args.output, 'run_all_exps.sh')
    with open(output_sh_file, 'w') as fd:
        for cmd in all_cmds:
            fd.write("{:s}\n".format(cmd))

################################################################################
if __name__ == '__main__':
    """Generate """
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Managing all experiments')
    parser.add_argument('config', metavar='DIR',
                        help='path to a reference config file')
    parser.add_argument('expcfg', metavar='DIR',
                        help='path to a experiment config file')
    parser.add_argument('--output', default='./exp_configs', type=str,
                        help='name of exp folder (default: ./exp_configs)')
    args = parser.parse_args()
    main(args)
