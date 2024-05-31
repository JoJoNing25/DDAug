import os
import util
import datetime
import shutil
import mlconfig
import torch
import json
import misc
import glob
from torch.utils.tensorboard import SummaryWriter


class ExperimentManager():
    # including_dir: *.py under these folder will be saved to experiments folder
    including_dir = [
        'losses',
        'network',
        'datasets',
        'datasets/SVHM/',
        'datasets/transform/',
    ]

    def __init__(self, exp_name, exp_path, config_file_path, eval_mode=False):
        if exp_name == '' or exp_name is None:
            exp_name = 'exp_at' + datetime.datetime.now()
        self.exp_name = exp_name
        self.exp_path = os.path.join(exp_path, exp_name)
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.log_filepath = os.path.join(self.exp_path, exp_name.split('/')[-1]) + ".log"
        self.stas_hist_path = os.path.join(self.exp_path, 'stats')
        self.stas_eval_path = os.path.join(self.exp_path, 'stats_eval')
        self.code_repo_path = os.path.join(self.exp_path, 'code_repo')
        self.tb_path = os.path.join(self.exp_path, 'tensorboard_logs')

        if misc.get_rank() == 0 and not eval_mode:
            self.build_dirs()

        if config_file_path is not None:
            dst = os.path.join(self.exp_path, exp_name+'.yaml')
            if dst != config_file_path and misc.get_rank() == 0 and not eval_mode:
                shutil.copyfile(config_file_path, dst)
            config = mlconfig.load(config_file_path)
            config.set_immutable()
        else:
            config = None
        self.config = config
        self.logger = None
        if misc.get_rank() == 0:
            self.logger = util.setup_logger(name=self.exp_path, log_file=self.log_filepath)
            self.tb_logger = SummaryWriter(log_dir=self.tb_path)

            # if not eval_mode:
            #     self.save_exp_files()

    def build_dirs(self):
        if misc.get_rank() == 0:
            util.build_dirs(self.exp_path)
            util.build_dirs(self.checkpoint_path)
            util.build_dirs(self.stas_hist_path)
            util.build_dirs(self.stas_eval_path)
            util.build_dirs(self.code_repo_path)
            util.build_dirs(self.tb_path)

    def save_exp_files(self):
        for name in glob.glob('./*'):
            if name.endswith('.py'):
                dst = os.path.join(self.code_repo_path, name)
                shutil.copyfile(name, dst)
                self.logger.info(name + ' saved to experiment folder')
        for sub_dir in self.including_dir:
            for name in glob.glob('./%s/*' % sub_dir):
                if name.endswith('.py'):
                    dst = os.path.join(self.code_repo_path, name)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copyfile(name, dst)
                    self.logger.info(name + ' saved to experiment folder')

    def save_eval_stats(self, exp_stats, name):
        filename = '%s_exp_stats_eval.json' % name
        filename = os.path.join(self.stas_eval_path, filename)
        with open(filename, 'w') as outfile:
            json.dump(exp_stats, outfile)
        return

    def load_eval_stats(self, name):
        filename = '%s_exp_stats_eval.json' % name
        filename = os.path.join(self.stas_eval_path, filename)
        if os.path.exists(filename):
            with open(filename, 'r') as json_file:
                data = json.load(json_file)
                return data
        else:
            return None

    def save_epoch_stats(self, epoch, exp_stats):
        filename = 'exp_stats_epoch_%d.json' % epoch
        filename = os.path.join(self.stas_hist_path, filename)
        with open(filename, 'w') as outfile:
            json.dump(exp_stats, outfile)
        return

    def load_epoch_stats(self, epoch=None):
        if epoch is not None:
            filename = 'exp_stats_epoch_%d.json' % epoch
            filename = os.path.join(self.stas_hist_path, filename)
            with open(filename, 'r') as json_file:
                data = json.load(json_file)
                return data
        else:
            epoch = self.config.epochs
            filename = 'exp_stats_epoch_%d.json' % epoch
            filename = os.path.join(self.stas_hist_path, filename)
            while not os.path.exists(filename) and epoch >= 0:
                epoch -= 1
                filename = 'exp_stats_epoch_%d.json' % epoch
                filename = os.path.join(self.stas_hist_path, filename)

            if not os.path.exists(filename):
                return None

            with open(filename, 'rb') as json_file:
                data = json.load(json_file)
                return data
        return None

    def save_state(self, target, name):
        if misc.get_rank() == 0:
            if isinstance(target, torch.nn.DataParallel):
                target = target.module
            filename = os.path.join(self.checkpoint_path, name) + '.pt'
            torch.save(target.state_dict(), filename)
            self.logger.info('%s saved at %s' % (name, filename))
        return

    def load_state(self, target, name, strict=True):
        filename = os.path.join(self.checkpoint_path, name) + '.pt'
        if not torch.cuda.is_available():
            map_location=torch.device('cpu')
            d = torch.load(filename, map_location)
        else:
            d = torch.load(filename)
        keys = []
        for k, v in d.items():
            if 'total_ops' in k or 'total_params' in k:
                keys.append(k)
        for k in keys:
            del d[k]

        if 'model' in name:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in d.items():
                if k.startswith('module.'):
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            target.load_state_dict(new_state_dict, strict=strict)
        else:
            target.load_state_dict(d, strict=strict)
        if misc.get_rank() == 0:
            self.logger.info('%s loaded from %s' % (name, filename))
        return target
