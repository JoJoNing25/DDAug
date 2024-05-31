import os
from .utils import dataset_options, transform_options
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.collate import SimCLRCollateFunction


class DatasetGenerator():
    def __init__(self, exp, train_bs=128, eval_bs=256, seed=0, n_workers=4,
                 train_d_type='SVHM_LC', train_tf_op='SVHM_LC960',
                 train_path='/data/gpfs/projects/punim1399/data/lc/darwin/finished',
                 val_d_type='SVHM_LC', val_tf_op='SVHM_LC960',
                 val_path='/data/gpfs/projects/punim1399/data/lc/darwin/finished',
                 test_d_type='SVHM_LC', test_tf_op='SVHM_LC960',
                 test_path='/data/gpfs/projects/punim1399/data/lc/darwin/finished',
                 **kwargs):

        if train_d_type not in dataset_options:
            raise('Unknown Dataset')
        if val_d_type not in dataset_options:
            raise('Unknown Dataset')
        if test_d_type not in dataset_options:
            raise('Unknown Dataset')

        self.train_bs = train_bs
        self.eval_bs = eval_bs
        self.n_workers = n_workers

        try:
            env_n_workers = os.environ['SLURM_CPUS_PER_TASK']
            if env_n_workers is not None:
                self.n_workers = int(env_n_workers)
            print('setting n_workers to', self.n_workers)
        except:
            print('setting n_workers base on SLURM failed, n_workers is {}'.format(self.n_workers))
        
        self.train_path = train_path
        self.test_path = test_path

        train_tf = transform_options[train_tf_op]["train_transform"]
        val_tf = transform_options[val_tf_op]["test_transform"]
        test_tf = transform_options[test_tf_op]["test_transform"]
        train_tf = transforms.Compose(train_tf)
        val_tf = transforms.Compose(val_tf)
        test_tf = transforms.Compose(test_tf)

        train_joint_tf = transform_options[train_tf_op]['train_joint_transform']
        val_joint_tf = transform_options[val_tf_op]['test_joint_transform']
        test_joint_tf = transform_options[test_tf_op]['test_joint_transform']

        self.train_set = dataset_options[train_d_type](
            seed, train_path, train_tf, train_joint_tf, 'train', kwargs)
        self.val_set = dataset_options[train_d_type](
            seed, val_path, val_tf, val_joint_tf, 'val', kwargs)
        self.test_set = dataset_options[test_d_type](
            seed, test_path, test_tf, test_joint_tf, 'test', kwargs)

    def get_loader(self, train_shuffle=True, train_sampler=None, val_sampler=None, test_sampler=None, collate_fn=None):
        if train_sampler is not None:
            if not collate_fn:
                train_loader = DataLoader(dataset=self.train_set, pin_memory=False,
                                          batch_size=self.train_bs, drop_last=True,
                                          num_workers=self.n_workers,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(dataset=self.train_set, pin_memory=False,
                                          batch_size=self.train_bs, drop_last=True,
                                          num_workers=self.n_workers,
                                          collate_fn=collate_fn,
                                          sampler=train_sampler)
            val_loader = DataLoader(dataset=self.val_set, pin_memory=False,
                                    batch_size=self.eval_bs, drop_last=False,
                                    num_workers=self.n_workers, sampler=val_sampler)
            test_loader = DataLoader(dataset=self.test_set, pin_memory=False,
                                     batch_size=self.eval_bs, drop_last=False,
                                     num_workers=self.n_workers, sampler=test_sampler)
        else:
            train_loader = DataLoader(dataset=self.train_set, pin_memory=False,
                                      batch_size=self.train_bs, drop_last=True,
                                      num_workers=self.n_workers,
                                      shuffle=train_shuffle)
            val_loader = DataLoader(dataset=self.val_set, pin_memory=False,
                                    batch_size=self.eval_bs, drop_last=False,
                                    num_workers=self.n_workers, shuffle=False)
            test_loader = DataLoader(dataset=self.test_set, pin_memory=False,
                                     batch_size=self.eval_bs, drop_last=False,
                                     num_workers=self.n_workers, shuffle=False)

        return train_loader, val_loader, test_loader
