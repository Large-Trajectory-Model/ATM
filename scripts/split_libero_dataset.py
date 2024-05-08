import argparse
import os
import glob
from natsort import natsorted

def split_pretrain_dataset(files, train_folder, val_folder, train_ratio):
    num_files = len(files)
    num_train = int(num_files * train_ratio)
    train_files = files[:num_train]
    test_files = files[num_train:]
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    for f in train_files:
        # Create relative symlinks
        os.system('ln -s {} {}'.format(os.path.relpath(f, train_folder), train_folder))

    for f in test_files:
        # Create relative symlinks
        os.system('ln -s {} {}'.format(os.path.relpath(f, val_folder), val_folder))

def split_bc_train_dataset(root_dir, pretrain_train_folder, num_trains=[10, 20, 40, 80]):
    pretrain_train_files = glob.glob(os.path.join(pretrain_train_folder, '*.hdf5'))
    pretrain_train_files = natsorted(pretrain_train_files)
    for num_train in num_trains:
        assert len(pretrain_train_files) >= num_train
        bc_train_folder = os.path.join(root_dir, 'bc_train_{}'.format(num_train))
        os.makedirs(bc_train_folder, exist_ok=True)
        bc_train_rest_folder = os.path.join(root_dir, 'bc_train_{}_rest'.format(num_train))
        os.makedirs(bc_train_rest_folder, exist_ok=True)
        train_files = pretrain_train_files[:num_train]
        unlabel_files = pretrain_train_files[num_train:]
        for f in train_files:
            # Create relative symlinks to the original files
            os.system('ln -s {} {}'.format(os.path.relpath(f, bc_train_folder), bc_train_folder))
        for f in unlabel_files:
            # Create relative symlinks to the original files
            os.system('ln -s {} {}'.format(os.path.relpath(f, bc_train_rest_folder), bc_train_rest_folder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='./data/atm_libero/')
    parser.add_argument('--train_ratio', type=float, default=0.9)
    args = parser.parse_args()

    for suite_name in os.listdir(args.folder):
        for task_name in os.listdir(os.path.join(args.folder, suite_name)):
            root_dir = os.path.join(args.folder, suite_name, task_name)
            files = glob.glob(os.path.join(root_dir, '*.hdf5'))
            if len(files) == 0:
                raise ValueError('No .hdf5 files found in {}'.format(args.folder))

            files = natsorted(files)

            train_folder = os.path.join(root_dir, 'train')
            val_folder = os.path.join(root_dir, 'val')

            if not os.path.exists(train_folder):
                split_pretrain_dataset(files, train_folder, val_folder, args.train_ratio)

                pretrain_train_folder = train_folder
                split_bc_train_dataset(root_dir, pretrain_train_folder, num_trains=[2, 5, 10, 20, 30, 40, 45])

