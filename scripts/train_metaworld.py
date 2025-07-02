from termcolor import colored
import subprocess

DATA_ROOT = './data/metaworld'

AGENTS = [
    # ('clip_bnmlp', 256),
    # ('vc1_bnmlp', 256),
    # ('r3m_bnmlp', 256),
    # ('spa_bnmlp', 256),
    # ('pointnet_bnmlp', 16),
    # ('point_next_bnmlp', 16),
    # ('pointnet_plus_plus_bnmlp', 16),
    ('lift3d_bnmlp', 16),
]

TASKS = [
    # 'assembly',
    # 'bin-picking',

    # "basketball",
    # "button-press-topdown",
    # "button-press-topdown-wall",
    # "button-press-wall",
    # "coffee-button",
    # "coffee-pull",
    # "coffee-push",
    # "disassemble",
    # "door-close",
    # "door-lock",
    # "door-open",
    # "door-unlock",
    # "drawer-close",
    # "faucet-close",
    # "faucet-open",
    # "handle-press-side",
    # "handle-press",
    # "handle-pull-side",
    # "peg-insert-side",
    # "pick-out-of-hole",
    # "pick-place",
    # "pick-place-wall",
    # "plate-slide-back-side",
    # "plate-slide-back",
    # "plate-slide-side",
    # "plate-slide",
    # # "push",
    # "push-wall",
    # "reach-wall",
    # "soccer",
    # "stick-pull",
    "stick-push",
    # "sweep",
    # "window-close",
    # "window-open",
]

CAMERAS = [
    'corner',
    # 'corner2',
]


def test_codebase():
    for agent, batch_size in AGENTS:
        for task in TASKS:
            for camera in CAMERAS:
                cmd = [
                    'python', '-m', 'lift3d.tools.train_policy',
                    '--config-name=train_metaworld',
                    f'agent={agent}',
                    f'task_name={task}',
                    f'camera_name={camera}', 
                    f"dataloader.batch_size={batch_size}",
                    f'dataset_dir={DATA_ROOT}/{task}_{camera}.zarr',
                    f"wandb.mode=offline",
                ]
                print(colored('[INFO]', 'blue'), ' '.join(cmd))
                subprocess.run(cmd)



if __name__ == '__main__':
    test_codebase()
