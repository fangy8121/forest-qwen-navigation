"""观测驱动的单行基线；mode=control 验证 Lee 速度与制动。"""
import hydra
from omni_drones.navigation.forest.local_flight import run_local


@hydra.main(version_base=None, config_path='../cfg', config_name='forest_local')
def main(cfg):
    run_local(cfg)


if __name__ == '__main__':
    main()
