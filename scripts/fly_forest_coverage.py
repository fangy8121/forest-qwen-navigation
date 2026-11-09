"""Run observed row-coverage and runtime commands in one physical flight."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hydra


@hydra.main(version_base=None, config_path='../cfg', config_name='forest_coverage')
def main(cfg):
    from omni_drones.navigation.forest.coverage_flight import run_coverage
    result = run_coverage(cfg)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
