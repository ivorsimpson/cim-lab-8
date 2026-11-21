from pathlib import Path
import pytest
from hydra import compose, initialize_config_dir

CONF = str(Path(__file__).parents[1] / "conf")


@pytest.fixture
def make_cfg():
    def make(*overrides):
        with initialize_config_dir(version_base=None, config_dir=CONF):
            return compose(config_name="config", overrides=list(overrides))
    return make
