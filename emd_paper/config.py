from pathlib import Path
from pydantic import BaseModel
from mackelab_toolbox.config import ValidatingConfig

class Config(ValidatingConfig):
    class Paths(BaseModel):
        smtproject : Path
        projectdir : Path
        datadir    : Path
        labnotesdir: Path
    PATHS: Paths

root = Path(__file__).parent.parent
config = Config(
    path_user_config   =root/"project.cfg",
    path_default_config=root/".project-defaults.cfg",
    package_name       ="emd-paper")
