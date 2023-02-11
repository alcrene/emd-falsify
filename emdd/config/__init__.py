# -*- coding: utf-8 -*-
import uuid
from pathlib import Path
from typing import Optional, ClassVar, Union, Literal
from configparser import ConfigParser

from pydantic import BaseModel, Field, validator, root_validator
from mackelab_toolbox.config import ValidatingConfig, prepend_rootdir, ensure_dir_exists
from mackelab_toolbox.config.holoviews import FiguresConfig
    # prepend_rootdir is a workaround because assigning automatically doesn’t currently work

# Possible improvement: If we could have nested config parsers, we might be 
#     able to rely more on the ConfigParser machinery, and less on a custom
#     Pydantic type, which should be easier for others to follow.
#     In particular, the `colors` field could remain a config parser, although
#     we would still want to allow dotted access.


class Config(ValidatingConfig):
    default_config_file : ClassVar = Path(__file__).parent/"defaults.cfg"
    config_module_name  : ClassVar = __name__

    package_name = "emdd"

    class paths:
        projectdir : Path
        configdir  : Path="config"
        # smtproject : Path
        # datadir    : Path
        # labnotesdir: Path
        figuresdir : Path

        # _prepent_rootdir = prepend_rootdir("figuresdir")

        # _ensure_dir_exists = ensure_dir_exists("figuresdir")
        
        _prepend_rootdir = validator("figuresdir",
                                     allow_reuse=True
                                    )(prepend_rootdir)
        _ensure_dir_exists = validator("figuresdir")(ensure_dir_exists)

        # @validator("figuresdir")
        # def ensure_dir_exists(cls, dirpath):
        #     if dirpath:
        #         os.makedirs(dirpath, exist_ok=True)
        #     return dirpath

    class random:
        entropy: int

    class caching:
        """
        Note that the `joblib` options are ignored when `use_disk_cache` is False.
        """
        use_disk_cache: bool=False

        class joblib:
            """
            these arguments are passed on to joblib.Memory.
            When `use_disk_cache` is True, functools.lru_cache is used instead of
            joblib.Memory, and the other config options are ignored.

            .. Notes::
               My reading of joblib's sources suggests that relevant values for
               `verbose` are 0, 1 and 2.
            """
            location: Path=".joblib-cache"
            verbose : int=0
            backend : str="local"
            mmap_mode: Optional[str]=None
            compress: Union[bool,int]=False

            _prepend_rootdir = validator("location", allow_reuse=True)(prepend_rootdir)

            @validator("location")
            def make_location_unique(cls, location):
                """
                Add a machine-specific unique folder to the cache location,
                to avoid collisions with other machines.
                (Caches are pickled data, so not machine-portable.)
                """
                alphabet = "abcdefghijklmnopqrstuvwxyz"
                num = uuid.getnode()
                # For legibility, replace the int by an equivalent string
                clst = []
                while num > 0:
                    clst.append(alphabet[num % 26])
                    num = num // 26
                hostdir = "host-"+"".join(clst)
                return location/hostdir

    figures: FiguresConfig
    # class figures:
    #     # class Config:
    #     #     extra = "allow"
    #     #     arbitrary_types_allowed = True  # B/c ConfigParser is not defined as a Pydantic type
    #     backend: Literal["matplotlib", "bokeh"]
    #     matplotlib: Optional[HoloConfig["matplotlib"]] = None
    #     bokeh: Optional[HoloConfig["bokeh"]]           = None

    #     @root_validator
    #     def load_backends(cls, values):
    #         """By preemptively loading the backends, we ensure that e.g.
    #         ``hv.opts(*config.figures.bokeh)`` does not raise an exception.
    #         """
    #         if "matplotlib" in values:
    #             renderer = hv.renderer("matplotlib")
    #             render_args = values["matplotlib"].renderer
    #             if render_args:
    #                 for kw, val in render_args.items():
    #                     setattr(renderer, kw, val)

    #         if "bokeh" in values:
    #             renderer = hv.renderer("bokeh")
    #             render_args = values["bokeh"].renderer
    #             if render_args:
    #                 for kw, val in render_args.items():
    #                     setattr(renderer, kw, val)

    #         return values

    #     @root_validator
    #     def set_defaults(cls, values):
    #         for backend in ["matplotlib", "bokeh"]:
    #             if backend in values and backend in hv.Store.renderers:  # If backend is not in `renderers`, than the best guess is that `load_backends` failed for that backend
    #                 hv.Store.set_current_backend(backend)  # Only to silence warnings
    #                 hv.opts.defaults(values[backend].all_opts)
    #         hv.Store.set_current_backend(values.get("backend"))
    #         return values

    #     def __getattr__(self, attr):
    #         "Use the config associated to `backend` as default."
    #         return getattr(getattr(self, self.backend), attr)


# config = Config(Path(__file__).parent/"defaults.cfg",
#                 config_module_name=__name__)
