# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional, Union, Literal
from pydantic import BaseModel, Field, validator, root_validator
from mackelab_toolbox.config import ValidatingConfig

from configparser import ConfigParser

# HoloConfig
import re
from typing import Any, List, Dict
from pydantic import PrivateAttr
import holoviews as hv

class DictModel(BaseModel):
    class Config:
        extra = "allow"

class HoloConfig(BaseModel):
    Curve  : Optional[Dict[str, Any]]
    Scatter: Optional[Dict[str, Any]]
    Overlay: Optional[Dict[str, Any]]
    Layout : Optional[Dict[str, Any]]
    # TODO: Add all hv Elements

    def __init__(self, **kwargs):
        """Extend `values` dict to include the capitalized forms used by HoloViews

        Example:
            {'curve': {'width': 200}}
        becomes
            {'curve': {'width': 200}, 'Curve': {'width': 200}}
        """
        elem_names = {field.name.lower(): field.name
                      for field in self.__fields__.values()}
        cap_vals = {}
        for option, value in kwargs.items():
            cap_name = elem_names.get(option.lower())
            if cap_name:
                cap_vals[cap_name] = value
        kwargs.update(cap_vals)
        super().__init__(**kwargs)

    @validator("Curve", "Scatter", "Overlay", "Layout",
               pre=True)
    def parse_element_options(cls, optstr, field):
        """Insert missing quotes in str dicts: {key: val}  ->  {'key': val}"""
        raw = re.sub(r"([{,])\s*(\w+)\s*:", r'\1 "\2":', optstr)
        return DictModel.parse_raw(raw).__dict__  # FIXME: Use Pydantic’s dict validator directly

    @property
    def all_opts(self):
        return {k:v for k, v in self.__dict__.items() if v}
        # return {fieldnm: getattr(self, fieldnm) for fieldnm in self.__fields__}

# TODO: Find a way to pass previous sections to the next one.

class Config(ValidatingConfig):
    package_name = "emd-paper"

    class paths:
        projectdir : Path
        configdir  : Path="config"
        smtproject : Path
        datadir    : Path
        labnotesdir: Path

    class random:
        entropy: int

    class figures:
        class Config:
            extra = "allow"
            arbitrary_types_allowed = True  # B/c ConfigParser is not defined as a Pydantic type
        backend: Literal["matplotlib", "bokeh"]
        colors: Union[Path, ConfigParser]
        matplotlib: Optional[HoloConfig] = None
        bokeh: Optional[HoloConfig]      = None

        @root_validator
        def load_backends(cls, values):
            """By preemptively loading the backends, we ensure that e.g.
            ``hv.opts(*config.figures.bokeh)`` does not raise an exception.
            """
            if "matplotlib" in values: hv.renderer("matplotlib")
            if "bokeh" in values: hv.renderer("bokeh")
            return values

        @root_validator
        def set_defaults(cls, values):
            for backend in ["matplotlib", "bokeh"]:
                if backend in values:
                    hv.Store.set_current_backend(backend)  # Only to silence warnings
                    hv.opts.defaults(values[backend].all_opts)
            hv.Store.set_current_backend(values["backend"])
            return values

        # # FIXME: Make this validator work
        # @validator("colors", pre=True)
        # def load_colors(cls, conffile):
        #     if isinstance(conffile, Path):
        #         # How to ensure that `conffile` is already an absolute path ??
        #         colors = ConfigParser()
        #         colors.read_file(open(conffile))
        #         return colors
        #     else:
        #         return val


# FIXME: Only works with a development install
config = Config(Path(__file__).parent.parent)

# Load colours from config file
# FIXME: Do this in a validator. The challange is that we need to wait for
#        ValidatingConfig to convert all relative paths to absolute
colors = ConfigParser()
# colors.read_file(open(config.paths.configdir/config.figures."paul_tol_colors.cfg"))
with open(config.figures.colors) as f:
    colors.read_file(f)
config.figures.colors = colors
