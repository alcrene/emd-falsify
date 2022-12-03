# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Optional, ClassVar, Union, Literal
from collections.abc import Mapping
from pydantic import BaseModel, Field, validator, root_validator
from mackelab_toolbox.config import ValidatingConfig, ValidatingConfigBase, prepend_rootdir  # prepend_rootdir is a workaround because assigning automatically doesn’t currently work
from mackelab_toolbox.utils import Singleton

from configparser import ConfigParser

# Possible improvement: If we could have nested config parsers, we might be 
#     able to rely more on the ConfigParser machinery, and less on a custom
#     Pydantic type, which should be easier for others to follow.
#     In particular, the `colors` field could remain a config parser, although
#     we would still want to allow dotted access.

# HoloConfig
import re
from typing import Any, List, Dict, Tuple
from pydantic import PrivateAttr#, StrictInt, StrictFloat, StrictBool
from pydantic.errors import PydanticErrorMixin
import holoviews as hv
import addict


# Generic parameter type for Holoviews options
HoloParam = Union[int, float, bool,
                  Tuple[Union[int, float, bool], ...],
                  str]
from pydantic.validators import int_validator, float_validator, bool_validator, str_validator
class GenericParam:
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, v):
        try:
            if isinstance(v, str):
                v = v.strip()

                # Check if v is a quoted string – either "…" or '…'
                # CAUTION: This won't catch malformed expressions like ""a" – the returned value would be '"a'
                if v.startswith('"') or v.endswith('"'):
                    if not v.startswith('"') and v.endswith('"'):
                        raise ValueError("Unclosed quotations")
                    return v[1:-1]
                if v.startswith("'") or v.endswith("'"):
                    if not v.startswith("'") and v.endswith("'"):
                        raise ValueError("Unclosed quotations")
                    return v[1:-1]

                # Check if v is a tuple
                if v.startswith("(") or v.endswith(")"):
                    if not v.startswith("(") and v.endswith(")"):
                        raise ValueError("Unclosed brackets")
                    return tuple(cls.validate(item.strip())
                                 for item in v[1:-1].split(","))
                # Check if v is a number:
                if v[0] in set("0123456789"):
                    vs = v
                    if "." in vs and "," in vs: # Format: 123,456.78  – Not recommended, because 123,456 would be converted to 123.456
                        vs = vs.replace(",", "")
                    elif "," in vs:  # Format: 123,78  – Convert to 123.78
                        vs = vs.replace(",", ".")
                    if vs.count(".") > 1:
                        raise TypeError(f"Number {v} contains more than one decimal indicator")
                    if "." in vs:
                        return float_validator(vs)
                    else:
                        return int_validator(vs)

                # Check if v is bool:
                if v.lower() in {"true", "false"}:
                    return bool_validator(v)

                # If nothing matches, return a string
                return v
            else:
                # If not a string, leave it as is.
                # However, still recurse into tuples / lists to validate their content
                if isinstance(v, (tuple, list)):
                    return type(v)(cls.validate(item) for item in v)
                else:
                    return v

        except (ValueError, TypeError, AssertionError) as e:
            # To help debugging, we add a message to errors which are caught by Pydantic
            raise type(e)(f"Error occurred while validating {cls}") from e


# class DictModel(BaseModel):
#     "Used to parse dictionaries exported as JSON"
#     class Config:
#         extra = "allow"

class _HoloConfigCreator(metaclass=Singleton):
    def __init__(self):
        self.config_types = {}
    def __getitem__(self, backend):
        try:
            return self.config_types[backend]
        except KeyError:
            self.config_types[backend] = type(
                f"HoloConfig[{backend}]", (HoloConfigBase,), {"_backend": backend})
            return self.config_types[backend]
HoloConfig = _HoloConfigCreator()

def make_addict(obj: Mapping) -> addict.Dict:
    d = addict.Dict(obj.items())
    for k, v in d.items():
        if isinstance(v, Mapping):
            d[k] = make_addict(v)
    return d

def split_cycles(d: dict) -> dict:
    for k, v in d.items():
        if k.lower() == "cycle" and isinstance(v, str):
            d[k] = v.split()
        elif isinstance(v, dict):
            d[k] = split_cycles(v)
    return d

def _replace_colors(value, colors):
    if isinstance(value, str) and value.startswith("colors."):
        c = colors
        for field in value.split(".")[1:]:
            c = c.get(field)
        return c
    elif isinstance(value, dict):
        for k, v in value.items():
            value[k] = _replace_colors(v, colors)
        return value
    else:
        return value

class HoloConfigBase(ValidatingConfigBase):
    _elem_names = {"Curve", "Scatter", "Overlay", "Layout", "Area"}
    _backend: ClassVar[str]

    colors: Union[Path, dict]
    defaults: Dict[str, GenericParam]={}

    Area   : Dict[str, GenericParam]={}
    Curve  : Dict[str, GenericParam]={}
    Layout : Dict[str, GenericParam]={}
    Overlay: Dict[str, GenericParam]={}
    Scatter: Dict[str, GenericParam]={}
    # TODO: Add all hv Elements

    renderer: Dict[str, Any]={}

    def __init__(self, **kwargs):
        """Extend `values` dict to include the capitalized forms used by HoloViews

        Example:
            {'curve': {'width': 200}}
        becomes
            {'curve': {'width': 200}, 'Curve': {'width': 200}}
        """
        elem_names = {field.name.lower(): field.name
                      for field in self.__fields__.values()}
                      # if field.name in self._elem_names}
        cap_vals = {}
        for option, value in kwargs.items():
            cap_name = elem_names.get(option.lower())
            if cap_name and cap_name not in cap_vals:
                cap_vals[cap_name] = value
        kwargs.update(cap_vals)
        super().__init__(**kwargs)

    _prepend_rootdir = validator("colors", allow_reuse=True)(prepend_rootdir)

    # @validator("*", pre=True)
    # def parse_element_options(cls, opts, field):
    #     """Support passing multiple values as a dictionary
        
    #     As a convenience, and to allow greater concision, plot options can be
    #     specified as a dictionary:
        
    #         [figures.matplotlib]
    #         Curve = {linewidth: 3, color: blue}

    #     which is equivalent to

    #         [figures.matplotlib.Curve]
    #         linewidth = 3
    #         color = blue

    #     Missing quotes are automatically inserted: ``{key: val}  ->  {'key': 'val'}``,
    #     but only if there are no quotes in option value.
    #     So ``{'key': val}`` would be left unchanged.
    #     `val` is wrapped iff it start with a letter or underscore.
    #     """
    #     if isinstance(opts, str) and opts.strip().startswith("{"):
    #         if not {"'", '"'}.intersection(opts):
    #             # If there are no quotes at all
    #             raw = re.sub(r"([{,])\s*(\w+)\s*:", r'\1 "\2":', opts)
    #             raw = re.sub(r":\s*([a-zA-Z_]+\w*)\s*([,}])", r': "\1" \2', raw)
    #         return DictModel.parse_raw(raw).__dict__  # FIXME: Use Pydantic’s dict validator directly
    #     else:
    #         return opts

    @validator("*")
    def test(cls, val, field):
        return val

    @validator("colors")
    def load_colors(cls, colors):
        if isinstance(colors, Path):
            colorcfg = ConfigParser()
            with open(colors) as f:
                colorcfg.read_file(f)
            colors = colorcfg
        if isinstance(colors, ConfigParser):
            # Convert to a dict
            colordict = make_addict(colorcfg)
            colordict.pop("DEFAULT", None)
            colors = colordict
        return colors

    @validator("colors")
    def convert_to_addict(cls, colors):
        return make_addict(colors)

    @validator("colors")
    def split_color_cycles(cls, colors):
        return split_cycles(colors)

    @validator("*")
    def replace_colors(cls, val, values):
        """
        Replace entries like `colors.pale.yellow` with the corresponding 
        value from the `colors` field.
        """
        colors = values.get("colors")
        if colors is not None:
            return _replace_colors(val, colors)
        else:
            return val

    @property
    def all_opts(self):
        """Return all set opts, for all plot Elements, including defaults."""
        allowed_opts = hv.opts._element_keywords(self._backend, self._elem_names)
        opts = {}
        for elem in self._elem_names:
            # List of default attributes we defined and which are applicable to this element
            opts[elem] = {}
            if getattr(self, "defaults"):
                opts[elem].update({k:v for k,v in self.defaults.items()
                                   if k in allowed_opts[elem]})
            elem_opts = getattr(self, elem, None)
            if elem_opts:
                opts[elem].update(elem_opts)
        return opts

class Config(ValidatingConfig):
    package_name = "emd-paper"

    class paths:
        projectdir : Path
        configdir  : Path="config"
        smtproject : Path
        datadir    : Path
        labnotesdir: Path
        figuresdir : Path

        _prepend_rootdir = validator("projectdir", "configdir", "smtproject",
                                     "datadir", "labnotesdir", "figuresdir",
                                     allow_reuse=True
                                    )(prepend_rootdir)

        @validator("figuresdir")
        def ensure_dir_exists(cls, dirpath):
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            return dirpath

    class random:
        entropy: int

    class figures:
        # class Config:
        #     extra = "allow"
        #     arbitrary_types_allowed = True  # B/c ConfigParser is not defined as a Pydantic type
        backend: Literal["matplotlib", "bokeh"]
        matplotlib: Optional[HoloConfig["matplotlib"]] = None
        bokeh: Optional[HoloConfig["bokeh"]]           = None

        @root_validator
        def load_backends(cls, values):
            """By preemptively loading the backends, we ensure that e.g.
            ``hv.opts(*config.figures.bokeh)`` does not raise an exception.
            """
            if "matplotlib" in values:
                renderer = hv.renderer("matplotlib")
                render_args = values["matplotlib"].renderer
                if render_args:
                    for kw, val in render_args.items():
                        setattr(renderer, kw, val)

            if "bokeh" in values:
                renderer = hv.renderer("bokeh")
                render_args = values["bokeh"].renderer
                if render_args:
                    for kw, val in render_args.items():
                        setattr(renderer, kw, val)

            return values

        @root_validator
        def set_defaults(cls, values):
            for backend in ["matplotlib", "bokeh"]:
                if backend in values and backend in hv.Store.renderers:  # If backend is not in `renderers`, than the best guess is that `load_backends` failed for that backend
                    hv.Store.set_current_backend(backend)  # Only to silence warnings
                    hv.opts.defaults(values[backend].all_opts)
            hv.Store.set_current_backend(values.get("backend"))
            return values


config = Config(Path(__file__).parent/".project-defaults.cfg",
                config_module_name=__name__)
