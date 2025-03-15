---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  formats: py:percent,md:myst
  notebook_metadata_filter: -jupytext.text_representation.jupytext_version
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python (emd-falsify-dev)
  language: python
  name: emd-falsify-dev
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Configuration options

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
import uuid
from pathlib import Path
from typing import Optional, ClassVar, Union, Literal, Dict
from configparser import ConfigParser
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
from pydantic import BaseModel, Field, validator, root_validator
from valconfig import ValConfig, ensure_dir_exists
from valconfig.contrib.holoviews import FiguresConfig, HoloMPLConfig, HoloBokehConfig, GenericParam
from scityping import Config as ScitypingConfig
```

```{code-cell}
class Config(ValConfig):
    __default_config_path__   = "defaults.cfg"

    class paths:
        figures : Path

        # This is typically used as a library: don’t create random paths on users’ computers
        #ensure_dir_exists = validator('figures', allow_reuse=True)(ensure_dir_exists)

    class mp:
        max_cores: int
        maxtasksperchild: Union[int,None]

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

            # _prepend_rootdir = validator("location", allow_reuse=True)(prepend_rootdir)

            # We could comment this out, since although pickled data are not machine portable, since the hash/filename is computed
            # from the pickle, if another machine tries to load to load from the same location, it should be OK.
            # However this potentially mixes 1000’s of files from different machines
            # in the same directory, making it almost impossible to later remove outputs from a specific machine.
            # (E.g. remove the laptop runs but keep the ones from the server)
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

    class viz(FiguresConfig):
        class matplotlib(HoloMPLConfig):
            prohibited_area : Dict[str, GenericParam]={}
            discouraged_area: Dict[str, GenericParam]={}
            calibration_curves: Dict[str, GenericParam]={}
        class bokeh(HoloBokehConfig):
            prohibited_area : Dict[str, GenericParam]={}
            discouraged_area: Dict[str, GenericParam]={}
            calibration_curves: Dict[str, GenericParam]={}

    scityping: ScitypingConfig={}
    
    @validator('scityping')
    def add_emd_falsify_safe_packages(scityping):
        scityping.safe_packages |= {"emd_falsify.tasks"}
        # scityping.safe_packages |= {"emd_falsify.models", "emd_falsify.tasks"}
        return scityping
```

config = Config(Path(__file__).parent/"defaults.cfg",
                config_module_name=__name__)

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [skip-execution]
---
config = Config()
```

# Default options

These are stored in the text file `defaults.cfg`.

```{include} defaults.cfg
:literal: true
```
