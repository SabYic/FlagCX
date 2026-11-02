# Load the real FlagCX Python wrapper from the local FlagCX checkout without
# importing the `flagcx` package itself (whose __init__ pulls in the torch
# plugin extension that is not built in this environment).

import importlib.util
import os
from pathlib import Path


_api_path = os.environ.get(
    "FLAGCX_API_PATH",
    "/root/FlagCX/src/flagcx/api.py",
)
_spec = importlib.util.spec_from_file_location("_flagcx_api", _api_path)
_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api)

FLAGCXLibrary = _api.FLAGCXLibrary
flagcxDevCommRequirements = _api.flagcxDevCommRequirements
flagcxUniqueId = _api.flagcxUniqueId
FLAGCX_WIN_COLL_SYMMETRIC = _api.FLAGCX_WIN_COLL_SYMMETRIC
