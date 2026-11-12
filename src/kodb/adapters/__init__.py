from .local import AutoTrust, CoCo, JevStyle, Kanana, Laya, Nari, PplxDecider, Thx
from .server import H2O, KevHttp, StartLux

ADAPTERS = {"laya": Laya, "kanana": Kanana, "pplx-decider": PplxDecider, "thx02": Thx, "jev-style": JevStyle,
            "nari": Nari, "coco": CoCo, "autotrust": AutoTrust, "kev-http": KevHttp, "startlux": StartLux, "h2o": H2O}
