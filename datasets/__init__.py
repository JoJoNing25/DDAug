import mlconfig
from . import dataset
from datasets import collate
from .aug_search_ops import AugPolicy
mlconfig.register(dataset.DatasetGenerator)
mlconfig.register(collate.SimCLRCollateFunction)
mlconfig.register(collate.SimCLRNoCropCollateFunction)
mlconfig.register(collate.BYOLCollateFunction)
mlconfig.register(collate.SimCLRCustomCollateFunction)
mlconfig.register(AugPolicy)