from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from .base_model import BaseModel
from .dlinear import DLinear
from .model import DLinearModel, MyModel
