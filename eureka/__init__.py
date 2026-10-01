'''
Eureka v0.0.3 | 2026

Eureka is a neuro-symbolic machine learning framework inspired by Kolmogorov-Arnold representation theory. It combines modern neural networks with symbolic formula extraction to build models that are accurate, interpretable, and suitable for research-driven experimentation.

GitHub: https://github.com/webshubham259/eureka
Docs: https://webshubham259.github.io/eureka/
'''

from .model import (
	Eureka,
    EUREKAClassifier,
	EUREKARegressor,
)
from .neural import TabularNet
from .elasticnet import ElasticNet

__all__ = [
	'Eureka',
	'EUREKAClassifier',
	'EUREKARegressor',
	'TabularNet',
	'ElasticNet',
]
__version__ = '0.0.3'