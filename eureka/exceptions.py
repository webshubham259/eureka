class EurekaError(Exception):
    """Base exception for Eureka library."""
    pass

class ModelNotFittedError(EurekaError):
    """Raised when a method requires a fitted model."""
    pass

class InvalidParameterError(EurekaError):
    """Raised when an invalid parameter value is provided."""
    pass

class DataDimensionError(EurekaError):
    """Raised when input data has incorrect dimensions."""
    pass

class NumericalInstabilityError(EurekaError):
    """Raised when numerical computations become unstable."""
    pass

class FeatureExtractionError(EurekaError):
    """Raised when feature extraction or transformation fails."""
    pass

class ModelSerializationError(EurekaError):
    """Raised when model saving/loading operations fail."""
    pass

class ConvergenceError(EurekaError):
    """Raised when the model fails to converge during training."""
    pass    