"""
Custom exceptions for dataset splitting operations.
"""


class SplittingError(Exception):
    """Base exception for splitting-related errors."""
    pass


class ConfigurationError(SplittingError):
    """Raised when configuration is invalid."""
    pass


class DataValidationError(SplittingError):
    """Raised when data validation fails."""
    pass


class InsufficientDataError(SplittingError):
    """Raised when there's insufficient data for splitting."""
    pass


class SplitRatioError(SplittingError):
    """Raised when split ratios are invalid."""
    pass
