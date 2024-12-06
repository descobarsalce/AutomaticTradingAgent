from functools import wraps
from typing import Callable, Dict, Any, get_type_hints, Union, Optional
import inspect
import numpy as np
from gymnasium import Env

def type_check(func: Callable) -> Callable:
    """
    Decorator for runtime type checking of function parameters and return value.
    
    Args:
        func: The function to be decorated
        
    Returns:
        Wrapped function with type checking
        
    Raises:
        TypeError: If parameter types don't match type hints
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get type hints for the function
        hints = get_type_hints(func)
        
        # Get the function's signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Check each parameter
        for param_name, param_value in bound_args.arguments.items():
            if param_name in hints:
                expected_type = hints[param_name]
                # Handle Optional types
                if hasattr(expected_type, "__origin__") and expected_type.__origin__ is Union:
                    if type(None) in expected_type.__args__:
                        if param_value is None:
                            continue
                        expected_type = next(t for t in expected_type.__args__ if t is not type(None))
                
                # Special handling for numpy arrays
                if expected_type == np.ndarray:
                    if not isinstance(param_value, np.ndarray):
                        raise TypeError(
                            f"Parameter '{param_name}' must be numpy.ndarray, "
                            f"got {type(param_value).__name__} instead"
                        )
                # Special handling for gym environments
                elif expected_type == Env:
                    if not isinstance(param_value, Env):
                        raise TypeError(
                            f"Parameter '{param_name}' must be a Gymnasium environment, "
                            f"got {type(param_value).__name__} instead"
                        )
                # Handle generic types
                elif not isinstance(param_value, expected_type):
                    raise TypeError(
                        f"Parameter '{param_name}' must be {expected_type.__name__}, "
                        f"got {type(param_value).__name__} instead"
                    )
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Check return value
        if 'return' in hints:
            return_type = hints['return']
            # Handle None return type
            if return_type is None:
                if result is not None:
                    raise TypeError(
                        f"Function should return None, got {type(result).__name__} instead"
                    )
            # Handle Optional return types
            elif hasattr(return_type, "__origin__") and return_type.__origin__ is Union:
                if type(None) in return_type.__args__:
                    if result is None:
                        return result
                    expected_type = next(t for t in return_type.__args__ if t is not type(None))
                    if not isinstance(result, expected_type):
                        raise TypeError(
                            f"Function should return {expected_type.__name__} or None, "
                            f"got {type(result).__name__} instead"
                        )
            # Handle numpy arrays
            elif return_type == np.ndarray:
                if not isinstance(result, np.ndarray):
                    raise TypeError(
                        f"Function should return numpy.ndarray, "
                        f"got {type(result).__name__} instead"
                    )
            # Handle generic return types
            elif not isinstance(result, return_type):
                raise TypeError(
                    f"Function should return {return_type.__name__}, "
                    f"got {type(result).__name__} instead"
                )
        
        return result
    
    return wrapper
