from functools import wraps
from typing import Callable, Dict, Any, get_type_hints, Union, Optional, _GenericAlias, List
import inspect
import numpy as np
from gymnasium import Env

def type_check(func: Callable) -> Callable:
    """
    Decorator for runtime type checking of function parameters and return value.
    Handles parameterized generics, numpy arrays, and Union types safely.
    
    Args:
        func: The function to be decorated
        
    Returns:
        Wrapped function with type checking
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get type hints for the function
        hints = get_type_hints(func)
        
        # Get the function's signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        def check_type(value: Any, expected_type: Any) -> bool:
            """Safe type checking with support for complex types"""
            try:
                # Handle None for Optional types
                if value is None:
                    if hasattr(expected_type, "__origin__") and expected_type.__origin__ is Union:
                        return type(None) in expected_type.__args__
                    return False
                
                # Handle numpy arrays specifically
                if expected_type is np.ndarray:
                    return isinstance(value, np.ndarray)
                
                # Handle gym environments
                if expected_type is Env:
                    return isinstance(value, Env)
                
                # Handle Union types
                if hasattr(expected_type, "__origin__") and expected_type.__origin__ is Union:
                    return any(check_type(value, t) for t in expected_type.__args__)
                
                # Handle parameterized generics (Dict, List, etc.)
                if hasattr(expected_type, "__origin__"):
                    origin = expected_type.__origin__
                    
                    # Verify basic type first
                    if not isinstance(value, origin):
                        return False
                    
                    # Handle dictionaries
                    if origin is dict:
                        key_type, value_type = expected_type.__args__
                        return all(
                            check_type(k, key_type) and check_type(v, value_type)
                            for k, v in value.items()
                        )
                    
                    # Handle sequences (list, tuple, etc.)
                    if origin in (list, tuple, set):
                        item_type = expected_type.__args__[0]
                        return all(check_type(item, item_type) for item in value)
                    
                    return True
                
                # Handle basic types
                return isinstance(value, expected_type)
                
            except Exception as e:
                print(f"Type checking error: {str(e)}")
                return False
        
        # Check each parameter
        for param_name, param_value in bound_args.arguments.items():
            if param_name in hints:
                expected_type = hints[param_name]
                if not check_type(param_value, expected_type):
                    if hasattr(expected_type, "__origin__"):
                        type_str = str(expected_type)
                    else:
                        type_str = expected_type.__name__
                    raise TypeError(
                        f"Parameter '{param_name}' must be {type_str}, "
                        f"got {type(param_value).__name__} instead"
                    )
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Check return value
        if 'return' in hints:
            # Skip return type check for __init__ methods
            if func.__name__ == '__init__':
                return result
                
            return_type = hints['return']
            # Special handling for None return type
            if return_type is type(None):
                if result is not None:
                    raise TypeError(
                        f"Function should return None, "
                        f"got {type(result).__name__} instead"
                    )
            elif not check_type(result, return_type):
                if hasattr(return_type, "__origin__"):
                    type_str = str(return_type)
                else:
                    type_str = return_type.__name__
                raise TypeError(
                    f"Function should return {type_str}, "
                    f"got {type(result).__name__} instead"
                )
        
        return result
    
    return wrapper
