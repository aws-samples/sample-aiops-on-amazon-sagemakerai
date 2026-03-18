"""Configuration module for the drift monitoring pipeline.

Exports configuration for use by other modules.
"""

try:
    from src.config.config import *  # noqa: F401, F403
except ImportError:
    # config.py not yet migrated; will be available after task 1.2
    pass
