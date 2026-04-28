"""Configure logging for simulations."""

import logging
from pathlib import Path


class LoggingConfigurator:
    """Handles simulation logging configuration."""
    
    LEVEL_MAP = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    @classmethod
    def configure(cls, config: dict) -> bool:
        """Configure logging based on config dict.
        
        Args:
            config: Simulation config dict
            
        Returns:
            Whether logging is enabled
        """
        log_config = config.get('simulation', {}).get('logging', {})
        
        enabled = log_config.get('enabled', True)
        level_str = log_config.get('level', 'INFO')
        log_file = log_config.get('file', None)
        console = log_config.get('console', True)
        
        if not enabled:
            logging.disable(logging.CRITICAL)
            return False
        
        # Configure handlers
        handlers = []
        if console:
            handlers.append(cls._create_console_handler())
        
        if log_file:
            handlers.append(cls._create_file_handler(log_file))
        
        level = cls.LEVEL_MAP.get(level_str.upper(), logging.INFO)
        
        logging.basicConfig(
            level=level,
            handlers=handlers,
            force=True
        )
        
        logging.getLogger('src.simulation.engine').setLevel(level)
        
        print(f"Logging configured: level={level_str}, console={console}, file={log_file}")
        
        return True
    
    @staticmethod
    def _create_console_handler() -> logging.Handler:
        """Create console logging handler."""
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        ))
        return handler
    
    @staticmethod
    def _create_file_handler(log_file: str) -> logging.Handler:
        """Create file logging handler."""
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        return handler