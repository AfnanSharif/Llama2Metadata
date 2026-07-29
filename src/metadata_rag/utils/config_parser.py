import os

def load_config():
    """Loads configuration from environment variables."""
    return {
        'env': os.getenv('APP_ENV', 'development'),
        'debug': os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
    }
