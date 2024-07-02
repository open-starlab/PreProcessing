# config.py
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'filename': 'preprocessing.log',
            'mode': 'a',
        },
    },
    'loggers': {
        'my_package': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        },
    }
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
