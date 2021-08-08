"""
Python script to define logger config
"""
import os
import logging

logging.basicConfig(
            filename=os.path.join(os.path.dirname(__file__), 'logs/churn_library.log'),
            level=logging.INFO,
            filemode='w',
            format='%(name)s - %(levelname)s - %(message)s')
