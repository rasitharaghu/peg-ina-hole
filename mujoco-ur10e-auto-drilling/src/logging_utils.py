from __future__ import annotations
import logging

def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
    return logging.getLogger('stable_drilling')
