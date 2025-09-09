from typing import Optional

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        backtrace=False,
        diagnose=False,
        colorize=True,
        enqueue=True,
    )


def get_logger(name: Optional[str] = None):
    return logger.bind(module=name or __name__)


