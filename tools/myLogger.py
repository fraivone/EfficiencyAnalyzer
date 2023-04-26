import logging
from functools import partial, partialmethod

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(20)


### Config the logger
## prevent numba from spitting out debug-levl logs
logger = logging
logger.MEMORY = logger.INFO - 1  # 19
logger.DEBUG2 = logger.DEBUG - 1
logging_level = logger.INFO

logger.basicConfig(
    format="[{asctime}] {levelname} - {message}",
    datefmt="%B %d - %H:%M:%S",
    level=logging_level,
    style="{",
    handlers=[logging.StreamHandler()],
)
logger.addLevelName(logger.DEBUG2, "\33[30m%s\033[1;0m" % "DEBUG2")
logger.addLevelName(
    logger.DEBUG, "\033[1;90m%s\033[1;0m" % logger.getLevelName(logger.DEBUG)
)
logger.addLevelName(logger.MEMORY, "\033[95m%s\033[1;0m" % "MEMORY")
logger.addLevelName(
    logger.INFO, "\033[1;92m%s\033[1;0m" % logger.getLevelName(logger.INFO)
)
logger.addLevelName(
    logger.WARNING, "\033[1;93m%s\033[1;0m" % logger.getLevelName(logger.WARNING)
)
logger.addLevelName(
    logger.ERROR, "\033[1;31m%s\033[1;0m" % logger.getLevelName(logger.ERROR)
)

logger.Logger.memory = partialmethod(logger.Logger.log, logging.MEMORY)
logger.memory = partial(logger.log, logger.MEMORY)
logger.Logger.debug2 = partialmethod(logger.Logger.log, logging.DEBUG2)
logger.debug2 = partial(logger.log, logger.DEBUG2)

###
