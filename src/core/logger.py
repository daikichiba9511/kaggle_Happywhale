from logging import getLogger, StreamHandler

def get_logger(level):
    logger = getLogger()
    handler = StreamHandler()
    handler.setLevel(level)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
