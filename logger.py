import logging


class Logger(logging.Logger):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        format_str = "%(asctime)s:%(levelname)s (%(name)s) || PID:%(process)s TID:%(thread)d|%(threadName)s: - " \
                     "[%(module)s.%(funcName)s:%(lineno)d] --- %(message)s"

        formatter = logging.Formatter(fmt=format_str)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.propagate = False
        self.addHandler(handler)
        self.setLevel(level)
        self.propagate = False
        self.addHandler(handler)



