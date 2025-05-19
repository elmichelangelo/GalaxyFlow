from datetime import datetime

import logging
import sys


class LoggerHandler:
    """
        Logger for logging information, error, steam and debug in
        different formats. The logger creates the following
        log files under ..\files\log\.

            info.log: logs only the information. Logging level - INFO
            error.log: logs the errors. Logging level - ERROR
            debug.log: logs all. Logging level - DEBUG
            logger.log: collect all logs (without debug) in one file.

            steam log is for console output
    """

    def __init__(self, logger_dict, log_folder_path):
        """
            Constructor
            :param logger_dict: dictionary of loggers.
                :key logger_name: Name of this logger. Item: String
                :key info_logger: Use information logger. Item: Boolean
                :key error_logger: Use information logger. Item: Boolean
                :key debug_logger: Use information logger. Item: Boolean
                :key steam_logger: Use information logger. Item: Boolean
                :key stream_logging_level: Logging level of stream logger
                        can be logging.DEBUG or logging.INFO
        """

        # Create variables
        _log_folder_path = log_folder_path
        _logger_name = logger_dict["logger_name"]

        # Set logging formats
        _info_form = logging.Formatter("{asctime} - "
                                       "Message: {message} - "
                                       "Logger name: {name} - "
                                       "Logger level: {levelname}",
                                       "%d.%m.%Y %H:%M:%S",
                                       style="{")
        _error_form = logging.Formatter("{asctime} - "
                                        "Message: {message} - "
                                        "Logger name: {name} - "
                                        "Logger level: {levelname} - "
                                        "Module: {module} - "
                                        "Function Name: {funcName} - "
                                        "Line number: {lineno} - "
                                        "Process: {process} - "
                                        "Thread: {thread}",
                                        "%d.%m.%Y %H:%M:%S",
                                        style="{")
        _debug_form = logging.Formatter("{asctime} - "
                                        "Message: {message} - "
                                        "Logger name: {name} - "
                                        "Logger level: {levelname} - "
                                        "File name: {filename} - "
                                        "Module: {module} - "
                                        "Function Name: {funcName} - "
                                        "Line number: {lineno} - "
                                        "Path name: {pathname} - "
                                        "Process: {process} - "
                                        "Thread: {thread}",
                                        "%d.%m.%Y %H:%M:%S",
                                        style="{")

        # Create empty list for all loggers
        self.lst_of_loggers = []
        self.time_now = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Initialize the information logger -----------------------------
        if logger_dict["info_logger"]:
            self.info_logger = self.init_file_logger(
                _logger_name + " info",
                logging.INFO,
                _log_folder_path,
                self.time_now + "-info.log",
                _info_form
            )

            # Append logger to list of loggers
            self.lst_of_loggers.append(self.info_logger)

        # Initialize the error logger -----------------------------------
        if logger_dict["error_logger"]:
            self.error_logger = self.init_file_logger(
                _logger_name + " error",
                logging.ERROR,
                _log_folder_path,
                self.time_now + "-error.log",
                _error_form
            )

            # Append logger to list of loggers
            self.lst_of_loggers.append(self.error_logger)

        # Initialize the debug logger -----------------------------------
        if logger_dict["debug_logger"]:
            self._debug_logger = self.init_file_logger(
                _logger_name + " debug",
                logging.DEBUG,
                _log_folder_path,
                self.time_now + "-debug.log",
                _debug_form
            )

            # Append logger to list of loggers
            self.lst_of_loggers.append(self._debug_logger)

        # Initialize the stream logger ----------------------------------
        if logger_dict["stream_logger"]:
            # Set logging format depending on given logging level
            if logger_dict["stream_logging_level"] == logging.DEBUG:
                self.stream_logger, self.stream_error_logger = self.init_stream_logger(
                    _logger_name + " stream",
                    logger_dict["stream_logging_level"],
                    _debug_form
                )
            else:
                self.stream_logger, self.stream_error_logger = self.init_stream_logger(
                    _logger_name + " stream",
                    logger_dict["stream_logging_level"],
                    _info_form
                )

            # Append logger to list of loggers
            self.lst_of_loggers.append(self.stream_logger)
            self.lst_of_loggers.append(self.stream_error_logger)

    def init_file_logger(self,
                         abs_logger_name,
                         logging_level,
                         log_folder_path,
                         log_file,
                         form
                         ):
        """
            Initialize the file logger (*.log) and the collecting logger
            (logger.log)

            :param abs_logger_name: Name of this logger
            :param logging_level: Level of this logger
            :param log_folder_path: Path of the log folder
            :param log_file: file name of the log file
            :param form: logging format
            :return: the file logger as instance
        """

        # Set up file handler
        file_handler_logger = logging.FileHandler(
            log_folder_path + log_file
        )
        file_handler_logging_collector = logging.FileHandler(
            log_folder_path + self.time_now + "-logger.log"
        )

        # Set up logging format
        file_handler_logger.setFormatter(form)
        file_handler_logging_collector.setFormatter(form)

        # Creating logger
        file_logger = logging.getLogger(abs_logger_name)

        # Add file handler to logger
        file_logger.addHandler(file_handler_logging_collector)
        file_logger.addHandler(file_handler_logger)

        # Set logging level
        file_logger.setLevel(logging_level)

        # return the file logger
        return file_logger

    @staticmethod
    def init_stream_logger(abs_logger_name, logging_level, form):
        """
            Initialize the stream logger for console output
            :param abs_logger_name: Name of this logger
            :param logging_level: Level of this logger
            :param form: logging format
            :return: the stream logger as instance and
                        stream_error_logger as instance for error output
        """

        # Set up stream handler
        stream_handler_logger = logging.StreamHandler(sys.stdout)
        stream_handler_error_logger = logging.StreamHandler(sys.stderr)

        # Set up logging format
        stream_handler_logger.setFormatter(form)
        stream_handler_error_logger.setFormatter(form)

        # Creating logger
        stream_logger = logging.getLogger(abs_logger_name)
        stream_error_logger = logging.getLogger(abs_logger_name + " stderr")

        # Add stream handler to logger
        stream_logger.addHandler(stream_handler_logger)
        stream_error_logger.addHandler(stream_handler_error_logger)

        # Set logging level
        stream_logger.setLevel(logging_level)
        stream_error_logger.setLevel(logging.ERROR)

        # return the stream logger
        return stream_logger, stream_error_logger

    def log_info(self, msg):
        if hasattr(self, "info_logger"):
            self.info_logger.info(msg)

    def log_error(self, msg):
        if hasattr(self, "error_logger"):
            self.error_logger.error(msg)

    def log_debug(self, msg):
        if hasattr(self, "_debug_logger"):
            self._debug_logger.debug(msg)

    def log_stream(self, msg):
        if hasattr(self, "stream_logger"):
            self.stream_logger.info(msg)