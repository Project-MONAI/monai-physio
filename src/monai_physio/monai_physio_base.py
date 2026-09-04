"""Base class for MONAI Physio providing standardized logging and debug settings.

This module provides the MONAIPhysioBase class that can be inherited by other
classes in the MONAI Physio package to provide consistent logging and messaging
functionality instead of scattered print statements.

All classes share a common logger called "MONAIPhysio" but include their class
name in log messages for identification. Users can filter which classes show logs.

Example:
    >>> from monai_physio.monai_physio_base import MONAIPhysioBase
    >>> import logging
    >>>
    >>> class MyClass(MONAIPhysioBase):
    ...     def __init__(self):
    ...         super().__init__(class_name='MyClass', log_level=logging.INFO)
    ...
    ...     def process(self):
    ...         self.log_info('Starting processing...')
    ...         self.log_debug('Debug details here')
    ...         self.log_warning('Something to be aware of')
    >>>
    >>> obj = MyClass()
    >>> obj.process()
    >>>
    >>> # Filter to show only specific classes
    >>> MONAIPhysioBase.set_log_classes(['MyClass', 'OtherClass'])
    >>>
    >>> # Show all classes again
    >>> MONAIPhysioBase.set_log_all_classes()
"""

import logging
import warnings
from typing import Any, Optional, cast


class ClassNameFilter(logging.Filter):
    """Filter to show logs only from specific class names.

    When enabled, only log messages from classes in the allowed list will be shown.
    """

    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message=".*SwigPyPacked.*"
    )
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message=".*swigvarlink.*"
    )

    def __init__(self) -> None:
        super().__init__()
        self.enabled: bool = False
        self.allowed_classes: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records based on class name."""
        if not self.enabled:
            return True  # Show all when filter is disabled

        # Extract class name from the message (format: "ClassName - message")
        if hasattr(record, "class_name"):
            return cast(str, getattr(record, "class_name")) in self.allowed_classes

        return True  # Show messages without class name attribute


class MONAIPhysioBase:
    """Base class providing standardized logging and debug settings.

    This class provides a consistent logging interface for all MONAI Physio
    classes. All classes share a common logger called "MONAIPhysio" but include
    their class name in log messages for identification.

    Class Attributes:
        _shared_logger (logging.Logger): Shared logger for all MONAI Physio classes
        _class_filter (ClassNameFilter): Filter for controlling which classes show logs
        _logger_initialized (bool): Whether the shared logger has been set up

    Instance Attributes:
        class_name (str): Name of the class for log message prefixing
        log_level (int): Current logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Example:
        >>> import logging
        >>> class MyRegistration(MONAIPhysioBase):
        ...     def __init__(self):
        ...         super().__init__(class_name='MyRegistration', log_level=logging.INFO)
        ...
        ...     def register(self):
        ...         self.log_info('Starting registration...')
        ...         self.log_debug('Using parameters: ...')
        >>>
        >>> # Filter to show only specific classes
        >>> MONAIPhysioBase.set_log_classes(['MyRegistration'])
        >>>
        >>> # Show all classes again
        >>> MONAIPhysioBase.set_log_all_classes()
    """

    # Class-level shared logger and filter
    _shared_logger: Optional[logging.Logger] = None
    _class_filter: Optional[ClassNameFilter] = None
    _logger_initialized: bool = False

    def __init__(
        self,
        class_name: Optional[str] = None,
        log_level: int | str = logging.INFO,
        log_to_file: Optional[str] = None,
    ):
        """Initialize the base class with logging configuration.

        Args:
            class_name: Name for the class (used in log messages). If None, uses
                the class name. Default: None
            log_level: Logging level. Can be an integer (logging.DEBUG, logging.INFO,
                logging.WARNING, logging.ERROR, logging.CRITICAL) or a string
                ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
                Default: logging.INFO
            log_to_file: Optional file path to write logs to in addition to
                console output. Default: None
        """
        # Set up class name for log message prefixing
        if class_name is None:
            class_name = self.__class__.__name__
        self.class_name = class_name

        # Initialize shared logger if not already done
        if not MONAIPhysioBase._logger_initialized:
            MONAIPhysioBase._initialize_shared_logger(log_level, log_to_file)

        # Get reference to shared logger
        assert MONAIPhysioBase._shared_logger is not None
        self.logger: logging.Logger = MONAIPhysioBase._shared_logger

        # Store the log level
        if isinstance(log_level, str):
            int_log_level = getattr(logging, log_level.upper())
        else:
            int_log_level = log_level
        self.log_level: int = int_log_level

    @classmethod
    def _initialize_shared_logger(
        cls, log_level: int | str, log_to_file: Optional[str] = None
    ) -> None:
        """Initialize the shared logger (called once)."""
        if cls._logger_initialized:
            return

        # Create the shared logger
        cls._shared_logger = logging.getLogger("MONAIPhysio")

        # Convert string log level to integer if needed
        if isinstance(log_level, str):
            int_log_level = getattr(logging, log_level.upper())
            cls._shared_logger.setLevel(int_log_level)
        else:
            cls._shared_logger.setLevel(log_level)

        # Remove existing handlers to avoid duplicates
        cls._shared_logger.handlers.clear()

        # Create class filter
        cls._class_filter = ClassNameFilter()

        # Create console handler with formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.addFilter(cls._class_filter)

        # Create formatter (includes class name in message)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

        # Add console handler to logger
        cls._shared_logger.addHandler(console_handler)

        # Optionally add file handler
        if log_to_file is not None:
            file_handler = logging.FileHandler(log_to_file)
            file_handler.setLevel(log_level)
            file_handler.addFilter(cls._class_filter)
            file_handler.setFormatter(formatter)
            cls._shared_logger.addHandler(file_handler)

        # Prevent propagation to root logger to avoid duplicate messages
        cls._shared_logger.propagate = False

        cls._logger_initialized = True

    @classmethod
    def set_log_level(cls, log_level: int | str) -> None:
        """Set the logging level for all MONAI Physio classes.

        Args:
            log_level: Logging level. Can be an integer (logging.DEBUG, logging.INFO,
                logging.WARNING, logging.ERROR, logging.CRITICAL) or a string
                ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').

        Example:
            >>> import logging
            >>> MONAIPhysioBase.set_log_level(logging.DEBUG)
            >>> # or
            >>> MONAIPhysioBase.set_log_level('DEBUG')
        """
        # Convert string log level to integer if needed
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper())

        if cls._shared_logger is not None:
            cls._shared_logger.setLevel(log_level)

            # Update all handlers
            for handler in cls._shared_logger.handlers:
                handler.setLevel(log_level)

    @classmethod
    def set_log_classes(cls, class_names: list[str]) -> None:
        """Set which classes should show their logging output.

        Only log messages from the specified classes will be displayed.
        All other classes will have their logs hidden.

        Args:
            class_names: List of class names to show logs from.
                Example: ["RegisterModelsPCA", "WorkflowFitStatisticalModelToPatient"]

        Example:
            >>> MONAIPhysioBase.set_log_classes(['RegisterModelsPCA'])
            >>> # Now only RegisterModelsPCA logs will be shown
        """
        if cls._class_filter is not None:
            cls._class_filter.enabled = True
            cls._class_filter.allowed_classes = set(class_names)

    @classmethod
    def set_log_all_classes(cls) -> None:
        """Enable logging output from all MONAI Physio classes.

        Disables the class filter so all classes show their logs.

        Example:
            >>> MONAIPhysioBase.set_log_all_classes()
            >>> # Now all classes will show their logs
        """
        if cls._class_filter is not None:
            cls._class_filter.enabled = False
            cls._class_filter.allowed_classes.clear()

    @classmethod
    def get_log_classes(cls) -> list[str]:
        """Get the list of classes currently showing logs.

        Returns:
            List of class names that are allowed to show logs.
            Empty list if filter is disabled (all classes shown).

        Example:
            >>> classes = MONAIPhysioBase.get_log_classes()
            >>> print(classes)
            ['RegisterModelsPCA', 'WorkflowFitStatisticalModelToPatient']
        """
        if cls._class_filter is not None and cls._class_filter.enabled:
            return sorted(cls._class_filter.allowed_classes)
        return []

    def log_debug(self, message: str, *args: Any) -> None:
        """Log a debug message with optional %-style formatting.

        Args:
            message: The debug message to log (can contain %-style placeholders)
            *args: Arguments for %-style string formatting

        Example:
            >>> self.log_debug('Processing %s with %d items', filename, count)
            >>> self.log_debug('Value is %(value)d', {'value': 42})
        """
        self._log(logging.DEBUG, message, *args)

    def log_info(self, message: str, *args: Any) -> None:
        """Log an info message with optional %-style formatting.

        Args:
            message: The info message to log (can contain %-style placeholders)
            *args: Arguments for %-style string formatting

        Example:
            >>> self.log_info('Loading file: %s', filepath)
            >>> self.log_info('Iteration %(iter)d of %(total)d', {'iter': 5, 'total': 10})
        """
        self._log(logging.INFO, message, *args)

    def log_warning(self, message: str, *args: Any) -> None:
        """Log a warning message with optional %-style formatting.

        Args:
            message: The warning message to log (can contain %-style placeholders)
            *args: Arguments for %-style string formatting

        Example:
            >>> self.log_warning('Memory usage at %d%%', usage_percent)
            >>> self.log_warning('Parameter %(name)s out of range', {'name': 'threshold'})
        """
        self._log(logging.WARNING, message, *args)

    def log_error(self, message: str, *args: Any) -> None:
        """Log an error message with optional %-style formatting.

        Args:
            message: The error message to log (can contain %-style placeholders)
            *args: Arguments for %-style string formatting

        Example:
            >>> self.log_error('Failed to load %s: %s', filename, error_msg)
            >>> self.log_error('Error code: %(code)d', {'code': 404})
        """
        self._log(logging.ERROR, message, *args)

    def log_critical(self, message: str, *args: Any) -> None:
        """Log a critical message with optional %-style formatting.

        Args:
            message: The critical message to log (can contain %-style placeholders)
            *args: Arguments for %-style string formatting

        Example:
            >>> self.log_critical('System failure at %s', timestamp)
            >>> self.log_critical('Critical error: %(msg)s', {'msg': 'Out of memory'})
        """
        self._log(logging.CRITICAL, message, *args)

    def _log(self, level: int, message: str, *args: Any) -> None:
        """Internal method to log with class name attached and %-style formatting.

        Args:
            level: Logging level
            message: Message to log (can contain %-style placeholders)
            *args: Arguments for %-style string formatting
        """
        # Prepend class name to message
        formatted_message = f"{self.class_name} {message}"

        # Create a log record with extra class_name attribute for filtering
        if self.logger.isEnabledFor(level):
            record = self.logger.makeRecord(
                self.logger.name,
                level,
                "(unknown file)",
                0,
                formatted_message,
                args,  # Pass args for lazy %-style formatting
                None,
            )
            record.class_name = self.class_name
            self.logger.handle(record)

    def log_section(
        self, title: str, *args: Any, width: int = 70, char: str = "="
    ) -> None:
        """Log a formatted section header with optional %-style formatting.

        Useful for visually separating major sections of output.

        Args:
            title: The section title (can contain %-style placeholders)
            *args: Arguments for %-style string formatting of title
            width: Total width of the header line. Default: 70
            char: Character to use for the header line. Default: '='

        Example:
            >>> self.log_section('Stage 1: Initialization')
            >>> self.log_section('Processing file: %s', filename)
            >>> self.log_section('Stage %(num)d: %(name)s', {'num': 2, 'name': 'Analysis'})
            # Outputs:
            # ======================================================================
            # Stage 2: Analysis
            # ======================================================================
        """
        separator = char * width
        self.log_info(separator)
        # Use log_info with args to leverage lazy formatting
        self.log_info(title, *args)
        self.log_info(separator)

    def log_progress(self, current: int, total: int, prefix: str = "Progress") -> None:
        """Log progress information.

        Args:
            current: Current step/iteration number
            total: Total number of steps/iterations
            prefix: Prefix text for the progress message. Default: 'Progress'

        Example:
            >>> for i in range(100):
            ...     self.log_progress(i + 1, 100)
            >>> self.log_progress(5, 10, prefix='Processing')

        Note:
            For custom formatted progress messages, use log_info() directly:
            >>> self.log_info('Loading %s: %d/%d', filename, current, total)
        """
        percentage = (current / total) * 100 if total > 0 else 0
        self.log_info("%s: %d/%d (%.1f%%)", prefix, current, total, percentage)
