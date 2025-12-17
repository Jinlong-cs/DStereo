import argparse
import contextlib
import logging
import os
import re
import subprocess
import time

import yaml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

IS_LOCAL = not os.path.exists("/running_package")


def init_logger(logger_name):
    head = "%(asctime)-15s %(levelname)s [Watchdog][%(filename)s:%(lineno)d] : %(message)s"  # noqa E501
    logger = logging.getLogger(logger_name)
    default_log_dir = "./watchdog" if IS_LOCAL else "/job_log"
    log_dir = os.getenv("JOB_LOG_PATH", default_log_dir)
    if os.path.exists(log_dir):
        log_file = os.path.join(log_dir, "watchdog.log")
    else:
        with contextlib.suppress(FileExistsError):
            os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(head)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger


logger = init_logger(__name__)

FILE_POSITION_RECORD = {}


class PatternType:
    RESTART = "restart"
    EXIT = "exit"
    CHECK_NCCL = "check_nccl"
    TMP = "123"


PATTERN_TYPES_DEFAULT_EXITCODE = {
    PatternType.RESTART: 200,
    PatternType.EXIT: -1,
    PatternType.CHECK_NCCL: 0,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-path",
        type=str,
        required=True,
        help="log file or dir path to monitor",
    )

    parser.add_argument(
        "--time-interval",
        type=int,
        default=1,
        help="monitoring time interval, in minutes.",
    )
    parser.add_argument(
        "--pattern-files",
        type=str,
        nargs="+",
        help="pattern file to handle exitcode.",
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def regex_search(in_str: str, pattern: str):
    """Find the first match object in the string that can \
        match the regular expression pattern.

    Args:
        in_str: Input str.
        pattern: Regular expression pattern.

    Returns:
        Regular expression pattern match result.
    """

    assert pattern is not None, "`pattern` can not be None."
    prog = re.compile(pattern)
    r = prog.search(in_str)
    return r.group() if r else None


def read_pattern(pattern_files, target_pattern_types):
    def _patch_pattern_with_exit_code(data, key, exit_code=-1):
        settings = data.get(key, None)

        pattern_with_code = {}
        if settings:
            patterns = settings.get("patterns", None)
            exit_codes = settings.get("exitcode", [])

            if patterns:
                if len(exit_codes) < len(patterns):
                    exit_codes += [exit_code] * (
                        len(patterns) - len(exit_codes)
                    )
                pattern_with_code = dict(zip(patterns, exit_codes))
        return pattern_with_code

    all_patterns = {}

    for pattern_file in pattern_files:
        assert os.path.exists(pattern_file)
        with open(pattern_file, "r") as file:
            data = yaml.safe_load(file)

        if data is None:
            continue

        for pattern_type in target_pattern_types:
            pattern_with_code = _patch_pattern_with_exit_code(
                data=data,
                key=pattern_type,
                exit_code=PATTERN_TYPES_DEFAULT_EXITCODE[pattern_type],
            )

            if len(pattern_with_code) > 0:
                if pattern_type in all_patterns.keys():
                    all_patterns[pattern_type].update(pattern_with_code)
                else:
                    all_patterns[pattern_type] = pattern_with_code

    # merge restart and exit
    if (
        PatternType.EXIT in target_pattern_types
        and PatternType.RESTART in target_pattern_types
    ):
        exit_patterns = all_patterns.get(PatternType.EXIT, None)
        restart_patterns = all_patterns.get(PatternType.RESTART, None)

        if exit_patterns and restart_patterns:
            for exit_pattern in list(exit_patterns.keys()):
                if exit_pattern in restart_patterns:
                    restart_patterns.pop(exit_pattern)

            all_patterns[PatternType.RESTART] = restart_patterns

    return all_patterns


def run_nccl_test():
    try:
        result = subprocess.run(
            "./nccl_test.sh",
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            env=os.environ.copy(),
        )
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"subprocess({e.cmd}) failed({e.returncode})! {e.output}.\n"
        )
        result = None

    return result


class LogMonitorHandler(FileSystemEventHandler):
    def __init__(
        self,
        log_path,
        log_regex,
        pattern_types,
        pattern_files,
    ):
        super().__init__()

        self.log_path = log_path
        self.log_regex = log_regex
        self.pattern_types = pattern_types
        self.patterns = read_pattern(pattern_files, pattern_types)

    def on_modified(self, event):
        if not event.is_directory:
            file_path = event.src_path
            logger.debug(f"Modified11: {file_path}")
            # monitor error log
            is_expected_file = self._is_expected_log_file(file_path)
            if is_expected_file:
                logger.debug(f"Modified: {file_path}")
                self._read_and_analysis_log(file_path=file_path)

    def _read_and_analysis_log(self, file_path):
        global FILE_POSITION_RECORD
        buffer_size = 8192
        last_position = FILE_POSITION_RECORD.get(file_path, 0)
        first_exception_flag = 0

        code = 0
        with open(file_path, "rb") as file:
            file.seek(last_position)
            while True:
                data = file.read(buffer_size)
                if not data:
                    break
                lines = data.decode().splitlines()
                matched_type, code = self._process_log_lines(lines)
                if code != 0 and first_exception_flag == 0:
                    first_exception_flag = code
            last_position = file.tell()
        FILE_POSITION_RECORD[file_path] = last_position

        if (
            matched_type in [PatternType.EXIT, PatternType.RESTART]
            and first_exception_flag != 0
        ):
            os._exit(first_exception_flag)

        if matched_type in [PatternType.CHECK_NCCL]:
            run_nccl_test()

    def _process_log_lines(self, lines):
        for line in lines:
            logger.debug(f"Find: {line}")
            # check -> exit -> restart
            for pattern_type in sorted(self.pattern_types):
                for pattern, exitcode in self.patterns.get(
                    pattern_type, {}
                ).items():
                    ret = regex_search(line, pattern)
                    if ret:
                        logger.warning(
                            f"Find `{ret}` in log file, will {pattern_type} and return {exitcode}"  # noqa E501
                        )
                        return pattern_type, exitcode
        return None, 0

    def _is_expected_log_file(self, file_path):
        pattern = re.compile(self.log_regex)
        if pattern.match(os.path.basename(file_path)):
            if os.path.isfile(self.log_path):
                return os.path.normpath(file_path) == os.path.normpath(
                    self.log_path
                )  # noqa E501
            else:
                assert os.path.isdir(self.log_path)
                return os.path.normpath(
                    os.path.dirname(file_path)
                ) == os.path.normpath(  # noqa E501
                    self.log_path
                )
        else:
            return False


def find_log_files(log_path: str, regex: str):
    if os.path.isdir(log_path):
        all_files = [
            os.path.join(log_path, f)
            for f in os.listdir(log_path)
            if os.path.isfile(os.path.join(log_path, f))
        ]
    else:
        all_files = [log_path]

    pattern = re.compile(regex)
    matched_files = [
        f for f in all_files if pattern.match(os.path.basename(f))
    ]

    return matched_files


class LogWatchDog:
    def __init__(
        self,
        log_dir: str,
        pattern_files: str,
        elastic_task: bool = False,
    ) -> None:
        self.log_dir = log_dir
        self.pattern_files = pattern_files
        self._observer = None
        self.log_files = []
        self.elastic_task = elastic_task

        if elastic_task:
            self.pattern_types = [PatternType.EXIT, PatternType.RESTART]
        else:
            self.pattern_types = [PatternType.CHECK_NCCL]
        self._chowned = False

    def wait(self):
        self._sleep_until_created_log_file()

    def _sleep_until_created_log_file(self):
        while True:
            if os.path.exists(self.log_dir):
                if not self.elastic_task:
                    self.log_files = find_log_files(
                        log_path=self.log_dir,
                        regex="^hobot-job.*task-\\d+\\.log$",
                    )
                else:
                    # elastic task
                    job_restart = int(os.environ.get("JOB_RESTART", "0"))
                    restart_log_dir = os.path.join(
                        self.log_dir, f"job_restart_{job_restart - 1}"
                    )
                    if (
                        job_restart > 0 and os.path.exists(restart_log_dir)
                    ) or job_restart == 0:
                        self.log_files = find_log_files(
                            self.log_dir,
                            regex="-err\\.log$",
                        )
                    else:
                        self.log_files = []
                if len(self.log_files) > 0:
                    break
            time.sleep(1)

    def maybe_chown(self):
        try:
            if not self._chowned:
                for f in self.log_files:
                    os.system(f"chmod 664 {f}")
        except Exception as e:
            logger.warning(f"chown failed: {e}")
        self._chowned = True

    def start(self):
        self.observer.start()

    def stop(self):
        self.observer.stop()

    def join(self):
        self.observer.join()

    def _init_monitor_handler(self):
        handler = LogMonitorHandler(
            log_path=self.log_files,
            log_regex="^hobot-job.*task-\\d+\\.log$",
            pattern_files=self.pattern_files,
            pattern_types=self.pattern_types,
        )
        return handler

    @property
    def observer(self):
        if self._observer is None:
            self._observer = Observer()
            handler = self._init_monitor_handler()
            self._observer.schedule(
                event_handler=handler,
                path=self.log_dir,
                recursive=True,
            )

        return self._observer


def main():
    args, _ = parse_args()

    log_dir = args.log_path
    log_watchdog = LogWatchDog(
        log_dir=log_dir,
        pattern_files=args.pattern_files,
    )

    # wait
    log_watchdog.wait()

    # start
    logger.info("=" * 50 + " START WATCHDOG " + "=" * 50)
    logger.info(f"files: {os.listdir(log_dir)}")
    log_watchdog.start()

    try:
        while True:
            time.sleep(int(args.time_interval) * 3)
            log_watchdog.maybe_chown()
    except Exception as e:
        log_watchdog.stop()
        logger.error(f"LogWatchDog failed: {e}")
    log_watchdog.join()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"watchdog failed: {e}")
