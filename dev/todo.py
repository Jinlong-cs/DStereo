#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Keep tracking of TODOs in code."""
import argparse
import os
import re
from typing import Dict, List, Optional

from termcolor import colored


class TODO:
    """One TODO."""

    def __init__(self, todo_str: str, filepath: str):
        self._todo_str = todo_str
        # grep names and time
        m = re.search("\(.*\)", todo_str)  # noqa: W605
        if m is None:
            self._names = None
            self._time = None
        else:
            matched = m.group(0)
            self._names = matched[1:-1].split(",")
            _time = self._names[-1].strip()
            if "?" in _time:
                self._time = None
                self._names = self._names[:-1]
            else:
                try:
                    self._time = float(_time)
                    self._names = self._names[:-1]
                except Exception:
                    self._time = None
        # grep todo info
        m = re.search(":.*", todo_str)
        if m is None:
            self._info = None
        else:
            self._info = m.group(0)[1:].strip()[:-1]
        self._filepath = filepath

    @property
    def names(self) -> List[str]:
        return self._names

    @property
    def time(self) -> float:
        return self._time

    @property
    def info(self) -> str:
        return self._info

    def __repr__(self) -> str:
        _str = colored(f"{self._names}", "red")
        _str += " | "
        _str += colored(f"{self.time}", "green")
        _str += " | "
        _str += colored(f"{self.info}", "blue")
        _str += " | "
        _str += colored(f"{self._filepath}", "white")
        return _str


HAT_MODULES = (
    "docs",
    "tests",
    "tools",
    "hat/utils",
    "hat/callbacks",
    "hat/data",
    "hat/engine",
    "hat/metrics",
    "hat/ops",
    "hat/optimizers",
    "hat/models/backbones",
    "hat/models/task_modules",
    "hat/models/losses",
    "hat/models/necks",
    "hat/models/structures",
    "OTHERS",
)


class TODOHelper:
    @staticmethod
    def names(todos: List[TODO]) -> List[str]:
        """Get names in todo list."""
        _names = []
        for todo in todos:
            _names.extend(todo.names if todo.names else [])
        return set(_names)

    @staticmethod
    def by_names(
        todos: List[TODO],
        names: Optional[List[str]] = None,
    ) -> Dict[str, List[TODO]]:
        """Get todo dict by name as key."""
        if names is None:
            names = TODOHelper.names(todos)
        todo_dict = {name: [] for name in names}
        for todo in todos:
            if not todo.names:
                continue
            for name in todo.names:
                if name in todo_dict:
                    todo_dict[name].append(todo)
        return todo_dict

    @staticmethod
    def by_hat_module(
        todos: List[TODO],
    ) -> Dict[str, List[TODO]]:
        """Get todo dict by hat module."""
        todo_dict = {key: [] for key in HAT_MODULES}
        todo_filepath = {todo._filepath: todo for todo in todos}
        for filepath in todo_filepath:
            matched = False
            for key in HAT_MODULES:
                if key in filepath:
                    todo_dict[key].append(todo_filepath[filepath])
                    matched = True
                    break
            if not matched:
                todo_dict["OTHERS"].append(todo_filepath[filepath])
        return todo_dict

    @staticmethod
    def print(todos: Dict[str, TODO], with_key: bool = False):
        """Print todo dict."""
        for key in todos:
            if len(todos[key]) == 0:
                continue
            if with_key:
                print(f"\n{key}")
            for todo in todos[key]:
                print(todo)

    @staticmethod
    def print_anonymous(todos: List[TODO]):
        """Print anonymous todos."""
        anonymous_todos = [todo for todo in todos if not todo.names]
        if len(anonymous_todos) > 0:
            print("\n### Anonymous TODO ###")
        for todo in anonymous_todos:
            print(todo)


def main(args):
    srcs = []
    black_list = {".git", "__pycache__", ".pytest_cache", "todo.py"}
    white_list = {".py", ".txt", ".rst", ".md"}
    for path, _dirs, files in os.walk(args.input, followlinks=True):
        if len(set(path.split("/")).intersection(set(black_list))) > 0:
            continue
        if len(set(files).intersection(set(black_list))) > 0:
            continue
        srcs += [
            os.path.join(path, x)
            for x in files
            if os.path.splitext(x)[1] in white_list
        ]

    todos = []
    for src in srcs:
        with open(src, "r") as fid:
            for line in fid.readlines():
                m = re.search("TODO.*", line)
                if m is not None:
                    output = m.group(0)
                    todos.append(TODO(output, src))
                else:
                    pass
    if args.by_module:
        todos_by_hat_module = TODOHelper.by_hat_module(todos)
        TODOHelper.print(todos_by_hat_module, with_key=True)
    else:
        todos_by_name = TODOHelper.by_names(todos)
        TODOHelper.print(todos_by_name)

    TODOHelper.print_anonymous(todos)
    print(f"todo sum: {len(todos)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="./", type=str, help="input code path"
    )
    # # TODO(min.du, 0.5): wip #
    # parser.add_argument('--by-module', dest='by_module', action='store_true',
    #                     help='group by module')
    args = parser.parse_args()
    main(args)
