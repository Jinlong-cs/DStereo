# print output of pytest-monitor
# pip3 install pytest-monitor
# pytest-monitor will be opened by default
import os
import sqlite3


def sqlite_to_list(filename):
    if not os.path.exists(filename):
        return

    cur_con = sqlite3.connect(filename)
    cur_from = cur_con.execute(
        "select ITEM, CPU_USAGE, MEM_USAGE from TEST_METRICS;"
    )
    cur_list = cur_from.fetchall()

    for case in cur_list:
        print(case)


if __name__ == "__main__":
    filename = ".pymon"
    sqlite_to_list(filename)
