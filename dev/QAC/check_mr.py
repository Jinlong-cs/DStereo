import os
import re

hint_message = """
Merge Request Title:

{mr_title}

The merge request title cannot match the Horizon commit message rule: https://gitlab.hobot.cc/tmg/se/-/blob/master/commit_message/commit_message.md#commit-header

The merge request title should match the pattern: {mr_pattern}

Merge Request Title要求: <类型>(<范围>): [JIRAID] <总结>

Examples:
fix(modules): [MATRIX5-56] Xxxxxxxx
feat(client): [AIDI-351] Add xxxxx
perf(lane module): [HAT-74] Xxxx
style(lane): [ADASJ2MONO-1399] Resolve QAC warnings xxxx
fix(client): [ADASJ2MONO-1566] Xxx
docs(workflow): [cr_id_skip] Add xxx

Possible reasons are:
1. 是否缺失JIRAID, 如果没有JIRA ID, 请使用cr_id_skip替代
2. 类型是否使用不符合规范? 可选值为feat|fix|bugfix|hotfix|docs|style|refactor|perf|test|chore
3. 总结语句首字母是否大写
4. JIRAID前后是否有一个以上的空格
5. 是否含有中文字符
"""  # noqa


def main():

    branch = os.environ.get("gitlabTargetBranch")

    if branch != "master":
        return

    merge_request_title = os.environ["gitlabMergeRequestTitle"]

    pattern = "^(feat|fix|bugfix|hotfix|docs|style|refactor|perf|test|chore)\(.*\): \[([a-zA-Z][a-zA-Z0-9_]+-[1-9][0-9]*|cr_id_skip)\] [A-Z]+.*"  # noqa

    if not re.match(pattern, merge_request_title):
        raise ValueError(
            hint_message.format(
                mr_title=merge_request_title, mr_pattern=pattern
            )
        )


if __name__ == "__main__":
    main()
