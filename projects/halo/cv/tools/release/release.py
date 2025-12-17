from public import public_dependencies, skip_file_list
from tasks import task_dependencies

all_dependencies = task_dependencies + public_dependencies

file_list = list(set(all_dependencies))
file_list.sort()

with open("halocv_release.py", "w") as writer:
    writer.writelines("file_list = [\n")
    for line in file_list:
        writer.writelines(f'    "{line}",\n')
    writer.writelines("]\n")

    writer.writelines("skip_file_list = [\n")
    for line in skip_file_list:
        writer.writelines(f'    "{line}",\n')
    writer.writelines("]\n")
