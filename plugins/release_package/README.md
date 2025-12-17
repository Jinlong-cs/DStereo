# [experimental] Release Package 
The tools of release package.

## Example file description
   We provide a simple example in `example`.

   `example/config.py` is the configuration file of release package. See more in [release package](https://horizonrobotics.feishu.cn/docs/doccne8Q6AuC9UIRvHcgKinQqNb).
   `example/code_strip.sh` is a shell script provided by the user, used to strip code and configure it into the config file.
   `example/codestrip_example.py` is the configuration file of the code strpping.
   `example/create_dockerfile.sh` is the script that generates the dockerfile, it is optional.
   `example/test_package.sh` is the script that use to test, it is optional.
   `example/add_user.sh` is the script that to add user in docker, which is used to access bucket.

## How to run
```bash

python3 release_package.py -release-config example/config.py --target-dir ./release_example
```
