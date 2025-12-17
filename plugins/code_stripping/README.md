# [experimental] code striping tools
The tools of code stripping is still experimental, which need more practice.

## How to run
```bash
python3 code_stripping.py --file-list configs/toolchain-file-list.py

python3 code_check.py
```

## Notes
All tests after stripped can be performed in the new release dir.

## init format
Current code stripping will refine __init__.py which must depend on code after autoformat.

__init__.py support stripping like this:

```python
   from . import x, y

   from . import (
       x,
       y,
   )

   from .x import X

   from .x import (
       X,
       Y,
   )

   __all__ = [
       X,
       Y,
   ]
```
