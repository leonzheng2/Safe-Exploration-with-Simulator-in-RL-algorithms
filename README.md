For Python Codec: the one in rl-glue community website is written in Python2. Please find a version ported in Python3, like https://github.com/steckdenis/rlglue-py3

Some little modifications are required in the code:

```python
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
```
instead of from io import StringIO

```python
StringIO()
```

instead of StringIO.StringIO('')


```python
from rlglue.agent.ClientAgent import ClientAgent
```

instead of from ClientAgent import Client Agent

```python
taskspec.decode()
```

instead of taskspec