For Python Codec: the one in rl-glue community website is written in Python2. Please find a version ported in Python3, like https://github.com/okkhoy/rlglue-python3-codec

Some little modifications are required in the code:

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

instead of from io import StringIO

StringIO()

instead of StringIO.StringIO('')

from rlglue.agent.ClientAgent import ClientAgent

instead of from ClientAgent import Client Agent