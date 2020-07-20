import io
from datetime import datetime

def config_log(input_path):
    now = datetime.now()
    global f
    f = open(input_path + "output" + now.strftime("%m_%d_%H_%M%S") + ".txt", "a")

def print2(*args, **kwargs):
    print(*args, **kwargs)
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    f.write(contents)
    f.flush()
    return contents