import sys


def print_and_log(logfile):
    class FileAndPrint:
        def __init__(self, f, out):
            self.file = f
            self.out = out

        def write(self, *args, **kwargs):
            self.out.write(*args, **kwargs)
            f = open(self.file, "a")
            f.write(*args, **kwargs)
            f.close()

        def flush(self, *args, **kwargs):
            self.out.flush(*args, **kwargs)

    sys.stdout = FileAndPrint(logfile, sys.stdout)
