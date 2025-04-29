import subprocess
import threading


class SubProcessThread(threading.Thread):
    def __init__(self, cmd, options=None):
        super().__init__()
        self.cmd = cmd
        self.options = options
        self.stdout = None  # stdout will contain the output of the command
        self.stderr = None  # stderr will contain the error message if any
        self.success = None  # success will contain True/False after running

    def run(self):
        try:
            result = subprocess.run(
                self.cmd,
                stdout=subprocess.PIPE if hasattr(self.options, 'stdoutPipe') and self.options.stdoutPipe == True else None,
                stderr=subprocess.PIPE if hasattr(self.options, 'stderrPipe') and self.options.stderrPipe == True else None,
                text=True if hasattr(self.options, 'text') and self.options.text else False,
                check=True if hasattr(self.options, 'check') and self.options.check else False)

            self.stdout = result.stdout if hasattr(self.options, 'stdoutPipe') and self.options.stdoutPipe == True else None
            self.stderr = result.stderr if hasattr(self.options, 'stderrPipe') and self.options.stderrPipe == True else None
            self.success = True
        except subprocess.CalledProcessError as e:
            self.stdout = result.stdout if hasattr(self.options, 'stdoutPipe') and self.options.stdoutPipe == True else None
            self.stderr = result.stderr if hasattr(self.options, 'stderrPipe') and self.options.stderrPipe == True else None
            self.success = False
            # print(f"Subprocess error: {e}")
        except Exception as e:
            self.stdout = None
            self.stderr = e
            self.success = False
            # print(f"Other error: {e}")
