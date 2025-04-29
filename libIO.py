from tkinter import Tk
from tkinter import filedialog


# upload video file
def fileOpenDialog(file_type):
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title=f"Select {file_type} file",
        filetypes=[(f"{file_type.upper()} files", f"*.{file_type.lower()}"), ("All files", "*.*")]
    )
    return file_path
