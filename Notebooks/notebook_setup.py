import sys
import os


def setup_paths(parent_steps=2):
    """
    Adds a parent directory to the system path to enable module imports from higher-level directories.

    This is useful when working in a subdirectory (e.g., a notebook or script) and you want to import
    modules from a `src/` or other folder located in a higher-level directory.

    Parameters
    ----------
    parent_steps : int, optional
        Number of directory levels to go up from the current working directory (default is 2).

    Returns
    -------
    None
        Modifies `sys.path` in-place and prints the absolute path added.
    """

    # Adjust the path to make sure Python can find your `src/` package
    path = os.getcwd()
    for _ in range(parent_steps):
        path = os.path.join(path, "..")

    abs_path = os.path.abspath(path)  # Remove the .. by going to parent directory
    sys.path.append(abs_path)
