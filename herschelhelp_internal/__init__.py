# -*- coding: utf-8 -*-

import subprocess
from os.path import dirname


def git_version():
    """Returns the git version of the module

    This function returns a string composed of the abbreviated Git hash of the
    module source, followed by the date of the last commit.  If the source has
    some local modifications, “ [with local modifications]” is added to the
    string.

    This is used to print the exact version of the source code that was used
    inside a Jupiter notebook.
    """
    module_dir = dirname(__file__)

    command_hash = "cd {} && git rev-list --max-count=1 " \
        "--abbrev-commit HEAD".format(module_dir)
    command_date = "cd {} && git log -1 --format=%cd" \
        .format(module_dir)
    command_modif = "cd {} && git diff-index --name-only HEAD" \
        .format(module_dir)

    try:
        commit_hash = subprocess.check_output(command_hash, shell=True)\
            .decode('ascii').strip()
        commit_date = subprocess.check_output(command_date, shell=True)\
            .decode('ascii').strip()
        commit_modif = subprocess.check_output(command_modif, shell=True)\
            .decode('ascii').strip()

        version = "{} ({})".format(commit_hash, commit_date)
        if commit_modif:
            version += " [with local modifications]"
    except subprocess.CalledProcessError:
        version = "Unable to determine version."

    return version
