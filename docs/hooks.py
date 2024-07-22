import logging
import os
import shutil
from textwrap import dedent

log = logging.getLogger("mkdocs")

# Global variable to store the paths of created __init__.py files
created_files = []


def on_startup(*args, **kwargs):
    log.info("Creating missing __init__.py files in rl4co package")
    for subdir, dirs, files in os.walk("rl4co"):
        if "__init__.py" not in files:
            init_file_path = os.path.join(subdir, "__init__.py")
            with open(init_file_path, "w"):
                pass  # empty file
            created_files.append(init_file_path)
    log.info(f"{len(created_files)} __init__.py files created")

    # huge trick: we save a backup of README.md and append CSS content to hide some elements
    log.info("Saving backup of README.md and appending CSS content")
    shutil.copyfile("README.md", "README_backup.md")
    # warning: don't touch any of the following. you have been warned :)
    def append_css_to_readme(file_path):
        css_content = dedent("""
            ---
            hide:
            - navigation
            - toc
            --- 

            <div>                        
            <style type="text/css">
            .md-typeset h1,
            .md-content__button {
                display: none;
            }
            </style>      
            </div> 
                                                            
            """)[1:] # remove "\n" from the beginning
        if not os.path.exists(file_path):
            print(f"Error: The file {file_path} does not exist.")
            return        
        with open(file_path, 'r') as original:
            data = original.read()
        with open(file_path, 'w') as modified:
            modified.write(css_content + data)
        print(f"CSS content has been appended to {file_path}")

    append_css_to_readme("README.md")


def on_shutdown(*args, **kwargs):
    log.info(f"Removing {len(created_files)} created __init__.py files")
    for file_path in created_files:
        if os.path.exists(file_path):
            os.remove(file_path)

    log.info("Replace README.md with README_backup.md")
    shutil.move("README_backup.md", "README.md")