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

    log.info("Copying README.md to docs/index.md and adding custom CSS")
    shutil.copyfile("README.md", "docs/index.md")

    def append_css_to_readme(file_path):
        css_content = dedent("""
        <style type="text/css">
        .md-typeset h1,
        .md-content__button {
            display: none;
        }
        </style>
        """)    
        if not os.path.exists(file_path):
            print(f"Error: The file {file_path} does not exist.")
            return        
        with open(file_path, 'a') as file:
            file.write(css_content)
        print(f"CSS content has been appended to {file_path}")

    append_css_to_readme("docs/index.md")


def on_shutdown(*args, **kwargs):
    log.info(f"Removing {len(created_files)} created __init__.py files")
    for file_path in created_files:
        if os.path.exists(file_path):
            os.remove(file_path)

    log.info("Removing docs/index.md")
    if os.path.exists("docs/index.md"):
        os.remove("docs/index.md")