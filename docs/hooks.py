import logging
import os
import shutil

log = logging.getLogger("mkdocs")

# Global variable to store the paths of created __init__.py files
created_files = []


def on_startup(*args, **kwargs):
    log.info("Copying examples directory to docs")
    shutil.copytree("examples", "docs/examples", dirs_exist_ok=True)
    log.info("Copying README.md to docs as index.md")
    shutil.copyfile("README.md", "docs/index.md")

    log.info("Creating missing __init__.py files in rl4co package")
    for subdir, dirs, files in os.walk("rl4co"):
        if "__init__.py" not in files:
            init_file_path = os.path.join(subdir, "__init__.py")
            with open(init_file_path, "w"):
                pass  # empty file
            created_files.append(init_file_path)
    log.info(f"{len(created_files)} __init__.py files created")


def on_shutdown(*args, **kwargs):
    log.info("Removing copied examples and index.md")
    shutil.rmtree("docs/examples")
    os.remove("docs/index.md")

    log.info(f"Removing {len(created_files)} created __init__.py files")
    for file_path in created_files:
        if os.path.exists(file_path):
            os.remove(file_path)
