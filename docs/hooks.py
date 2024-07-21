import shutil
import logging, os

log = logging.getLogger('mkdocs')

def on_startup(*args, **kwargs):
    log.info('Copying examples directory to docs')
    shutil.copytree('examples', 'docs/examples', dirs_exist_ok=True)
    log.info('Copying README.md to docs as index.md')
    shutil.copyfile('README.md', 'docs/index.md')
    
    
def on_shutdown(*args, **kwargs):
    log.info('Removing copied examples and index.md')
    shutil.rmtree('docs/examples')
    os.remove('docs/index.md')
