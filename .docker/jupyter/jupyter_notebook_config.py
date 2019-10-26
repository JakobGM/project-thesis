# Configuration file for jupyter-notebook running in development

# Answer yes to any prompts.
c.JupyterApp.answer_yes = True

# Full path of a config file.
c.JupyterApp.config_file = "/root/.jupyter/jupyter_notebook_config.py"

# Generate default config file.
c.JupyterApp.generate_config = False

# Allow requests where the Host header doesn't point to a local server
c.NotebookApp.allow_remote_access = True

# Whether to allow the user to run the notebook as root.
c.NotebookApp.allow_root = True

# The base URL for the notebook server.
c.NotebookApp.base_url = "/"

# Whether to enable MathJax for typesetting math/TeX
c.NotebookApp.enable_mathjax = True

# The IP address the notebook server will listen on.
c.NotebookApp.ip = "0.0.0.0"

# Hostnames to allow as local when allow_remote_access is False.
# Local IP addresses (such as 127.0.0.1 and ::1) are automatically accepted as
# local as well.
c.NotebookApp.local_hostnames = ["localhost", "0.0.0.0"]

# The MathJax.js configuration file that is to be used.
c.NotebookApp.mathjax_config = "TeX-AMS-MML_HTMLorMML-full,Safe"

# The directory to use for notebooks and kernels.
c.NotebookApp.notebook_dir = "/code"

# Whether to open in a browser after starting. The specific browser used is
# platform dependent and determined by the python standard library `webbrowser`
# module, unless it is overridden using the --browser (NotebookApp.browser)
# configuration option.
c.NotebookApp.open_browser = False

# The port the notebook server will listen on.
c.NotebookApp.port = 8888

# If True, display a button in the dashboard to quit (shutdown the notebook
#  server).
c.NotebookApp.quit_button = False

# Set to False to disable terminals.
c.NotebookApp.terminals_enabled = True

# Disable password/token functionality
c.NotebookApp.token = ""
c.NotebookApp.password = ""

# Make the link clickable when run through docker
c.NotebookApp.custom_display_url = "http://localhost:8888/"

# Redirect to Jupyter Lab on entry
c.NotebookApp.default_url = "/lab"
