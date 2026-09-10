import os
import subprocess
import sys

# Folder that CONTAINS dot.exe (not the exe itself)
graphviz_bin = r"C:\Users\wesveng\Documents\PhD\Projects\2026-007_microglia_ECS\analysis\diffusion\mcm\Graphviz-10.0.1-win64\bin"

# Inject Graphviz into PATH for this kernel session
if graphviz_bin not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + graphviz_bin

print("Python executable:", sys.executable)
print("Graphviz bin:", graphviz_bin)

# This must work
subprocess.run(["dot", "-V"], check=True)