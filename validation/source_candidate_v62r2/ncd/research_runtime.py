"""Configure the optional project-local Julia runtime without global settings."""
from pathlib import Path
import os

def configure_julia(project_root=None):
    root=Path(project_root or os.environ["NCD_PROJECT_ROOT"]).resolve() if (project_root or os.environ.get("NCD_PROJECT_ROOT")) else Path(__file__).resolve().parents[1]
    depot=root/".julia";project=depot/"environments"/"pysr"
    binary=project/"pyjuliapkg"/"install"/"bin"
    os.environ.setdefault("JULIA_DEPOT_PATH",str(depot))
    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT",str(project))
    # Explicit paths avoid libjulia discovery's mojibake on non-ASCII Windows paths.
    if (binary/"julia.exe").is_file():
        os.environ.setdefault("PYTHON_JULIACALL_EXE",str(binary/"julia.exe"))
        os.environ.setdefault("PYTHON_JULIACALL_PROJECT",str(project))
        os.environ.setdefault("PYTHON_JULIACALL_LIB",str(binary/"libjulia.dll"))
        os.environ.setdefault("PYTHON_JULIACALL_BINDIR",str(binary))
    os.environ.setdefault("JULIA_NUM_THREADS","2")
    return {"depot":str(depot),"project":str(project)}
