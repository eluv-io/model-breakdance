import os

def setup_checkpoints_symlink():
    target = "~/.cache/imagebind/checkpoints"
    link_name = ".checkpoints"
    os.makedirs(target, exist_ok=True)

    if os.path.islink(link_name) or os.path.isdir(link_name):
        return  # already set

    if os.path.exists(link_name):
        raise RuntimeError(f"{link_name} exists and is not a symlink")

    os.symlink(target, link_name)

# imagebind hardcodes the checkpoints to working dir, so we link it to ~/.cache which will be mounted
# in the container
setup_checkpoints_symlink()