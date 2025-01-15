import rich

_log_styles = {
    "RTGSLAM": "bold green",
    "GUI": "bold red",
    "Eval": "bold red",
    "Tracker": "bold cyan",
    "FrontEnd_1": "bold purple",
    "Mapper" : "bold yellow"
}

def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag_msg=None, tag="GUI"):
    style = get_style(tag)
    if not tag_msg:
        tag_msg = tag
    rich.print(f"[{style}]{tag_msg}:[/{style}]", *args)