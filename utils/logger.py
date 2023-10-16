"""
A simple logger used to display debug info
"""


class Logger:
    def warning(s: str):
        print(f"warning: {s}")

    def debug(s: str):
        print(s)

    def progress(s: str):
        limit = 50
        if len(s) <= limit:
            fill = " " * (limit - len(s))
            log = s + fill
        else:
            log = s[:limit]
        print(log, end="\r", flush=True)
