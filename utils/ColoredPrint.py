COLORS = {
    "red": "\033[91m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "green": "\033[92m",
    "reset": "\033[0m"
}

def print_red(message: str):
    print(f"{COLORS['red']}{message}{COLORS['reset']}")

def print_yel(message: str):
    print(f"{COLORS['yellow']}{message}{COLORS['reset']}")

def print_blu(message: str):
    print(f"{COLORS['blue']}{message}{COLORS['reset']}")

def print_gre(message: str):
    print(f"{COLORS['green']}{message}{COLORS['reset']}")
