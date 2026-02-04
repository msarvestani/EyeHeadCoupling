from __future__ import annotations

import tkinter as tk
from tkinter import filedialog


def select_folder() -> str:
    """Open a dialog for the user to select a folder."""
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory()


def select_file() -> str:
    """Open a dialog for the user to select a file."""
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename()


def choose_option(option1: str, option2: str, option3: str, option4: str) -> str:
    """Prompt the user to choose one of four options."""
    result: dict[str, str] = {}

    def select(choice: str) -> None:
        result["value"] = choice
        root.destroy()

    root = tk.Tk()
    root.title("Choose the type of visual stim")
    tk.Label(root, text="Please choose the type of visual stim:").pack(pady=10)
    tk.Button(root, text=option1, width=12, command=lambda: select(option1)).pack(side="left", padx=10, pady=10)
    tk.Button(root, text=option2, width=12, command=lambda: select(option2)).pack(side="left", padx=10, pady=10)
    tk.Button(root, text=option3, width=12, command=lambda: select(option3)).pack(side="left", padx=10, pady=10)
    tk.Button(root, text=option4, width=12, command=lambda: select(option4)).pack(side="left", padx=10, pady=10)

    while "value" not in result:
        root.update()

    return result["value"]


__all__ = ["select_folder", "select_file", "choose_option"]
