import os

import matplotlib.pyplot as plt
import pandas as pd


class OutputWriter:
    """
    Saves the figures, tables and console log produced by the analysis.

    Every message passed to ``log`` is printed and also kept in memory so the
    full run can be written to a text file at the end. Figures are saved as PNG
    files and tables as CSV files under a single output directory.

    Attributes:
        root (str): Root output directory.
        figures_dir (str): Directory for saved figures (``root/figures``).
        tables_dir (str): Directory for saved CSV tables (``root/tables``).
        log_lines (list[str]): Every message logged so far, in order.
    """
    def __init__(self, root: str = "outputs"):
        """
        Creates the output directories if they do not already exist.

        Args:
            root (str, optional): Root output directory. Defaults to "outputs".

        Returns:
            None
        """
        self.root = root
        self.figures_dir = os.path.join(root, "figures")
        self.tables_dir = os.path.join(root, "tables")
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
        self.log_lines = []

    def log(self, message: str = "") -> None:
        """
        Prints a message and stores it for the summary log.

        Args:
            message (str, optional): Text to print. Defaults to an empty line.
        """
        print(message)
        self.log_lines.append(str(message))

    def header(self, title: str) -> None:
        """
        Logs a section title between two horizontal rules.

        Args:
            title (str): Section title.
        """
        self.log("\n" + "=" * 78)
        self.log(title)
        self.log("=" * 78)

    def save_figure(self, name: str) -> None:
        """
        Saves the current matplotlib figure as a PNG and displays it.

        Args:
            name (str): File name without the extension.
        """
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, f"{name}.png"))
        plt.show()
        self.log(f"  [figure] {name}.png")

    def save_table(self, df: pd.DataFrame, name: str, index: bool = True, float_format: str = "%.3f") -> None:
        """
        Saves a DataFrame as a CSV file in the tables directory.

        Args:
            df (pd.DataFrame): Table to save.
            name (str): File name without the extension.
            index (bool, optional): Whether to write the index. Defaults to True.
            float_format (str, optional): Format string for floats. Defaults to "%.3f".
        """
        df.to_csv(os.path.join(self.tables_dir, f"{name}.csv"), index=index, float_format=float_format)
        self.log(f"  [table]  {name}.csv")

    def save_log(self, file_name: str = "eda_summary.txt") -> None:
        """
        Writes every logged message to a text file in the root directory.

        Args:
            file_name (str, optional): Name of the log file. Defaults to "eda_summary.txt".
        """
        with open(os.path.join(self.root, file_name), "w", encoding="utf-8") as file_out:
            file_out.write("\n".join(self.log_lines))
