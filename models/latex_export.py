import os
from glob import glob

import numpy as np
import pandas as pd


def tex_escape(text: str, underscore: str = r"\_") -> str:
    """
    Escapes LaTeX special characters and converts <= and >= to math symbols.

    Args:
        text (str): Text to escape.
        underscore (str, optional): Replacement for "_". Use " " for column
            headers so names read as words. Defaults to r"\\_".

    Returns:
        str: LaTeX-safe text.
    """
    text = str(text)
    replacements = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                    ("_", underscore), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                    ("^", r"\textasciicircum{}")]
    for old, new in replacements:
        text = text.replace(old, new)
    return text.replace("<=", r"$\leq$").replace(">=", r"$\geq$")


def format_cell(value) -> str:
    """
    Formats one table value for LaTeX.

    Integers get thousands separators, floats of 1,000 or more are rounded to
    whole numbers, very small floats use scientific notation, other floats get
    three decimals, and text is escaped.

    Args:
        value: Cell value.

    Returns:
        str: Formatted cell text (empty for missing values).
    """
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        if value != 0 and abs(value) < 1e-3:
            return f"{value:.1e}"
        return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.3f}"
    return tex_escape(value)


class LatexTableExporter:
    """
    Converts the saved CSV tables into booktabs LaTeX tables for the report.

    Each CSV in ``tables_dir`` becomes a ``tabular`` environment in
    ``tex_dir`` with the same file stem, ready to ``\\input{}`` in the report.

    Attributes:
        tables_dir (str): Directory holding the CSV tables.
        tex_dir (str): Directory the LaTeX tables are written to.
        column_formats (dict[str, str]): Column specs for tables that need
            something other than one left column followed by right columns
            (e.g. wrapped p-columns for long text).
    """
    def __init__(self, tables_dir: str, tex_dir: str, column_formats: dict = None):
        """
        Stores the directories and creates the LaTeX directory if needed.

        Args:
            tables_dir (str): Directory holding the CSV tables.
            tex_dir (str): Directory the LaTeX tables are written to.
            column_formats (dict[str, str], optional): Column specs keyed by table name. Defaults to None.

        Returns:
            None
        """
        self.tables_dir = tables_dir
        self.tex_dir = tex_dir
        self.column_formats = column_formats or {}
        os.makedirs(tex_dir, exist_ok=True)

    def csv_to_tex(self, csv_path: str) -> str:
        """
        Converts one CSV table to LaTeX.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            str: The LaTeX ``tabular`` code.
        """
        name = os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path)
        df.columns = ["" if str(c).startswith("Unnamed") else c for c in df.columns]
        column_format = self.column_formats.get(name, "l" + "r" * (df.shape[1] - 1))

        lines = [r"\begin{tabular}{" + column_format + "}", r"\toprule",
                 " & ".join(tex_escape(c, underscore=" ") for c in df.columns) + r" \\", r"\midrule"]
        for row in df.itertuples(index=False):
            cells = [format_cell(v) for v in row]
            if name == "data_dictionary":
                cells[0] = r"\texttt{" + cells[0] + "}"
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    def export_all(self) -> int:
        """
        Converts every CSV in the tables directory and writes the LaTeX files.

        Returns:
            int: Number of LaTeX tables in the output directory.
        """
        for csv_path in sorted(glob(os.path.join(self.tables_dir, "*.csv"))):
            name = os.path.splitext(os.path.basename(csv_path))[0]
            with open(os.path.join(self.tex_dir, f"{name}.tex"), "w", encoding="utf-8") as file_out:
                file_out.write(self.csv_to_tex(csv_path))
        return len(glob(os.path.join(self.tex_dir, "*.tex")))
