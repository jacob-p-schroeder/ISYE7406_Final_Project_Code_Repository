import json
import os
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

from .output_writer import OutputWriter

# U.S. CPI-U, U.S. city average, annual averages (1982-84 = 100).
# Source: U.S. Bureau of Labor Statistics, https://www.bls.gov/cpi/
CPI_U = {
    1960: 29.6, 1961: 29.9, 1962: 30.2, 1963: 30.6, 1964: 31.0, 1965: 31.5,
    1966: 32.4, 1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 49.3, 1975: 53.8, 1976: 56.9, 1977: 60.6,
    1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9, 1982: 96.5, 1983: 99.6,
    1984: 103.9, 1985: 107.6, 1986: 109.6, 1987: 113.6, 1988: 118.3, 1989: 124.0,
    1990: 130.7, 1991: 136.2, 1992: 140.3, 1993: 144.5, 1994: 148.2, 1995: 152.4,
    1996: 156.9, 1997: 160.5, 1998: 163.0, 1999: 166.6, 2000: 172.2, 2001: 177.1,
    2002: 179.9, 2003: 184.0, 2004: 188.9, 2005: 195.3, 2006: 201.6, 2007: 207.3,
    2008: 215.3, 2009: 214.5, 2010: 218.1, 2011: 224.9, 2012: 229.6, 2013: 233.0,
    2014: 236.7, 2015: 237.0, 2016: 240.0, 2017: 245.1,
}

ERA_ORDER = ["<=1979", "1980s", "1990s", "2000s", "2010s"]


def parse_json(text: str) -> list:
    """
    Parses a JSON string from the TMDB files, returning an empty list on failure.

    Args:
        text (str): JSON-encoded list (e.g. the cast, crew or genres column).

    Returns:
        list: The parsed list, or an empty list if the text is missing or invalid.
    """
    try:
        return json.loads(text)
    except Exception:
        return []


def normalize_title(title: str) -> str:
    """
    Normalizes a film title so TMDB and Oscar titles can be matched.

    Strips accents, lower-cases, replaces "&" with "and", removes punctuation,
    collapses whitespace and drops a leading "the", "a" or "an".

    Args:
        title (str): Raw film title.

    Returns:
        str: Normalized title key.
    """
    title = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode()
    title = title.lower().replace("&", " and ")
    title = re.sub(r"[^a-z0-9 ]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return re.sub(r"^(the|a|an) ", "", title)


def era_label(year: int) -> str:
    """
    Maps a release year to its era label.

    Args:
        year (int): Release year.

    Returns:
        str: "<=1979" for years before 1980, otherwise the decade (e.g. "1990s").
    """
    return "<=1979" if year < 1980 else f"{(int(year) // 10) * 10}s"


class TMDBDataLoader:
    """
    Loads, engineers, cleans and inflation-adjusts the TMDB 5000 data, and
    merges in Academy Award nominations and wins.

    The pipeline joins the movies and credits files on movie ID, parses the
    nested JSON columns into numeric features (cast size, crew size, crew counts
    by department, genres), filters the films through a cleaning funnel,
    converts dollar amounts to constant ``base_year`` dollars, and matches Oscar
    records on a normalized title and release year.

    Attributes:
        writer (OutputWriter): Writer used to log messages and save tables.
        data_dir (str): Directory holding the raw CSV files.
        min_year (int): First release year kept in the analysis set.
        max_year (int): Last release year kept in the analysis set.
        base_year (int): Year whose dollars all amounts are converted to.
        min_dollars (int): Smallest budget and revenue treated as valid.
        oscar_year_tolerance (int): Allowed difference, in years, between the
            Oscar film year and the TMDB release year when matching.
        era_order (list[str]): Ordered era labels.
        full (pd.DataFrame): Joined and engineered data before cleaning.
        funnel (pd.DataFrame): Films remaining and dropped at each cleaning step.
        analysis (pd.DataFrame): Cleaned, inflation-adjusted analysis set.
        has_awards (bool): Whether the Oscar data was found and merged.
    """
    def __init__(self, writer: OutputWriter, data_dir: str = "Data", min_year: int = 1960, max_year: int = 2017,
                 base_year: int = 2017, min_dollars: int = 10_000, oscar_year_tolerance: int = 1,
                 era_order: list = ERA_ORDER):
        """
        Stores the configuration and checks the year range against the CPI table.

        Args:
            writer (OutputWriter): Writer used to log messages and save tables.
            data_dir (str, optional): Directory holding the raw CSV files. Defaults to "Data".
            min_year (int, optional): First release year kept. Must be at least 1960. Defaults to 1960.
            max_year (int, optional): Last release year kept. Defaults to 2017.
            base_year (int, optional): Year whose dollars all amounts are converted to. Defaults to 2017.
            min_dollars (int, optional): Smallest valid budget and revenue. Defaults to 10,000.
            oscar_year_tolerance (int, optional): Allowed year difference when matching Oscar records. Defaults to 1.
            era_order (list[str], optional): Ordered era labels. Defaults to ERA_ORDER.

        Returns:
            None
        """
        assert min_year >= min(CPI_U) and max_year <= max(CPI_U), "year range outside CPI table"
        self.writer = writer
        self.data_dir = data_dir
        self.min_year = min_year
        self.max_year = max_year
        self.base_year = base_year
        self.min_dollars = min_dollars
        self.oscar_year_tolerance = oscar_year_tolerance
        self.era_order = era_order
        self.has_awards = False

    def load(self) -> pd.DataFrame:
        """
        Joins the movies and credits files and engineers features from the JSON columns.

        Creates cast size, crew size, crew counts by department, the number of
        distinct crew departments, the genre list, number of genres, primary
        genre, number of production companies, an English-language flag, and
        release year and month.

        Returns:
            pd.DataFrame: The joined and engineered data, also stored as ``self.full``.
        """
        self.writer.header("LOADING DATA")
        movies = pd.read_csv(os.path.join(self.data_dir, "tmdb_5000_movies.csv"))
        credits = pd.read_csv(os.path.join(self.data_dir, "tmdb_5000_credits.csv"))
        self.writer.log(f"movies:  {movies.shape}   credits: {credits.shape}")

        credits = credits.rename(columns={"movie_id": "id"}).drop(columns=["title"], errors="ignore")
        df = movies.merge(credits, on="id", how="inner", validate="one_to_one").reset_index(drop=True)
        self.writer.log(f"merged on movie id: {df.shape}")

        # Cast and crew: nested JSON to numeric features
        cast_lists = df["cast"].map(parse_json)
        crew_lists = df["crew"].map(parse_json)
        df["cast_size"] = cast_lists.map(len)
        df["crew_size"] = crew_lists.map(len)
        departments = pd.DataFrame([Counter(member.get("department", "Unknown") for member in crew)
                                    for crew in crew_lists]).fillna(0).astype(int)
        departments.columns = ["crew_" + re.sub(r"[^A-Za-z0-9]+", "_", c).strip("_").lower()
                               for c in departments.columns]
        df = pd.concat([df, departments], axis=1)
        df["crew_departments"] = (departments > 0).sum(axis=1)

        # Genres, production companies and language
        df["genres_list"] = df["genres"].map(lambda s: [g["name"] for g in parse_json(s)])
        df["n_genres"] = df["genres_list"].map(len)
        df["primary_genre"] = df["genres_list"].map(lambda g: g[0] if g else "Unknown")
        df["n_prod_companies"] = df["production_companies"].map(lambda s: len(parse_json(s)))
        df["is_english"] = (df["original_language"] == "en").astype(int)

        # Release dates
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year
        df["release_month"] = df["release_date"].dt.month

        self.full = df
        return df

    def clean(self) -> pd.DataFrame:
        """
        Filters the joined data through the cleaning funnel and adjusts for inflation.

        Keeps released films within the year range that have a budget and revenue
        of at least ``min_dollars``, a positive runtime, and non-empty cast and
        crew lists. Saves the funnel as ``cleaning_funnel.csv``, then adds real
        (inflation-adjusted) budget and revenue, return on budget, log versions
        of the dollar and size variables, and the era.

        Returns:
            pd.DataFrame: The analysis set, also stored as ``self.analysis``.
        """
        self.writer.header("CLEANING FUNNEL")
        df = self.full
        funnel = [("Raw movies x credits (joined)", len(df))]

        df = df[df["status"] == "Released"]
        funnel.append(("Status == Released", len(df)))
        df = df[df["release_year"].between(self.min_year, self.max_year)]
        funnel.append((f"Release year in [{self.min_year}, {self.max_year}]", len(df)))
        df = df[(df["budget"] >= self.min_dollars) & (df["revenue"] >= self.min_dollars)]
        funnel.append((f"Budget and revenue >= ${self.min_dollars:,}", len(df)))
        df = df[df["runtime"] > 0]
        funnel.append(("Runtime > 0", len(df)))
        df = df[(df["cast_size"] > 0) & (df["crew_size"] > 0)]
        funnel.append(("Non-empty cast and crew lists", len(df)))

        funnel_table = pd.DataFrame(funnel, columns=["Step", "Films remaining"])
        funnel_table["Dropped"] = -funnel_table["Films remaining"].diff().fillna(0).astype(int)
        self.writer.log(funnel_table.to_string(index=False))
        self.writer.save_table(funnel_table, "cleaning_funnel", index=False, float_format="%.0f")
        self.funnel = funnel_table

        analysis = df.copy()
        analysis["cpi_factor"] = analysis["release_year"].map(lambda y: CPI_U[self.base_year] / CPI_U[int(y)])
        analysis["budget_real"] = analysis["budget"] * analysis["cpi_factor"]
        analysis["revenue_real"] = analysis["revenue"] * analysis["cpi_factor"]
        analysis["roi"] = analysis["revenue_real"] / analysis["budget_real"]
        for column in ["budget", "revenue", "budget_real", "revenue_real", "roi", "cast_size", "crew_size"]:
            analysis[f"ln_{column}"] = np.log(analysis[column])
        analysis["era"] = pd.Categorical(analysis["release_year"].map(era_label),
                                         categories=self.era_order, ordered=True)

        self.analysis = analysis
        return analysis

    def merge_oscars(self) -> pd.DataFrame:
        """
        Adds Oscar nomination and win counts to the analysis set.

        Oscar records are grouped by normalized title and film year, then matched
        to each analysis film on its normalized title and release year, trying
        the exact year first and then years up to ``oscar_year_tolerance`` away.
        Films with no match count as not nominated. Oscar-nominated films in the
        window that did not match are saved to ``oscar_unmatched_check.csv`` for
        manual review. If the Oscar file is missing, the analysis set is
        returned unchanged and ``has_awards`` stays False.

        Returns:
            pd.DataFrame: The analysis set with ``n_noms``, ``n_wins``,
            ``oscar_categories``, ``nominated`` and ``won`` columns.
        """
        self.writer.header("MERGING OSCAR DATA")
        path = os.path.join(self.data_dir, "the_oscar_award.csv")
        if not os.path.exists(path):
            self.writer.log(f"  !! {path} not found -- skipping awards analysis.")
            return self.analysis

        oscars = pd.read_csv(path)
        year_col = "year_film" if "year_film" in oscars.columns else "year"
        oscars = oscars.dropna(subset=["film"]).copy()
        oscars["winner"] = oscars["winner"].astype(str).str.lower().isin(["true", "1", "yes"])
        oscars["key"] = oscars["film"].map(normalize_title)
        self.writer.log(f"Oscar rows (with a film): {len(oscars):,}   "
                        f"years {oscars[year_col].min()}-{oscars[year_col].max()}")

        by_film = (oscars.groupby(["key", year_col])
                         .agg(n_noms=("winner", "size"), n_wins=("winner", "sum"),
                              categories=("category", lambda s: "; ".join(sorted(set(s)))))
                         .reset_index())
        lookup = {(row["key"], int(row[year_col])): row for _, row in by_film.iterrows()}

        analysis = self.analysis
        keys = analysis["title"].map(normalize_title)
        year_offsets = sorted(range(-self.oscar_year_tolerance, self.oscar_year_tolerance + 1), key=abs)
        hits, matched_keys = [], set()
        for key, year in zip(keys, analysis["release_year"].astype(int)):
            record = None
            for offset in year_offsets:
                record = lookup.get((key, year + offset))
                if record is not None:
                    matched_keys.add((key, year + offset))
                    break
            hits.append(record)

        analysis["n_noms"] = [int(h["n_noms"]) if h is not None else 0 for h in hits]
        analysis["n_wins"] = [int(h["n_wins"]) if h is not None else 0 for h in hits]
        analysis["oscar_categories"] = [h["categories"] if h is not None else "" for h in hits]
        analysis["nominated"] = (analysis["n_noms"] > 0).astype(int)
        analysis["won"] = (analysis["n_wins"] > 0).astype(int)

        # Oscar-nominated films in the window that did not match an analysis film
        in_window = by_film[by_film[year_col].between(analysis["release_year"].min(), analysis["release_year"].max())]
        unmatched = in_window[[(k, int(y)) not in matched_keys for k, y in zip(in_window["key"], in_window[year_col])]]
        titles = oscars.drop_duplicates("key").set_index("key")["film"]
        unmatched = unmatched.assign(film=unmatched["key"].map(titles))[["film", year_col, "n_noms", "n_wins"]]
        unmatched.to_csv(os.path.join(self.writer.tables_dir, "oscar_unmatched_check.csv"), index=False)

        self.writer.log(f"Oscar-nominated films in window: {len(in_window):,}; "
                        f"matched to analysis set: {len(matched_keys):,}")
        self.writer.log("  (Unmatched films are expected: TMDB-5000 is a subset, and films lacking "
                        "budget/revenue were dropped.")
        self.writer.log("   Review tables/oscar_unmatched_check.csv for title-format mismatches, "
                        "e.g. subtitles or punctuation.)")
        self.writer.log(f"Analysis films nominated: {analysis['nominated'].sum()}  |  "
                        f"won >=1 award: {analysis['won'].sum()}")

        self.analysis = analysis
        self.has_awards = True
        return analysis
