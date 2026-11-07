import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint

from .data_loader import ERA_ORDER, era_label
from .output_writer import OutputWriter

# Seed for the bootstrap confidence bands drawn by seaborn regplot, so figures are reproducible
REGPLOT_SEED = 42


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple:
    """
    Computes a confidence interval for a Pearson correlation using the Fisher z-transform.

    Args:
        r (float): Sample correlation.
        n (int): Sample size.
        alpha (float, optional): Significance level. Defaults to 0.05 (a 95% interval).

    Returns:
        tuple[float, float]: Lower and upper bounds, or (NaN, NaN) if n < 4 or |r| = 1.
    """
    if n < 4 or abs(r) >= 1:
        return np.nan, np.nan
    z, se, z_crit = np.arctanh(r), 1 / np.sqrt(n - 3), stats.norm.ppf(1 - alpha / 2)
    return np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)


def partial_correlation(df: pd.DataFrame, x: str, y: str, controls: list) -> tuple:
    """
    Computes the partial correlation of two variables after removing linear effects of controls.

    Both variables are regressed on the controls (plus an intercept) by least
    squares, and the Pearson correlation of the two residual vectors is returned.

    Args:
        df (pd.DataFrame): Data containing x, y and the controls.
        x (str): First variable.
        y (str): Second variable.
        controls (list[str]): Variables to adjust for.

    Returns:
        tuple[float, float, int]: Partial correlation, its p-value, and the number of complete rows.
    """
    d = df[[x, y] + controls].dropna()
    Z = np.column_stack([np.ones(len(d))] + [d[c].values for c in controls])
    x_resid = d[x].values - Z @ np.linalg.lstsq(Z, d[x].values, rcond=None)[0]
    y_resid = d[y].values - Z @ np.linalg.lstsq(Z, d[y].values, rcond=None)[0]
    r, p = stats.pearsonr(x_resid, y_resid)
    return r, p, len(d)


def scatter_lowess(ax: plt.Axes, x: pd.Series, y: pd.Series, ylog: bool = False, frac: float = 0.5,
                   color: str = "#4C72B0") -> None:
    """
    Draws a scatter plot with a LOESS smoother on the given axes.

    Args:
        ax (plt.Axes): Axes to draw on.
        x (pd.Series): Horizontal values.
        y (pd.Series): Vertical values.
        ylog (bool, optional): Smooth log(y) and use a log y-axis. Defaults to False.
        frac (float, optional): LOESS span (share of points in each local fit). Defaults to 0.5.
        color (str, optional): Point color. Defaults to "#4C72B0".
    """
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    ax.scatter(d.x, d.y, s=7, alpha=0.25, color=color, edgecolor="none")
    y_fit = np.log(d.y) if ylog else d.y
    smooth = lowess(y_fit, d.x, frac=frac, return_sorted=True)
    ax.plot(smooth[:, 0], np.exp(smooth[:, 1]) if ylog else smooth[:, 1], lw=2.2, color="crimson")
    if ylog:
        ax.set_yscale("log")


class ExploratoryAnalysis:
    """
    Runs the exploratory data analysis for the project midterm report.

    Each method covers one part of Sections 2.2 and 2.3 of the report: it logs
    the key numbers, saves the tables as CSV files and saves the figures as PNG
    files through the shared ``OutputWriter``.

    Attributes:
        full (pd.DataFrame): Joined and engineered data before cleaning.
        analysis (pd.DataFrame): Cleaned analysis set. ``awards`` adds a
            ``budget_q`` (real-budget quartile) column to it.
        writer (OutputWriter): Writer used to log messages and save outputs.
        has_awards (bool): Whether Oscar columns are present in the analysis set.
        min_dollars (int): Smallest budget and revenue treated as valid.
        era_order (list[str]): Ordered era labels.
    """
    def __init__(self, full: pd.DataFrame, analysis: pd.DataFrame, writer: OutputWriter, has_awards: bool = True,
                 min_dollars: int = 10_000, era_order: list = ERA_ORDER):
        """
        Stores the data and settings used by every analysis step.

        Args:
            full (pd.DataFrame): Joined and engineered data before cleaning.
            analysis (pd.DataFrame): Cleaned analysis set.
            writer (OutputWriter): Writer used to log messages and save outputs.
            has_awards (bool, optional): Whether Oscar columns are present. Defaults to True.
            min_dollars (int, optional): Smallest valid budget and revenue. Defaults to 10,000.
            era_order (list[str], optional): Ordered era labels. Defaults to ERA_ORDER.

        Returns:
            None
        """
        self.full = full
        self.analysis = analysis
        self.writer = writer
        self.has_awards = has_awards
        self.min_dollars = min_dollars
        self.era_order = era_order

    def crew_department_columns(self) -> list:
        """
        Lists the crew-by-department count columns.

        Returns:
            list[str]: Column names starting with "crew_", excluding crew_size and crew_departments.
        """
        return [c for c in self.analysis.columns
                if c.startswith("crew_") and c not in ("crew_size", "crew_departments")]

    def data_description(self) -> None:
        """
        Produces the Section 2.2 data description outputs.

        Saves the data dictionary, missing-value rates before cleaning, the share
        of films with usable budget and revenue by era (a selection-bias check),
        summary statistics with skewness, mean crew counts by department, and a
        bar chart of films per release year.
        """
        self.writer.header("SECTION 2.2  DATA DESCRIPTION")
        full, an = self.full, self.analysis

        dictionary = [
            ("budget / budget_real", "Production budget, nominal USD / 2017 USD (CPI-adjusted)", "TMDB movies"),
            ("revenue / revenue_real", "Worldwide box office, nominal USD / 2017 USD", "TMDB movies"),
            ("roi", "revenue_real / budget_real", "Engineered"),
            ("release_year, era", "Year and era (<=1979, 1980s, ..., 2010s) of release", "TMDB movies"),
            ("runtime", "Runtime in minutes", "TMDB movies"),
            ("vote_average, vote_count", "TMDB user rating and number of ratings", "TMDB movies"),
            ("popularity", "TMDB popularity score", "TMDB movies"),
            ("primary_genre, n_genres", "First-listed genre and number of genres", "Parsed from JSON"),
            ("n_prod_companies, is_english", "Number of production companies; English-language flag", "Parsed from JSON"),
            ("cast_size", "Number of credited cast members", "Parsed from credits JSON"),
            ("crew_size", "Number of credited crew members", "Parsed from credits JSON"),
            ("crew_<department>", "Crew counts by department (Production, Sound, Art, ...)", "Parsed from credits JSON"),
            ("crew_departments", "Number of distinct crew departments credited", "Parsed from credits JSON"),
        ]
        if self.has_awards:
            dictionary += [("n_noms, n_wins", "Oscar nominations / wins per film (all categories)", "Oscar dataset (merged)"),
                           ("nominated, won", "1 if film has >=1 nomination / >=1 win", "Engineered")]
        self.writer.save_table(pd.DataFrame(dictionary, columns=["Variable", "Description", "Source"]),
                               "data_dictionary", index=False)

        self.writer.log(f"\nFull joined dataset: {len(full):,} films x {full.shape[1]} columns")
        self.writer.log(f"Analysis dataset:    {len(an):,} films ({an['release_year'].min()}-{an['release_year'].max()})")

        # Missing or zero values among released films, before cleaning
        released = full[full["status"] == "Released"]
        missing = pd.DataFrame({
            "pct_missing_or_zero": [
                (released["budget"].fillna(0) <= 0).mean() * 100,
                (released["revenue"].fillna(0) <= 0).mean() * 100,
                (released["runtime"].fillna(0) <= 0).mean() * 100,
                released["release_date"].isna().mean() * 100,
                (released["cast_size"] == 0).mean() * 100,
                (released["crew_size"] == 0).mean() * 100,
                released["homepage"].isna().mean() * 100,
                released["tagline"].isna().mean() * 100,
            ]}, index=["budget (0 = missing)", "revenue (0 = missing)", "runtime", "release_date",
                       "cast list empty", "crew list empty", "homepage", "tagline"])
        self.writer.log("\nMissing / zero values among released films (pre-cleaning), %:")
        self.writer.log(missing.round(2).to_string())
        self.writer.save_table(missing, "missingness")

        # Selection check: share of films with usable budget and revenue, by era
        in_range = released[released["release_year"].between(an["release_year"].min(),
                                                              an["release_year"].max())].copy()
        in_range["era"] = in_range["release_year"].map(era_label)
        in_range["usable"] = ((in_range["budget"] >= self.min_dollars)
                              & (in_range["revenue"] >= self.min_dollars)).astype(int)
        usable = in_range.groupby("era")["usable"].agg(films="size", usable="sum", pct_usable="mean")
        usable["pct_usable"] *= 100
        usable = usable.reindex([e for e in self.era_order if e in usable.index])
        self.writer.log("\nShare of films with usable budget & revenue, by era (selection-bias check):")
        self.writer.log(usable.round(1).to_string())
        self.writer.save_table(usable, "usable_by_era")

        # Summary statistics
        numeric = ["budget", "budget_real", "revenue", "revenue_real", "roi", "runtime", "vote_average",
                   "vote_count", "popularity", "cast_size", "crew_size", "crew_departments", "n_genres",
                   "n_prod_companies"]
        summary = an[numeric].describe().T
        summary["skew"] = an[numeric].skew()
        self.writer.log("\nSummary statistics (analysis set):")
        self.writer.log(summary.round(2).to_string())
        self.writer.save_table(summary, "summary_statistics", float_format="%.2f")

        # Crew department breakdown
        department_means = (an[self.crew_department_columns()].mean()
                            .sort_values(ascending=False).to_frame("mean_crew_count"))
        self.writer.log("\nMean crew count by department:")
        self.writer.log(department_means.round(2).to_string())
        self.writer.save_table(department_means, "crew_department_means")

        # Films per release year
        fig, ax = plt.subplots(figsize=(9, 3.4))
        an["release_year"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#4C72B0", width=0.85)
        ax.set_xlabel("Release year")
        ax.set_ylabel("Films")
        ax.set_title("Films in analysis set by release year")
        ax.set_xticks(ax.get_xticks()[::4])
        self.writer.save_figure("fig_films_per_year")

    def distributions(self) -> None:
        """
        Plots histograms of the main numeric variables and logs skewness before and after a log transform.
        """
        self.writer.header("2.3  EDA - UNIVARIATE DISTRIBUTIONS")
        an = self.analysis
        specs = [("budget_real", "Budget (2017 USD)", True), ("revenue_real", "Revenue (2017 USD)", True),
                 ("cast_size", "Cast size", True), ("crew_size", "Crew size", True),
                 ("runtime", "Runtime (min)", False), ("vote_average", "TMDB vote average", False)]

        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
        for ax, (column, label, log_x) in zip(axes.ravel(), specs):
            x = an[column].dropna()
            bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 40) if log_x else 40
            ax.hist(x, bins=bins, color="#4C72B0", edgecolor="white")
            if log_x:
                ax.set_xscale("log")
            median_label = f"median = {x.median():,.0f}" if x.median() > 20 else f"median = {x.median():.2f}"
            ax.axvline(x.median(), color="crimson", ls="--", lw=1.5, label=median_label)
            ax.set_xlabel(label + (" (log scale)" if log_x else ""))
            ax.set_ylabel("Films")
            ax.legend(frameon=False)
        self.writer.save_figure("fig_distributions")

        self.writer.log("  Skewness (raw -> log): " + ", ".join(
            f"{c}: {an[c].skew():.1f} -> {np.log(an[c]).skew():.2f}"
            for c in ["budget_real", "revenue_real", "cast_size", "crew_size"]))

    def time_trends(self) -> None:
        """
        Plots budget, revenue, cast size, crew size and ROI against release year (RQ1).

        Also saves the era summary table, Spearman trend tests against release
        year, and a stacked bar chart of crew composition by era.
        """
        self.writer.header("2.3  EDA - TIME TRENDS (RQ1)")
        an = self.analysis
        year = an["release_year"]

        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
        scatter_lowess(axes[0, 0], year, an["budget_real"], ylog=True)
        axes[0, 0].set_title("Budget (2017 USD) vs year")
        axes[0, 0].set_ylabel("Budget (log scale)")

        medians = an.groupby("release_year")[["budget", "budget_real"]].median()
        axes[0, 1].plot(medians.index, medians["budget"], label="Nominal", lw=2)
        axes[0, 1].plot(medians.index, medians["budget_real"], label="2017 USD", lw=2)
        axes[0, 1].set_title("Median budget by year")
        axes[0, 1].set_ylabel("USD")
        axes[0, 1].legend(frameon=False)

        scatter_lowess(axes[0, 2], year, an["revenue_real"], ylog=True)
        axes[0, 2].set_title("Revenue (2017 USD) vs year")
        axes[0, 2].set_ylabel("Revenue (log scale)")

        scatter_lowess(axes[1, 0], year, an["cast_size"], ylog=True)
        axes[1, 0].set_title("Cast size vs year")
        axes[1, 0].set_ylabel("Cast size (log scale)")

        scatter_lowess(axes[1, 1], year, an["crew_size"], ylog=True)
        axes[1, 1].set_title("Crew size vs year")
        axes[1, 1].set_ylabel("Crew size (log scale)")

        scatter_lowess(axes[1, 2], year, an["roi"], ylog=True)
        axes[1, 2].axhline(1, color="grey", ls=":", lw=1)
        axes[1, 2].set_title("Return on budget (revenue/budget) vs year")
        axes[1, 2].set_ylabel("ROI (log scale)")
        for ax in axes.ravel():
            ax.set_xlabel("Release year")
        self.writer.save_figure("fig_time_trends")

        # Era summary
        by_era = an.groupby("era", observed=True)
        era_summary = pd.DataFrame({
            "films": by_era.size(),
            "median_budget_nominal": by_era["budget"].median(),
            "median_budget_real": by_era["budget_real"].median(),
            "median_revenue_real": by_era["revenue_real"].median(),
            "median_roi": by_era["roi"].median(),
            "pct_budget_ge_100M_real": by_era["budget_real"].apply(lambda s: (s >= 1e8).mean() * 100),
            "median_cast": by_era["cast_size"].median(),
            "median_crew": by_era["crew_size"].median(),
            "median_runtime": by_era["runtime"].median(),
        })
        self.writer.log(era_summary.round(2).to_string())
        self.writer.save_table(era_summary, "era_summary", float_format="%.2f")

        # Spearman trend tests against release year
        rows = []
        for column in ["budget_real", "revenue_real", "roi", "cast_size", "crew_size"]:
            rho, p = stats.spearmanr(an["release_year"], an[column])
            rows.append((column, rho, p))
        trends = pd.DataFrame(rows, columns=["variable", "spearman_rho_vs_year", "p_value"]).set_index("variable")
        self.writer.log("\nMonotonic trend vs release year:")
        self.writer.log(trends.round(4).to_string())
        self.writer.save_table(trends, "trend_tests", float_format="%.4f")

        # Crew composition by era
        top = an[self.crew_department_columns()].mean().sort_values(ascending=False).head(7).index.tolist()
        composition = an.groupby("era", observed=True)[top].mean()
        ax = composition.plot(kind="bar", stacked=True, figsize=(8.5, 5), colormap="tab20", width=0.8)
        ax.set_ylabel("Mean credited crew per film")
        ax.set_xlabel("")
        ax.set_title("Crew composition by era (top departments)")
        ax.legend([c.replace("crew_", "") for c in top], frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=0)
        self.writer.save_figure("fig_crew_composition_by_era")

    def budget_revenue_by_era(self) -> None:
        """
        Measures how the budget-revenue relationship changes by era (RQ1).

        Plots log revenue against log budget for each era, saves the correlation
        and log-log slope (elasticity) by era with confidence intervals, logs a
        preliminary year-interaction regression with HC3 standard errors, and
        saves return on budget by real-budget tier.
        """
        self.writer.header("2.3  EDA - BUDGET vs REVENUE BY ERA (RQ1)")
        an = self.analysis
        eras = [e for e in self.era_order if (an["era"] == e).any()]

        # Log-log scatter by era
        fig, axes = plt.subplots(1, len(eras), figsize=(3.6 * len(eras), 3.9), sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, era in zip(axes, eras):
            group = an[an["era"] == era]
            sns.regplot(data=group, x="ln_budget_real", y="ln_revenue_real", ax=ax,
                        scatter_kws={"s": 10, "alpha": 0.35}, line_kws={"color": "crimson"}, seed=REGPLOT_SEED)
            ax.set_title(f"{era} (n={len(group)})")
            ax.set_xlabel("ln budget (2017 USD)")
            ax.set_ylabel("ln revenue (2017 USD)" if ax is axes[0] else "")
        self.writer.save_figure("fig_budget_revenue_by_era")

        # Correlation and slope by era, with confidence intervals
        rows = []
        for era in eras:
            group = an[an["era"] == era]
            r, _ = stats.pearsonr(group["ln_budget_real"], group["ln_revenue_real"])
            rho, _ = stats.spearmanr(group["ln_budget_real"], group["ln_revenue_real"])
            ci_lo, ci_hi = fisher_ci(r, len(group))
            fit = stats.linregress(group["ln_budget_real"], group["ln_revenue_real"])
            rows.append((era, len(group), r, ci_lo, ci_hi, rho, fit.slope, fit.stderr, r ** 2))
        by_era = pd.DataFrame(rows, columns=["era", "n", "pearson_r", "ci_lo", "ci_hi", "spearman_rho",
                                             "ols_slope", "slope_se", "r_squared"]).set_index("era")
        self.writer.log(by_era.round(3).to_string())
        self.writer.save_table(by_era, "budget_revenue_by_era")

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        axes[0].errorbar(by_era.index, by_era["pearson_r"],
                         yerr=[by_era["pearson_r"] - by_era["ci_lo"], by_era["ci_hi"] - by_era["pearson_r"]],
                         fmt="o-", capsize=4, lw=2)
        axes[0].set_ylabel("Pearson r (ln budget, ln revenue)")
        axes[0].set_title("Strength of budget-revenue association")
        axes[1].errorbar(by_era.index, by_era["ols_slope"], yerr=1.96 * by_era["slope_se"], fmt="s-",
                         color="darkorange", capsize=4, lw=2)
        axes[1].set_ylabel("Log-log slope (elasticity)")
        axes[1].set_title("Revenue elasticity to budget (95% CI)")
        self.writer.save_figure("fig_correlation_by_era")

        # Preliminary RQ1 model: budget x centered-year interaction
        centered = an.assign(year_c=an["release_year"] - an["release_year"].mean())
        model = smf.ols("ln_revenue_real ~ ln_budget_real * year_c", data=centered).fit(cov_type="HC3")
        self.writer.log("\nQuick preview of RQ1 interaction (full modeling goes in Section 5):")
        self.writer.log(model.summary().tables[1].as_text())

        # Return on budget by real-budget tier
        tiers = pd.cut(an["budget_real"], [0, 1e7, 5e7, 1e8, np.inf], labels=["<$10M", "$10-50M", "$50-100M", ">$100M"])
        roi_tiers = an.groupby(tiers, observed=True)["roi"].agg(
            films="size", median_roi="median",
            pct_unprofitable=lambda s: (s < 1).mean() * 100,
            iqr=lambda s: s.quantile(.75) - s.quantile(.25))
        self.writer.log("\nReturn on budget by budget tier (2017 USD):")
        self.writer.log(roi_tiers.round(2).to_string())
        self.writer.save_table(roi_tiers, "roi_by_budget_tier", float_format="%.2f")

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x=tiers, y=an["roi"], ax=ax, showfliers=False, color="#8FB4D9")
        ax.set_yscale("log")
        ax.axhline(1, color="crimson", ls="--", lw=1)
        ax.set_xlabel("Budget tier")
        ax.set_ylabel("Revenue / budget (log scale)")
        ax.set_title("Return on budget by budget tier (fliers hidden)")
        self.writer.save_figure("fig_roi_by_budget_tier")

    def scale_vs_budget(self) -> None:
        """
        Compares cast and crew size with budget and checks their link to revenue beyond budget (RQ2).

        Plots log cast and crew size against log budget, saves raw and partial
        correlations with log revenue (adjusting for log budget, then log budget
        and release year), and saves variance inflation factors for the core predictors.
        """
        self.writer.header("2.3  EDA - PRODUCTION SCALE vs BUDGET (RQ2)")
        an = self.analysis

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for ax, column, label in zip(axes[:2], ["ln_cast_size", "ln_crew_size"], ["ln cast size", "ln crew size"]):
            sns.regplot(data=an, x="ln_budget_real", y=column, ax=ax,
                        scatter_kws={"s": 8, "alpha": 0.3}, line_kws={"color": "crimson"}, seed=REGPLOT_SEED)
            r = an[["ln_budget_real", column]].corr().iloc[0, 1]
            ax.set_title(f"{label} vs ln budget (r = {r:.2f})")
        sns.regplot(data=an, x="ln_cast_size", y="ln_crew_size", ax=axes[2],
                    scatter_kws={"s": 8, "alpha": 0.3}, line_kws={"color": "crimson"}, seed=REGPLOT_SEED)
        r = an[["ln_cast_size", "ln_crew_size"]].corr().iloc[0, 1]
        axes[2].set_title(f"ln crew vs ln cast (r = {r:.2f})")
        self.writer.save_figure("fig_scale_vs_budget")

        rows = []
        for column in ["ln_cast_size", "ln_crew_size"]:
            r_raw = an[[column, "ln_revenue_real"]].corr().iloc[0, 1]
            r_budget, p_budget, n = partial_correlation(an, column, "ln_revenue_real", ["ln_budget_real"])
            r_year, p_year, _ = partial_correlation(an, column, "ln_revenue_real", ["ln_budget_real", "release_year"])
            rows.append((column, r_raw, r_budget, p_budget, r_year, p_year, n))
        partials = pd.DataFrame(rows, columns=["scale_variable", "raw_r_with_ln_revenue", "partial_r_given_budget",
                                               "p_budget", "partial_r_given_budget_year", "p_budget_year",
                                               "n"]).set_index("scale_variable")
        self.writer.log("Does scale carry signal about revenue beyond budget? (partial correlations)")
        self.writer.log(partials.round(4).to_string())
        self.writer.save_table(partials, "partial_correlations", float_format="%.4f")

        # Multicollinearity check
        X = an[["ln_budget_real", "ln_cast_size", "ln_crew_size", "release_year", "runtime"]].dropna()
        X_const = np.column_stack([np.ones(len(X)), X.values])
        vif = pd.Series([variance_inflation_factor(X_const, i + 1) for i in range(X.shape[1])],
                        index=X.columns, name="VIF")
        self.writer.log("\nVariance inflation factors (relevant for Lasso/Ridge motivation in RQ2):")
        self.writer.log(vif.round(2).to_string())
        self.writer.save_table(vif.to_frame(), "vif", float_format="%.2f")

    def correlations(self) -> None:
        """
        Saves and plots the Spearman correlation matrix of the key variables and logs each one's correlation with log revenue.
        """
        self.writer.header("2.3  EDA - CORRELATION MATRIX")
        columns = ["ln_budget_real", "ln_revenue_real", "ln_cast_size", "ln_crew_size", "crew_departments", "runtime",
                   "vote_average", "vote_count", "popularity", "release_year", "n_genres", "n_prod_companies",
                   "is_english"]
        corr = self.analysis[columns].corr(method="spearman")
        self.writer.save_table(corr, "spearman_correlations", float_format="%.3f")

        fig, ax = plt.subplots(figsize=(9.5, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                    cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
        ax.set_title("Spearman correlation matrix")
        self.writer.save_figure("fig_correlation_heatmap")

        with_revenue = corr["ln_revenue_real"].drop("ln_revenue_real").sort_values(key=abs, ascending=False)
        self.writer.log("Spearman correlation with ln revenue (sorted by |rho|):")
        self.writer.log(with_revenue.round(3).to_string())

    def genre(self) -> None:
        """
        Summarizes films, budget, revenue, ROI, cast and crew by genre, and plots the twelve most common genres.

        A film counts once in every genre it lists.
        """
        self.writer.header("2.3  EDA - GENRE")
        exploded = self.analysis.explode("genres_list").rename(columns={"genres_list": "genre"})
        exploded = exploded[exploded["genre"].notna()]
        by_genre = exploded.groupby("genre").agg(
            films=("title", "size"), median_budget_real=("budget_real", "median"),
            median_revenue_real=("revenue_real", "median"), median_roi=("roi", "median"),
            median_cast=("cast_size", "median"), median_crew=("crew_size", "median"))
        by_genre = by_genre.sort_values("films", ascending=False)
        self.writer.log(by_genre.round(2).to_string())
        self.writer.save_table(by_genre, "genre_summary", float_format="%.2f")

        top = by_genre.head(12)
        order = top.index.tolist()
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
        axes[0].barh(order, top["films"], color="#4C72B0")
        axes[0].set_title("Films per genre")
        axes[1].barh(order, top["median_budget_real"] / 1e6, color="#DD8452")
        axes[1].set_title("Median budget ($M, 2017)")
        axes[2].barh(order, top["median_roi"], color="#55A868")
        axes[2].axvline(1, color="grey", ls=":")
        axes[2].set_title("Median ROI (revenue / budget)")
        axes[0].invert_yaxis()
        self.writer.save_figure("fig_genre_summary")

    def awards(self) -> None:
        """
        Describes the nomination and win stages of the awards question (RQ4).

        Saves the class balance, nomination rates by real-budget quartile and win
        rates among nominees by quartile (with Wilson intervals), nomination rates
        by era, a chi-square test of budget quartile against nomination, and
        Mann-Whitney tests comparing nominated vs. other films and winners vs.
        losing nominees. Adds a ``budget_q`` column to the analysis set.
        """
        self.writer.header("2.3  EDA - AWARDS (RQ4)")
        an = self.analysis
        n_films, n_nominated, n_won = len(an), int(an["nominated"].sum()), int(an["won"].sum())
        balance = pd.DataFrame({
            "stage": ["All films", "Nominated (>=1 nomination)", "Won (>=1 win) among nominated"],
            "n": [n_films, n_nominated, n_won],
            "rate": [np.nan, n_nominated / n_films, n_won / n_nominated if n_nominated else np.nan]})
        self.writer.log(balance.to_string(index=False))
        self.writer.save_table(balance, "awards_class_balance", index=False)
        if n_won < 30:
            self.writer.log(f"  !! Only {n_won} winners among {n_nominated} nominees -- the win-stage model will "
                            "have low power. Consider regularization / Firth logistic and report CIs.")

        # Real-budget quartiles
        labels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
        an["budget_q"] = pd.qcut(an["budget_real"], 4, labels=labels)
        edges = an["budget_real"].quantile([0, .25, .5, .75, 1.0]).values / 1e6
        self.writer.log("Budget quartile edges ($M, 2017): " + ", ".join(f"{e:.1f}" for e in edges))

        stage_a = an.groupby("budget_q", observed=True)["nominated"].agg(n="size", events="sum")
        nominees = an[an["nominated"] == 1]
        stage_b = nominees.groupby("budget_q", observed=True)["won"].agg(n="size", events="sum")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        stages = [("nomination", stage_a, "P(nominated) by budget quartile"),
                  ("win", stage_b, "P(win | nominated) by budget quartile")]
        for ax, (name, table, title) in zip(axes[:2], stages):
            table = table.copy()
            table["rate"] = table["events"] / table["n"].replace(0, np.nan)
            intervals = [proportion_confint(e, k, method="wilson") if k > 0 else (np.nan, np.nan)
                         for e, k in zip(table["events"], table["n"])]
            table["ci_lo"], table["ci_hi"] = [c[0] for c in intervals], [c[1] for c in intervals]
            ax.bar(table.index.astype(str), table["rate"],
                   yerr=[table["rate"] - table["ci_lo"], table["ci_hi"] - table["rate"]],
                   color="#4C72B0", capsize=4)
            for i, (k, e) in enumerate(zip(table["n"], table["events"])):
                ax.text(i, 0.01, f"{int(e)}/{int(k)}", ha="center", color="white", fontsize=8)
            ax.set_title(title)
            ax.set_ylabel("Proportion (95% Wilson CI)")
            ax.tick_params(axis="x", rotation=20)
            self.writer.save_table(table, f"awards_{name}_by_budget_quartile")
            self.writer.log(f"\n{title}:")
            self.writer.log(table.round(3).to_string())

        # Nomination rate by era (confounded with time)
        by_era = an.groupby("era", observed=True)["nominated"].mean() * 100
        axes[2].bar(by_era.index.astype(str), by_era.values, color="#C44E52")
        axes[2].set_title("% of films nominated, by era")
        axes[2].set_ylabel("% nominated")
        self.writer.log("\n% nominated by era:")
        self.writer.log(by_era.round(2).to_string())
        self.writer.save_figure("fig_awards_by_budget_quartile")

        chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(an["budget_q"], an["nominated"]))
        self.writer.log(f"\nChi-square test (budget quartile x nominated): chi2={chi2:.2f}, p={p:.4g}")

        # Distribution comparisons and Mann-Whitney tests
        variables = [("ln_budget_real", "ln budget (2017 USD)"), ("vote_average", "TMDB vote average"),
                     ("cast_size", "Cast size"), ("crew_size", "Crew size")]
        comparisons = [("A: nominated vs not", an, "nominated"),
                       ("B: won vs lost (nominees only)", nominees, "won")]
        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        rows = []
        for row_idx, (stage, data, flag) in enumerate(comparisons):
            for col_idx, (variable, label) in enumerate(variables):
                ax = axes[row_idx, col_idx]
                sns.boxplot(data=data, x=flag, y=variable, ax=ax, showfliers=False, palette=["#B9CFE6", "#4C72B0"])
                ax.set_xlabel(("Nominated" if flag == "nominated" else "Won") + " (0 = no, 1 = yes)")
                ax.set_ylabel(label)
                group_0 = data.loc[data[flag] == 0, variable].dropna()
                group_1 = data.loc[data[flag] == 1, variable].dropna()
                if len(group_0) > 1 and len(group_1) > 1:
                    test = stats.mannwhitneyu(group_0, group_1, alternative="two-sided")
                    rows.append((stage, variable, len(group_0), len(group_1),
                                 group_0.median(), group_1.median(), test.pvalue))
                ax.set_title(f"{stage.split(':')[0]}: {label}", fontsize=10)
        self.writer.save_figure("fig_awards_distributions")

        mann_whitney = pd.DataFrame(rows, columns=["stage", "variable", "n_group0", "n_group1", "median_group0",
                                                   "median_group1", "mannwhitney_p"])
        self.writer.log("\nMann-Whitney tests:")
        self.writer.log(mann_whitney.round(4).to_string(index=False))
        self.writer.save_table(mann_whitney, "awards_mannwhitney", index=False, float_format="%.4f")

        top_genres = nominees.explode("genres_list")["genres_list"].value_counts().head(8)
        self.writer.log("\nTop genres among nominated films:")
        self.writer.log(top_genres.to_string())

    def save_clean_dataset(self, output_file: str) -> pd.DataFrame:
        """
        Writes the analysis set to CSV without the raw JSON and free-text columns.

        Args:
            output_file (str): Path of the CSV file to write.

        Returns:
            pd.DataFrame: The saved analysis-ready dataset.
        """
        drop = ["genres", "keywords", "cast", "crew", "production_companies", "production_countries",
                "spoken_languages", "homepage", "tagline", "overview", "genres_list"]
        clean = self.analysis.drop(columns=[c for c in drop if c in self.analysis.columns])
        clean.to_csv(output_file, index=False)
        return clean
