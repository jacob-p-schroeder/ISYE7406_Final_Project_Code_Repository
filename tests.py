import os
import tempfile

from models import OutputWriter
from models import TMDBDataLoader


def test_TMDBDataLoader(data_dir):
    with tempfile.TemporaryDirectory() as output_dir:
        tester = TMDBDataLoader(OutputWriter(output_dir), data_dir=data_dir)
        full = tester.load()
        analysis = tester.clean()
        analysis = tester.merge_oscars()

        assert len(full) == 4803, f"expected 4,803 joined films, got {len(full):,}"
        assert len(analysis) == 3139, f"expected 3,139 analysis films, got {len(analysis):,}"
        assert analysis["release_year"].between(1960, 2016).all(), "release years outside 1960-2016"
        assert (analysis["budget_real"] > 0).all() and (analysis["revenue_real"] > 0).all(), "non-positive dollars"
        assert tester.has_awards, "Oscar data was not merged"
        assert analysis["nominated"].sum() == 747, f"expected 747 nominated films, got {analysis['nominated'].sum()}"
        assert analysis["won"].sum() == 300, f"expected 300 winning films, got {analysis['won'].sum()}"

    return f"Analysis films     : {len(analysis):,}\nNominated / won    : {analysis['nominated'].sum()} / {analysis['won'].sum()}"


if __name__ == "__main__":
    data_path = os.path.join("Data")
    loader_results = test_TMDBDataLoader(data_path)
    print(loader_results)
