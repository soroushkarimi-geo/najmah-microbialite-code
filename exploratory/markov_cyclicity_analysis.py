"""Permutation test for stratigraphic order in the facies log: does the
MF-E -> MF-B transition occur more often than in a random ordering?

Hypothesis: within facies association F6 (shallow subtidal to lower
intertidal) of Bayet-Goll et al. (2022, Geological Journal), thrombolite (MF-E)
records higher water energy and stromatolite (MF-B) lower energy or more
restricted conditions, so MF-E -> MF-B transitions up-section are the expected
signature of shallowing-upward cycles. The test addresses this one transition
(and its reverse, for comparison), not cyclicity in general.

Method: the labeled specimens of a section are ordered by sample number,
collapsed to an embedded sequence (repeats of the same facies merged, as in
embedded Markov-chain analysis) and the number of MF-E -> MF-B transitions is
counted. The same count is computed for 10000 random reorderings of the
specimens (seed 42), which keep the facies proportions but remove any order.
The one-sided p-value is the share of reorderings with at least as many
transitions. A permutation test avoids the degrees-of-freedom corrections of
the classical chi-square test for embedded Markov chains (Powers and
Easterling, 1982). Peritidal successions that look cyclic often show no
statistically significant order when tested this way (Wilkinson, Diedrich and
co-workers, "Stratal order in peritidal carbonate sequences"), so a
non-significant result is an informative outcome.

Limitations:
  - Sample number is a proxy for stratigraphic position, not measured height.
  - Dudan and Kezi are tested separately and never pooled: TR5-TR6 and their
    bounding surfaces SB5/SB6 occur only in the Dudan (= Dawan) section; Kezi
    contains TR4 only (Bayet-Goll et al., 2022).
  - Only labeled specimens enter the sequence; MF-A is not labeled.

Output: markov_permutation_test_results.csv in --output-dir.

Run from the repository root:
    PYTHONPATH=. python exploratory/markov_cyclicity_analysis.py --input-csv data/image_metadata_auto_label.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from class_correlation_analysis import parse_sample_number

UNLABELED = "unlabeled"
N_PERMUTATIONS = 10000
RANDOM_SEED = 42


def build_ordered_sequence(df: pd.DataFrame, section_prefix: str) -> list[str]:
    """Facies sequence of one section, ordered by sample number.

    Args:
        df: Labeled image-level table with ``sample_code`` and ``label``;
            reduced to one row per specimen.
        section_prefix: ``'SD'`` or ``'SK'``.

    Returns:
        list of str: Labels from the base to the top of the section.
        Specimens without a sample number are left out.
    """
    per_sample = df.drop_duplicates("sample_code").copy()
    per_sample["section_prefix"] = per_sample["sample_code"].str.split("_").str[0].str.split("-").str[0]
    section_df = per_sample[per_sample["section_prefix"] == section_prefix].copy()
    section_df["sample_number"] = section_df["sample_code"].apply(parse_sample_number)
    section_df = section_df.dropna(subset=["sample_number"]).sort_values("sample_number")
    return section_df["label"].tolist()


def embedded_transitions(sequence: list[str]) -> list[tuple[str, str]]:
    """Transitions between different facies in an ordered sequence.

    Consecutive repeats are merged first (embedded Markov-chain convention),
    so ``[B, B, E, E, E, B]`` gives ``[(B, E), (E, B)]``.

    Args:
        sequence: Facies labels in stratigraphic order.

    Returns:
        list of tuple: ``(from_state, to_state)`` for every change of facies.
    """
    collapsed = [sequence[0]] if sequence else []
    for label in sequence[1:]:
        if label != collapsed[-1]:
            collapsed.append(label)
    return list(zip(collapsed[:-1], collapsed[1:]))


def permutation_test_transition(
    sequence: list[str], from_state: str, to_state: str, n_permutations: int, rng: np.random.Generator
) -> dict:
    """One-sided permutation test: is a transition more frequent than by chance?

    The observed number of ``from_state`` -> ``to_state`` embedded
    transitions is compared with its distribution over random reorderings
    of the same sequence. p = (number of reorderings with a count >= the
    observed count + 1) / (n_permutations + 1).

    Args:
        sequence: Facies labels in stratigraphic order.
        from_state: Lower facies of the transition, e.g. ``'MF-E'``.
        to_state: Upper facies of the transition, e.g. ``'MF-B'``.
        n_permutations: Number of random reorderings.
        rng: ``numpy.random.Generator``.

    Returns:
        dict: ``from_state``, ``to_state``, ``observed_count``,
        ``null_mean``, ``null_std`` and ``p_value``.
    """
    observed_transitions = embedded_transitions(sequence)
    observed_count = sum(1 for a, b in observed_transitions if a == from_state and b == to_state)

    null_counts = []
    seq_array = np.array(sequence)
    for _ in range(n_permutations):
        shuffled = rng.permutation(seq_array).tolist()
        shuffled_transitions = embedded_transitions(shuffled)
        count = sum(1 for a, b in shuffled_transitions if a == from_state and b == to_state)
        null_counts.append(count)

    null_counts = np.array(null_counts)
    p_value = float((null_counts >= observed_count).sum() + 1) / (n_permutations + 1)  # one-sided, +1 avoids p=0

    return {
        "from_state": from_state,
        "to_state": to_state,
        "observed_count": observed_count,
        "null_mean": float(null_counts.mean()),
        "null_std": float(null_counts.std()),
        "p_value": p_value,
    }


def run(input_csv: str, output_dir: Path) -> None:
    """Test MF-E -> MF-B and MF-B -> MF-E in the Dudan and Kezi sections.

    A section is skipped if it does not contain both MF-B and MF-E. One
    random generator (seed 42) is shared by all tests, in a fixed order.

    Args:
        input_csv: Labeled feature table (output of ``auto_label.py``). Rows
            labeled ``'unlabeled'`` are ignored.
        output_dir: Output folder (``pathlib.Path``); created if needed.
    """
    df = pd.read_csv(input_csv)
    labeled_df = df[df["label"] != UNLABELED].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    all_results = []
    for section_prefix, section_name in [("SD", "Dudan (= Dawan; TR4-TR6 all present)"),
                                           ("SK", "Kezi (TR4 only, per Bayet-Goll et al. 2022)")]:
        sequence = build_ordered_sequence(labeled_df, section_prefix)
        print(f"\n=== {section_name} ===")
        print(f"Ordered facies sequence ({len(sequence)} specimens): {sequence}")

        present_classes = sorted(set(sequence))
        if "MF-B" not in present_classes or "MF-E" not in present_classes:
            print("MF-B and/or MF-E not both present in this section - skipping transition test.")
            continue

        embedded = embedded_transitions(sequence)
        print(f"Embedded transitions (self-transitions collapsed, {len(embedded)} genuine changes): {embedded}")

        print("\nTesting the Bayet-Goll et al. (2022)-grounded hypothesis: "
              "MF-E -> MF-B occurs more often than chance (shallowing signature)")
        result_forward = permutation_test_transition(sequence, "MF-E", "MF-B", N_PERMUTATIONS, rng)
        result_reverse = permutation_test_transition(sequence, "MF-B", "MF-E", N_PERMUTATIONS, rng)

        for result in (result_forward, result_reverse):
            print(f"  {result['from_state']} -> {result['to_state']}: "
                  f"observed={result['observed_count']}, "
                  f"random-shuffle mean={result['null_mean']:.2f} (sd={result['null_std']:.2f}), "
                  f"p={result['p_value']:.4f}")
            result["section"] = section_name
            all_results.append(result)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / "markov_permutation_test_results.csv", index=False)
        print(f"\nSaved -> {output_dir / 'markov_permutation_test_results.csv'}")

    print(
        "\nInterpretation: p < 0.05 means the observed transition count is unlikely to arise from "
        "a random ordering of the same facies proportions - evidence of genuine stratigraphic order. "
        "A NON-significant result is a legitimate, reportable finding (per Wilkinson et al.'s own "
        "landmark caution that qualitative 'looks cyclic' assessments often do not survive rigorous "
        "testing) - do not keep adjusting parameters until something looks significant."
    )


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv`` and ``output_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/tables/cyclicity")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(input_csv=args.input_csv, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()