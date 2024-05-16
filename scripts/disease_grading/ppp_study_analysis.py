import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import swifter  # noqa

import src.general.common_plot as cplot
import src.segmentation.mask_utils as mask_utils

my_parser = argparse.ArgumentParser(
    description="Load and process PPP study segmentation results.",
    usage="python -m scripts.disease_grading.ppp_study_analysis --preds /path/to/preds",
)
my_parser.add_argument(
    "--preds",
    type=str,
    required=True,
    help="Path to the directory with segmentation results.",
)
my_parser.add_argument(
    "--min-obj-size",
    type=int,
    default=26,
    help="Minimum object size to consider.",
)
my_parser.add_argument(
    "--output",
    type=str,
    help="Path to the output csv.",
)
my_parser.add_argument(
    "--categories",
    type=str,
    nargs="+",
    default=["other", "healthy", "pustules", "spots"],
    help="Prediction categories.",
)
my_parser.add_argument(
    "--object_indices",
    type=int,
    nargs="+",
    default=[2, 3],
    help="Object indices matching prediction categories.",
)


def load_data(path):
    logger.info(f"Loading data from {path}.")
    masks = list(path.glob("*_preds.png"))
    patients = sorted(set([mask.name.split("_")[0] for mask in masks]))
    locs = sorted(set([mask.name.split("_")[1] for mask in masks]))
    df = pd.DataFrame(
        [
            {
                "patient_id": pid,
                **{loc: path / f"{pid}_{loc}_preds.png" for loc in locs},
            }
            for pid in patients
        ]
    )
    df[locs] = df[locs].map(lambda x: str(x) if x.exists() else np.nan)
    return df, patients, locs


def compute_counts(path, obj_indices, min_size):
    df, patients, locs = load_data(path)
    logger.info(
        f"Computing object counts from {path} with indices {obj_indices} and min size {min_size}."
    )
    col_suffixes = ["_pustules_count", "_spots_count", "_combined_count"]
    count_cols = [f"{loc}{suffix}" for loc in locs for suffix in col_suffixes]
    df[count_cols] = df.swifter.apply(
        lambda x: [
            c for loc in locs for c in _compute_counts(x[loc], obj_indices, min_size)
        ],
        axis=1,
        result_type="expand",
    )
    for suffix in col_suffixes:
        df[f"HANDS{suffix}"] = df[f"A{suffix}"] + df[f"B{suffix}"]
        df[f"FEET{suffix}"] = df[f"C{suffix}"] + df[f"D{suffix}"]

    df[locs] = df[locs].map(
        lambda x: x.replace(str(path), path.name) if x is not np.nan else np.nan
    )
    return df


def _compute_counts(mask, obj_indices, min_size):
    if mask is np.nan:
        return (np.nan,) * (len(obj_indices) + 1)
    mask = mask_utils.load_mask_array(mask)
    mask = mask_utils.rm_small_objs_from_non_bin_mask(
        non_binary_mask=mask, min_size=min_size, cats_idxs=obj_indices, bg=0
    )
    counts = [
        mask_utils.nb_objs(non_binary_mask=mask, cls_id=idx, bg=0)
        for idx in obj_indices
    ]
    return *counts, sum(counts)


def plot_counts(df, prefixes):
    logger.info(f"Plotting object counts with prefixes {prefixes}.")
    count_cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    df = df[["patient_id"] + count_cols]
    df = df.melt(
        id_vars="patient_id",
        value_vars=count_cols,
        var_name="count_name",
        value_name="count_value",
    )
    df["location"] = df["count_name"].apply(lambda x: x.split("_")[0])
    df["category"] = df["count_name"].apply(lambda x: x.split("_")[1])
    return cplot.plot_distribution_summary(
        df,
        x="count_value",
        row="location",
        col="category",
        use_row_as_color=False,
        binwidth=10,
        row_title="Location",
        col_title="Category",
        fig_title="Distribution of object counts by location and category",
    )


def save_results(fig, df, output_dir):
    logger.info(f"Saving results to {output_dir}.")
    fig.savefig(output_dir / "object_count_distribution.png", bbox_inches="tight")
    fig.savefig(output_dir / "object_count_distribution.svg", bbox_inches="tight")
    df.to_excel(output_dir / "object_count_distribution.xlsx", index=False)


if __name__ == "__main__":
    my_args = my_parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s",
    )
    logger = logging.getLogger("PPPStudyAnalysisScript")
    logger.info("Running with args: %s", my_args)

    preds = Path(my_args.preds)
    assert preds.exists(), f"Path {my_args.preds} does not exist."
    output = (
        Path(my_args.output)
        if my_args.output
        else preds.parent / f"{preds.name}_analysis"
    )
    output.mkdir(exist_ok=True)

    count_df = compute_counts(preds, my_args.object_indices, my_args.min_obj_size)
    figure = plot_counts(
        count_df,
        prefixes=["HANDS", "FEET"],
    )
    save_results(figure, count_df, output)
