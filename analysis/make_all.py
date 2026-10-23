#!/usr/bin/env python3
"""Regenerate every paper figure, table and statistic (tables are printed as Markdown and LaTeX).

Run with the project virtualenv (stats_rw_gradient needs statsmodels):

    .venv/bin/python -m analysis.make_all
"""
from analysis import (
    fig_bridge_pricer_vs_estimator,
    fig_dataset_cost_quality,
    fig_h1_band_vs_adaptive,
    fig_h1_ladder_trajectories,
    fig_h1_llm_conditions,
    fig_h1_llm_margin_traces,
    fig_h1_llm_variant_trajectories,
    fig_h2_conditions,
    fig_h2_mechanism,
    fig_h2_trajectories,
    stats_multiplicity,
    stats_outcomes,
    stats_rw_gradient,
    table_decision_cost,
    table_h1_pricing_behaviour,
    table_dataset,
    table_h2_estimation_accuracy,
    table_h2_oracle_abstention,
)

FIGURES = [
    table_dataset,
    fig_dataset_cost_quality,
    fig_h1_ladder_trajectories,
    fig_h1_band_vs_adaptive,
    fig_h1_llm_variant_trajectories,
    fig_h1_llm_conditions,
    fig_h1_llm_margin_traces,
    table_h1_pricing_behaviour,
    fig_h2_trajectories,
    fig_h2_conditions,
    fig_h2_mechanism,
    table_h2_oracle_abstention,
    table_h2_estimation_accuracy,
    table_decision_cost,
    fig_bridge_pricer_vs_estimator,
    stats_outcomes,
    stats_multiplicity,
    stats_rw_gradient,
]


def main():
    for module in FIGURES:
        print(f"\n=== {module.__name__.rsplit('.', 1)[-1]} ===")
        module.main()


if __name__ == "__main__":
    main()
