#!/usr/bin/env python3
"""Regenerate every figure, table and statistic of the paper (tables are printed as Markdown and LaTeX).

    python -m analysis.make_all
"""
from analysis import (
    fig_bridge_pricer_vs_estimator,
    fig_dataset_cost_quality,
    fig_rq1_band_vs_adaptive,
    fig_rq1_ladder_trajectories,
    fig_rq1_llm_conditions,
    fig_rq1_llm_margin_traces,
    fig_rq1_llm_variant_trajectories,
    fig_rq2_conditions,
    fig_rq2_mechanism,
    fig_rq2_trajectories,
    stats_multiplicity,
    stats_outcomes,
    stats_rw_gradient,
    table_decision_cost,
    table_rq1_pricing_behaviour,
    table_dataset,
    table_rq2_estimation_accuracy,
    table_rq2_oracle_abstention,
)

FIGURES = [
    table_dataset,
    fig_dataset_cost_quality,
    fig_rq1_ladder_trajectories,
    fig_rq1_band_vs_adaptive,
    fig_rq1_llm_variant_trajectories,
    fig_rq1_llm_conditions,
    fig_rq1_llm_margin_traces,
    table_rq1_pricing_behaviour,
    fig_rq2_trajectories,
    fig_rq2_conditions,
    fig_rq2_mechanism,
    table_rq2_oracle_abstention,
    table_rq2_estimation_accuracy,
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
