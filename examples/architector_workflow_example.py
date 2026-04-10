"""Example workflow for high-throughput architector calculations.

This script demonstrates how to:
1. Create a workflow from an architector CSV
2. Split a CSV into multiple workflow DBs for multi-cluster distribution
3. Submit jobs to HPC
4. Monitor progress with the dashboard
5. Update job statuses
"""

from oact_utilities.workflows import (
    JobStatus,
    create_split_workflows,
    create_workflow,
)

# =============================================================================
# Step 1: Initialize workflow from architector CSV
# =============================================================================


def setup_workflow():
    """Create a new workflow from architector CSV file."""

    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-14_08-15-32_actinides_sample.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-14_08-15-32_actinides.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-14_08-15-32_non_actinides_sample.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-14_08-15-32_non_actinides.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-14_08-15-32.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-14_08-15-32_actinides_largest.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-14_08-15-32_non_actinides_largest.csv

    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-21_09-49-34_non_actinides.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-21_09-49-34_non_actinides_tuo.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-21_09-49-34_non_actinides_dod.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-21_09-49-34.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-21_09-49-34_actinides_dod.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-21_09-49-34_actinides_tuo.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-21_09-49-34_actinides.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/4_06_pull/santi_results_compiled_full_sane_smax_4_6_26_actinides_tuo.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/4_06_pull/santi_results_compiled_full_sane_smax_4_6_26_non_actinides_tuo.csv

    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/4_06_pull/santi_results_compiled_full_sane_smax_4_6_26_actinides_dod.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/4_06_pull/santi_results_compiled_full_sane_smax_4_6_26_non_actinides_dod.csv

    csv_path = "/Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/4_06_pull/santi_results_compiled_full_sane_smax_4_6_26_non_actinides_tuo.csv"  # Your architector CSV
    db_path = "./4_06/santi_results_compiled_full_sane_smax_4_6_26_non_actinides_tuo.db"  # SQLite database

    # Create workflow (chunks CSV and initializes database)
    db_path, workflow = create_workflow(
        csv_path=csv_path,
        db_path=db_path,
        geometry_column="structure",  # Column with XYZ geometries
        charge_column="charge",
        spin_column="spinmult",
        batch_size=10000,
        extra_columns={"metal": "TEXT"},  # Example extra columns to store
    )

    print("Workflow initialized!")
    print(f"  Database: {db_path}")

    # Show initial summary
    summary = workflow.get_summary()
    print("\nInitial status:")
    print(summary)
    # show row from db as an example
    example_job = workflow.get_jobs_by_status(JobStatus.TO_RUN)[0]
    print("\nExample job record:")
    print(example_job)

    workflow.close()
    return db_path


# =============================================================================
# Step 2 (alternative): Split a CSV into multiple workflow DBs at creation time
# =============================================================================


def setup_split_workflow():
    """Split a CSV into three equal workflow DBs for multi-cluster distribution.

    Creates three independent SQLite databases from the non_actinides_dod CSV,
    each containing roughly one third of the rows. The databases can then be
    submitted to separate HPC allocations or handed to collaborators.

    Output DBs:
        ./4_06/non_actinides_dod_chunk0.db  (~1/3 of rows)
        ./4_06/non_actinides_dod_chunk1.db  (~1/3 of rows)
        ./4_06/non_actinides_dod_chunk2.db  (~1/3 of rows)
    """
    csv_path = "/Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/4_06_pull/santi_results_compiled_full_sane_smax_4_6_26_actinides_dod.csv"

    shards = create_split_workflows(
        csv_path=csv_path,
        db_dir="./4_06",
        db_name="actinides_dod",
        split_names=["chunk0", "chunk1", "chunk2"],
        # no fractions -> equal random split
        geometry_column="structure",
        charge_column="charge",
        spin_column="spinmult",
        extra_columns={"metal": "TEXT"},
        seed=42,
    )

    print("\nSplit workflow created:")
    for db_path, workflow in shards:
        summary = workflow.get_summary()
        n_ready = summary.get("to_run", 0)
        print(f"  {db_path.name}: {n_ready} jobs ready")
        workflow.close()

    return [db_path for db_path, _ in shards]


if __name__ == "__main__":
    # Uncomment to run
    # db_path = setup_workflow()
    db_paths = setup_split_workflow()
    print(__doc__)
    print("\nSee function docstrings for usage examples.")
