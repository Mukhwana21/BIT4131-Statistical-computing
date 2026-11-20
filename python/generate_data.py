"""Generate the shared synthetic teaching dataset used by both language runners."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "student_performance.csv"


def main() -> None:
    rng = np.random.default_rng(4131)
    n = 240
    month_index = np.repeat(np.arange(24), 10)
    groups = rng.choice(["A", "B", "C"], size=n, p=[0.34, 0.33, 0.33])
    gender = rng.choice(["Female", "Male"], size=n)
    study_hours = rng.uniform(1, 10, size=n)
    attendance = rng.uniform(55, 100, size=n)
    group_effect = pd.Series(groups).map({"A": 0.0, "B": 2.0, "C": -1.5}).to_numpy()
    seasonal = 2.0 * np.sin(2 * np.pi * (month_index % 12) / 12)
    score = (
        28
        + 3.2 * study_hours
        + 0.28 * attendance
        + group_effect
        + seasonal
        + rng.normal(0, 7, size=n)
    )
    score = np.clip(score, 0, 100)
    passed = (score >= 60).astype(int)
    dates = pd.to_datetime("2024-01-15") + pd.to_timedelta(
        month_index * 30, unit="D"
    )
    score_with_missing = score.copy()
    hours_with_missing = study_hours.copy()
    score_missing = rng.choice(n, size=14, replace=False)
    hours_missing = rng.choice(n, size=10, replace=False)
    score_with_missing[score_missing] = np.nan
    hours_with_missing[hours_missing] = np.nan
    passed_with_missing = passed.astype(float)
    passed_with_missing[score_missing] = np.nan

    frame = pd.DataFrame(
        {
            "student_id": np.arange(1, n + 1),
            "date": dates.strftime("%Y-%m-%d"),
            "group": groups,
            "gender": gender,
            "study_hours": hours_with_missing.round(3),
            "attendance_pct": attendance.round(2),
            "score": score_with_missing.round(2),
            "passed": passed_with_missing,
            "score_complete": score.round(2),
        }
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False, na_rep="")
    print(f"Wrote {len(frame)} records to {OUTPUT.relative_to(ROOT)}")
    print(f"Missing score values: {int(frame['score'].isna().sum())}")
    print(f"Missing study-hour values: {int(frame['study_hours'].isna().sum())}")


if __name__ == "__main__":
    main()
