"""Run the 23 BIT 4131 examples against the shared teaching dataset."""

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "student_performance.csv"
OUT = ROOT / "results" / "python"
FIGURES = OUT / "figures"
SEED = 4131
TOPICS = [
    "Measures of central tendency and dispersion",
    "Probability and conditional probability",
    "Probability distributions",
    "Sampling distributions and Central Limit Theorem",
    "Confidence intervals",
    "Hypothesis testing",
    "Correlation and covariance",
    "Linear regression and least squares",
    "Matrix algebra in statistics",
    "Eigenvalues, eigenvectors and PCA",
    "Numerical optimization",
    "Random numbers and Monte Carlo methods",
    "Time-series analysis",
    "Statistical data visualization",
    "Data transformation and normalization",
    "ANOVA and experimental design",
    "Chi-square and categorical data analysis",
    "Logistic regression",
    "Non-parametric statistical methods",
    "Bootstrap and resampling methods",
    "Multivariate statistical analysis",
    "Missing-data analysis and imputation",
    "Statistical reporting and reproducible analysis",
]


def save_figure(number: int, figure: plt.Figure) -> None:
    figure.tight_layout()
    figure.savefig(FIGURES / f"topic_{number:02d}.png", dpi=150)
    plt.close(figure)


def complete(column: str, data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna(subset=[column]).copy()


def run(data: pd.DataFrame) -> list[str]:
    results: list[str] = []
    scores = data["score_complete"].to_numpy()
    complete_scores = complete("score", data)["score"].to_numpy()
    paired = data.dropna(subset=["study_hours", "score_complete"])
    pass_data = data.dropna(subset=["passed"]).copy()
    rng = np.random.default_rng(SEED)

    def topic1() -> str:
        s = pd.Series(scores)
        q1, q3 = s.quantile([0.25, 0.75])
        mode = s.mode().iloc[0]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(scores, orientation="horizontal")
        ax.set(title="Complete assessment-score distribution", xlabel="Score")
        save_figure(1, fig)
        return (
            f"n={len(s)}; mean={s.mean():.3f}; median={s.median():.3f}; "
            f"mode={mode:.2f}; sample variance={s.var(ddof=1):.3f}; "
            f"SD={s.std(ddof=1):.3f}; Q1={q1:.2f}; Q3={q3:.2f}; "
            f"IQR={q3-q1:.2f}. The median and IQR are robust summaries; "
            "compare mean and median when describing skew."
        )

    def topic2() -> str:
        p_pass = float(pass_data["passed"].mean())
        group_b = pass_data[pass_data["group"] == "B"]
        conditional = float(group_b["passed"].mean())
        simulated = rng.binomial(1, p_pass, size=10000)
        return (
            f"P(pass)={p_pass:.4f}; P(pass | group B)={conditional:.4f}; "
            f"10,000-trial simulated proportion={simulated.mean():.4f}. "
            "The simulated rate should be close to the observed rate, with "
            "remaining variation due to Monte Carlo sampling."
        )

    def topic3() -> str:
        mu, sigma = stats.norm.fit(complete_scores)
        probability = stats.norm.cdf(60, loc=mu, scale=sigma)
        x = np.linspace(0, 100, 300)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(complete_scores, bins=18, density=True, alpha=0.55, label="Observed")
        ax.plot(x, stats.norm.pdf(x, mu, sigma), color="darkred", label="Fitted normal")
        ax.set(title="Normal fit to observed scores", xlabel="Score", ylabel="Density")
        ax.legend()
        save_figure(3, fig)
        return (
            f"Fitted normal mu={mu:.3f}, sigma={sigma:.3f}; "
            f"fitted P(score <= 60)={probability:.4f}. Compare the curve with "
            "the histogram; the normal model is an approximation to bounded scores."
        )

    def topic4() -> str:
        means = np.mean(rng.choice(scores, size=(5000, 30), replace=True), axis=1)
        expected_se = scores.std(ddof=1) / np.sqrt(30)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(means, bins=35, density=True, alpha=0.7)
        ax.axvline(scores.mean(), color="darkred", linestyle="--", label="Population-data mean")
        ax.set(title="Sampling distribution of means (n=30)", xlabel="Sample mean")
        ax.legend()
        save_figure(4, fig)
        return (
            f"5,000 resamples of size 30: mean of sample means={means.mean():.3f}; "
            f"simulated SE={means.std(ddof=1):.3f}; theoretical SE={expected_se:.3f}. "
            "The sampling distribution is more concentrated and approximately normal."
        )

    def topic5() -> str:
        ci = stats.t.interval(
            0.95, df=len(scores) - 1, loc=scores.mean(),
            scale=stats.sem(scores),
        )
        return (
            f"Mean={scores.mean():.3f}; 95% t interval=({ci[0]:.3f}, {ci[1]:.3f}); "
            "under the sampling assumptions, this method has 95% long-run coverage "
            "for the population mean."
        )

    def topic6() -> str:
        test = stats.ttest_1samp(scores, popmean=60)
        decision = "reject H0" if test.pvalue < 0.05 else "fail to reject H0"
        return (
            f"H0: mean score=60; t({len(scores)-1})={test.statistic:.4f}; "
            f"two-sided p={test.pvalue:.6g}; alpha=0.05: {decision}. "
            "This is evidence relative to H0, not proof about an individual student."
        )

    def topic7() -> str:
        matrix = paired[["study_hours", "score_complete"]].cov()
        correlation = paired["study_hours"].corr(paired["score_complete"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(paired["study_hours"], paired["score_complete"], alpha=0.65)
        ax.set(title="Study hours and assessment score", xlabel="Study hours", ylabel="Score")
        save_figure(7, fig)
        return (
            f"Complete pairs n={len(paired)}; covariance={matrix.iloc[0,1]:.4f}; "
            f"Pearson r={correlation:.4f}. A positive association is not evidence "
            "that study hours alone caused score differences."
        )

    def topic8() -> str:
        fit = ols(
            "score_complete ~ study_hours + attendance_pct",
            data=data.dropna(subset=["study_hours", "score_complete"]),
        ).fit()
        predictions = fit.fittedvalues
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(fit.model.endog, predictions, alpha=0.65)
        ax.plot([0, 100], [0, 100], "r--")
        ax.set(title="Observed versus fitted score", xlabel="Observed", ylabel="Fitted")
        save_figure(8, fig)
        coeff = fit.params
        pvals = fit.pvalues
        return (
            f"n={int(fit.nobs)}; equation: score={coeff['Intercept']:.3f} "
            f"+ {coeff['study_hours']:.3f}*study_hours "
            f"+ {coeff['attendance_pct']:.3f}*attendance; "
            f"p-values (intercept, hours, attendance)="
            f"({pvals['Intercept']:.4g}, {pvals['study_hours']:.4g}, "
            f"{pvals['attendance_pct']:.4g}); R^2={fit.rsquared:.4f}; "
            f"residual SD={np.sqrt(fit.mse_resid):.3f}. Slopes are adjusted associations."
        )

    def topic9() -> str:
        a = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([[1.0, 4.0], [2.0, 0.0]])
        rhs = np.array([5.0, 7.0])
        solution = np.linalg.solve(a, rhs)
        x = np.column_stack(
            [np.ones(len(paired)), paired["study_hours"], paired["score_complete"]]
        )
        y = paired["attendance_pct"].to_numpy()
        beta = np.linalg.solve(x.T @ x, x.T @ y)
        return (
            f"A+B={a+b}; A@B={a@b}; A.T={a.T}; det(A)={np.linalg.det(a):.3f}; "
            f"inverse(A)={np.linalg.inv(a)}; "
            f"solution of A x=b is {solution}; residual norm="
            f"{np.linalg.norm(a @ solution-rhs):.2e}. Regression normal-equation "
            f"coefficients (intercept, hours, score for attendance)={beta.round(4)}."
        )

    def topic10() -> str:
        x = data.dropna(subset=["study_hours", "score_complete"])[
            ["study_hours", "attendance_pct", "score_complete"]
        ].to_numpy()
        z = StandardScaler().fit_transform(x)
        pca = PCA().fit(z)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.arange(1, 4), pca.explained_variance_ratio_, "o-")
        ax.set(title="PCA scree plot", xlabel="Principal component", ylabel="Explained variance ratio")
        ax.set_xticks([1, 2, 3])
        save_figure(10, fig)
        return (
            f"Explained variance ratios={pca.explained_variance_ratio_.round(4)}; "
            f"eigenvalues={pca.explained_variance_.round(4)}; "
            f"eigenvectors/components (rows)={pca.components_.round(4)}. "
            "Standardization prevents the original units from dominating PCA."
        )

    def topic11() -> str:
        x = paired["study_hours"].to_numpy()
        y = paired["score_complete"].to_numpy()

        def objective(beta: np.ndarray) -> float:
            return float(np.sum((y - beta[0] - beta[1] * x) ** 2))

        opt = optimize.minimize(objective, x0=np.zeros(2), method="BFGS")
        closed = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x, y, alpha=0.55, label="Observed")
        ax.plot(grid, opt.x[0] + opt.x[1] * grid, color="darkred", label="Optimized line")
        ax.set(title="Least-squares optimization", xlabel="Study hours", ylabel="Score")
        ax.legend()
        save_figure(11, fig)
        return (
            f"Converged={opt.success}; message={opt.message}; intercept={opt.x[0]:.4f}; "
            f"slope={opt.x[1]:.4f}; sum of squared errors={opt.fun:.3f}; "
            f"closed-form intercept/slope=({closed[1]:.4f}, {closed[0]:.4f})."
        )

    def topic12() -> str:
        n = 50000
        points = rng.random((n, 2))
        inside = (points[:, 0] ** 2 + points[:, 1] ** 2) <= 1
        estimates = 4 * np.cumsum(inside) / np.arange(1, n + 1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(1, n + 1), estimates, linewidth=0.7)
        ax.axhline(np.pi, color="red", linestyle="--", label="pi")
        ax.set(title="Monte Carlo approximation of pi", xlabel="Draws", ylabel="Estimate")
        ax.legend()
        save_figure(12, fig)
        q = float(inside.mean())
        se = 4 * np.sqrt(q * (1 - q) / n)
        return f"N={n}; estimate pi={4*q:.6f}; approximate MC SE={se:.6f}."

    def topic13() -> str:
        monthly = data.assign(date=pd.to_datetime(data["date"])).set_index("date")[
            "score_complete"
        ].resample("MS").mean().dropna()
        time = np.arange(len(monthly))
        trend = stats.linregress(time, monthly.to_numpy())
        forecast = trend.intercept + trend.slope * len(monthly)
        acf1 = float(monthly.autocorr(lag=1))
        seasonal = monthly.groupby(monthly.index.month).mean()
        trend_interpretation = (
            "The linear trend is statistically detectable at alpha=0.05."
            if trend.pvalue < 0.05
            else "The linear trend is not statistically detectable at alpha=0.05."
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(monthly.index, monthly.values, marker="o")
        ax.plot(monthly.index, trend.intercept + trend.slope * time, "--", label="Linear trend")
        ax.set(title="Monthly mean complete score", ylabel="Mean score")
        ax.legend()
        save_figure(13, fig)
        return (
            f"Monthly observations={len(monthly)}; first={monthly.index[0].date()}, "
            f"last={monthly.index[-1].date()}; trend={trend.slope:.4f} score/month "
            f"(p={trend.pvalue:.4g}); lag-1 autocorrelation={acf1:.4f}; "
            f"calendar-month mean-score range={seasonal.min():.2f} to "
            f"{seasonal.max():.2f}; "
            f"next-month linear-trend forecast={forecast:.2f}. "
            f"{trend_interpretation} Forecast is illustrative."
        )

    def topic14() -> str:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].hist(scores, bins=18, color="steelblue", edgecolor="white")
        axes[0, 0].set(title="Score histogram", xlabel="Score")
        data.boxplot(column="score_complete", by="group", ax=axes[0, 1])
        axes[0, 1].set(title="Scores by group", xlabel="Group", ylabel="Score")
        axes[0, 1].get_figure().suptitle("")
        axes[1, 0].scatter(paired["study_hours"], paired["score_complete"], alpha=0.6)
        axes[1, 0].set(title="Study hours vs score", xlabel="Hours", ylabel="Score")
        monthly = data.assign(date=pd.to_datetime(data["date"])).set_index("date")[
            "score_complete"
        ].resample("MS").mean()
        axes[1, 1].plot(monthly.index, monthly.values, marker=".")
        axes[1, 1].set(title="Monthly mean score", ylabel="Mean score")
        save_figure(14, fig)
        return (
            "Created histogram, group boxplot, scatterplot, and time-series line "
            "chart with labeled axes; choose charts to match variable types."
        )

    def topic15() -> str:
        z = stats.zscore(scores, ddof=1)
        scaled = (scores - scores.min()) / (scores.max() - scores.min())
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        axes[0].hist(scores, bins=18, color="steelblue")
        axes[0].set(title="Original", xlabel="Score")
        axes[1].hist(z, bins=18, color="darkorange")
        axes[1].set(title="Z-score", xlabel="Standardized value")
        axes[2].hist(scaled, bins=18, color="seagreen")
        axes[2].set(title="Min-max", xlabel="Scaled value")
        fig.suptitle("Score transformations")
        save_figure(15, fig)
        return (
            f"Original mean/SD=({scores.mean():.3f}, {scores.std(ddof=1):.3f}); "
            f"z-score mean/SD=({z.mean():.3g}, {z.std(ddof=1):.3f}); "
            f"min-max range=({scaled.min():.1f}, {scaled.max():.1f})."
        )

    def topic16() -> str:
        grouped = data.dropna(subset=["score_complete"])
        fit = ols("score_complete ~ C(group)", data=grouped).fit()
        table = anova_oneway(
            grouped["score_complete"].to_numpy(),
            groups=grouped["group"].to_numpy(),
            use_var="equal",
        )
        anova_table = anova_lm(fit, typ=2)
        tukey = pairwise_tukeyhsd(grouped["score_complete"], grouped["group"])
        means = grouped.groupby("group")["score_complete"].mean().round(3).to_dict()
        interpretation = (
            "At alpha=0.05, the omnibus test indicates group means differ."
            if table.pvalue < 0.05
            else "At alpha=0.05, the omnibus test does not detect a group-mean difference."
        )
        return (
            f"Group means={means}; F({table.df_num:.0f},{table.df_denom:.0f})="
            f"{table.statistic:.4f}; p={table.pvalue:.6g}; "
            f"ANOVA table:\n{anova_table.round(4).to_string()}\n"
            f"Tukey HSD:\n{tukey}\n"
            f"{interpretation} Interpret with ANOVA assumptions and nonrandom "
            "group assignment in mind."
        )

    def topic17() -> str:
        table = pd.crosstab(pass_data["group"], pass_data["passed"])
        observed, p, dof, expected = stats.chi2_contingency(table, correction=False)
        interpretation = (
            "Evidence of an association at alpha=0.05."
            if p < 0.05
            else "No statistically significant association detected at alpha=0.05."
        )
        return (
            f"Observed table (columns 0=not pass, 1=pass):\n{table.to_string()}\n"
            f"Expected counts:\n{pd.DataFrame(expected, index=table.index, columns=table.columns).round(2).to_string()}\n"
            f"chi-square({dof})={observed:.4f}; p={p:.6g}. {interpretation} "
            "Assess expected counts and remember association is not causation."
        )

    def topic18() -> str:
        model_data = data.dropna(subset=["study_hours", "passed"])
        x = model_data[["study_hours", "attendance_pct"]].to_numpy()
        y = model_data["passed"].astype(int).to_numpy()
        model = LogisticRegression(random_state=SEED, max_iter=1000).fit(x, y)
        pred = model.predict(x)
        representative_probability = model.predict_proba(
            np.array([[
                model_data["study_hours"].median(),
                model_data["attendance_pct"].median(),
            ]])
        )[0, 1]
        cm = confusion_matrix(y, pred, labels=[0, 1])
        return (
            f"Intercept={model.intercept_[0]:.4f}; coefficients="
            f"{model.coef_[0].round(4)}; odds ratios={np.exp(model.coef_[0]).round(4)}; "
            f"predicted pass probability at median predictors="
            f"{representative_probability:.4f}; "
            f"predictors=study_hours, attendance_pct; apparent confusion matrix "
            f"(rows/columns 0,1)={cm.tolist()}; accuracy={accuracy_score(y,pred):.3f}; "
            f"precision={precision_score(y,pred,zero_division=0):.3f}; "
            f"recall={recall_score(y,pred,zero_division=0):.3f}; "
            f"F1={f1_score(y,pred,zero_division=0):.3f}. "
            "Training-set performance is optimistic; validate on held-out data."
        )

    def topic19() -> str:
        female = data.loc[data["gender"] == "Female", "score_complete"]
        male = data.loc[data["gender"] == "Male", "score_complete"]
        u = stats.mannwhitneyu(female, male, alternative="two-sided")
        t = stats.ttest_ind(female, male, equal_var=False)
        interpretation = (
            "The rank test detects a distributional difference at alpha=0.05."
            if u.pvalue < 0.05
            else "The rank test does not detect a distributional difference at alpha=0.05."
        )
        return (
            f"Female n={len(female)}, median={female.median():.2f}; "
            f"male n={len(male)}, median={male.median():.2f}; "
            f"Mann-Whitney U={u.statistic:.2f}, p={u.pvalue:.5f}; "
            f"Welch t={t.statistic:.4f}, p={t.pvalue:.5f}. {interpretation} "
            "The rank test compares distributions, not necessarily medians alone."
        )

    def topic20() -> str:
        boot = np.mean(rng.choice(scores, size=(5000, len(scores)), replace=True), axis=1)
        ci = np.quantile(boot, [0.025, 0.975])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(boot, bins=35, color="steelblue", edgecolor="white")
        ax.axvline(scores.mean(), color="darkred", linestyle="--", label="Observed mean")
        ax.set(title="Bootstrap sampling distribution of the mean", xlabel="Bootstrap mean")
        ax.legend()
        save_figure(20, fig)
        return (
            f"Resamples=5,000; bootstrap mean={boot.mean():.3f}; "
            f"bootstrap SE={boot.std(ddof=1):.3f}; percentile 95% CI="
            f"({ci[0]:.3f}, {ci[1]:.3f}). Bootstrap cannot remove sampling bias."
        )

    def topic21() -> str:
        x = data.dropna(subset=["study_hours", "score_complete"])[
            ["study_hours", "attendance_pct", "score_complete"]
        ].to_numpy()
        z = StandardScaler().fit_transform(x)
        labels = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit_predict(z)
        counts = np.bincount(labels)
        corr = np.corrcoef(x, rowvar=False)
        covariance = np.cov(x, rowvar=False)
        return (
            f"Variables=(study_hours, attendance_pct, score_complete); "
            f"correlation matrix:\n{np.array2string(corr, precision=3)}\n"
            f"covariance matrix:\n{np.array2string(covariance, precision=3)}\n"
            f"Standardized k-means cluster counts={counts.tolist()}. "
            "Cluster IDs are arbitrary and depend on scaling and initialization."
        )

    def topic22() -> str:
        columns = ["study_hours", "attendance_pct", "score"]
        missing = data[columns].isna().sum()
        matrix = data[columns].isna().to_numpy().T
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Greys")
        ax.set(yticks=np.arange(len(columns)), yticklabels=columns, xlabel="Record")
        ax.set_title("Missingness map (dark = missing)")
        save_figure(22, fig)
        imputed = SimpleImputer(strategy="median").fit_transform(data[columns])
        scores_imputed = imputed[:, 2]
        return (
            f"Missing counts:\n{missing.to_string()}\n"
            f"Missing values imputed per column="
            f"{dict(zip(columns, map(int, data[columns].isna().sum().to_numpy())))}; "
            f"observed score mean={data['score'].mean():.3f}; "
            f"score mean after median imputation={scores_imputed.mean():.3f}. "
            "Median imputation is illustrative and understates uncertainty."
        )

    def topic23() -> str:
        return (
            "The runner records methods and generated results topic by topic, "
            "uses fixed random seeds, reads the shared CSV, and saves figures "
            "alongside this report. For a submission, include this report, the "
            "topic guide, and both source runners in the group's GitHub repository."
        )

    functions = [
        topic1, topic2, topic3, topic4, topic5, topic6, topic7, topic8,
        topic9, topic10, topic11, topic12, topic13, topic14, topic15,
        topic16, topic17, topic18, topic19, topic20, topic21, topic22, topic23,
    ]
    for number, (title, function) in enumerate(zip(TOPICS, functions), start=1):
        result = function()
        results.append(f"## {number}. {title}\n\n{result}\n")
        print(f"\n[{number:02d}] {title}\n{result}")
    return results


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}. Run python/generate_data.py.")
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    required = {
        "date", "group", "gender", "study_hours", "attendance_pct",
        "score", "passed", "score_complete",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        results = run(data)
    figure_topics = {1, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 20, 22}
    for number in figure_topics:
        results[number - 1] += (
            f"\n\n![Topic {number} visualization](figures/topic_{number:02d}.png)\n"
        )
    preamble = (
        "# BIT 4131 reproducible results (Python)\n\n"
        f"Dataset: [`student_performance.csv`](../../data/student_performance.csv) "
        f"({len(data)} synthetic records). Results and figures were generated by "
        "[`run_all.py`](../../python/run_all.py); see the "
        "[topic guide](../../docs/topic_guide.md) for assumptions, theory, and "
        "equations. These are educational examples, not real-student findings.\n\n"
    )
    conclusion = (
        "\n## Overall conclusion and limitations\n\n"
        "In this generated dataset, study hours and attendance are positively "
        "associated with assessment score, and the fitted models summarize those "
        "relationships. Group-level tests provide mixed evidence, illustrating "
        "why effect estimates, assumptions, and post-hoc comparisons matter. "
        "All records are synthetic, models are demonstrations rather than "
        "causal claims, and the reported training-sample classification metrics "
        "are not out-of-sample performance estimates.\n"
    )
    (OUT / "report.md").write_text(
        preamble + "\n".join(results) + conclusion, encoding="utf-8"
    )
    print(f"\nReport written to {OUT.relative_to(ROOT)}/report.md")


if __name__ == "__main__":
    main()
