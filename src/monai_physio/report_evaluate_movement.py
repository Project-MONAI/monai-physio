"""Rendering of a movement evaluation: its CSVs, its plot and its report.

:class:`ReportEvaluateMovement` turns the metric rows
:class:`monai_physio.WorkflowEvaluateMovement` scores into the artifacts a
reader looks at.  It computes nothing --- every number it prints was measured by
the workflow --- so a change to how a result is *presented* never touches how it
is *measured*.

Kept separate from the workflow for the same reason the metrics were moved out
of the inference workflows: one file, one job.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .monai_physio_base import MONAIPhysioBase


class ReportEvaluateMovement(MONAIPhysioBase):
    """Write the CSV, the volume plot and the markdown report of one evaluation.

    Args:
        log_level: Logging level. Default: ``logging.INFO``.
    """

    # Volume-plot series colors, assigned in this order and never cycled: eight
    # hues whose neighbors stay apart under the common color-vision deficiencies.
    _SERIES_COLORS = (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#4a3aa7",
        "#e34948",
    )

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

    # ─────────────────────────── Public API ────────────────────────────────
    def report(
        self,
        rows: list[dict[str, Any]],
        provenance: dict[str, Any],
        stages: list[float],
        smoothing_sigma_mm: float,
        evaluation_spacing_mm: float,
        displacement_statistics: list[dict[str, Any]],
        pooled: dict[str, float],
        out_dir: Path,
    ) -> dict[str, Path]:
        """Write every artifact of one evaluation and return their paths.

        Args:
            rows: One metric row per stage and structure, as scored by
                :meth:`monai_physio.WorkflowEvaluateMovement.process`.
            provenance: Case name, shape parameters and network weights.
            stages: Every stage scored, in order.
            smoothing_sigma_mm: Recorded in the report's run section.
            evaluation_spacing_mm: Recorded in the report's run section.
            displacement_statistics: One entry per stage, empty when the
                point-by-point error was not measured.
            pooled: That error over every point and stage.
            out_dir: Directory the artifacts are written to.

        Returns:
            Dict with ``csv_file``, ``volume_plot_file`` and ``report_file``.
        """
        csv_file = self._write_csv(rows, out_dir)
        plot_file = self._write_volume_plot(rows, out_dir)
        report_file = self._write_report(
            rows,
            provenance,
            stages,
            smoothing_sigma_mm,
            evaluation_spacing_mm,
            plot_file,
            displacement_statistics,
            pooled,
            out_dir,
        )
        return {
            "csv_file": csv_file,
            "volume_plot_file": plot_file,
            "report_file": report_file,
        }

    # ───────────────────────────── Internals ───────────────────────────────
    @staticmethod
    def _write_csv(rows: list[dict[str, Any]], out_dir: Path) -> Path:
        """Write every metric row, provenance included, to one CSV."""
        csv_file = out_dir / "evaluation_metrics.csv"
        with csv_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return csv_file

    def _write_volume_plot(self, rows: list[dict[str, Any]], out_dir: Path) -> Path:
        """Plot the acquired and predicted volume of every structure against stage.

        One color per structure, taken in a fixed order from a hue set separable
        under color-vision deficiency; the acquired volume is solid and the
        predicted volume dashed, so the two never rest on color alone.
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        plot_file = out_dir / "volume_vs_stage.png"
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        try:
            ends: list[tuple[float, float, str]] = []
            for index, label in enumerate(
                sorted({int(row["label_id"]) for row in rows})
            ):
                matching = sorted(
                    (row for row in rows if row["label_id"] == label),
                    key=lambda row: float(row["stage"]),
                )
                stages = [float(row["stage"]) for row in matching]
                truth = [float(row["volume_truth_mm3"]) / 1000.0 for row in matching]
                predicted = [
                    float(row["volume_predicted_mm3"]) / 1000.0 for row in matching
                ]
                # More structures than hues reuses them; the end labels, not the
                # color, are what name each line.
                color = self._SERIES_COLORS[index % len(self._SERIES_COLORS)]
                ax.plot(stages, truth, color=color, linewidth=2.0, marker="o", ms=5)
                ax.plot(stages, predicted, color=color, linewidth=2.0, linestyle="--")
                ends.append((stages[-1], truth[-1], str(matching[0]["label_name"])))

            # Three of the hues fall below 3:1 against a white page, so each line
            # is named where it ends rather than in a color key alone. Structures
            # of similar size end on top of each other, so the names are pushed
            # apart, largest first, before they are drawn.
            span = float(np.ptp(ax.get_ylim()))
            previous = float("inf")
            for x_end, y_end, name in sorted(ends, key=lambda end: -end[1]):
                text_y = min(y_end, previous - 0.05 * span)
                ax.annotate(
                    name,
                    xy=(x_end, text_y),
                    xytext=(6, 0),
                    textcoords="offset points",
                    color="#52514e",
                    fontsize=9,
                    va="center",
                )
                previous = text_y

            ax.set_xlabel("Stage", color="#52514e")
            ax.set_ylabel("Volume (mL)", color="#52514e")
            ax.grid(True, color="#e1e0d9", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#c3c2b7")
            ax.spines["bottom"].set_color("#c3c2b7")
            ax.tick_params(colors="#898781", labelsize=9)
            ax.legend(
                handles=[
                    Line2D([], [], color="#898781", linewidth=2.0, label="acquired"),
                    Line2D(
                        [],
                        [],
                        color="#898781",
                        linewidth=2.0,
                        linestyle="--",
                        label="predicted",
                    ),
                ],
                frameon=False,
                loc="best",
                fontsize=9,
                labelcolor="#52514e",
            )
            fig.savefig(str(plot_file), bbox_inches="tight", dpi=150)
        finally:
            plt.close(fig)
        self.log_info("Volume plot: %s", plot_file)
        return plot_file

    def _write_report(
        self,
        rows: list[dict[str, Any]],
        provenance: dict[str, Any],
        stages: list[float],
        smoothing_sigma_mm: float,
        evaluation_spacing_mm: float,
        plot_file: Path,
        displacement_statistics: list[dict[str, Any]],
        pooled: dict[str, float],
        out_dir: Path,
    ) -> Path:
        """Write the markdown report beside the CSV."""
        coefficients = [
            provenance[key] for key in sorted(provenance) if key.startswith("pca_c")
        ]
        # Both tables carry whichever metrics the rows were scored with.
        has_dice = "dice" in rows[0]
        lines = [
            f"# Movement accuracy: {provenance['case_id']}",
            "",
            "## Run",
            "",
            f"- Hold-out case: `{provenance['case_id']}`",
            f"- Stages evaluated: {len(stages)} "
            f"({', '.join(f'{stage:.2f}' for stage in stages)})",
            f"- Shape parameters: `{provenance['shape_parameters_file']}`",
            "- Shape parameters (standard deviations): "
            + json.dumps([round(value, 4) for value in coefficients]),
            f"- Network weights: `{provenance['network_weights_file']}`",
            f"- Network weights created: {provenance['network_weights_created']}",
            f"- Network weights modified: {provenance['network_weights_modified']}",
            f"- Network epoch: {provenance['network_epoch']}",
            f"- Deformation smoothing sigma: {smoothing_sigma_mm:.1f} mm",
            f"- Evaluation grid pitch: {evaluation_spacing_mm:.2f} mm isotropic",
            "",
            "Every score compares the reference frame carried into that stage "
            "with the inferred deformation against the frame acquired there.",
            "",
            "## Volume over the stages",
            "",
            f"![Structure volume against stage]({plot_file.name})",
            "",
            "Solid: the volume acquired at that stage. Dashed: the volume the "
            "prediction carries there.",
            "",
            "## Per structure, over the stages",
            "",
            "Mean beside worst case: the stage a structure is predicted worst at "
            "is what a plausible mean can hide.",
            "",
        ]
        metrics = (["Dice"] if has_dice else []) + [
            "Volume difference (%)",
            "Surface RMSE (mm)",
        ]
        has_displacement = "displacement_95th_mm" in rows[0]
        summary_metrics = (["Dice (mean)", "Dice (min)"] if has_dice else []) + [
            "Volume difference (mean %)",
            "Volume difference (worst %)",
            "Surface RMSE (mean mm)",
            "Surface RMSE (max mm)",
        ]
        if has_displacement:
            summary_metrics += [
                "Displacement RMS (mean mm)",
                "Displacement 95th (mean mm)",
                "Displacement max (mm)",
            ]
        lines += self._table_header(["Structure"], summary_metrics)
        for label in sorted({int(row["label_id"]) for row in rows}):
            matching = [row for row in rows if row["label_id"] == label]
            cells = [str(matching[0]["label_name"])]
            if has_dice:
                cells += [
                    f"{self._mean(matching, 'dice'):.4f}",
                    f"{self._min(matching, 'dice'):.4f}",
                ]
            cells += [
                f"{self._mean(matching, 'volume_difference_percent'):+.2f}",
                f"{self._worst(matching, 'volume_difference_percent'):+.2f}",
                f"{self._mean(matching, 'surface_rmse_mm'):.3f}",
                f"{self._worst(matching, 'surface_rmse_mm'):.3f}",
            ]
            if has_displacement:
                cells += [
                    f"{self._mean(matching, 'displacement_rms_mm'):.3f}",
                    f"{self._mean(matching, 'displacement_95th_mm'):.3f}",
                    f"{self._worst(matching, 'displacement_max_mm'):.3f}",
                ]
            lines.append("| " + " | ".join(cells) + " |")

        lines += ["", "## Worst case, over every structure and stage", ""]
        if has_dice:
            lines.append(self._worst_case_line(rows, "dice", "Dice", "{:.4f}", "min"))
        lines += [
            self._worst_case_line(
                rows,
                "volume_difference_percent",
                "Volume difference",
                "{:+.2f} %",
                "magnitude",
            ),
            self._worst_case_line(
                rows, "surface_rmse_mm", "Surface RMSE", "{:.3f} mm", "magnitude"
            ),
        ]
        if has_displacement:
            lines.append(
                self._worst_case_line(
                    rows,
                    "displacement_95th_mm",
                    "Displacement 95th percentile",
                    "{:.3f} mm",
                    "magnitude",
                )
            )
        lines += self._displacement_error_section(displacement_statistics, pooled)

        lines += ["", "## Per stage", ""]
        lines += self._table_header(["Stage", "Structure"], metrics)
        for row in rows:
            cells = [f"{row['stage']:.2f}", str(row["label_name"])]
            if has_dice:
                cells.append(f"{row['dice']:.4f}")
            cells += [
                f"{row['volume_difference_percent']:+.2f}",
                f"{row['surface_rmse_mm']:.3f}",
            ]
            lines.append("| " + " | ".join(cells) + " |")

        report_file = out_dir / "evaluation_report.md"
        report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.log_info("Report: %s", report_file)
        return report_file

    @staticmethod
    def _worst_case_line(
        rows: list[dict[str, Any]], key: str, name: str, fmt: str, pick: str
    ) -> str:
        """Bullet naming the structure and stage one metric is worst at.

        ``pick`` is ``"min"`` for a metric whose worst value is its smallest, and
        ``"magnitude"`` for one whose worst value is the furthest from zero in
        either direction.
        """
        measured = [row for row in rows if not np.isnan(float(row[key]))]
        if not measured:
            return f"- Worst {name}: not measured"
        worst = (
            min(measured, key=lambda row: float(row[key]))
            if pick == "min"
            else max(measured, key=lambda row: abs(float(row[key])))
        )
        return (
            f"- Worst {name}: {fmt.format(float(worst[key]))} "
            f"({worst['label_name']}, stage {float(worst['stage']):.2f})"
        )

    def _displacement_error_section(
        self, statistics: list[dict[str, Any]], pooled: dict[str, float]
    ) -> list[str]:
        """Per-stage displacement-vector error, when the true surfaces were given."""
        if not statistics:
            return []
        lines = [
            "",
            "## Displacement error",
            "",
            "Distance between where the network puts each mesh point and where "
            "the shape model fitted it in the frame acquired at that stage. "
            "Unlike the labelmap metrics above, this is measured point by point, "
            "so it does not average a displacement away against its opposite.",
            "",
            f"- RMS over every point and stage: {pooled['rms_mm']:.3f} mm",
            f"- 95th percentile over every point and stage: {pooled['p95_mm']:.3f} mm",
            f"- Max over every point and stage: {pooled['max_mm']:.3f} mm",
            "",
            "The 95th percentile is the figure to quote: the maximum is one "
            "point of one stage, so it moves with a single badly placed vertex, "
            "while the RMS is pulled down by the bulk of the surface that "
            "hardly moves at all. Every figure here is over the whole shape "
            "model; the same error split by structure is in the table above.",
            "",
        ]
        lines += self._table_header(
            ["Stage"], ["RMS (mm)", "95th percentile (mm)", "Max (mm)"]
        )
        for row in statistics:
            lines.append(
                f"| {float(row['stage']):.2f} | {float(row['rms_error_mm']):.3f} "
                f"| {float(row['p95_error_mm']):.3f} "
                f"| {float(row['max_error_mm']):.3f} |"
            )
        return lines

    @staticmethod
    def _table_header(keys: list[str], metrics: list[str]) -> list[str]:
        """Markdown header and alignment rows: keys left, metrics right."""
        return [
            "| " + " | ".join(keys + metrics) + " |",
            "| " + " | ".join(["---"] * len(keys) + ["---:"] * len(metrics)) + " |",
        ]

    @staticmethod
    def _column(rows: list[dict[str, Any]], key: str) -> list[float]:
        """One column's values, dropping the rows it could not be measured on."""
        return [float(row[key]) for row in rows if not np.isnan(float(row[key]))]

    @staticmethod
    def _mean(rows: list[dict[str, Any]], key: str) -> float:
        """Mean of one column, ignoring the rows where it could not be measured."""
        values = ReportEvaluateMovement._column(rows, key)
        return float(np.mean(values)) if values else float("nan")

    @staticmethod
    def _min(rows: list[dict[str, Any]], key: str) -> float:
        """Smallest value of one column."""
        values = ReportEvaluateMovement._column(rows, key)
        return float(np.min(values)) if values else float("nan")

    @staticmethod
    def _worst(rows: list[dict[str, Any]], key: str) -> float:
        """Value of one column furthest from zero, its sign kept."""
        values = ReportEvaluateMovement._column(rows, key)
        return float(max(values, key=abs)) if values else float("nan")
