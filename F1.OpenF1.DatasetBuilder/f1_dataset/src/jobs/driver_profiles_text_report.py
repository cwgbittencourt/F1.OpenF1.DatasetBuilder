from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from config.settings import ensure_paths, load_settings


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "driver_profiles_text.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _init_mlflow(tracking_uri: str | None, experiment_name: str) -> bool:
    if not tracking_uri:
        return False
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return True


def _find_latest_profiles(artifacts_dir: Path) -> Path:
    root = artifacts_dir / "modeling" / "driver_profiles"
    if not root.exists():
        raise FileNotFoundError(f"Nenhum relatorio encontrado em {root}")

    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        reverse=True,
        key=lambda p: p.name,
    )
    for candidate in candidates:
        csv_path = candidate / "driver_profiles.csv"
        if csv_path.exists():
            return csv_path
    raise FileNotFoundError("driver_profiles.csv nao encontrado.")


def _percentile_rank(series: pd.Series) -> pd.Series:
    if len(series) <= 1:
        return pd.Series([0.5] * len(series), index=series.index)
    return series.rank(pct=True)


def _build_text_row(row: pd.Series, strengths: list[str], weaknesses: list[str], other: list[str]) -> str:
    header = f"Piloto {row['driver_name']} (Equipe {row['team_name']})"
    if "lap_mean_delta_to_meeting_mean" in row and pd.notna(row.get("meeting_lap_mean_avg", np.nan)):
        base = (
            f"{header}: ritmo relativo {row['lap_mean_delta_to_meeting_mean']:.3f}s "
            f"(media pista {row['meeting_lap_mean_avg']:.3f}s), "
            f"consistencia {row['lap_std']:.3f}s, "
            f"qualidade de volta {row['lap_quality_good_rate']:.2f}, "
            f"anomalias {row['anomaly_rate']:.2f}."
        )
    else:
        base = (
            f"{header}: ritmo medio {row['lap_mean']:.3f}s, "
            f"consistencia {row['lap_std']:.3f}s, "
            f"qualidade de volta {row['lap_quality_good_rate']:.2f}, "
            f"anomalias {row['anomaly_rate']:.2f}."
        )
    if "finish_rate" in row and pd.notna(row.get("finish_rate", np.nan)):
        base += (
            f" confiabilidade {row['finish_rate']:.2f} "
            f"(completude {row.get('lap_completion_mean', np.nan):.2f})."
        )
    if "points_total" in row and pd.notna(row.get("points_total", np.nan)):
        base += (
            f" pontos {row['points_total']:.0f} "
            f"(race {row.get('points_race', 0):.0f}, sprint {row.get('points_sprint', 0):.0f})."
        )
    strengths_txt = "Pontos fortes: " + (", ".join(strengths) if strengths else "sem destaques fortes.")
    weaknesses_txt = "Pontos fracos: " + (", ".join(weaknesses) if weaknesses else "sem fragilidades fortes.")
    other_txt = "Outras valencias: " + (", ".join(other) if other else "na media no restante.")
    style = f"Estilo(cluster)={row.get('driver_style_cluster', np.nan)}"
    circuit = (
        f"Circuitos(cluster dominante)={row.get('dominant_circuit_cluster', np.nan)} "
        f"({row.get('dominant_circuit_cluster_pct', np.nan):.2f})"
        if pd.notna(row.get("dominant_circuit_cluster_pct", np.nan))
        else "Circuitos(cluster dominante)=n/a"
    )
    return f"{base} {strengths_txt} {weaknesses_txt} {other_txt} {style}. {circuit}."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera texto de performance por piloto."
    )
    parser.add_argument("--config", help="Caminho do arquivo de configuracao")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    config_path = args.config or os.getenv("CONFIG_PATH") or "./config/config.yaml"
    settings = load_settings(config_path)
    ensure_paths(settings)
    _setup_logging(settings.paths.logs_dir)

    artifacts_root = Path(settings.paths.artifacts_dir)
    profiles_path = _find_latest_profiles(artifacts_root)
    df = pd.read_csv(profiles_path)

    if "delta_pace_median" in df.columns:
        df["delta_pace_median_abs"] = df["delta_pace_median"].abs()
    else:
        df["delta_pace_median_abs"] = df["delta_pace_mean"].abs()

    pace_metric = "lap_mean"
    pace_label = "ritmo medio"
    if "lap_mean_delta_to_meeting_mean" in df.columns:
        pace_metric = "lap_mean_delta_to_meeting_mean"
        pace_label = "ritmo relativo a pista (delta)"

    metrics = {
        pace_metric: (pace_label, False),
        "lap_std": ("consistencia", False),
        "lap_quality_good_rate": ("qualidade de volta", True),
        "anomaly_rate": ("estabilidade (anomalias)", False),
        "degradation_mean": ("degradacao media", False),
        "degradation_slope": ("sensibilidade ao desgaste", False),
        "delta_pace_median_abs": ("estabilidade entre stints (mediana)", False),
        "rank_percentile_mean": ("posicao relativa media", False),
        "rank_percentile_median": ("posicao relativa mediana", False),
        "finish_rate": ("confiabilidade (terminos)", True),
        "lap_completion_mean": ("completude de voltas", True),
        "points_total": ("pontos totais", True),
    }

    metric_scores: dict[str, pd.Series] = {}
    for metric, (_, higher_is_better) in metrics.items():
        if metric not in df.columns:
            continue
        series = df[metric].copy()
        series = series.fillna(series.median())
        pct = _percentile_rank(series)
        metric_scores[metric] = pct if higher_is_better else 1.0 - pct

    score_df = pd.DataFrame(metric_scores)

    summaries: list[dict[str, object]] = []
    for idx, row in df.iterrows():
        metric_vals = []
        for metric, (label, _) in metrics.items():
            if metric not in score_df.columns:
                continue
            metric_vals.append(
                (label, float(score_df.loc[idx, metric]), metric)
            )

        metric_vals.sort(key=lambda x: x[1], reverse=True)
        strengths = [m[0] for m in metric_vals[: args.top_k]]
        weaknesses = [m[0] for m in metric_vals[-args.top_k :]]

        other = []
        for label, score, metric in metric_vals[args.top_k : -args.top_k]:
            if score >= 0.6:
                other.append(label + " acima da media")
            elif score <= 0.4:
                other.append(label + " abaixo da media")
            else:
                other.append(label + " na media")

        text = _build_text_row(row, strengths, weaknesses, other[:5])
        summaries.append(
            {
                "driver_number": row["driver_number"],
                "driver_name": row["driver_name"],
                "team_name": row.get("team_name", ""),
                "summary_text": text,
            }
        )

    summary_df = pd.DataFrame(summaries)

    run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = (
        Path(settings.paths.artifacts_dir)
        / "modeling"
        / "driver_profiles"
        / run_timestamp
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifacts_dir / "driver_profiles_text.csv"
    summary_df.to_csv(csv_path, index=False)

    tracking_uri = settings.mlflow.tracking_uri
    if not tracking_uri and settings.output.register_mlflow:
        tracking_path = Path(settings.paths.artifacts_dir) / "mlruns"
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{tracking_path}"

    if settings.output.register_mlflow and _init_mlflow(
        tracking_uri, settings.mlflow.experiment_name
    ):
        with mlflow.start_run(run_name="driver_profiles_text_report") as run:
            mlflow.set_tags({"task": "driver_profiles_text_report"})
            mlflow.log_params({"top_k": args.top_k, "source": str(profiles_path)})
            mlflow.log_artifact(str(csv_path))
            logging.getLogger(__name__).info(
                "Relatorio de texto registrado no MLflow: %s", run.info.run_id
            )

    logging.getLogger(__name__).info("Relatorio gerado: %s", csv_path)


if __name__ == "__main__":
    main()
