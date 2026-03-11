from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


def _latest_file(base_dir: Path, filename: str) -> Path:
    matches = list(base_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Arquivo nao encontrado: {filename} em {base_dir}")
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _post_json(url: str, payload: Dict, timeout_s: int) -> Tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            status = resp.getcode()
            content = resp.read().decode("utf-8")
            return status, content
    except HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except URLError as e:
        return 0, str(e)


def _build_prompt(row: pd.Series, summary_text: str) -> Dict:
    def _safe_float(key: str) -> float | None:
        if key in row and pd.notna(row[key]):
            return float(row[key])
        return None

    delta_pace_abs = None
    if "delta_pace_median_abs" in row:
        delta_pace_abs = float(row["delta_pace_median_abs"])
    elif "delta_pace_mean_abs" in row:
        delta_pace_abs = float(row["delta_pace_mean_abs"])

    lap_mean_delta = None
    meeting_lap_mean_avg = None
    lap_mean_z = None
    if "lap_mean_delta_to_meeting_mean" in row:
        lap_mean_delta = float(row["lap_mean_delta_to_meeting_mean"])
    if "meeting_lap_mean_avg" in row:
        meeting_lap_mean_avg = float(row["meeting_lap_mean_avg"])
    if "lap_mean_z_to_meeting_mean" in row:
        lap_mean_z = float(row["lap_mean_z_to_meeting_mean"])

    finish_rate = _safe_float("finish_rate")
    lap_completion_mean = _safe_float("lap_completion_mean")
    points_total = _safe_float("points_total")
    points_race = _safe_float("points_race")
    points_sprint = _safe_float("points_sprint")
    races_count = _safe_float("races_count")
    sprints_count = _safe_float("sprints_count")

    delta_pace_count = None
    if "delta_pace_count" in row:
        try:
            delta_pace_count = int(row["delta_pace_count"])
        except Exception:
            delta_pace_count = None

    metrics = {
        "overall_rank": int(row["overall_rank"]),
        "overall_score": float(row["overall_score"]),
        "lap_mean": _safe_float("lap_mean"),
        "lap_mean_delta_to_meeting_mean": lap_mean_delta,
        "meeting_lap_mean_avg": meeting_lap_mean_avg,
        "lap_mean_z_to_meeting_mean": lap_mean_z,
        "lap_std": _safe_float("lap_std"),
        "lap_quality_good_rate": _safe_float("lap_quality_good_rate"),
        "anomaly_rate": _safe_float("anomaly_rate"),
        "degradation_mean": _safe_float("degradation_mean"),
        "degradation_slope": _safe_float("degradation_slope"),
        "delta_pace_median_abs": delta_pace_abs,
        "delta_pace_count": delta_pace_count,
        "rank_percentile_mean": _safe_float("rank_percentile_mean"),
        "rank_percentile_median": _safe_float("rank_percentile_median"),
        "finish_rate": finish_rate,
        "lap_completion_mean": lap_completion_mean,
        "points_total": points_total,
        "points_race": points_race,
        "points_sprint": points_sprint,
        "races_count": races_count,
        "sprints_count": sprints_count,
    }
    system = (
        "Voce e um analista de desempenho da F1. Gere uma analise por topicos em portugues.\n"
        "Regras: use exatamente estes topicos, cada um em uma linha iniciando com '- ':\n"
        "1) Desempenho geral (ranking/score)\n"
        "2) Ritmo relativo a pista (use lap_mean_delta_to_meeting_mean e meeting_lap_mean_avg; negativo = mais rapido; inclua lap_mean_z_to_meeting_mean)\n"
        "3) Consistencia (variabilidade)\n"
        "4) Qualidade de volta\n"
        "5) Anomalias\n"
        "6) Degradacao de pneus (nivel e tendencia)\n"
        "7) Delta pace (mencione o numero de stints usados)\n"
        "8) Posicionamento relativo (use rank_percentile_mean e rank_percentile_median)\n"
        "9) Sintese final (2 a 3 frases, inclua confiabilidade: finish_rate e lap_completion_mean, e pontos: points_total/points_race/points_sprint com contagens)\n"
        "Padrao de formato (obrigatorio):\n"
        "- Use sempre 2 casas decimais.\n"
        "- Valores em segundos com sufixo 's'.\n"
        "- Percentuais multiplicar por 100 e usar '%'.\n"
        "- Z-score sempre como 'z=...'.\n"
        "- Se algum valor vier nulo, escreva 'n/a'.\n"
        "Nao invente fatos fora dos dados fornecidos."
    )
    user = (
        f"Piloto: {row['driver_name']} (#{row['driver_number']}), Equipe: {row['team_name']}.\n"
        f"Resumo tecnico: {summary_text}\n"
        f"Metricas (numeros): {json.dumps(metrics, ensure_ascii=False)}\n"
        "Valores obrigatorios por topico:\n"
        "- Desempenho geral: overall_rank, overall_score (score com 2 casas)\n"
        "- Ritmo relativo a pista: lap_mean_delta_to_meeting_mean (s), meeting_lap_mean_avg (s), lap_mean_z_to_meeting_mean (z=..)\n"
        "- Consistencia: lap_std (s)\n"
        "- Qualidade de volta: lap_quality_good_rate (%)\n"
        "- Anomalias: anomaly_rate (%)\n"
        "- Degradacao de pneus: degradation_mean (s), degradation_slope (s/volta)\n"
        "- Delta pace: delta_pace_median_abs (s), delta_pace_count\n"
        "- Posicionamento relativo: rank_percentile_mean (%), rank_percentile_median (%)\n"
        "- Confiabilidade (na sintese): finish_rate (%), lap_completion_mean (%)\n"
        "- Pontos (na sintese): points_total, points_race, points_sprint, races_count, sprints_count\n"
        "Gere o texto final seguindo as regras."
    )
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    }


def _load_existing(output_csv: Path) -> Dict[int, str]:
    if not output_csv.exists():
        return {}
    df = pd.read_csv(output_csv)
    existing = {}
    for _, r in df.iterrows():
        try:
            existing[int(r["driver_number"])] = str(r.get("llm_text", ""))
        except Exception:
            continue
    return existing


def _append_row(output_csv: Path, row: Dict) -> None:
    is_new = not output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "driver_number",
                "driver_name",
                "team_name",
                "overall_rank",
                "overall_score",
                "llm_text",
                "error",
            ],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera texto por piloto via MLflow Gateway.")
    parser.add_argument(
        "--base-dir",
        default="",
        help="Diretorio base de artifacts.",
    )
    parser.add_argument("--ranking-csv", default="", help="Caminho para driver_overall_ranking.csv")
    parser.add_argument("--profiles-text-csv", default="", help="Caminho para driver_profiles_text.csv")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:5000/gateway/gemini/mlflow/invocations",
        help="Endpoint do MLflow Gateway.",
    )
    parser.add_argument("--output-dir", default="", help="Diretorio de saida.")
    parser.add_argument("--timeout-s", type=int, default=60, help="Timeout por requisicao (s).")
    parser.add_argument("--max-retries", type=int, default=4, help="Tentativas por piloto.")
    parser.add_argument("--sleep-base", type=float, default=1.5, help="Backoff base (s).")
    args = parser.parse_args()

    default_base = Path(os.getenv("ARTIFACTS_DIR", "/app/artifacts")) / "modeling" / "driver_profiles"
    base_dir = Path(args.base_dir) if args.base_dir else default_base
    ranking_csv = Path(args.ranking_csv) if args.ranking_csv else _latest_file(base_dir, "driver_overall_ranking.csv")
    profiles_csv = (
        Path(args.profiles_text_csv)
        if args.profiles_text_csv
        else _latest_file(base_dir, "driver_profiles_text.csv")
    )

    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "llm_reports" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "driver_profiles_llm.csv"

    ranking = pd.read_csv(ranking_csv)
    profiles = pd.read_csv(profiles_csv)
    merged = ranking.merge(profiles, on=["driver_number", "driver_name", "team_name"], how="left")

    existing = _load_existing(output_csv)
    total = len(merged)
    for idx, row in merged.iterrows():
        driver_number = int(row["driver_number"])
        if driver_number in existing and existing[driver_number].strip():
            continue

        summary_text = str(row.get("summary_text", "")).strip()
        payload = _build_prompt(row, summary_text)

        llm_text = ""
        error = ""
        for attempt in range(1, args.max_retries + 1):
            status, content = _post_json(args.endpoint, payload, args.timeout_s)
            if status == 200:
                try:
                    data = json.loads(content)
                    llm_text = data["choices"][0]["message"]["content"].strip()
                    error = ""
                    break
                except Exception as e:
                    error = f"parse_error: {e}"
            else:
                error = f"http_{status}: {content[:200]}"

            sleep_s = args.sleep_base * (2 ** (attempt - 1))
            time.sleep(sleep_s)

        _append_row(
            output_csv,
            {
                "driver_number": driver_number,
                "driver_name": row["driver_name"],
                "team_name": row["team_name"],
                "overall_rank": int(row["overall_rank"]),
                "overall_score": float(row["overall_score"]),
                "llm_text": llm_text,
                "error": error,
            },
        )

        if (idx + 1) % 10 == 0:
            print(f"Processados {idx + 1}/{total}")

    print(f"Concluido. Arquivo: {output_csv}")


if __name__ == "__main__":
    main()
