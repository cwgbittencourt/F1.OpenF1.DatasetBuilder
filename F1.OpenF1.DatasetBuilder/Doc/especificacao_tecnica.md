# Especificacao Tecnica — OpenF1 Dataset Builder (Estado Atual)

## 1. Objetivo Atual

- Coletar dados historicos da API OpenF1 de forma incremental e controlada.
- Construir datasets analiticos em camadas bronze, silver e gold.
- Registrar execucoes, metricas e artefatos no MLflow.
- Gerar relatorios e rankings de pilotos.
- Disponibilizar uma API FastAPI para orquestrar relatorios e importacao de temporadas.

## 2. Stack E Execucao

- Python 3.11 com pipeline modular e jobs de analytics.
- FastAPI para endpoints de orquestracao.
- MLflow para tracking de execucoes e artefatos.
- Dockerfile e Docker Compose como caminho oficial de execucao.
- Visual Studio 2026 como IDE recomendada para desenvolvimento local.

## 3. Fonte De Dados

Base URL: `https://api.openf1.org/v1`.
Endpoints usados no pipeline:
- `meetings`
- `sessions`
- `drivers`
- `laps`
- `car_data`
- `location`
- `stints`
- `weather`

## 4. Arquitetura Funcional Atual

Fluxo resumido:
1. Descoberta dinamica de meetings, sessions e drivers.
2. Coleta dos endpoints por unidade de processamento.
3. Persistencia em bronze (raw).
4. Normalizacao em silver.
5. Engenharia de atributos e gold (uma linha por volta).
6. Validacao de qualidade.
7. Publicacao no MLflow.
8. Consolidacao e jobs de modelagem/relatorios.

## 5. Campos Adicionais No Gold

- `meeting_date_start`: data/hora de inicio da corrida (meeting).
- `track_temperature`: temperatura da pista por volta (quando disponivel).
- `air_temperature`: temperatura do ar por volta (quando disponivel).
- `weather_date`: timestamp da amostra de clima associada a volta.

## 6. Classificacao De Circuito

- `dominant_circuit_speed_class` em `driver_profiles.csv` com valores `low`, `medium`, `high`.
- A classe e calculada por tercis da velocidade media por meeting.

## 7. Unidade De Processamento E Orquestracao

Unidade logica: `(season, meeting_key, session_key, driver_number)`.
O runner respeita paralelismo configuravel, rate limit, retry com backoff e checkpoints por unidade.

## 8. Configuracao

Arquivo principal: `config/config.yaml`.
Campos relevantes:
- `seasons`: lista de temporadas.
- `session_name`: `Race` ou `Sprint`.
- `drivers.include` e `drivers.exclude`: filtros por nome.
- `meetings.mode`: `all`, `first_of_season`, `by_key`, `by_name`.
- `meetings.include`: lista de keys ou nomes dependendo do modo.
- `execution`: paralelismo, retry e rate limit.
- `output.formats`: `parquet`, `csv`.
- `paths`: diretorios de dados, logs, checkpoints e artifacts.
- `mlflow`: tracking uri e nome do experimento.
- `api.base_url`: base da OpenF1.

Exemplo minimo:
```yaml
seasons:
  - 2023
session_name: Race

drivers:
  include: []
  exclude: []

meetings:
  mode: all
  include: []

execution:
  max_parallel_drivers: 1
  max_http_connections: 5
  min_request_interval_ms: 600
  retry_attempts: 5
  retry_backoff_seconds: 2
  rate_limit_cooldown_seconds: 60

output:
  formats:
    - parquet
    - csv
  register_mlflow: true

paths:
  data_dir: ./f1_dataset/data
  logs_dir: ./f1_dataset/data/logs
  checkpoints_dir: ./f1_dataset/data/checkpoints
  artifacts_dir: ./f1_dataset/data/artifacts

mlflow:
  tracking_uri: null
  experiment_name: OpenF1Dataset

api:
  base_url: https://api.openf1.org/v1
```

## 9. API Do Sistema

Endpoints atuais:
- `GET /health`: healthcheck.
- `POST /driver-profiles`: gera relatorios por meeting. Campos: `season`, `meeting_key`, `session_name` (Race, Sprint ou all), `include_llm`, `llm_endpoint`.
- `POST /import-season`: cria job assincrono por temporada. Campos: `season`, `session_name` (Race ou Sprint), `include_llm`, `llm_endpoint`.
- `GET /jobs/{job_id}`: status do job.
- `GET /jobs/{job_id}/logs?lines=200`: ultimas linhas do log do job.

## 10. Jobs Implementados

Pipeline e consolidacao:
- `build_openf1_dataset.py`
- `process_meeting.py`
- `consolidate_gold_dataset.py`
- `batch_import_season.py`

Modelagem e analytics:
- `train_lap_time_regression.py`
- `train_lap_time_ranking.py`
- `train_relative_position.py`
- `train_stint_delta_pace.py`
- `train_tyre_degradation.py`
- `train_lap_quality_classifier.py`
- `train_lap_anomaly.py`
- `train_driver_style_clustering.py`
- `train_circuit_segmentation.py`
- `compare_lap_time_runs.py`
- `compare_extended_experiments.py`

Relatorios e LLM:
- `driver_profiles_report.py`
- `driver_profiles_rankings.py`
- `driver_profiles_overall_ranking.py`
- `driver_profiles_text_report.py`
- `generate_driver_performance_llm.py`
- `import_season_job.py`

## 11. Saidas E Particionamento

- Bronze: `f1_dataset/data/bronze/season=.../meeting_key=.../session_key=.../driver_number=.../`.
- Silver: `f1_dataset/data/silver/season=.../meeting_key=.../session_key=.../driver_number=.../`.
- Gold: `f1_dataset/data/gold/season=.../meeting_key=.../session_key=.../driver_number=.../dataset.parquet`.
- Consolidado: `f1_dataset/data/gold/consolidated.parquet`.
- Artefatos de relatorios: `f1_dataset/data/artifacts/modeling/driver_profiles`.

## 12. Observabilidade E Resiliencia

- Logs do pipeline em `f1_dataset/data/logs/pipeline.log`.
- Logs de jobs assincronos em `f1_dataset/data/logs/jobs`.
- Checkpoints em `f1_dataset/data/checkpoints`.
- Retry com backoff e cooldown para rate limit.

## 13. Backlog Nao Implementado

- Endpoints adicionais da OpenF1 como `pit`, `position`, `intervals` e `race_control`.
- Persistencia consolidada em banco ou lakehouse.
