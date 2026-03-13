# Especificacao Tecnica — OpenF1 Dataset Builder (Estado Atual)

## 1. Objetivo Atual

- Coletar dados historicos da API OpenF1 de forma incremental e controlada.
- Construir datasets analiticos em camadas bronze, silver e gold.
- Registrar execucoes, metricas e artefatos no MLflow.
- Gerar relatorios e rankings de pilotos.
- Disponibilizar uma API FastAPI para orquestrar relatorios e importacao de temporadas.
- Expor consultas ao gold consolidado (perguntas em linguagem natural e catalogo de meetings).

## 2. Stack E Execucao

- Python 3.11 com pipeline modular e jobs de analytics.
- FastAPI para endpoints de orquestracao.
- MLflow para tracking de execucoes e artefatos.
- MLflow Gateway para respostas de LLM no endpoint de perguntas do gold.
- Dockerfile e Docker Compose como caminho oficial de execucao.
- Visual Studio 2026 como IDE recomendada para desenvolvimento local.

Execucao com Docker Compose (com build):
```bash
cd F1.OpenF1.DatasetBuilder
docker compose up --build openf1-dataset
```

```bash
cd F1.OpenF1.DatasetBuilder
docker compose up --build openf1-api
```

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
- `GET /health`: healthcheck simples da API, confirma que o serviço está respondendo (não valida dependências externas).
- `GET /health/dependencies`: verifica dependências externas (MLflow, MinIO/S3 e OpenF1) e retorna status por dependência.
- `GET /gold/meetings`: lista meetings existentes no gold (meeting_key/meeting_name/sessions).
- `POST /gold/questions`: responde perguntas usando o gold consolidado (pt-BR garantido).
- `POST /train/stint-delta-pace`: treino assincrono do modelo de delta de ritmo (com filtros, MLflow obrigatorio). (Machine Learning)
- `POST /driver-profiles`: gera relatorios por meeting. Campos: `season`, `meeting_key`, `session_name` (Race, Sprint ou all), `include_llm`, `llm_endpoint`.
- `POST /driver-profiles/season`: gera relatorios por temporada (multiplas sessoes). Campos: `seasons`, `session_names`, `include_llm`, `llm_endpoint`, `drivers_include`, `drivers_exclude`.
- `POST /import-season`: cria job assincrono por temporada. Campos: `season`, `session_name` (Race ou Sprint), `include_llm`, `llm_endpoint`, `resume_job_id` (opcional).
- `POST /import-season/resume`: cria job assincrono a partir de um job anterior. Campos: `resume_job_id`, `include_llm` (opcional), `llm_endpoint` (opcional).
- `POST /data-lake/sync`: sincroniza bronze/silver/gold com MinIO (upload/download).
- `GET /jobs/{job_id}`: status do job.
- `GET /jobs/{job_id}/logs?lines=200`: ultimas linhas do log do job.

Detalhes do `/train/stint-delta-pace` (Machine Learning):
Objetivo: treinar um modelo de regressao para prever o delta de ritmo entre stints usando o gold consolidado.
Requisitos: gold consolidado local e MLflow configurado para log de parametros, metricas e artefatos.
Parametros principais: `target_mode` (`prev_stint_mean` para delta entre stints consecutivos; `stint_start_mean` para delta vs media das primeiras voltas do stint), `baseline_laps` (usado no `stint_start_mean`), `group_col` (coluna usada para split por grupo e evitar vazamento), `test_size`, `random_state`, `n_estimators`, `max_depth`, `min_samples_leaf`.
Filtros opcionais: `season`, `meeting_key`, `session_name` (Race, Sprint ou all), `driver_number`, `constructor`.
Retorno: `job_id` para acompanhamento via `/jobs/{job_id}` e logs em `f1_dataset/data/logs/jobs`.

Exemplo (com filtros):
```bash
curl -X POST http://localhost:7077/train/stint-delta-pace \
  -H "Content-Type: application/json" \
  -d '{"season":2024,"session_name":"Race","target_mode":"stint_start_mean","baseline_laps":3,"constructor":"McLaren"}'
```

Exemplo (minimo, sem filtros):
```bash
curl -X POST http://localhost:7077/train/stint-delta-pace \
  -H "Content-Type: application/json" \
  -d '{"target_mode":"prev_stint_mean","baseline_laps":3}'
```

Exemplo de resposta:
```json
{
  "status": "queued",
  "job_id": "6f7b4c0a9c8b4f9f9b0f2f6c7a8d9e10"
}
```

Exemplo de acompanhamento em `/jobs/{job_id}`:
```json
{
  "job_id": "6f7b4c0a9c8b4f9f9b0f2f6c7a8d9e10",
  "job_type": "train_stint_delta_pace",
  "status": "running",
  "created_at": "2026-03-13T12:10:15.123456",
  "filters": {
    "season": 2024,
    "meeting_key": null,
    "session_name": "Race",
    "driver_number": null,
    "constructor": "McLaren"
  },
  "params": {
    "target_mode": "stint_start_mean",
    "baseline_laps": 3,
    "group_col": "meeting_key",
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 300,
    "max_depth": null,
    "min_samples_leaf": 1
  },
  "log_file": "f1_dataset/data/logs/jobs/6f7b4c0a9c8b4f9f9b0f2f6c7a8d9e10.log",
  "status_file": "f1_dataset/data/logs/jobs/6f7b4c0a9c8b4f9f9b0f2f6c7a8d9e10.status.json"
}
```

Exemplo de logs em `/jobs/{job_id}/logs?lines=200`:
```json
{
  "job_id": "6f7b4c0a9c8b4f9f9b0f2f6c7a8d9e10",
  "lines": 5,
  "log": "2026-03-13 12:10:16,021 INFO root - Carregando gold consolidado\n2026-03-13 12:10:18,442 INFO root - Aplicando filtros: season=2024, session_name=Race\n2026-03-13 12:10:21,107 INFO root - Treinando RandomForestRegressor\n2026-03-13 12:10:29,553 INFO root - Metrics: mae=1.23, rmse=2.34, r2=0.78, mape=0.04\n2026-03-13 12:10:30,112 INFO root - Run finalizado"
}
```

Exemplo de resume:
```bash
curl -X POST http://localhost:7077/import-season/resume \
  -H "Content-Type: application/json" \
  -d '{"resume_job_id":"SEU_JOB_ID","include_llm": true}'
```

Exemplo de health de dependencias:
```bash
curl http://localhost:7077/health/dependencies
```

Exemplo de resposta do `/health/dependencies`:
```json
{
  "status": "degraded",
  "dependencies": {
    "mlflow": { "status": "ok", "tracking_uri": "http://mlflow:5000", "latency_ms": 120 },
    "minio": { "status": "ok", "endpoint": "http://minio:9000", "bucket": "openf1-datalake", "latency_ms": 95 },
    "openf1": {
      "status": "degraded",
      "status_code": 429,
      "message": "OpenF1 pode ficar indisponivel para nao-assinantes em horario de eventos."
    }
  },
  "checked_at": "2026-03-13T13:45:58.906279Z"
}
```
Interpretacao rapida:
- `ok`: dependencia acessivel.
- `degraded`: respondeu com restricao (ex.: 401/403/429) ou configuracao parcial.
- `down`: indisponivel.
- `not_configured`: variaveis/credenciais nao configuradas.

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
