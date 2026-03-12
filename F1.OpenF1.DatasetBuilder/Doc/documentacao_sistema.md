Documentacao do Sistema OpenF1 Dataset Builder
Esta documentacao descreve arquitetura, modulos, fluxo de dados, configuracao, operacao e detalhes dos jobs e da API. O README de uso rapido fica em `F1.OpenF1.DatasetBuilder/README.md`.

**Visao Geral**
A solucao coleta dados da API OpenF1, organiza em camadas bronze/silver/gold, valida qualidade, publica no MLflow e executa jobs de modelagem e relatorios analiticos. Ha tambem uma API FastAPI para orquestrar a geracao de relatorios e importacao de temporadas com status e logs.

**Arquitetura Funcional**
Fluxo macro:
1. Descoberta dinamica de meetings, sessions e drivers.
2. Coleta de endpoints por unidade de processamento.
3. Persistencia em bronze (raw).
4. Normalizacao em silver.
5. Engenharia de atributos e gold (dataset analitico).
6. Validacao de qualidade.
7. Publicacao de artefatos e metricas no MLflow.
8. Consolidacao, modelagem, rankings e relatorios.

**Diagramas E Fluxos**
Fluxo macro do pipeline:
```text
OpenF1 API
  -> Descoberta (meetings/sessions/drivers)
    -> Coleta por unidade
      -> Bronze (raw)
        -> Silver (normalizado)
          -> Gold (features por volta)
            -> Validacao
              -> Publicacao (MLflow + artefatos)
                -> Consolidacao/Modelagem/Relatorios
```

Fluxo macro do pipeline (Mermaid):
```mermaid
flowchart TD
  A[OpenF1 API] --> B[Descoberta]
  B --> C[Coleta por unidade]
  C --> D[Bronze]
  D --> E[Silver]
  E --> F[Gold]
  F --> G[Validacao]
  G --> H[Publicacao MLflow + Artefatos]
  H --> I[Consolidacao]
  I --> J[Modelagem]
  I --> K[Relatorios/Rankings]
```

Fluxo por unidade de processamento:
```text
(season, meeting_key, session_key, driver_number)
  -> coleta laps
  -> coleta car_data
  -> coleta location
  -> coleta stints
  -> coleta weather
  -> escreve bronze (5 datasets)
  -> normaliza (silver)
  -> gera features e gold
  -> valida qualidade
  -> registra MLflow
  -> checkpoint completed
```

Fluxo da API `/driver-profiles`:
```text
POST /driver-profiles
  -> tenta baixar gold do data lake (quando faltarem)
  -> verifica dados gold (season/meeting/session)
  -> se faltando: executa pipeline e consolida gold
  -> gera driver_profiles.csv
  -> gera rankings e texto
  -> opcional: gera LLM e merge
  -> sincroniza bronze/silver/gold com o data lake e limpa locais (se habilitado)
  -> responde com URIs de artifacts no MLflow
```

Fluxo da API `/driver-profiles` (Mermaid):
```mermaid
flowchart TD
  A[POST /driver-profiles] --> B[Download data lake se necessario]
  B --> C[Check gold data]
  C -->|missing| D[Run pipeline]
  D --> E[Consolidate gold]
  C -->|exists| F[Generate driver_profiles]
  E --> F
  F --> G[Generate rankings]
  G --> H[Generate text report]
  H --> I{Include LLM?}
  I -->|yes| J[Generate LLM + merge]
  I -->|no| K[Skip LLM]
  J --> L[Sync data lake + cleanup local]
  K --> L
  L --> M[Return MLflow artifact URIs]
```

Fluxo da API `/import-season`:
```text
POST /import-season
  -> cria job_id e arquivos de status/log
  -> executa importacao em background
  -> para cada meeting: garante gold, gera relatorios, opcional LLM
  -> atualiza status com progresso por meeting
```

Fluxo da API `/import-season` (Mermaid):
```mermaid
flowchart TD
  A[POST /import-season] --> B[Create job files]
  B --> C[Spawn background process]
  C --> D[Iterate meetings]
  D --> E[Ensure gold data]
  E --> F[Generate reports]
  F --> G{Include LLM?}
  G -->|yes| H[Generate LLM]
  G -->|no| I[Skip LLM]
  H --> J[Update status]
  I --> J
  J --> K[Complete job]
```

Paralelismo e checkpoints:
```text
Runner
  -> build_processing_units()
  -> cria N workers (max_parallel_drivers)
    -> para cada unidade:
      -> consulta checkpoint
      -> se status=completed: skip
      -> status=running
      -> processa pipeline da unidade
      -> status=completed
      -> se erro: status=failed
```

Paralelismo e checkpoints (Mermaid):
```mermaid
flowchart TD
  A[Runner] --> B[build_processing_units]
  B --> C{max_parallel_drivers}
  C --> D[Worker pool]
  D --> E[Fetch checkpoint]
  E -->|completed| F[Skip unit]
  E -->|not completed| G[Set status=running]
  G --> H[Process unit pipeline]
  H --> I[Set status=completed]
  H -->|error| J[Set status=failed]
```

Retry, backoff e rate limiting:
```text
OpenF1Client.get()
  -> wait(min_request_interval_ms)
  -> request
    -> 429: cooldown + retry
    -> 5xx: backoff + retry
    -> 404: retorna vazio
    -> 200: retorna dados
  -> apos N tentativas: erro
```

Retry, backoff e rate limiting (Mermaid):
```mermaid
flowchart TD
  A[OpenF1Client.get] --> B[RateLimiter.wait]
  B --> C[HTTP request]
  C -->|200| D[Return JSON]
  C -->|404| E[Return empty list]
  C -->|429| F[Cooldown]
  C -->|5xx or exception| G[Backoff]
  F --> H{Attempts left?}
  G --> H
  H -->|yes| B
  H -->|no| I[Raise error]
```

**Componentes Principais**
- `clients/`: clientes HTTP para OpenF1 e MLflow.
- `discovery/`: descoberta de meetings, sessions e drivers.
- `collectors/`: coleta de endpoints brutos.
- `processors/`: normalizacao e limpeza basica.
- `feature_engineering/`: agregacoes por volta e features.
- `validators/`: validacao de qualidade do gold.
- `publishers/`: escrita de datasets e publicacao no MLflow.
- `orchestration/`: runner, paralelismo e checkpoints.
- `modeling/`: utilitarios de dataset e pre-processamento.
- `jobs/`: scripts de pipeline, modelagem e relatorios.
- `api/`: FastAPI para orquestracao de relatorios e jobs.

**Unidade De Processamento**
Unidade logica: `(season, meeting_key, session_key, driver_number)`.
Isso permite reprocessar apenas um piloto em uma sessao especifica, com checkpoint independente.

**Orquestracao E Resiliencia**
Arquivos principais: `f1_dataset/src/orchestration/runner.py`, `f1_dataset/src/orchestration/checkpoint_store.py`.
Mecanismos:
- Paralelismo controlado por `execution.max_parallel_drivers`.
- Rate limiting por `execution.min_request_interval_ms`.
- Retry e backoff exponencial por `execution.retry_attempts` e `execution.retry_backoff_seconds`.
- Cooldown em caso de HTTP 429 por `execution.rate_limit_cooldown_seconds`.
- Checkpoints por unidade processada para retomar execucoes.

**Camadas De Dados**
Bronze:
- Dados crus por endpoint (`laps`, `car_data`, `location`, `stints`, `weather`).
- Uso principal: auditoria e replay.
Silver:
- JSON achatado, tipos e datas padronizadas.
Gold:
- Dataset analitico por volta, pronto para modelagem.

**Modelo Gold**
Colunas base:
- Identificadores e contexto: `season`, `meeting_key`, `meeting_name`, `meeting_date_start`, `session_key`, `session_name`, `driver_number`, `driver_name`, `team_name`.
- Volta e alvo: `lap_number`, `lap_duration`, `duration_sector_1`, `duration_sector_2`, `duration_sector_3`.
- Indicadores adicionais: `i1_speed`, `i2_speed`, `st_speed`, `is_pit_out_lap`.

Features de telemetria:
- `avg_speed`, `max_speed`, `min_speed`, `speed_std`.
- `avg_rpm`, `max_rpm`, `min_rpm`, `rpm_std`.
- `avg_throttle`, `max_throttle`, `min_throttle`, `throttle_std`.

Features de controle do carro:
- `full_throttle_pct`, `brake_pct`, `brake_events`, `hard_brake_events`.
- `drs_pct`, `gear_changes`.

Features de trajetoria:
- `distance_traveled`, `trajectory_length`, `trajectory_variation`.

Flags e cobertura de dados:
- `telemetry_points`, `trajectory_points`, `has_telemetry`, `has_trajectory`.

Contexto de stint:
- `stint_number`, `compound`, `stint_lap_start`, `stint_lap_end`.
- `tyre_age_at_start`, `tyre_age_at_lap`.

Clima por volta (quando disponivel):
- `track_temperature`, `air_temperature`, `weather_date`.

**Validacao De Qualidade**
Arquivo: `f1_dataset/src/validators/quality.py`.
Metricas:
- `rows`: total de linhas do gold.
- `null_pct`: percentual medio de nulos.
- `valid_laps`: voltas com `lap_number` valido.
- `discarded_laps`: linhas descartadas por falta de `lap_number`.
Uso: garantir completude minima antes de publicar e modelar.

**Publicacao E MLflow**
Arquivo: `f1_dataset/src/publishers/mlflow_publisher.py`.
Registros principais:
- Parametros da execucao (temporada, meeting, session, driver).
- Metricas de qualidade do gold.
- Artefatos: parquet/csv, relatorio de qualidade, amostras e logs.
Objetivo: rastreabilidade, reproducibilidade e comparacao entre execucoes.

**Jobs Do Sistema**
Pipeline e consolidacao:
- `build_openf1_dataset.py`: executa o pipeline completo por unidades.
- `process_meeting.py`: processa um meeting especifico.
- `consolidate_gold_dataset.py`: consolida gold em um unico arquivo.
- `batch_import_season.py`: gera configs por batches e opcionalmente executa o pipeline.

Modelagem e analytics:
- `train_lap_time_regression.py`: regressao de tempo de volta.
- `train_lap_time_ranking.py`: regressao + ranking de pilotos.
- `train_relative_position.py`: predicao de rank_percentile.
- `train_stint_delta_pace.py`: delta de pace entre stints.
- `train_tyre_degradation.py`: degradacao de pneus.
- `train_lap_quality_classifier.py`: classificacao de qualidade de volta.
- `train_lap_anomaly.py`: deteccao de anomalias por volta.
- `train_driver_style_clustering.py`: clustering de estilo de pilotagem.
- `train_circuit_segmentation.py`: segmentacao de circuitos.
- `compare_lap_time_runs.py`: comparacao dos modelos de lap time no MLflow.
- `compare_extended_experiments.py`: comparacao de experimentos estendidos.

Relatorios e rankings:
- `driver_profiles_report.py`: gera metricas consolidadas por piloto.
- `driver_profiles_rankings.py`: rankings por metrica.
- `driver_profiles_overall_ranking.py`: ranking geral por score composto.
- `driver_profiles_text_report.py`: resumo textual baseado em percentis.
- `generate_driver_performance_llm.py`: texto por piloto via MLflow Gateway.
- `import_season_job.py`: executor do job assincrono criado pela API.

**Metricas Modelagem**
Regressoes (`train_lap_time_regression`, `train_lap_time_ranking`, `train_relative_position`, `train_stint_delta_pace`, `train_tyre_degradation`):
- `mae`, `rmse`, `r2`, `mape`.
Motivo: medir erro absoluto, penalizar erros grandes, variancia explicada e erro relativo.

Ranking de lap time (`train_lap_time_ranking`):
- `rank_spearman_mean`, `rank_ndcg_mean`, `rank_meeting_count`, `rank_driver_mean`.
Motivo: avaliar concordancia de ordenacao e qualidade do ranking por corrida.

Classificacao (`train_lap_quality_classifier`):
- `accuracy`, `precision`, `recall`, `f1`, `roc_auc`.
Motivo: medir performance global, equilibrio entre classes e poder discriminativo.

Anomalias (`train_lap_anomaly`):
- `rows`, `anomaly_count`, `anomaly_rate`, `score_min`, `score_max`, `score_mean`.
Motivo: quantificar incidencia e distribuicao do score.

Clustering (`train_driver_style_clustering`, `train_circuit_segmentation`):
- `clusters`, `rows`, `silhouette`, `davies_bouldin`.
Motivo: avaliar coesao e separacao dos grupos.

**Metricas Pilotos**
Saida principal: `driver_profiles.csv`.
Metricas agregadas por piloto:
- Ritmo e consistencia: `lap_mean`, `lap_std`, `lap_mean_delta_to_meeting_mean`, `lap_mean_z_to_meeting_mean`, `meeting_lap_mean_avg`.
- Qualidade e estabilidade: `lap_quality_good_rate`, `lap_quality_bad_rate`, `anomaly_rate`.
- Desgaste e stints: `degradation_mean`, `degradation_p95`, `degradation_slope` (performance ao longo do stint), `stint_performance_delta_mean`, `stint_performance_delta_slope` (aliases com nome mais claro), `tyre_wear_slope` (desgaste isolado), `delta_pace_mean`, `delta_pace_median`, `delta_pace_std`, `delta_pace_count`.
- Posicionamento relativo: `rank_percentile_mean`, `rank_percentile_median`.
- Confiabilidade: `finish_rate`, `lap_completion_mean`, `dnf_rate`.
- Resultados: `points_total`, `points_race`, `points_sprint`, `races_count`, `sprints_count`, `results_count`.
- Contexto adicional: `laps_total`, `meetings_total`, `pit_out_rate`, `driver_style_cluster`, `dominant_circuit_cluster`, `dominant_circuit_cluster_pct`.
- Contexto de corrida: `meeting_date_start`, `dominant_circuit_speed_class`, `dominant_circuit_speed_class_pct`, `track_temperature_mean`, `track_temperature_min`, `track_temperature_max`, `track_temperature_std`, `air_temperature_mean`, `air_temperature_min`, `air_temperature_max`, `air_temperature_std`.
Uso: alimentar rankings, comparacoes, textos e pontuacao geral.

**API FastAPI**
Arquivo: `f1_dataset/src/api/app.py`.
Endpoints:
- `GET /health`: healthcheck.
- `POST /driver-profiles`: gera relatorios e rankings para um meeting. Aceita `season`, `meeting_key`, `session_name` (Race, Sprint ou all), `include_llm`, `llm_endpoint`.
- `POST /driver-profiles/season`: gera relatorios por temporada e multiplas sessoes. Aceita `seasons`, `session_names` (vazio = todas), `include_llm`, `llm_endpoint`, `drivers_include`, `drivers_exclude`.
- `POST /import-season`: cria job assincrono por temporada. Aceita `season`, `session_name` (Race ou Sprint), `include_llm`, `llm_endpoint`.
- `POST /data-lake/sync`: sincroniza bronze/silver/gold com MinIO (upload/download).
- `GET /jobs/{job_id}`: status do job.
- `GET /jobs/{job_id}/logs?lines=200`: ultimas linhas do log.
Saidas do `/driver-profiles`: URIs no MLflow para `driver_overall_ranking.csv`, `driver_profiles_text.csv` e, se solicitado, `driver_profiles_llm.csv` e `driver_overall_ranking_llm.csv`.
Saidas do `/driver-profiles/season`: `artifacts` por temporada (URIs MLflow), `summaries` por temporada e `top_drivers` por temporada.

**Exemplos de chamadas**

```bash
curl http://localhost:7077/health
```

```bash
curl -X POST http://localhost:7077/driver-profiles \
  -H "Content-Type: application/json" \
  -d '{"season": 2023, "meeting_key": "1141", "session_name": "Race", "include_llm": true}'
```

```bash
curl -X POST http://localhost:7077/driver-profiles/season \
  -H "Content-Type: application/json" \
  -d '{"seasons":[2023,2024], "session_names":["Race","Sprint"], "include_llm": false, "drivers_include": [], "drivers_exclude": []}'
```

```bash
curl -X POST http://localhost:7077/driver-profiles/season \
  -H "Content-Type: application/json" \
  -d '{"seasons":[2023], "session_names":[], "include_llm": false}'
```

```bash
curl -X POST http://localhost:7077/import-season \
  -H "Content-Type: application/json" \
  -d '{"season": 2023, "session_name": "Race", "include_llm": false}'
```

```bash
curl -X POST http://localhost:7077/data-lake/sync \
  -H "Content-Type: application/json" \
  -d '{"direction":"upload","subdirs":["bronze","silver","gold"],"cleanup_local":true}'
```

```bash
curl -X POST http://localhost:7077/data-lake/sync \
  -H "Content-Type: application/json" \
  -d '{"direction":"download","subdirs":["gold"],"only_if_missing":true}'
```

```bash
curl http://localhost:7077/jobs/{job_id}
```

```bash
curl "http://localhost:7077/jobs/{job_id}/logs?lines=200"
```
Saidas do `/import-season`: `job_id` e arquivos de status com progresso por meeting.

**Configuracao**
Arquivo principal: `config/config.yaml`.
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
  max_http_connections: 10
  min_request_interval_ms: 300
  retry_attempts: 4
  retry_backoff_seconds: 2
  rate_limit_cooldown_seconds: 30

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
  tracking_uri: ""
  experiment_name: OpenF1Dataset

api:
  base_url: https://api.openf1.org/v1
```

Modos de meetings:
- `all`: processa todos os meetings da temporada.
- `first_of_season`: processa apenas o primeiro meeting (o runner interrompe apos o primeiro).
- `by_key`: filtra por `meeting_key`.
- `by_name`: filtra por `meeting_name`.

Overrides por variaveis de ambiente:
- `DATA_DIR`, `LOG_DIR`, `CHECKPOINT_DIR`, `ARTIFACTS_DIR`.
- `REGISTER_MLFLOW`, `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT`, `MLFLOW_CREATE_EXPERIMENT`.
- `OPENF1_BASE_URL`.
- `CONFIG_PATH`, `CONFIG_DIR`.
- `JOBS_DIR`.
- `MLFLOW_GATEWAY_ENDPOINT`.
- `CLEANUP_LOCAL_ARTIFACTS`, `SYNC_DATA_LAKE`, `DOWNLOAD_DATA_LAKE`, `CLEANUP_LOCAL_DATA`.
- `DATA_LAKE_BUCKET`, `DATA_LAKE_PREFIX`, `DATA_LAKE_S3_ENDPOINT`.
- `DATA_LAKE_SUBDIRS`, `DATA_LAKE_DOWNLOAD_SUBDIRS`, `DATA_LAKE_CREATE_BUCKET`.

**Diretorios E Artefatos**
- Dados locais: `f1_dataset/data/bronze`, `silver`, `gold` (temporarios, podem ser limpos apos sync).
- Data lake (MinIO/S3): bucket/prefix configurados por `DATA_LAKE_BUCKET` + `DATA_LAKE_PREFIX`.
- Logs: `f1_dataset/data/logs` e `f1_dataset/data/logs/jobs`.
- Checkpoints: `f1_dataset/data/checkpoints`.
- Artefatos: `f1_dataset/data/artifacts` (temporario; publicado no MLflow).
- MLflow local (quando aplicavel): `f1_dataset/data/artifacts/mlruns`.

**Execucao**
Pipeline:
```bash
python -m jobs.build_openf1_dataset --config ./config/config.yaml
```
Consolidacao:
```bash
python -m jobs.consolidate_gold_dataset
```
API:
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
Jobs de modelagem e relatorios seguem a mesma convencao `python -m jobs.<nome>`.

**Observabilidade**
- Logs por job em `data/logs`.
- Logs de jobs assincronos da API em `data/logs/jobs`.
- Checkpoints por unidade em `data/checkpoints`.
- MLflow registra parametros, metricas e artefatos.

**Falhas E Reprocessamento**
- Falhas sao registradas no checkpoint com status `failed`.
- Para reprocessar, remova o checkpoint da unidade ou altere o filtro de entrada.
- O pipeline ignora unidades com status `completed`.
