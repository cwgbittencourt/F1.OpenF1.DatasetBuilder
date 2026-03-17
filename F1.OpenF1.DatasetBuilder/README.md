# OpenF1 Dataset Builder

Pipeline container-first para ingestao da API OpenF1, construcao de datasets e publicacao no MLflow, com API FastAPI para orquestracao de relatorios.

## Uso rapido (Docker Compose)

1. Ajuste `.env` (ja criado) conforme necessario.
2. Ajuste a configuracao em `config/config.yaml`.
3. Execute o servico desejado:

```bash
docker compose up --build openf1-dataset
```

```bash
docker compose up --build openf1-api
```

O servico da API fica disponivel em `http://localhost:7077/docs`.

## API (FastAPI)

Endpoints:
- `GET /health`: healthcheck simples da API, confirma que o serviço está respondendo. Obs: não valida dependências externas (MLflow, MinIO/S3, OpenF1).
- `GET /health/dependencies`: verifica dependências externas (MLflow, MinIO/S3 e OpenF1) e retorna status por dependência. Obs: OpenF1 pode ficar indisponível para não-assinantes em horários de eventos.
- `GET /catalog/bronze`: lista registros da camada Bronze (dados crus, origem, path, sync opcional via `check_sync=true`); aceita `season` para filtrar.
- `GET /catalog/silver`: lista registros da camada Silver (normalização, schema, nulls, path); aceita `season` para filtrar.
- `GET /catalog/gold`: lista registros do Gold por volta (dataset por piloto); suporta `include_schema=true` e aceita `season` para filtrar.
- `GET /gold/meetings`: lista meetings disponíveis no gold consolidado, com `meeting_key`, `meeting_name` e sessões; aceita filtros por `season` e `session_name`. Obs: requer `f1_dataset/data/gold/consolidated.parquet` local; se não existir retorna 404.
- `GET /gold/lap`: retorna os dados de todos os pilotos para uma volta específica. Requer `season`, `lap_number` e `meeting_key` ou `meeting_name`. Gera colunas adicionais `lap_duration_min` (mm:ss:fff), `lap_duration_total` (hh:mm:ss:fff) e `lap_duration_gap` (hh:mm:ss:fff); ordena por `lap_duration_total`.
- `GET /gold/laps/max`: retorna o número máximo de voltas para uma corrida/sessão, usado para preencher o combo de voltas.
- `POST /gold/questions`: responde perguntas em linguagem natural usando o gold consolidado; aplica filtros e retorna `answer` (pt-BR) e `summary` estatístico do recorte. Tenta DuckDB (LLM -> SQL -> execucao) antes do LLM narrativo. Summary inclui `fastest_lap`, `slowest_lap`, `records`, quantis de tempo, top por meeting/piloto/equipe e cobertura de telemetria/trajectoria. Perguntas sobre “volta mais rápida” são respondidas de forma determinística. Se o LLM retornar “Sem dados no gold.”, tenta fallback web via DuckDuckGo (desativável com `WEB_FALLBACK_PROVIDER=disabled`). DuckDB pode ser desativado com `GOLD_QUESTIONS_DUCKDB=false`. Obs: requer gold consolidado local e endpoint LLM acessível via `MLFLOW_GATEWAY_ENDPOINT`; falhas no LLM retornam 502.
- `GET /ui/gold-lap`: tela web para consultar voltas do gold por temporada + meeting + lap_number, com seleção de colunas e ordenação.
- `GET /jobs`: lista jobs assíncronos recentes (id, status, tipo, datas, mensagem). Status possíveis: `queued`, `running`, `waiting`, `completed`, `failed`, `resumed`.
- Datas de jobs (`created_at`, `started_at`, `finished_at`) são gravadas em UTC (ISO 8601 com timezone).
- `POST /train/stint-delta-pace`: dispara treino assíncrono do modelo de delta de ritmo entre stints; retorna `job_id` para acompanhamento. Obs: job roda em background e grava logs em `f1_dataset/data/logs/jobs`. (Machine Learning)
- `POST /train/lap-time-regression`: treino assíncrono de regressão de tempo de volta. (Machine Learning)
- `POST /train/lap-time-ranking`: treino assíncrono de ranking de lap time. (Machine Learning)
- `POST /train/relative-position`: treino assíncrono de posição relativa por meeting. (Machine Learning)
- `POST /train/tyre-degradation`: treino assíncrono de degradação de pneus. (Machine Learning)
- `POST /train/lap-quality-classifier`: treino assíncrono de classificação de qualidade de volta. (Machine Learning)
- `POST /train/lap-anomaly`: treino assíncrono de detecção de anomalias por volta. (Machine Learning)
- `POST /train/driver-style-clustering`: treino assíncrono de clustering de estilo de pilotagem. (Machine Learning)
- `POST /train/circuit-segmentation`: treino assíncrono de segmentação de circuitos. (Machine Learning)
- `POST /driver-profiles`: gera relatórios e rankings de pilotos para um meeting específico; garante gold, gera artifacts e retorna URIs no MLflow. Obs: pode executar pipeline se faltarem dados e depende de MLflow disponível.
- `POST /driver-profiles/season`: gera relatórios por temporada e múltiplas sessões em lote; retorna artifacts e sumários por temporada. Obs: processamento pode demorar em temporadas múltiplas.
- `POST /import-season`: inicia job assíncrono para importar e processar uma temporada inteira (com opção de LLM); retorna `job_id`. Obs: acompanha via `/jobs` e `/jobs/{job_id}/logs`. Quando encontra etapa futura, pausa com `status=waiting` e `next_meeting`.
- `POST /import-season/resume`: retoma um job de importação anterior usando `resume_job_id`, pulando meetings já concluídos. Obs: requer `resume_job_id` válido de importação. O job anterior passa para `status=resumed` com `resumed_job_id`.
- `POST /data-lake/sync`: sincroniza bronze/silver/gold com o data lake (upload/download) e controla limpeza local. Obs: requer credenciais/endpoint S3/MinIO configurados; `cleanup_local=true` apaga dados locais após sync.
- `GET /jobs/{job_id}`: consulta status atual do job assíncrono (ex.: importação ou treino). Obs: válido para jobs criados por `/import-season` e `/train/stint-delta-pace`.
- `GET /jobs/{job_id}/logs?lines=200`: retorna as últimas linhas do log do job assíncrono. Obs: `lines` controla a quantidade de linhas retornadas.
- `GET /mlflow/runs`: lista runs do MLflow com métricas, parâmetros e artefatos.
- `GET /minio/objects`: lista objetos do MinIO/S3 (bucket, prefixo, tamanho, camada, URI).

Campos principais:
- `/gold/questions`: `question`, `season`, `meeting_key` (opcional), `session_name` (Race, Sprint ou all), `driver_name` (opcional), `driver_number` (opcional). Resposta sempre em pt-BR.
- `/gold/lap`: `season`, `lap_number`, `meeting_key` ou `meeting_name`, `session_name` (Race, Sprint ou all). Retorna também `lap_duration_min`, `lap_duration_total`, `lap_duration_gap` formatados.
- `/gold/laps/max`: `season`, `meeting_key` ou `meeting_name`, `session_name` (Race, Sprint ou all). Retorna `max_lap_number`.
- `/train/stint-delta-pace` (Machine Learning): treina modelo de regressao para delta de ritmo entre stints; `target_mode` (`prev_stint_mean` ou `stint_start_mean`), `baseline_laps` (usado no `stint_start_mean`), `group_col` (split por grupo), `test_size`, `random_state`, `n_estimators`, `max_depth`, `min_samples_leaf` + filtros (`season`, `meeting_key`, `session_name`, `driver_number`, `constructor`).
  Validacao: `mae`, `rmse`, `r2`, `mape` medem a qualidade da previsao (erro absoluto, penalizacao de erros grandes, variancia explicada e erro percentual medio).
- `/train/lap-time-regression` (Machine Learning): `include_sectors`, `group_col`, `test_size`, `random_state`, `n_estimators`, `max_depth`, `min_samples_leaf`.
- `/train/lap-time-ranking` (Machine Learning): `include_sectors`, `group_col`, `driver_col`, `test_size`, `random_state`, `n_estimators`, `max_depth`, `min_samples_leaf`.
- `/train/relative-position` (Machine Learning): `group_col`, `test_size`, `random_state`, `n_estimators`, `max_depth`, `min_samples_leaf`.
- `/train/tyre-degradation` (Machine Learning): `include_sectors`, `group_col`, `test_size`, `random_state`, `n_estimators`, `max_depth`, `min_samples_leaf`.
- `/train/lap-quality-classifier` (Machine Learning): `include_sectors`, `group_col`, `test_size`, `random_state`, `n_estimators`.
- `/train/lap-anomaly` (Machine Learning): `contamination`, `n_estimators`, `random_state`.
- `/train/driver-style-clustering` (Machine Learning): `clusters`, `random_state`.
- `/train/circuit-segmentation` (Machine Learning): `clusters`, `random_state`.
- `/driver-profiles`: `season`, `meeting_key`, `session_name` (Race, Sprint ou all), `include_llm`, `llm_endpoint`.
- `/driver-profiles/season`: `seasons` (lista), `session_names` (lista; vazio = todas), `include_llm`, `llm_endpoint`, `drivers_include`, `drivers_exclude`.
- `/import-season`: `season`, `session_name` (Race ou Sprint), `include_llm`, `llm_endpoint`, `resume_job_id` (opcional).
- `/import-season/resume`: `resume_job_id`, `include_llm` (opcional), `llm_endpoint` (opcional).
- `/data-lake/sync`: `direction` (upload/download), `subdirs`, `cleanup_local`, `only_if_missing`.

Exemplo de chamada:

```bash
curl -X POST http://localhost:7077/driver-profiles \
  -H "Content-Type: application/json" \
  -d '{"season": 2024, "meeting_key": "1252", "session_name": "Race", "include_llm": false}'
```

Exemplo de resposta do `/train/stint-delta-pace`:
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
  "created_at": "2026-03-13T12:10:15.123456+00:00",
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

Os relatorios por piloto incluem classificacao dominante de circuito por velocidade (`dominant_circuit_speed_class`), performance no stint (`stint_performance_delta_*`), desgaste do pneu (`tyre_wear_slope`) e estatisticas de temperatura.
No LLM, a analise separa "performance no stint" de "desgaste do pneu" e contextualiza com temperatura da pista.
No endpoint `/driver-profiles/season`, a resposta inclui:
- `artifacts` por temporada (URIs do MLflow)
- `summaries` por temporada (drivers, laps, meetings, sessions)
- `top_drivers` por temporada (top 5 do ranking geral)

## Exemplos de endpoints

```bash
curl http://localhost:7077/health
```

```bash
curl http://localhost:7077/health/dependencies
```

```bash
curl http://localhost:7077/ui/gold-lap
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

```bash
curl "http://localhost:7077/gold/meetings?season=2024&session_name=Race"
```

```bash
curl "http://localhost:7077/gold/lap?season=2024&meeting_name=Italian%20Grand%20Prix&lap_number=12"
```

```bash
curl "http://localhost:7077/gold/laps/max?season=2024&meeting_name=Italian%20Grand%20Prix"
```

```bash
curl -X POST http://localhost:7077/gold/questions \
  -H "Content-Type: application/json" \
  -d '{"question":"Faça um resumo da temporada de 2024","season":2024,"session_name":"all"}'
```

```bash
curl -X POST http://localhost:7077/train/stint-delta-pace \
  -H "Content-Type: application/json" \
  -d '{"season":2024,"session_name":"Race","target_mode":"stint_start_mean","baseline_laps":3,"constructor":"McLaren"}'
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
curl -X POST http://localhost:7077/import-season/resume \
  -H "Content-Type: application/json" \
  -d '{"resume_job_id":"SEU_JOB_ID","include_llm": true}'
```
Observacao: `/import-season/resume` reutiliza `season` e `session_name` do job anterior e pula meetings com status `ok` ou `skipped`. O job anterior passa para `status=resumed` com `resumed_job_id`. Se encontrar etapa futura, o novo job pausa com `status=waiting` e `next_meeting`.

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

```bash
curl "http://localhost:7077/catalog/gold?include_schema=true"
curl "http://localhost:7077/catalog/gold?season=2025&include_schema=true"
```

```bash
curl "http://localhost:7077/mlflow/runs?limit=10"
```

```bash
curl "http://localhost:7077/minio/objects?prefix=openf1/gold&limit=20"
```

## MLflow (container separado)

O MLflow esta na rede `infra_default` com alias `mlflow`. Por isso o pipeline acessa `http://mlflow:5000`.
Isso ja esta definido em `.env`. Se voce mover o MLflow para outra rede, ajuste o compose e o `.env`.

### Artefatos no MinIO/S3

Se o MLflow estiver configurado com artefatos em MinIO/S3, o container do pipeline precisa das credenciais.
As variaveis abaixo ja estao em `.env` (ajuste conforme seu ambiente):

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `MLFLOW_S3_ENDPOINT_URL`

## Variaveis de ambiente relevantes

- `CONFIG_PATH` (default: `/app/config/config.yaml`)
- `CONFIG_DIR` (default: `/app/config`)
- `DATA_DIR`, `LOG_DIR`, `CHECKPOINT_DIR`, `ARTIFACTS_DIR`
- `CLEANUP_LOCAL_ARTIFACTS` (default: `true` para limpar artifacts locais apos logar no MLflow)
- `SYNC_DATA_LAKE` (default: `true` para sincronizar bronze/silver/gold no MinIO)
- `DOWNLOAD_DATA_LAKE` (default: `true` para baixar bronze/silver/gold do MinIO quando faltarem)
- `CLEANUP_LOCAL_DATA` (default: `true` para limpar bronze/silver/gold locais apos sync)
- `DATA_LAKE_BUCKET`, `DATA_LAKE_PREFIX`, `DATA_LAKE_S3_ENDPOINT`
- `DATA_LAKE_SUBDIRS` (upload, default: bronze,silver,gold)
- `DATA_LAKE_DOWNLOAD_SUBDIRS` (download, default: gold)
- `DATA_LAKE_CREATE_BUCKET` (default: `true` para criar o bucket se nao existir)
- `JOBS_DIR` (default: `{logs_dir}/jobs`)
- `OPENF1_BASE_URL`
- `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT`, `REGISTER_MLFLOW`
- `MLFLOW_CREATE_EXPERIMENT` (default: `true`)
- `MLFLOW_GATEWAY_ENDPOINT` (para LLM)

## Estrutura

- `f1_dataset/src`: codigo modular do pipeline
- `f1_dataset/data`: bronze/silver/gold + logs/checkpoints (temporarios, podem ser limpos apos sync)
- `config/config.yaml`: configuracao do recorte e execucao

No gold, cada volta inclui `meeting_date_start` e temperaturas (`track_temperature`, `air_temperature`) quando disponiveis.
No data lake (MinIO/S3), os datasets bronze/silver/gold ficam no bucket/prefix configurados em `.env`.

## Execucao local (IDE)

O arquivo `F1.OpenF1.DatasetBuilder.py` inicia o job principal usando o config definido em `CONFIG_PATH`.

## MLflow

Se `REGISTER_MLFLOW=true`, o pipeline registra parametros, metricas e artefatos no tracking URI.
Quando `MLFLOW_TRACKING_URI` estiver vazio, o tracking local sera usado.
