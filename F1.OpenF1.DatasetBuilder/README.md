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

O servico da API fica disponivel em `http://localhost:7077`.

## API (FastAPI)

Endpoints:
- `GET /health`
- `POST /driver-profiles`
- `POST /driver-profiles/season`
- `POST /import-season`
- `POST /data-lake/sync`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/logs?lines=200`

Campos principais:
- `/driver-profiles`: `season`, `meeting_key`, `session_name` (Race, Sprint ou all), `include_llm`, `llm_endpoint`.
- `/driver-profiles/season`: `seasons` (lista), `session_names` (lista; vazio = todas), `include_llm`, `llm_endpoint`, `drivers_include`, `drivers_exclude`.
- `/import-season`: `season`, `session_name` (Race ou Sprint), `include_llm`, `llm_endpoint`.
- `/data-lake/sync`: `direction` (upload/download), `subdirs`, `cleanup_local`, `only_if_missing`.

Exemplo de chamada:

```bash
curl -X POST http://localhost:7077/driver-profiles \
  -H "Content-Type: application/json" \
  -d '{"season": 2024, "meeting_key": "1252", "session_name": "Race", "include_llm": false}'
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
