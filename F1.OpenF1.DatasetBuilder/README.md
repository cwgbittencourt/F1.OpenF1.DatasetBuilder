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
- `POST /import-season`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/logs?lines=200`

Campos principais:
- `/driver-profiles`: `season`, `meeting_key`, `session_name` (Race, Sprint ou all), `include_llm`, `llm_endpoint`.
- `/import-season`: `season`, `session_name` (Race ou Sprint), `include_llm`, `llm_endpoint`.

Exemplo de chamada:

```bash
curl -X POST http://localhost:7077/driver-profiles \
  -H "Content-Type: application/json" \
  -d '{"season": 2024, "meeting_key": "1252", "session_name": "Race", "include_llm": false}'
```

Os relatorios por piloto incluem classificacao dominante de circuito por velocidade (`dominant_circuit_speed_class`).

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
- `JOBS_DIR` (default: `{logs_dir}/jobs`)
- `OPENF1_BASE_URL`
- `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT`, `REGISTER_MLFLOW`
- `MLFLOW_GATEWAY_ENDPOINT` (para LLM)

## Estrutura

- `f1_dataset/src`: codigo modular do pipeline
- `f1_dataset/data`: bronze/silver/gold + logs/checkpoints
- `config/config.yaml`: configuracao do recorte e execucao

No gold, cada volta inclui `meeting_date_start` e temperaturas (`track_temperature`, `air_temperature`) quando disponiveis.

## Execucao local (IDE)

O arquivo `F1.OpenF1.DatasetBuilder.py` inicia o job principal usando o config definido em `CONFIG_PATH`.

## MLflow

Se `REGISTER_MLFLOW=true`, o pipeline registra parametros, metricas e artefatos no tracking URI.
Quando `MLFLOW_TRACKING_URI` estiver vazio, o tracking local sera usado.
