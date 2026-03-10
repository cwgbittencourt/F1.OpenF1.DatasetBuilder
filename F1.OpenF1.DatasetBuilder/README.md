# OpenF1 Dataset Builder

Pipeline container-first para ingestao da API OpenF1, construcao de datasets e publicacao no MLflow.

## Uso rapido (Docker Compose)

1. Ajuste `.env` (ja criado) conforme necessario.
2. Ajuste a configuracao em `config/config.yaml`.
3. Execute:

```bash
docker compose up --build
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

## Estrutura

- `f1_dataset/src`: codigo modular do pipeline
- `f1_dataset/data`: bronze/silver/gold + logs/checkpoints
- `config/config.yaml`: configuracao do recorte e execucao

## Execucao local (IDE)

O arquivo `F1.OpenF1.DatasetBuilder.py` inicia o job principal usando o config definido em `CONFIG_PATH`.

## MLflow

Se `REGISTER_MLFLOW=true`, o pipeline registra parametros, metricas e artefatos no tracking URI.
Quando `MLFLOW_TRACKING_URI` estiver vazio, o tracking local sera usado.
