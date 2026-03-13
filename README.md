F1 OpenF1 Dataset Builder - Fases, Metricas e API
Este README descreve as fases executadas pela solucao (pipeline de dados, modelagem, relatorios e API) e as metricas aplicadas em cada caso, com o motivo de uso.

**Escopo**
Solucao principal em `F1.OpenF1.DatasetBuilder/` com pipeline OpenF1 -> bronze/silver/gold, publicacao no MLflow, jobs de modelagem/analytics e uma API FastAPI para orquestrar relatorios e importacao de temporadas.

**Fases Da Solucao**
1. Configuracao e orquestracao
Descricao: leitura de configuracao, controle de paralelismo, rate limit e checkpoints. Entrypoint em `F1.OpenF1.DatasetBuilder/f1_dataset/src/orchestration/runner.py`.
Metricas: nenhuma metrica numerica registrada aqui.
Por que: esta fase controla execucao e resiliencia; as metricas relevantes sao calculadas nas fases de dados e modelagem.

2. Descoberta dinamica
Descricao: descoberta de meetings, sessions e drivers via API OpenF1.
Metricas: nenhuma.
Por que: etapa deterministica para selecionar o recorte de dados sem hardcode.

3. Coleta / ingestao
Descricao: coleta dos endpoints `laps`, `car_data`, `location`, `stints`, `weather`.
Metricas: nenhuma metrica registrada no MLflow.
Por que: o foco e capturar dados crus para auditoria e replay; a qualidade e medida posteriormente.

4. Camada Bronze
Descricao: persistencia raw por unidade (season/meeting/session/driver).
Metricas: nenhuma.
Por que: bronze e camada de backup/forense, sem transformacoes.

5. Normalizacao (Silver)
Descricao: flatten de JSON, padronizacao de colunas e tipos.
Metricas: nenhuma.
Por que: a etapa prepara os dados para features; a avaliacao de qualidade ocorre no gold.

6. Engenharia de atributos e Gold
Descricao: agregacao por volta e enriquecimento com telemetria, trajetoria, dados de stint e clima.
Metricas de telemetria (por volta): `avg_speed`, `max_speed`, `min_speed`, `speed_std`, `avg_rpm`, `max_rpm`, `min_rpm`, `rpm_std`, `avg_throttle`, `max_throttle`, `min_throttle`, `throttle_std`.
Por que: capturam ritmo medio e variabilidade de performance durante a volta.
Metricas de controle do carro: `full_throttle_pct`, `brake_pct`, `brake_events`, `hard_brake_events`, `drs_pct`, `gear_changes`.
Por que: representam agressividade, uso de acelerador/freio/DRS e estilo de pilotagem.
Metricas de trajetoria: `distance_traveled`, `trajectory_length`, `trajectory_variation`.
Por que: quantificam linha percorrida e estabilidade espacial.
Metricas de cobertura de dados: `telemetry_points`, `trajectory_points`, `has_telemetry`, `has_trajectory`.
Por que: indicam se a volta tem dados suficientes para analise confiavel.
Contexto de stint: `stint_number`, `compound`, `stint_lap_start`, `stint_lap_end`, `tyre_age_at_start`, `tyre_age_at_lap`.
Por que: necessario para estudos de degradacao e delta de pace.
Contexto de corrida: `meeting_date_start`.
Por que: permite recortes temporais por data da corrida.
Temperatura por volta: `track_temperature`, `air_temperature`.
Por que: correlaciona desempenho com condicoes de pista.

7. Validacao de qualidade do Gold
Descricao: checagem de colunas obrigatorias e completude.
Metricas: `rows`, `null_pct`, `valid_laps`, `discarded_laps`.
Por que: medem volume gerado, percentual de nulos e quantas voltas sao aproveitaveis.

8. Publicacao e rastreio (MLflow)
Descricao: publicacao de artefatos e log de metricas/params.
Metricas: as metricas de qualidade do gold sao logadas no MLflow, junto com parametros de execucao.
Por que: garante rastreabilidade e comparacao entre execucoes.

9. Consolidacao Gold
Descricao: unifica datasets gold em um unico arquivo consolidado.
Metricas: nenhuma.
Por que: etapa de uniao e distribuicao, sem avaliacao estatistica.

10. Modelagem e experimentos (jobs em `F1.OpenF1.DatasetBuilder/f1_dataset/src/jobs`)
- `train_lap_time_regression`: `mae`, `rmse`, `r2`, `mape`. Motivo: medir erro absoluto, penalizar erros grandes, variancia explicada e erro relativo.
- `train_lap_time_ranking`: `mae`, `rmse`, `r2`, `mape`, `rank_spearman_mean`, `rank_ndcg_mean`, `rank_meeting_count`, `rank_driver_mean`. Motivo: avaliar erro de tempo e qualidade do ranking de pilotos por corrida.
- `train_relative_position`: `mae`, `rmse`, `r2`, `mape`, `rank_spearman_mean`. Motivo: avaliar previsao do rank_percentile e correlacao de ordenacao.
- `train_stint_delta_pace`: `mae`, `rmse`, `r2`, `mape`. Motivo: medir erro na previsao do delta de ritmo entre stints.
- `train_tyre_degradation`: `mae`, `rmse`, `r2`, `mape`. Motivo: medir erro na previsao da degradacao ao longo do stint.
- `train_lap_quality_classifier`: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`. Motivo: medir acuracia global, equilibrio entre falsos positivos/negativos e capacidade discriminativa.
- `train_lap_anomaly`: `rows`, `anomaly_count`, `anomaly_rate`, `score_min`, `score_max`, `score_mean`. Motivo: quantificar incidencias e distribuicao dos scores de anomalia.
- `train_driver_style_clustering`: `clusters`, `rows`, `silhouette`, `davies_bouldin`. Motivo: avaliar coesao e separacao dos clusters de estilo.
- `train_circuit_segmentation`: `clusters`, `rows`, `silhouette`, `davies_bouldin`. Motivo: avaliar segmentacao de circuitos por comportamento agregado.

11. Relatorios e rankings de pilotos
Descricao: consolidacao de metricas por piloto e geracao de rankings e textos.
Metricas base por piloto (driver_profiles_report): `laps_total`, `meetings_total`, `lap_mean`, `lap_std`, `finish_rate`, `lap_completion_mean`, `dnf_rate`, `points_total`, `points_race`, `points_sprint`, `races_count`, `sprints_count`, `lap_quality_good_rate`, `lap_quality_bad_rate`, `anomaly_rate`, `degradation_mean`, `degradation_p95`, `degradation_slope`, `stint_performance_delta_mean`, `stint_performance_delta_slope`, `tyre_wear_slope`, `delta_pace_mean`, `delta_pace_median`, `delta_pace_std`, `delta_pace_count`, `rank_percentile_mean`, `rank_percentile_median`, `meeting_lap_mean_avg`, `lap_mean_delta_to_meeting_mean`, `lap_mean_z_to_meeting_mean`, `pit_out_rate`, `driver_style_cluster`, `dominant_circuit_cluster`, `dominant_circuit_cluster_pct`, `meeting_date_start`, `dominant_circuit_speed_class`, `dominant_circuit_speed_class_pct`, `track_temperature_mean`, `track_temperature_min`, `track_temperature_max`, `track_temperature_std`, `air_temperature_mean`, `air_temperature_min`, `air_temperature_max`, `air_temperature_std`.
Por que: cobrem ritmo, consistencia, qualidade, anomalias, desgaste, estabilidade entre stints, posicionamento relativo e confiabilidade.
Observacao: `stint_performance_delta_*` sao aliases mais indicativos das metricas de performance ao longo do stint (mesmo calculo de `degradation_*`). `tyre_wear_slope` isola o desgaste com base em `tyre_age_at_lap`, excluindo a 1a volta de cada stint.
Rankings (driver_profiles_rankings/overall_ranking): usam as metricas acima com direcao (maior e melhor ou menor e melhor) para gerar ranking por criterio e score composto.
Por que: transformar metricas isoladas em comparativos por piloto.
Relatorio textual (driver_profiles_text_report / generate_driver_performance_llm): converte percentis e metricas em texto, mantendo as mesmas fontes numericas.
Por que: facilitar consumo humano e gerar narrativa padronizada.
O LLM explicita condicoes de pista e separa performance no stint de desgaste do pneu.

12. Comparacao de experimentos
Descricao: consolida comparativos de runs no MLflow (lap time e experimentos extendidos).
Metricas: reutiliza as metricas ja logadas nos runs do MLflow.
Por que: facilitar escolha do melhor modelo sem recalcular.

13. API de orquestracao (FastAPI)
Descricao: endpoints para gerar perfis de pilotos por corrida, importar temporadas em background e consultar status/logs.
Metricas: nao cria novas metricas; reaproveita as mesmas metricas dos relatorios e do MLflow.
Por que: a API e camada de automacao, nao de avaliacao estatistica.

**Endpoints Atuais Da API**
- `GET /health`: healthcheck simples da API, confirma que o serviço está respondendo. Obs: não valida dependências externas (MLflow, MinIO/S3, OpenF1).
- `GET /health/dependencies`: verifica dependências externas (MLflow, MinIO/S3 e OpenF1) e retorna status por dependência. Obs: OpenF1 pode ficar indisponível para não-assinantes em horários de eventos.
- `GET /gold/meetings`: lista meetings disponíveis no gold consolidado, com `meeting_key`, `meeting_name` e sessões; aceita filtros por `season` e `session_name`. Obs: requer `f1_dataset/data/gold/consolidated.parquet` local; se não existir retorna 404.
- `POST /gold/questions`: responde perguntas em linguagem natural usando o gold consolidado; aplica filtros e retorna `answer` (pt-BR) e `summary` estatístico do recorte. Obs: requer gold consolidado local e endpoint LLM acessível via `MLFLOW_GATEWAY_ENDPOINT`; falhas no LLM retornam 502.
- `POST /train/stint-delta-pace`: dispara treino assíncrono do modelo de delta de ritmo entre stints; retorna `job_id` para acompanhamento. Obs: job roda em background e grava logs em `f1_dataset/data/logs/jobs`.
- `POST /driver-profiles`: gera relatórios e rankings por meeting específico; garante gold, gera artifacts e retorna URIs no MLflow. Obs: pode executar pipeline se faltarem dados e depende de MLflow disponível.
- `POST /driver-profiles/season`: gera relatórios por temporada e múltiplas sessões em lote; retorna artifacts e sumários por temporada. Obs: processamento pode demorar em temporadas múltiplas.
- `POST /import-season`: inicia job assíncrono para importar e processar uma temporada inteira (com opção de LLM); retorna `job_id`. Obs: acompanha via `/jobs` e `/jobs/{job_id}/logs`.
- `POST /import-season/resume`: retoma um job de importação anterior usando `resume_job_id`, pulando meetings já concluídos. Obs: requer `resume_job_id` válido de importação.
- `POST /data-lake/sync`: sincroniza bronze/silver/gold com o MinIO (upload/download) e controla limpeza local. Obs: requer credenciais/endpoint S3/MinIO configurados; `cleanup_local=true` apaga dados locais após sync.
- `GET /jobs/{job_id}`: consulta status atual do job assíncrono (ex.: importação ou treino). Obs: válido para jobs criados por `/import-season` e `/train/stint-delta-pace`.
- `GET /jobs/{job_id}/logs?lines=200`: retorna as últimas linhas do log do job assíncrono. Obs: `lines` controla a quantidade de linhas retornadas.

**Exemplos de chamadas**

```bash
curl http://localhost:7077/health
```

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

```bash
curl "http://localhost:7077/gold/meetings?season=2024&session_name=Race"
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
Observacao: `/import-season/resume` reutiliza `season` e `session_name` do job anterior e pula meetings com status `ok` ou `skipped`.

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

**Artefatos e trilhas**
- Dados locais: `F1.OpenF1.DatasetBuilder/f1_dataset/data/bronze`, `silver`, `gold` (temporarios, podem ser limpos apos sync).
- Data lake (MinIO/S3): bucket/prefix configurados por `DATA_LAKE_BUCKET` + `DATA_LAKE_PREFIX`.
- Artefatos e relatorios: `F1.OpenF1.DatasetBuilder/f1_dataset/data/artifacts` (temporario; publicado no MLflow).
- Logs e checkpoints: `F1.OpenF1.DatasetBuilder/f1_dataset/data/logs`, `checkpoints` e `logs/jobs`.

Se quiser detalhes de execucao, veja o README da aplicacao em `F1.OpenF1.DatasetBuilder/README.md`.
