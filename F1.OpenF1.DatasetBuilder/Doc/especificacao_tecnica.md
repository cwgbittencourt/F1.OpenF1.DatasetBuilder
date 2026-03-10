# Especificação Técnica — Pipeline OpenF1 → Dataset Python → MLflow → Consumo ASP.NET/C#

## 1. Objetivo

Construir uma solução evolutiva para:

- coletar dados históricos da API OpenF1;
- transformar os dados em datasets analíticos em Python;
- registrar datasets, execuções e artefatos no MLflow;
- disponibilizar os resultados finais para consumo em uma aplicação ASP.NET/C#.

A solução **não deve nascer amarrada** a um único piloto, corrida ou temporada.  
O recorte inicial da V1 será apenas um caso de uso de validação.

---

## 2. Diretrizes obrigatórias

### 2.1 Arquitetura genérica
A arquitetura deve ser desenhada para suportar naturalmente:

- vários pilotos;
- várias corridas;
- várias temporadas;
- futuramente, outros tipos de sessão além de `Race`.

### 2.2 Recorte inicial separado da arquitetura-alvo
O recorte inicial da primeira entrega será:

- piloto: **Lewis Hamilton**;
- temporada: **2023**;
- corrida: **primeira corrida da temporada**;
- sessão: **Race**.

Esse recorte deve existir apenas como **configuração inicial de execução**, e não como regra fixa da lógica central.

### 2.3 Nada hardcoded
Não fixar no código:

- `meeting_key`;
- `session_key`;
- `driver_number`;
- ids internos específicos;
- piloto único;
- uma única temporada.

Tudo deve ser descoberto dinamicamente a partir da API, usando filtros e parâmetros.

### 2.4 Ingestão incremental e controlada
Ao evoluir para múltiplos pilotos/corridas/temporadas, a solução **não deve baixar tudo de uma única vez**.

A ingestão deve ocorrer em unidades menores, preferencialmente por piloto, com:

- `foreach` ou paralelismo controlado;
- limitação explícita de concorrência;
- proteção contra rate limit;
- retry com exponential backoff;
- checkpoint para retomada;
- persistência incremental a cada unidade concluída.

### 2.5 Performance com segurança
O objetivo é privilegiar performance sem:

- sobrecarregar o servidor;
- disparar requisições simultâneas em excesso;
- atingir rate limit;
- correr risco de bloqueio.

---

## 3. Stack definida

### 3.0 Ambiente oficial de desenvolvimento e execução
O stack Python deste projeto deve seguir obrigatoriamente este padrão:

- desenvolvimento no **Visual Studio 2026**;
- execução oficial em **container Docker**;
- orquestração local por **Docker Compose**;
- persistência de dados, logs, checkpoints e artefatos em **volumes**;
- configuração externa por `.env` e arquivos de configuração.

O ambiente local do sistema operacional não deve ser tratado como ambiente oficial de execução do pipeline.  
O host será usado como ambiente de edição, depuração, versionamento e orquestração, enquanto a execução do pipeline deve ocorrer no container.


### 3.1 Python
Python será responsável por:


- coleta de dados na OpenF1 API;
- transformação e normalização;
- construção do dataset;
- engenharia de atributos;
- persistência analítica;
- envio e rastreio no MLflow.

### 3.2 ASP.NET/C#
ASP.NET/C# será responsável por:

- consumir os resultados finais;
- exibir informações ao usuário;
- servir APIs de consulta;
- dashboards, rankings, comparações e visualizações finais.


### 3.3 Containerização obrigatória
A solução Python deve nascer preparada para rodar dentro de container, com:

- `Dockerfile`;
- `docker-compose.yml`;
- `.dockerignore`;
- `.env`;
- volumes persistentes;
- comando de inicialização explícito;
- suporte a passagem de parâmetros de execução.

Essa containerização não é opcional. Ela faz parte da arquitetura-base do projeto.

### 3.4 Benefícios esperados da abordagem container-first
A abordagem com Visual Studio 2026 + Python + Docker Compose deve garantir:

- reprodutibilidade do ambiente;
- isolamento de dependências;
- facilidade de versionamento;
- menor risco de divergência entre máquinas;
- controle melhor de logs, dados e checkpoints;
- caminho mais seguro para evolução e produção.


### 3.6 Visual Studio 2026 como IDE padrão
O desenvolvimento do stack Python deve considerar explicitamente o uso do **Visual Studio 2026** como IDE principal, incluindo:

- edição de código;
- depuração;
- gerenciamento do projeto;
- integração com containers;
- execução controlada via Docker Compose.

A estrutura do projeto deve ser amigável para abertura e manutenção dentro do Visual Studio 2026.

### 3.5 Regra arquitetural
O projeto deve ser desenhado como **container-first**.

Isso significa:

- não depender da instalação manual de pacotes Python no host como forma principal de execução;
- não pressupor ambiente local “pré-ajustado”;
- tratar o container como ambiente oficial do pipeline;
- garantir que a solução rode de forma consistente por `docker compose up` ou comando equivalente.

---

## 4. Fonte de dados

A fonte primária será a API pública OpenF1:

- Base URL: `https://api.openf1.org/v1`

Principais endpoints previstos:

- `meetings`
- `sessions`
- `drivers`
- `laps`
- `car_data`
- `location`

Evolução futura:

- `weather`
- `stints`
- `pit`
- `position`
- `intervals`
- `race_control`
- `session_result`

---

## 5. Visão macro da arquitetura

```text
Visual Studio 2026
   ->
Código Python
   ->
Container Docker
   ->
Docker Compose
   ->
OpenF1 API
   ->
Camada Bronze
   ->
Camada Silver
   ->
Camada Gold (dataset analítico)
   ->
MLflow
   ->
ASP.NET/C#
```

---

## 6. Arquitetura funcional

## 6.1 Descoberta
Responsável por descobrir dinamicamente:

- meetings por temporada;
- sessões por meeting;
- sessão `Race`;
- pilotos de cada sessão.

## 6.2 Coleta
Responsável por buscar os dados brutos da API.

## 6.3 Normalização
Responsável por:

- flatten de JSON;
- padronização de nomes de colunas;
- conversão de tipos;
- normalização de datas;
- limpeza básica.

## 6.4 Enriquecimento
Responsável por:

- associar telemetria a uma volta;
- associar trajetória a uma volta;
- gerar métricas agregadas;
- preparar dataset final.

## 6.5 Validação
Responsável por:

- checagem de colunas obrigatórias;
- nulos;
- duplicidades;
- consistência mínima;
- quantidade mínima de pontos por volta.

## 6.6 Publicação
Responsável por:

- salvar parquet/csv;
- registrar dataset no MLflow;
- salvar artefatos;
- registrar metadados da execução.

---

## 7. Princípio de escalabilidade

A solução deve ser desenhada para suportar a seguinte evolução:

### Fase 1
- 1 piloto
- 1 corrida
- 1 temporada
- sessão `Race`

### Fase 2
- 1 piloto
- várias corridas
- 1 temporada
- sessão `Race`

### Fase 3
- vários pilotos
- várias corridas
- 1 temporada
- sessão `Race`

### Fase 4
- vários pilotos
- várias corridas
- várias temporadas
- sessão `Race`

### Fase 5
- expansão para outros endpoints e contextos estratégicos
- eventual suporte a outros tipos de sessão

---

## 8. Unidade de processamento

A menor unidade lógica de processamento deve ser:

```text
(season, meeting_key, session_key, driver_number)
```

Essa unidade deve ser tratada como:

- pequena;
- rastreável;
- reprocessável;
- persistível de forma isolada;
- segura para checkpoint.

---

## 9. Estratégia de ingestão incremental

### 9.1 Regra central
Não baixar temporada inteira com todos os pilotos de uma vez.

### 9.2 Estratégia correta
Processar assim:

```text
temporada
  -> meeting
    -> sessão Race
      -> piloto
        -> laps
        -> car_data
        -> location
        -> processa
        -> salva
```

### 9.3 Benefícios
- menor consumo de memória;
- menor risco de falha catastrófica;
- retomada mais simples;
- menor chance de rate limit;
- melhor governança da execução.

---

## 10. Paralelismo controlado

### 10.1 Requisito
Deve ser possível processar pilotos em paralelo, mas com controle configurável.

### 10.2 Regras
- não executar paralelismo irrestrito;
- limitar workers máximos;
- limitar requisições simultâneas;
- implementar backoff em caso de erro temporário;
- reduzir agressividade quando houver sinais de rate limiting.

### 10.3 Estratégia recomendada
Começar com:

- V1: processamento sequencial por piloto;
- V2: paralelismo controlado com baixo grau de concorrência;
- V3: política adaptativa baseada em erros e latência.

### 10.4 Configurações esperadas
Exemplos de parâmetros configuráveis:

- `max_parallel_drivers`
- `max_http_connections`
- `min_request_interval_ms`
- `retry_attempts`
- `retry_backoff_seconds`
- `rate_limit_cooldown_seconds`

---

## 11. Resiliência operacional

A solução deve possuir:

### 11.1 Retry
Retry automático para falhas transitórias.

### 11.2 Exponential backoff
Ao receber erro temporário, aumentar progressivamente o tempo entre tentativas.

### 11.3 Checkpoint
Registrar status por unidade processada:

- pendente;
- em execução;
- concluído;
- falhou.

### 11.4 Retomada
A execução deve poder continuar a partir do último ponto salvo.

### 11.5 Persistência incremental
Após concluir o processamento de um piloto, salvar imediatamente os resultados e liberar memória.

---

## 12. Estrutura de projeto Python

Sugestão de estrutura:

```text
f1_dataset/
  src/
    config/
    clients/
    discovery/
    collectors/
    processors/
    feature_engineering/
    validators/
    publishers/
    orchestration/
    jobs/
  data/
    bronze/
    silver/
    gold/
    checkpoints/
    logs/
  docker/
  tests/
  .env
  .env.example
  .dockerignore
  docker-compose.yml
  Dockerfile
  pyproject.toml
  README.md
```

---

## 13. Responsabilidade de cada módulo

### `config/`
- variáveis e parâmetros do projeto;
- defaults de execução;
- nomes de partições e diretórios.

### `clients/`
- cliente HTTP da OpenF1;
- cliente do MLflow.

### `discovery/`
- descoberta de meetings;
- descoberta de sessões;
- descoberta de pilotos.

### `collectors/`
- coleta dos endpoints brutos.

### `processors/`
- limpeza e transformação básica;
- parsing de timestamps;
- padronização de colunas.

### `feature_engineering/`
- agregações por volta;
- métricas temporais;
- métricas espaciais.

### `validators/`
- regras de qualidade do dataset.

### `publishers/`
- escrita em parquet/csv;
- registro no MLflow;
- exportação de artefatos.

### `orchestration/`
- fila de execução;
- controle de concorrência;
- checkpoints;
- políticas de retry.

### `jobs/`
- entry points do pipeline.


### `docker/`
- scripts auxiliares de inicialização;
- configurações de ambiente;
- arquivos de apoio ao container, quando necessário.

### Arquivos de infraestrutura obrigatórios
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `.env.example`

Esses arquivos devem fazer parte da primeira versão do projeto.

---

## 14. Jobs esperados

A solução não deve nascer com jobs amarrados a um único piloto.

Exemplo correto:

- `build_openf1_dataset.py`
- `process_meeting.py`
- `consolidate_gold_dataset.py`

Evitar nomes presos a um recorte único, como:

- `build_dataset_hamilton_2023.py`

Esse recorte deve existir apenas como configuração.

---

## 15. Entrada por configuração

A execução deve aceitar configuração externa, por arquivo ou parâmetros.

### Entradas mínimas
- temporadas;
- tipo de sessão;
- filtro de meetings;
- filtro de pilotos;
- modo de execução;
- limite de concorrência;
- diretório de saída;
- opção de registro no MLflow.

- parâmetros de container e volumes;
- flags para execução local controlada via compose.

### Exemplo conceitual de configuração

```yaml
seasons:
  - 2023

session_name: Race

drivers:
  include:
    - Lewis Hamilton

meetings:
  mode: first_of_season

execution:
  max_parallel_drivers: 1
  retry_attempts: 4
  retry_backoff_seconds: 2
  min_request_interval_ms: 300

output:
  format:
    - parquet
    - csv
  register_mlflow: true
```


### Exemplo conceitual de infraestrutura com Compose

```yaml
services:
  openf1-dataset:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: openf1-dataset
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./checkpoints:/app/checkpoints
    command: >
      python -m src.jobs.build_openf1_dataset
```

### Regras para volumes
Devem existir volumes ou bind mounts para persistir fora do container, no mínimo:

- `data/`
- `logs/`
- `checkpoints/`

Opcionalmente, também:

- `artifacts/`
- `reports/`

Sem isso, o projeto corre risco de perder dados e rastros operacionais ao recriar o container.

---

## 16. Descoberta dinâmica dos dados

### Ordem obrigatória da descoberta
1. buscar meetings da temporada;
2. selecionar o meeting alvo;
3. buscar sessões do meeting;
4. selecionar a sessão `Race`;
5. buscar pilotos da sessão;
6. selecionar pilotos conforme filtro;
7. coletar endpoints por piloto.

### Observação
O código não deve depender de ids previamente conhecidos.

---

## 17. Endpoints mínimos da V1

### 17.1 `meetings`
Para localizar os GPs da temporada.

### 17.2 `sessions`
Para localizar a sessão `Race` do meeting.

### 17.3 `drivers`
Para obter os pilotos e o `driver_number`.

### 17.4 `laps`
Base da volta e do alvo principal.

### 17.5 `car_data`
Telemetria temporal do carro.

### 17.6 `location`
Trajetória espacial.

---

## 18. Camadas de dados

### 18.1 Bronze
Dados crus da API, praticamente sem transformação.

Exemplos:
- respostas originais;
- datasets achatados minimamente;
- persistência para replay e auditoria.

### 18.2 Silver
Dados limpos e normalizados.

Exemplos:
- datas convertidas;
- nomes padronizados;
- colunas consistentes;
- tipos numéricos tratados.

### 18.3 Gold
Dataset analítico final, pronto para ML e consumo.

Exemplos:
- uma linha por volta;
- features agregadas;
- partições por temporada/corrida/sessão/piloto.

---

## 19. Particionamento de saída

A saída deve ser particionada para evitar arquivos gigantes e facilitar reprocessamento.

Estrutura recomendada:

```text
gold/
  season=2023/
    meeting_key=XXXX/
      session_key=YYYY/
        driver_number=44/
          dataset.parquet
```

Essa estrutura permite:

- consolidar por corrida;
- consolidar por temporada;
- consolidar por piloto;
- reprocessar apenas uma unidade falha.

---

## 20. Modelo de dataset inicial

A primeira versão do dataset deve ser em **nível de volta**.

### Chaves mínimas
- `season`
- `meeting_key`
- `meeting_name`
- `session_key`
- `session_name`
- `driver_number`
- `driver_name`
- `team_name`
- `lap_number`

### Alvo inicial
- `lap_duration`

### Features iniciais sugeridas
- `avg_speed`
- `max_speed`
- `min_speed`
- `speed_std`
- `avg_rpm`
- `max_rpm`
- `rpm_std`
- `avg_throttle`
- `throttle_std`
- `full_throttle_pct`
- `brake_pct`
- `brake_events`
- `hard_brake_events`
- `drs_pct`
- `gear_changes`
- `distance_traveled`
- `trajectory_length`
- `trajectory_variation`

### Contexto adicional quando disponível
- tempos de setor;
- indicadores de pit in/out;
- flags de completude da volta.

---

## 21. Relação entre os endpoints

### `laps`
Define a base da volta e o alvo principal.

### `car_data`
Mostra como o carro foi conduzido ao longo do tempo.

### `location`
Mostra onde o carro estava na pista.

### Síntese
- `laps` = resultado da volta
- `car_data` = comportamento do carro
- `location` = trajetória

---

## 22. Lógica de associação por volta

A associação entre os dados deve seguir esta lógica:

1. `laps` define a estrutura das voltas;
2. `car_data` é associado à volta por janela temporal;
3. `location` é associado à volta por janela temporal;
4. os pontos temporais são agregados em métricas por volta;
5. a linha final representa a volta como um todo.

---

## 23. Engenharia de atributos inicial

### 23.1 Telemetria
- médias;
- máximos;
- mínimos;
- desvios padrão;
- percentuais de uso;
- eventos de frenagem;
- trocas de marcha.

### 23.2 Trajetória
- distância percorrida;
- comprimento do traçado;
- variação de trajetória;
- estabilidade da linha.

### 23.3 Qualidade
- quantidade de amostras por volta;
- flag de volta válida;
- flag de telemetria suficiente.

---

## 24. Caso inicial de execução da V1

O primeiro cenário usado para validar o pipeline será:

- temporada: `2023`
- meeting: `primeira corrida da temporada`
- sessão: `Race`
- piloto: `Lewis Hamilton`

Esse cenário deve ser configurado externamente.

A arquitetura deve continuar preparada para crescer sem refatoração estrutural.

---

## 25. Estratégia de evolução

### 25.1 Primeiro passo
Validar o pipeline de ponta a ponta com um único piloto.

### 25.2 Segundo passo
Repetir para várias corridas do mesmo piloto.

### 25.3 Terceiro passo
Adicionar múltiplos pilotos.

### 25.4 Quarto passo
Adicionar múltiplas temporadas.

### 25.5 Quinto passo
Adicionar contexto extra:
- clima;
- stint;
- pit stop;
- posição;
- race control.

---

## 26. Registro no MLflow

### 26.1 O que registrar
Mesmo antes do treino de modelos, registrar:

#### Parâmetros
- temporadas processadas;
- meeting_key;
- session_key;
- pilotos processados;
- total de voltas;
- total de registros por endpoint;
- versão do pipeline;
- configuração de execução.

#### Métricas de qualidade
- linhas geradas;
- colunas geradas;
- percentual de nulos;
- voltas válidas;
- voltas descartadas;
- tempo total de execução;
- tempo por piloto.

#### Artefatos
- parquet final;
- csv final;
- dicionário de dados;
- relatório de qualidade;
- logs resumidos.

### 26.2 Finalidade
- rastreabilidade;
- governança;
- reprodutibilidade;
- comparação entre versões do pipeline.

---

## 27. Contrato esperado para o ASP.NET/C#

O ASP.NET/C# não será responsável pela montagem do dataset.

Ele deve consumir resultados já processados, por exemplo:

- arquivos consolidados;
- tabelas persistidas em banco;
- metadados registrados no MLflow;
- métricas e rankings pré-calculados.

### Casos de uso do ASP.NET/C#
- exibir ranking de voltas;
- comparar pilotos;
- mostrar evolução por corrida;
- dashboards;
- APIs para consulta;
- páginas analíticas.

---

## 28. Persistência recomendada

### Inicial
- Parquet como formato principal;
- CSV como apoio para inspeção rápida.

- persistência fora do container por volumes ou bind mounts.

### Evolução
- persistir versão consolidada em banco relacional ou lakehouse;
- manter o particionamento analítico em parquet.

---

## 29. Requisitos não funcionais

### 29.1 Performance
- ingestão incremental;
- baixo consumo de memória;
- processamento por unidade pequena.

### 29.2 Escalabilidade
- arquitetura preparada para múltiplos pilotos/corridas/temporadas.

### 29.3 Resiliência
- retry;
- backoff;
- checkpoint;
- retomada.

### 29.4 Observabilidade
- logs estruturados;
- métricas por etapa;
- rastreio de falhas por piloto.

### 29.5 Reprodutibilidade
- execução por configuração;
- versões do dataset;
- registro no MLflow.

### 29.6 Manutenibilidade
- módulos bem separados;
- baixo acoplamento;
- nada hardcoded na lógica central.


### 29.7 Portabilidade operacional
- execução consistente via Docker Compose;
- independência do ambiente Python instalado no host;
- facilidade de subida em outra máquina sem reconfiguração manual extensa.

---

## 30. Regras obrigatórias de implementação

1. Não fixar ids da OpenF1 no código.
2. Não prender a solução a um único piloto.
3. Não prender a solução a uma única temporada.
4. Não baixar tudo de uma vez ao escalar.
5. Processar incrementalmente por piloto.
6. Permitir paralelismo, mas sempre controlado.
7. Implementar retry com exponential backoff.
8. Implementar checkpoint e retomada.
9. Persistir resultados incrementalmente.
10. Particionar a saída por temporada/corrida/sessão/piloto.
11. Registrar datasets e execução no MLflow.
12. Manter o ASP.NET/C# desacoplado da engenharia de dados.
13. Implementar Dockerfile e docker-compose.yml desde a V1.
14. Persistir dados, logs e checkpoints fora do container.
15. Tratar o container como ambiente oficial de execução.
16. Garantir compatibilidade com desenvolvimento no Visual Studio 2026.

---

## 31. Resultado esperado da V1

Ao final da primeira versão, a solução deve ser capaz de:

- descobrir dinamicamente a primeira corrida de 2023;
- localizar a sessão `Race`;
- localizar Lewis Hamilton nessa sessão;
- coletar `laps`, `car_data` e `location`;
- gerar um dataset por volta;
- salvar o dataset em parquet/csv;
- registrar a execução no MLflow;
- deixar o resultado pronto para consumo futuro pelo ASP.NET/C#.
- subir a execução oficial por container com Docker Compose.

---

## 32. Resultado esperado da arquitetura-alvo

Ao evoluir, a solução deve ser capaz de:

- processar múltiplos pilotos;
- processar múltiplas corridas;
- processar múltiplas temporadas;
- controlar concorrência;
- respeitar rate limits;
- retomar execuções interrompidas;
- consolidar datasets progressivamente;
- alimentar aplicações analíticas em ASP.NET/C#.

---

## 33. Prompt operacional resumido para geração da solução

Construir um projeto Python de ingestão e preparação de datasets a partir da API OpenF1, com arquitetura modular, genérica e escalável. O projeto deve descobrir dinamicamente meetings, sessões e pilotos; coletar `laps`, `car_data` e `location`; transformar os dados em dataset analítico por volta; persistir em camadas bronze, silver e gold; registrar parâmetros, métricas e artefatos no MLflow; e preparar a saída para consumo por uma aplicação ASP.NET/C#. A arquitetura não pode ficar presa a um único piloto, corrida ou temporada. O recorte inicial da V1 será Lewis Hamilton na primeira corrida de 2023, sessão Race, mas isso deve existir apenas como configuração. Ao escalar, o pipeline deve processar incrementalmente por piloto, com paralelismo controlado, retry com exponential backoff, checkpoint, retomada e proteção contra rate limit.
