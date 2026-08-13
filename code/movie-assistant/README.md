# Movie Assistant

![Screenshot 1](/screenshots/print1.png?raw=true "Screenshot 1")

A Retrieval-Augmented Generation (RAG) movie assistant that answers questions about movies released between 2016 and 2026.

The application uses a movie knowledge base, vector search, an LLM, and a Streamlit interface. It also includes retrieval and LLM evaluation, user feedback, and monitoring with Prometheus, Grafana, and Grafana Tempo.

> **Author's note:** This project received significant assistance from ChatGPT during development, particularly for evaluation, monitoring, testing and this documentation. Vector Search was the chosen method and explanation is available later on this document.

## Dataset

The project uses the **Wikipedia Movies Dataset 2016–2026**:

https://www.kaggle.com/datasets/lakshyaupadhyaya/wikipedia-movies-dataset-2016-2026/

The dataset contains movie information such as:

* title
* description
* director
* writers
* producers
* cast
* release date
* country
* language

The processed data used by the application will also be uploaded to the repository. Therefore, users do **not** need to rebuild the knowledge base or embeddings to reproduce the project.

## How it works

The application follows this flow:

```text
User question
     ↓
Streamlit UI
     ↓
Vector search
     ↓
Top movie documents
     ↓
Context construction
     ↓
LLM
     ↓
Answer
     ↓
User feedback
     ↓
Prometheus / Grafana
```

The final RAG uses vector search because it performed best during retrieval evaluation.

## Project structure

```text
.
├── src/
│   ├── app.py
│   ├── rag.py
│   ├── search.py
│   ├── evaluation.py
│   ├── monitoring.py
│   └── ...
│
├── data/
│   └── processed/
│       ├── documents.json
│       └── embeddings.npy
│
├── grafana/
│   ├── dashboards/
│   │   └── ...
│   └── provisioning/
│       └── ...
│
├── prometheus.yml
├── tempo.yaml
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

### Main files

**`app.py`**

Streamlit application. Loads the movie data and embeddings, accepts user questions, displays the RAG response, shows retrieved movies, and collects useful/not useful feedback.

**`rag.py`**

Contains the main RAG logic:

* context construction
* prompt construction
* LLM call
* RAG tracing
* RAG metrics
* user-facing answer generation

**`search.py`**

Contains the retrieval functionality:

* document loading
* text index creation
* embedding model loading
* vector search
* text search
* hybrid search

**`evaluation.py`**

Evaluates the retrieval methods and LLM prompts using predefined test questions.

**`monitoring.py`**

Defines the OpenTelemetry tracing and Prometheus metrics used by the application.

## Requirements

* Python 3.12+
* [uv](https://docs.astral.sh/uv/)
* Docker
* Docker Compose
* OpenAI API key

The project uses **uv** for dependency management. `pip install` is not required.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

Install the project dependencies with uv:

```bash
uv sync
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5.4-mini
```

Do not commit `.env` or other files containing secrets.

## Data

If the processed dataset and embeddings are already present in the repository, no ingestion step is required.

The application expects the processed files used by `search.py`.

<b>If you want to regenerate them, remove the existing processed files and run the ingestion process provided by the project.</b>

This is useful when:

* using a different version of the dataset;
* changing the embedding model;
* rebuilding the knowledge base.

## Running the application

Build and start the complete application and monitoring stack with:

```bash
docker compose up -d --build
```

The Docker build installs the application dependencies and prepares the Streamlit application inside the image.

> **Note:** The Docker image build can take approximately **5–8 minutes**, mainly because some of the Python dependencies are large. The application is only started after the image has been successfully built.

Once the containers are running, the application can be accessed at:

```text
http://localhost:8501
```
![Screenshot 2](/screenshots/print2.png?raw=true "Screenshot 2")

The monitoring services are available at:

```text
Grafana:     http://localhost:3000
Prometheus:  http://localhost:9090
Tempo:       http://localhost:3200
```

Example questions:

```text
Who directed Oppenheimer?

Who stars in Parasite?

What is Blade Runner 2049 about?

What language is Parasite in?

Tell me about Dune: Part Two.
```

The application returns an answer based on the retrieved movie documents.

The retrieved movie titles can also be inspected through the interface.

## User feedback

Each RAG response can be classified by the user as:

* 🟢 Useful
* 🔴 Not useful

The feedback is recorded as a Prometheus metric and displayed in Grafana.

This provides a simple way to monitor how useful the generated answers are from the user's perspective.

## Evaluation

The project evaluates both retrieval and final LLM responses.

### Retrieval evaluation

Three retrieval approaches were compared:

* Text search
* Vector search
* Hybrid search

The evaluation uses **Hit Rate@5** and **MRR@5**.

Results:

| Method        | Hit Rate@5 |    MRR@5 |
| ------------- | ---------: | -------: |
| Text search   |       0.80 |     0.49 |
| Vector search |   **0.93** | **0.90** |
| Hybrid search |       0.93 |     0.71 |

Vector search was selected for the final RAG because it achieved the highest MRR while matching the best Hit Rate.

Hybrid search was also evaluated, but did not improve the ranking quality over vector search.

### LLM evaluation

Two prompt approaches were evaluated using a separate test set.

| Prompt   | Accuracy |
| -------- | -------: |
| Prompt A |     0.60 |
| Prompt B | **0.80** |

Prompt B was selected for the final application.

The selected prompt explicitly instructs the model to:

* use only the retrieved context;
* avoid unsupported information;
* say that it does not know when the context does not contain the answer;
* keep answers concise and factual.

## Monitoring

The project uses:

* OpenTelemetry for application tracing;
* Grafana Tempo for traces;
* Prometheus for metrics;
* Grafana for dashboards.

The monitoring configuration files are located in the project root:

* `prometheus.yml` — Prometheus configuration and application scrape target.
* `tempo.yaml` — Grafana Tempo configuration.
* `docker-compose.yml` — starts the application and monitoring services.

Grafana dashboards are provisioned automatically from the project files. This means the dashboard is recreated automatically when the Grafana container is recreated.
![Screenshot 3](/screenshots/print3.png?raw=true "Screenshot 3")

The application exposes Prometheus metrics and OpenTelemetry traces while running.

### Monitoring architecture

```text
Movie Assistant
      │
      ├── OpenTelemetry ──→ Tempo
      │
      └── Prometheus metrics
                  │
                  ↓
               Prometheus
                  │
                  ↓
                Grafana
```

The monitoring services are provided through Docker Compose.

## Running monitoring

The monitoring stack is started together with the application:

```bash
docker compose up -d --build
```

This starts the application and monitoring dependencies defined in the project's Docker Compose configuration.

Prometheus collects application metrics and Tempo receives application traces while the Movie Assistant is running.

A pre-built dashboard named "Basic Monitoring Dashboard" can be found in Grafana under the "Movie Assistant" folder.
![Screenshot 4](/screenshots/print4.png?raw=true "Screenshot 4")

## Reproducing the evaluation

Run:

```bash
uv run python src/evaluation.py
```

The script evaluates:

1. text search;
2. vector search;
3. hybrid search;
4. Hit Rate@5;
5. MRR@5;
6. Prompt A;
7. Prompt B.

The evaluation requires the processed documents, embeddings, and a valid OpenAI API key.

## Limitations

The current system has several limitations:

* Retrieval is limited to the information available in the dataset.
* The LLM should not be expected to answer questions that are not supported by the retrieved context.
* The current retrieval evaluation uses a relatively small manually defined test set.
* LLM evaluation also uses a manually defined test set.
* User feedback is subjective and does not represent a formal correctness evaluation.
* The system currently uses vector search in the final RAG even though hybrid search was evaluated.
* There is currently no document re-ranking stage.
* There is currently no explicit user query rewriting stage.
* Movie information depends on the quality and completeness of the source dataset.
* The application is not intended to replace a comprehensive movie database.

## Best practices for using the assistant

Questions work best when they are specific and related to information available in the dataset.

Good examples:

```text
Who directed Oppenheimer?

Who stars in Parasite?

When was Dune: Part Two released?

What country is Parasite from?

What is Everything Everywhere All at Once about?
```

Less useful questions include:

```text
What is the best movie ever?

Should I watch this movie?

What movie should I watch tonight?
```

These questions require subjective or external information that is not necessarily available in the knowledge base.

## Best practices implemented

The project currently includes:

* vector search;
* text search evaluation;
* hybrid search evaluation;
* retrieval ranking evaluation using MRR;
* multiple LLM prompt evaluation;
* user feedback;
* application monitoring;
* distributed tracing;
* metrics dashboards;
* automatic Grafana dashboard provisioning.

Document re-ranking and explicit query rewriting are not currently implemented.

## Reproducibility

Configure the `.env` file:

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5.4-mini
```

Ensure the processed dataset and embeddings are available.

Build and start the complete application:

```bash
docker compose up -d --build
```

The Streamlit application is included in the Docker image and starts automatically with the container.

The initial Docker image build may take approximately **5–8 minutes** because of the size of some Python dependencies.

The project also provisions the Grafana datasources and dashboard automatically.

Run the evaluation when required:

```bash
uv run python src/evaluation.py
```

The project dependencies and their versions are defined in `pyproject.toml` and the uv lock file.

## License

This project was developed as part of the LLM Zoomcamp final project.
