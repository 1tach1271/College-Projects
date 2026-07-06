# GPU-Accelerated Financial Market Regime & Risk Intelligence Engine

This project implements a full end-to-end, high-performance financial market intelligence system. Built upon NVIDIA's **RAPIDS** ecosystem (`cuDF`, `cuML`, `cuGraph`) and **CuPy**, it processes over 2.3 million trading records from the JPX Tokyo Stock Exchange dataset to detect market regimes, surface volatility bursts, construct sectoral correlation graphs, and evaluate systemic risk—all meticulously optimized for execution on consumer-level GPUs (specifically constrained to 6 GB VRAM on an RTX 4050).

---

## 📂 System Architecture & Script Explanations

The intelligence engine is modularized into 13 Python scripts. Below is a detailed explanation of each script, including its purpose in the pipeline and the prominent built-in functions or library methods it implements.

### 1. `config.py` (Configuration)
**Purpose**: Serves as the centralized registry for hyperparameters, file paths, and environment settings.
**Key Functions/Logic**:
*   `pathlib.Path(...)`: Built-in Python library function used to construct operating-system-agnostic file paths for input datasets and output intermediate artifacts.
*   Declares constants like `RANDOM_STATE = 42` and model hyperparameters (Random Forest depths, XGBoost booster settings) to ensure absolute determinism across executions.

### 2. `gpu_utils.py` (Hardware Operations)
**Purpose**: Handles GPU memory monitoring, dynamic CPU-fallback dispatching, and hardware-context detection. Prevents Over-Of-Memory (OOM) errors during heavy cuGraph / cuDF tasks.
**Key Functions**:
*   `pynvml.nvmlInit()` / `nvmlDeviceGetMemoryInfo()`: Functions from the NVIDIA Management Library bridging the Python API directly to the physical GPU architecture to read live VRAM utilization.
*   `@contextlib.contextmanager`: Python built-in decorator handler used to create the `gpu_context("TaskName")` scope, allowing us to seamlessly wrap heavy pipeline functions and log their specific memory footprints before and after execution.

### 3. `utils.py` (Shared Utilities)
**Purpose**: Provides standard systemic hooks such as logging configuration, generic pipeline execution timing, and physical disk caching.
**Key Functions**:
*   `time.perf_counter()`: A highly accurate Python OS clock used to generate granular benchmarks measuring execution wall-time across various pipeline steps.
*   `pickle.dump(obj, file)` / `pickle.load(file)`: Standard Python built-in binary serialization methods. Crucial for saving the large 1.5 GB intermediate state of the pipeline to disk natively, breaking the pipeline into restartable modules safely.

### 4. `ingestion.py` (Data Pipeline 1)
**Purpose**: Responsible for bulk-loading historical financial data into memory and unifying disparate supplementary tables (e.g., sector groupings).
**Key Functions**:
*   `cudf.read_csv()`: The GPU-equivalent of pandas `read_csv`. Dispatches IO parsing natively to thousands of CUDA cores inside the RTX 4050, resulting in significantly faster file ingestions.
*   `pd.merge(..., how="left")`: Relational join built-in to Pandas, mapped here on "SecuritiesCode" to map the 17 unique JPX Sector configurations onto our price timeseries dataframe.

### 5. `cleaning.py` (Data Pipeline 2)
**Purpose**: Standardizes raw asset pricing. Applies stock-split factors, drops poorly-quoted companies, and mitigates dirty data points via Z-score bounds. 
**Key Functions**:
*   `df.fillna(method="ffill")` / `.bfill()`: "Forward/Backward fill" operations carrying the last valid observation forward. Essential for fixing discontinuous gap trading.
*   `np.abs(stats.zscore(df))` > Threshold: Computes the Gaussian absolute Z-score of numerical returns to statistically identify anomalies (e.g. data-entry bugs) and purges any returns violating ±10σ.

### 6. `features.py` (Data Pipeline 3)
**Purpose**: The central feature engineering engine. Generates 27 indicators spanning Momentum, Cross-Sectional Ranking, liquidity (Amihud), and various Volatility windows.
**Key Functions**:
*   `df.groupby("SecuritiesCode").rolling(window=X).std()`: The structural Pandas timeseries aggregation function used to calculate moving averages.
*   `pd.qcut(df, q=3, labels=[0, 1, 2])`: Pandas "Quantile Cut". A crucial discretization function deployed here cross-sectionally per day to classify stocks evenly into "High, Medium, and Low" volatility regimes simultaneously, constructing the primary Y-target for Machine Learning.

### 7. `signals.py` (Spectral Intelligence)
**Purpose**: Transposes prices from the time domain cleanly into the frequency domain. Creates market 'bandgap' filters (Low, Mid, High signals) to detect wave-like volatility bursts.
**Key Functions**:
*   `cupy.fft.fft(data)`: The Fast Fourier Transform. Deployed heavily across thousands of isolated CPU threads on the GPU to decompose closing prices into constituent wave frequencies natively utilizing CuPy (NVIDIA's numpy equivalent).
*   `np.hanning(window)`: A windowing function built to multiply arrays by bell-curve weights prior to FFT execution, severely reducing "spectral leakage" at endpoint boundaries.

### 8. `graph.py` (Network Intelligence)
**Purpose**: Formulates the correlation risk engine. Treats individual stocks as 'Nodes' and their correlated rolling returns as 'Edges' to map unseen systemic market ties.
**Key Functions**:
*   `cupy.corrcoef()`: Extremely fast GPU cosine-similarity built-in, evaluating thousands of correlations within a few milliseconds.
*   `cugraph.Graph.from_cudf_edgelist()`: GPU Graph constructor generating an optimized CSR (Compressed Sparse Row) matrix natively.
*   `cugraph.louvain()`: A complex community-detection heuristic clustering stocks dynamically based purely on mathematical edge-weight similarities instead of their actual industry names.
*   `cugraph.pagerank()`: The classic Google centrality algorithm evaluating cascading graph risks. Surfaces specific stock identifiers that present the highest structural threat to the wider network.

### 9. `ml.py` (Predictive Layer Deep-Dive)
**Purpose**: Acts as the objective Machine Learning predictive engine, bridging heavily engineered hybrid features and signal processing into actionable quant logic. It trains two distinct predictive regimes sequentially via tree-based algorithms. 

**End-to-End Operational Flow**:
1.  **Strict Temporal Cross-Validation (`time_split()`)**: 
    Before any models are instantiated, the script splits the finalized 2.3M aggregated observations (X) across the time domain chronologically (80% Train, 20% validation). To absolutely guarantee zero target-leakage and prevent look-ahead bias, it introduces a strictly calculated `PURGE_GAP_DAYS` (defaulting to 5 days). This physical gap purges structural autocorrelation overlap occurring between rolling indicators across the train/test boundaries.
2.  **Target Extraction & Normalization (`_prepare_xy()`)**: 
    Generates structured memory matrices by sweeping the data arrays, standardizing missing attributes by casting NaN limits locally, and converting target vectors into natively recognized `float32` datatypes to squeeze max GPU cache throughputs efficiently.
3.  **Task A: Market Regime Classification (`train_regime_classifier()`)**:
    Detects whether the market is shifting into a Low, Medium, or High volatility environment over a leading chronological timeframe. Deploys three distinct algorithms to ensemble-vote confidence:
    *   **Logistic Regression**: Calculates basic stochastic gradient descent baselines.
    *   **Random Forest**: Extracts non-linear dependencies. Uses configuration flags controlling strict `max_depth` to enforce generalization natively bridging via cuML (`cuml.ensemble.RandomForestClassifier`).
    *   **XGBoost Engine**: A highly-tuned GPU gradient-booster using `tree_method="gpu_hist"`, operating exclusively in `multi:softprob` space to determine mathematically-smooth probability bounds for all three specific regime conditions.
4.  **Task B: Volatility Spike Prediction (`train_vol_spike_predictor()`)**:
    Constructs an independent binary classification boundary attempting to flag impending systemic flash-surges. Because volatility spikes are inherently extremely rare events, standard binary metrics mathematically collapse. To overcome severe target sparsity, the pipeline calculates class prevalence (`spike_ratio`) mathematically weighting minority-class iterations internally via strict `scale_pos_weight` gradients mapping non-symmetric loss penalties natively into the XGBoost boosters.
5.  **Multi-Dimensional Evaluation (`_evaluate()`)**:
    Bypasses standard Accuracy metrics and isolates algorithmic competence via rigorous calculations determining precise Harmonic Mean `f1_macro` metrics and multi-class One-vs-Rest `roc_auc_score` matrices ensuring strict predictive integrity limits against historical financial data structures natively.

**Key Built-In Functions**:
*   `xgboost.train(..., tree_method="gpu_hist")`: XGBoost native initialization loading memory directly onto distributed NVIDIA VRAM to compute histogram bin splits optimally while entirely skipping the primary OS CPU threads. 
*   `sklearn.metrics.f1_score()` & `roc_auc_score()`: External evaluation mathematical standards generating reliable binary vs multivariate metric calculations correctly interpreting imbalanced precision bounds globally across the arrays.

### 10. `risk.py` (Aggregation Engine)
**Purpose**: Distills the massive data tables from the earlier pipeline states into a clear suite of centralized numerical "threat" thresholds.
**Key Functions**:
*   `scipy.stats.percentileofscore()`: Translates arbitrary numerical dimensions (like pure centrality and drawdown percentages) into flattened 0-100 uniformly bounded percentages allowing different logic ranges to be mathematically added together directly into composite ranks.

### 11. `benchmark.py` (Validation)
**Purpose**: The objective performance tester. Isolates identically structured CPU logic (Pandas/Scipy) alongside GPU logic (cuDF/CuPy) and compares wall-time speeds.
**Key Functions**:
*   `tracemalloc.start()` / `tracemalloc.get_traced_memory()`: Standard python library tools measuring Python's absolute dynamic RAM allocation peaks. 
*   `np.mean(times)`: Validates tests across configurable iteration counts to prevent caching noise overheads.

### 12. `dashboard.py` (Front-End Viewer)
**Purpose**: A highly interactive, dark-themed HTML/React framework built entirely natively in Python displaying all processed output artifacts gracefully.
**Key Functions**:
*   `dash.Dash(__name__)`: Instantiates the backend Flask WSGI server wrapping the application components simultaneously.
*   `plotly.graph_objects.Figure()` & `go.Scatter(...)` / `go.Bar(...)`: Structural charting objects that compile JSON strings into rich JS interactive canvases in the browser securely.
*   `dash.callback`: Asynchronous JS-React listener decorators enabling the front-end to trigger structural callback functions (e.g., swapping rendered graphics) natively inside Python when a user clicks dropdowns.

### 13. `main.py` (Orchestrator)
**Purpose**: The central nervous system. Strings together all modules conceptually, cleans VRAM securely between states globally, and logs massive metrics visually.
**Key Functions**:
*   `del dataframe` & GPU Memory Contexts: The explicit destruction of Python objects triggering the Garbage Collector correctly prior to moving between logic blocks, strictly keeping the processing footprint inside our structural 6 GB VRAM limits.
*   `__name__ == "__main__"`: The python execution barrier blocking execution during modular imports but triggering sequentially when strictly deployed as a script.
