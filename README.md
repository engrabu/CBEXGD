# Federated Learning Experiment Runner

This project contains the full experimental setup and visualization tools for evaluating various deduplication and privacy-preserving techniques in Federated Learning (FL) systems.

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Last Updated](https://img.shields.io/badge/updated-May%202025-brightgreen.svg)

---

## ✨ Overview

* **`run_all_experiments.py`**: Executes simulations across datasets (MNIST, FMNIST, CIFAR10) with various deduplication and privacy strategies.
* **`improved_code.py`**: Contains models, deduplication mechanisms, and privacy wrappers.
* **`visualize_result.py`**: Generates comparative visualizations from the results.

---

## ⚙ Installation

### Python Requirements

* Python 3.8+
* Install dependencies:

```bash
pip install -r requirements.txt
```

### Sample `requirements.txt`

```txt
numpy
pandas
matplotlib
seaborn
torch
torchvision
```

---

## 🌐 Running Experiments

Use the following command to execute all experiments:

```bash
python run_all_experiments.py \
  --clients 100 \
  --rounds 10 \
  --output results \
  --dataset all \
  --privacy all \
  --distribution all
```

### Optional Flags

| Flag              | Description                                 |
| ----------------- | ------------------------------------------- |
| `--clients`       | Number of FL clients (default: 100)         |
| `--rounds`        | Communication rounds (default: 10)          |
| `--output`        | Base directory for output files             |
| `--dataset`       | Dataset: MNIST, FMNIST, CIFAR10, or all     |
| `--privacy`       | Privacy mode: none, dp, secure\_agg, or all |
| `--distribution`  | Data distribution: iid, non-iid, or all     |
| `--deduplication` | Run a specific deduplication method         |
| `--quick`         | Only run essential methods for debugging    |

### Quick Debug Mode

```bash
python run_all_experiments.py --quick
```

---

## 📊 Visualizing Results

1. Update the file paths in `visualize_result.py` under `load_data()` to match your result folder.

```python
accuracy_df = pd.read_csv('results_<timestamp>/accuracy_by_method_dataset.csv')
summary_df = pd.read_csv('results_<timestamp>/comprehensive_summary.csv')
comparison_df = pd.read_csv('results_<timestamp>/interim_results/comparison_table.csv')
```

2. Run visualization:

```bash
python visualize_result.py
```

3. Output includes:

* `method_accuracy_comparison.png`
* `heatmap_<dataset>.png`
* `radar_chart_<dataset>.png`
* `metric_comparison_<metric>.png`
* `method_profile_<method>.png`

---

## 📃 Output Structure

```
results_<timestamp>/
├── Overall/
│   ├── accuracy_by_method_dataset.csv
│   ├── method_accuracy_comparison.png
├── interim_results/
│   └── comparison_table.csv
├── comprehensive_summary.csv
└── error_log.txt  (if any)
```

---

## 🌐 Example Minimal Test

```bash
python run_all_experiments.py \
  --clients 10 \
  --rounds 5 \
  --dataset MNIST \
  --privacy none \
  --distribution iid \
  --deduplication EXGD
```

---

## 🏆 Citation

If you use this framework in your research, please cite the relevant work and acknowledge the implementation support.

---

## ⚖ License

MIT License

---

## 👥 Contributors

* **Aliyu Garba** – *Lead Developer & Researcher*
  Universiti Teknologi PETRONAS
  [garba\_24000002@utp.edu.my](mailto:garba_24000002@utp.edu.my)

> Feel free to open issues or submit pull requests for improvements!
