# Friend Recommendation System — Enron Email Dataset

A graph-theory based friend recommendation system built on the real-world **Enron Email Dataset**.  
Compares two methods — **Jaccard Similarity** (local) vs **Eigenvector Centrality** (global) — to identify potential new connections in a communication network.

---

## Problem Statement

In large social or communication networks, it is difficult for users to identify new meaningful connections.  
This project models the Enron email network as a graph and recommends potential friendships by analyzing:
- Mutual connections between users (Jaccard Similarity)
- Global importance of users in the network (Eigenvector Centrality)

---

## Dataset

**Enron Email Dataset** — real corporate email data, publicly available.

| Property | Detail |
|----------|--------|
| Source | Kaggle — [Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) |
| File | `emails.csv` (~1.3 GB) |
| Total emails | ~500,000 |
| Employees | ~150 Enron staff |
| Period | 1999 – 2002 |

> The dataset is **not included** in this repository (file too large for GitHub).  
> Download `emails.csv` from the Kaggle link above and place it inside the `data/` folder.

---

## Graph Modeling

Each email interaction is modelled as an edge in an undirected weighted graph.

| Element | Meaning |
|---------|---------|
| Node | Email user / Enron employee |
| Edge | Email communication between two users |
| Edge weight | Number of emails exchanged |
| Graph type | Undirected, Weighted |

**After cleaning:**

| Stat | Value |
|------|-------|
| Emails parsed | 50,000 |
| Raw edges extracted | 69,465 |
| Edges after noise filter | 7,421 |
| Nodes in final graph | 3,939 |
| Edges in final graph | 7,185 |

---

## Methods Compared

### Method 1 — Jaccard Similarity (local)

Measures how much the neighborhoods of two users overlap.

```
Jaccard(A, B) = |N(A) ∩ N(B)| / |N(A) ∪ N(B)|
```

| | |
|---|---|
| Type | Local — only looks at direct neighbors |
| Score range | 0.0 (no overlap) to 1.0 (complete overlap) |
| Strength | Simple, fast, interpretable |
| Weakness | Treats all mutual friends equally — a shared friend who is a CEO counts the same as an intern |

---

### Method 2 — Eigenvector Centrality (global)

Measures global importance of every user in the network, then scores candidate pairs based on how influential they and their mutual friends are.

```
score(A, B) = ec(A) + ec(B) + alpha × Σ ec(mutual_neighbors)
```

| | |
|---|---|
| Type | Global — uses the entire network structure |
| Inspired by | Google PageRank |
| Strength | Rewards connections between influential people; unique score per pair |
| Weakness | More compute-intensive; harder to explain intuitively |

---

## Key Difference

| | Jaccard | Eigenvector Centrality |
|---|---|---|
| Question asked | How many friends do A and B share? | How important are A, B, and their mutual friends? |
| Perspective | Local neighborhood | Entire network |
| Problem | Too many pairs score 1.0 — cannot rank | Every pair gets a unique score |

---

## Project Structure

```
enron-friend-recommendation/
│
├── data/
│   └── emails.csv                  ← Download from Kaggle (not in repo)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              ← Load and parse raw emails
│   ├── cleaning.py                 ← Clean addresses, filter noise
│   ├── graph_builder.py            ← Build NetworkX graph, train/test split
│   ├── jaccard.py                  ← Jaccard similarity scorer
│   ├── eigenvector.py              ← Eigenvector centrality scorer
│   ├── evaluation.py               ← AUC-ROC, Precision@K comparison
│   └── visualizations.py           ← All plots and charts
│
├── outputs/                        ← Auto-generated PNG charts (after running)
│
├── notebooks/
│   └── exploration.ipynb           ← EDA and graph exploration
│
├── main.py                         ← Run this to execute full pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/enron-friend-recommendation.git
cd enron-friend-recommendation
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
- Go to: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
- Download `emails.csv`
- Place it inside the `data/` folder

### 5. Run the full pipeline
```bash
python main.py
```

> If `emails.csv` is not found, the script automatically runs on a synthetic graph so you can verify the code works before downloading the dataset.

---

## How Evaluation Works

We use a **link removal** strategy (no labelled data needed):

1. Remove 10% of real edges from the graph (718 edges) — these become the **ground truth**
2. Train both methods on the remaining 90% of edges
3. Ask both methods to score 5,000 candidate non-edge pairs
4. Check if the removed edges appear in the top-ranked predictions
5. Measure performance using AUC-ROC and Precision@K

---

## Results

| Method | AUC-ROC | Precision@10 | Precision@50 |
|--------|---------|--------------|--------------|
| Jaccard Similarity | 0.3739 | 0.0000 | 0.0000 |
| **Eigenvector Centrality** | **0.5400** | **0.0000** | **0.0400** |

### Interpretation

| Score | Meaning |
|-------|---------|
| AUC = 0.5 | Random guessing — baseline |
| Jaccard AUC = 0.37 | Worse than random — too many tied scores at 1.0, cannot rank meaningfully |
| EC AUC = 0.54 | Better than random — unique scores allow meaningful ranking |
| EC P@50 = 0.04 | 2 out of top-50 EC recommendations were real hidden connections |

### Conclusion

> Eigenvector Centrality outperforms Jaccard Similarity on the Enron email dataset.  
> Jaccard fails because many user pairs share all mutual friends (score = 1.0), making ranking impossible.  
> Eigenvector Centrality uses global network influence to assign unique scores to every pair,  
> successfully predicting real connections above the random baseline.

---

## Output Charts

After running `main.py`, the `outputs/` folder contains:

| File | Description |
|------|-------------|
| `degree_distribution.png` | Log-log degree distribution — confirms power-law network structure |
| `graph_visualization.png` | Network graph — node size and colour = eigenvector centrality score |
| `top_recommendations.png` | Top-10 recommended pairs side by side for both methods |
| `method_comparison.png` | AUC / P@10 / P@50 bar chart comparison |

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| Python 3.13 | Core language |
| NetworkX | Graph construction and eigenvector centrality |
| Pandas | CSV loading and email parsing |
| NumPy | Numerical operations |
| Matplotlib | All visualizations |
| Scikit-learn | AUC-ROC evaluation |

---

## Author

**Sharmi**  
GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

---

## References

- Enron Email Dataset — Carnegie Mellon University / Kaggle
- Newman, M. E. J. (2010). *Networks: An Introduction*. Oxford University Press.
- Page, L. et al. (1999). *The PageRank Citation Ranking: Bringing Order to the Web*.
- Liben-Nowell, D. & Kleinberg, J. (2007). *The Link Prediction Problem for Social Networks*. JASIST.
