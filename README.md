# LGCAR

LGCAR is a Mashup–API recommendation model that combines LightGCN graph propagation, Mashup functional intent, API service semantics, two levels of contrastive learning, and masked reconstruction tasks. This repository provides the full model for the main experiments on the PWA and HGA datasets.

## Environment

Use a Python environment with PyTorch and PyTorch Geometric installed. Other dependencies include NumPy, pandas, tqdm, NLTK, and PrettyTable.

The default device is `cuda:0`, which can be changed in the configuration file for each dataset. The code downloads the NLTK `stopwords` and `wordnet` resources when needed.

## Project Structure

```text
LGCAR/
├── train.py                 # Training entry point
├── evaluate.py              # Recommendation metric evaluation
├── utils.py                 # Data loading and preprocessing
├── models/
│   ├── Model.py             # LGCAR model
│   └── loss_func.py         # Loss functions
├── config/
│   ├── pwa.json             # PWA configuration
│   └── HGA.json             # HGA configuration
└── data/
    ├── pwa/
    └── HGA/
```

Each dataset directory should contain the following files:

- `train.txt`: Training Mashup–API interactions.
- `test.txt`: Test Mashup–API interactions.
- `api_list.txt`: API list and ID mappings.
- `mashup_list.txt`: Mashup list and ID mappings.
- `entity_list.txt`: Knowledge graph entity list and ID mappings.
- `relation_list.txt`: Knowledge graph relation types and ID mappings.
- `kg_final_four.txt`: Knowledge graph containing four types of API attribute relations.
- `mashup_functional_intent.txt`: Associations between Mashups and functional intent nodes.

## Running the Experiments

Run the following commands from the repository root.

### PWA

```bash
python train.py --dataset pwa --seed 2025
```

### HGA

```bash
python train.py --dataset HGA --seed 2025
```


## Configuration

PWA and HGA load `config/pwa.json` and `config/HGA.json`, respectively. Set hyperparameters such as the number of layers, learning rate, edge dropout rate, contrastive loss weight, reconstruction loss weights, and masking rates in these configuration files.


## Evaluation Metrics

Only the following six metrics are computed and reported:

- Recall@10, Recall@20
- NDCG@10, NDCG@20
- Hit Ratio@10, Hit Ratio@20


## Results

Each run creates a separate results directory:

```text
results/{dataset}_{timestamp}/
├── best_model.pth
├── best_results.json
└── hyperparameters.json
```

- `best_model.pth`: Parameters of the best model.
- `best_results.json`: Best epoch, model selection score, random seed, and the six recommendation metrics.
- `hyperparameters.json`: Configuration values used for the run, including command-line overrides.
