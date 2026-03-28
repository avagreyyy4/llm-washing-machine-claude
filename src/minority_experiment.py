import torch
import torch.nn.functional as F
import json
from pathlib import Path
import transformer_lens as tl
import numpy as np
from collections import defaultdict
from manual_sae import ManualSAE
from scipy import stats
import matplotlib.pyplot as plt

'''
This analysis aims to look how minority compounds are stored and if 
that differs compared to the original experiment.

Part 1 repeats parts of the original experiment now including minority compounds 
Part 2 attemptions to directly answer if minority compounds are where the model "fails to predict"

1a: Next token analysis for majority and minority compounds 
1b: Cosine similarity for majority and minority compounds at a layerwise level
1c: SAE feature analysis for majority and minority compounds at a layerwise level

2: Given three compound families (with varying association strength), see how the model peforms on sentences
with slightly more context
'''

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

COMPOUNDS = [
    ("washing", "machine"),
    ("washing", "powder"),
    ("washing", "method"),
    ("washing", "cycle"),
    ("coffee", "table"),
    ("coffee", "shop"),
    ("coffee", "maker"),
    ("coffee", "mug"),
    ("swimming", "pool"),
    ("swimming", "suit"),
    ("swimming", "lesson"),
    ("swimming", "team"),
    ("parking", "lot"),
    ("parking", "meter"),
    ("parking", "garage"),
    ("parking", "ticket"),
    ("hot", "dog"),
    ("hot", "tub"),
    ("hot", "spring"),
    ("hot", "sauce"),
    ("shooting", "star"),
    ("shooting", "range"),
    ("shooting", "gallery"),
    ("shooting", "guard"),
    ("living", "room"),
    ("living", "wage"),
    ("living", "standard"),
    ("living", "conditions"),
    ("driving", "license"),
    ("driving", "force"),
    ("driving", "rain"),
    ("driving", "seat"),
    ("guinea", "pig"),
    ("guinea", "fowl"),
    ("guinea", "worm"),
    ("guinea", "coast"),
    ("brick", "house"),
    ("brick", "wall"),
    ("brick", "layer"),
    ("brick", "road"),
    ("mountain", "cabin"),
    ("mountain", "bike"),
    ("mountain", "lion"),
    ("mountain", "range"),
    ("garden", "hose"),
    ("garden", "shed"),
    ("garden", "fence"),
    ("garden", "party"),
    ("water", "bottle"),
    ("water", "supply"),
    ("water", "heater"),
    ("water", "fall"),
    ("steel", "bridge"),
    ("steel", "frame"),
    ("steel", "cage"),
    ("steel", "plate"),
    ("chocolate", "cake"),
    ("chocolate", "chip"),
    ("chocolate", "factory"),
    ("chocolate", "bar"),
    ("door", "handle"),
    ("door", "bell"),
    ("door", "frame"),
    ("door", "step"),
    ("blue", "berry"),
    ("blue", "print"),
    ("blue", "bell"),
    ("blue", "bird"),
    ("snow", "man"),
    ("snow", "storm"),
    ("snow", "board"),
    ("snow", "fall"),
    ("sun", "flower"),
    ("sun", "burn"),
    ("sun", "beam"),
    ("sun", "screen"),
]

TEMPLATES = [
    "The {compound} was",
    "She bought a {compound} for",
    "I saw a {compound} in the",
    "There is a {compound} near the",
    "He fixed the {compound} with",
    "A new {compound} arrived",
    "The old {compound} needed",
    "We need a {compound} to",
]

def load_model(model_name="gpt2"):
    model = tl.HookedTransformer.from_pretrained(model_name, device=DEVICE)
    model.eval()
    return model

def get_token_ids(model, word):
    tokens = model.to_tokens(f" {word}", prepend_bos=False)
    return tokens[0].tolist()

# ── PART 1a: Compound Association Scores ──────────────────────────────────────────────

def filter_compounds(scored):
    """
    Drop groups where the majority compound (highest priming_ratio) has rank > 50.
    Also drop individual compounds with rank > 3000 (likely single-token cases e.g. waterfall).
    """
    groups = defaultdict(list)
    for entry in scored:
        groups[entry["word1"]].append(entry)

    valid_word1s = set()
    for word1, entries in groups.items():
        majority = max(entries, key=lambda e: e["priming_ratio"])
        if majority["rank"] <= 50:
            valid_word1s.add(word1)
        else:
            print(f"  Dropping group '{word1}': majority '{majority['word2']}' at rank {majority['rank']:.0f}")

    filtered = []
    for entry in scored:
        if entry["word1"] not in valid_word1s:
            continue
        if entry["rank"] > 3000:
            print(f"  Dropping ({entry['word1']}, {entry['word2']}): rank {entry['rank']:.0f} -- likely single token")
            continue
        filtered.append(entry)

    print(f"\n  Kept {len(filtered)} compounds across {len(valid_word1s)} groups after filtering")
    return filtered


def score_compounds(model):
    """
    For each (word1, word2) pair in COMPOUNDS, compute probability, rank,
    and priming ratio using the same template approach as experiment1.
    Filters out groups where the majority compound has rank > 50, and
    individual compounds with rank > 3000 (single-token cases).
    Normalizes priming ratios within each word1 group to get a 0-1 scale.
    Also computes log_rank for cross-group comparison.
    Returns scored list with normalized_association and log_rank attached to each entry.
    """
    print("\n" + "="*70)
    print("COMPOUND SCORING")
    print("="*70)

    # baseline P(word2 | "The")
    bos_tokens = model.to_tokens("The")
    with torch.no_grad():
        logits = model(bos_tokens)
    base_probs = F.softmax(logits[0, -1], dim=-1)

    scored = []
    for word1, word2 in COMPOUNDS:
        word2_tokens = get_token_ids(model, word2)
        if len(word2_tokens) != 1:
            print(f"  Skipping ({word1}, {word2}): word2 is multi-token")
            continue
        word2_id = word2_tokens[0]

        avg_p = 0.0
        avg_rank = 0.0
        for template in TEMPLATES:
            prompt = template.split("{compound}")[0] + word1
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                logits = model(tokens)
            probs = F.softmax(logits[0, -1], dim=-1)
            avg_p += probs[word2_id].item()
            avg_rank += (probs > probs[word2_id]).sum().item() + 1
        avg_p /= len(TEMPLATES)
        avg_rank /= len(TEMPLATES)

        base_p = base_probs[word2_id].item()
        priming_ratio = avg_p / max(base_p, 1e-10)

        scored.append({
            "word1": word1,
            "word2": word2,
            "probability": round(avg_p, 6),
            "rank": round(avg_rank, 1),
            "priming_ratio": round(priming_ratio, 2),
        })

        print(f"  {word1:12s} {word2:12s} | p={avg_p:.4f} | rank={avg_rank:.0f} | ratio={priming_ratio:.1f}x")

    # filter before normalization
    print("\n" + "-"*70)
    print("FILTERING")
    print("-"*70)
    scored = filter_compounds(scored)

    # normalize within each word1 group and compute log_rank
    groups = defaultdict(list)
    for entry in scored:
        groups[entry["word1"]].append(entry)

    for word1, entries in groups.items():
        max_ratio = max(e["priming_ratio"] for e in entries)
        for entry in entries:
            entry["normalized_association"] = round(entry["priming_ratio"] / max_ratio, 4)
            entry["log_rank"] = round(np.log(entry["rank"]), 4)

    with open(RESULTS_DIR / "compound_scores.json", "w") as f:
        json.dump(scored, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR / 'compound_scores.json'}")
    return scored


# ── PART 1b: Cosine Similarity Layerwise ──────────────────────────────────────────────

def compute_cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10))

def get_residual_at_head(model, prompt, head_word, layer=-1):
    """
    Run prompt through model and extract residual stream at the head word position.
    layer=-1 means final layer.
    """
    tokens = model.to_tokens(prompt)
    token_strs = [model.to_string([t]) for t in tokens[0]]

    head_str = f" {head_word}"
    head_pos = None
    for pos, ts in enumerate(token_strs):
        if ts.lower() == head_str.lower():
            head_pos = pos
            break

    if head_pos is None:
        return None

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    layer_key = f"blocks.{model.cfg.n_layers - 1 if layer == -1 else layer}.hook_resid_post"
    return cache[layer_key][0, head_pos].cpu().numpy()


def get_mean_residual(model, prompts, head_word, layer):
    """
    Extract residual stream at head_word position across multiple prompts
    and return the mean vector.
    """
    vecs = []
    for prompt in prompts:
        vec = get_residual_at_head(model, prompt, head_word, layer=layer)
        if vec is not None:
            vecs.append(vec)
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def experiment_majority_minority_cosine(model, data):
    print("\n" + "="*70)
    print("COSINE SIMILARITY VS ASSOCIATION STRENGTH (ALL LAYERS)")
    print("="*70)

    n_layers = model.cfg.n_layers
    results = []

    for entry in data:
        word1 = entry["word1"]
        word2 = entry["word2"]
        normalized_association = entry["normalized_association"]
        log_rank = entry["log_rank"]
        priming_ratio = entry["priming_ratio"]

        # build one prompt per template for compound and isolation
        compound_prompts  = [t.format(compound=f"{word1} {word2}") for t in TEMPLATES]
        isolation_prompts = [t.format(compound=word2) for t in TEMPLATES]

        cos_per_layer = []
        skip = False

        for layer in range(n_layers):
            compound_vec  = get_mean_residual(model, compound_prompts, word2, layer)
            isolation_vec = get_mean_residual(model, isolation_prompts, word2, layer)

            if compound_vec is None or isolation_vec is None:
                skip = True
                break

            cos_sim = compute_cosine_similarity(compound_vec, isolation_vec)
            cos_per_layer.append(round(cos_sim, 4))

        if skip:
            print(f"  Skipping ({word1}, {word2}): token not found")
            continue

        results.append({
            "word1": word1,
            "word2": word2,
            "normalized_association": normalized_association,
            "log_rank": log_rank,
            "priming_ratio": priming_ratio,
            "cosine_per_layer": cos_per_layer,
        })

        print(f"  {word1:12s} {word2:12s} | norm_assoc: {normalized_association:.4f} | "
              f"log_rank: {log_rank:.4f} | "
              f"cos layer 0: {cos_per_layer[0]:.4f} -> layer 11: {cos_per_layer[-1]:.4f}")

    with open(RESULTS_DIR / "cosine_vs_association_all_layers.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR / 'cosine_vs_association_all_layers.json'}")
    return results


def plot_cosine_vs_association_all_layers(results):
    """
    Four plots (2x2):
    Row 0: normalized_association vs cosine similarity (slope across layers + high/low split)
    Row 1: log_rank vs cosine similarity (slope across layers + high/low split)
    Note: for log_rank, higher value = weaker association, so labels are flipped.
    """


    n_layers = len(results[0]["cosine_per_layer"])

    predictors = [
        ("normalized_association", "Normalized Association", False),
        ("log_rank",               "log(Rank)",             True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, (key, label, higher_is_weaker) in enumerate(predictors):
        x = np.array([r[key] for r in results])

        slopes, p_values = [], []
        for layer in range(n_layers):
            y = np.array([r["cosine_per_layer"][layer] for r in results])
            slope, _, _, p, _ = stats.linregress(x, y)
            slopes.append(slope)
            p_values.append(p)

        ax = axes[row, 0]
        ax.plot(range(n_layers), slopes, 'b-o', linewidth=2, markersize=6)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.4)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Slope", fontsize=12)
        ax.set_title(f"Slope of {label} vs Cosine Similarity", fontsize=11)

        median = np.median(x)
        if higher_is_weaker:
            strong = [r for r in results if r[key] <= median]
            weak   = [r for r in results if r[key] >  median]
            strong_label = "Low log(rank) — strong assoc"
            weak_label   = "High log(rank) — weak assoc"
        else:
            strong = [r for r in results if r[key] >= median]
            weak   = [r for r in results if r[key] <  median]
            strong_label = "High association"
            weak_label   = "Low association"

        strong_mean = [np.mean([r["cosine_per_layer"][l] for r in strong]) for l in range(n_layers)]
        weak_mean   = [np.mean([r["cosine_per_layer"][l] for r in weak])   for l in range(n_layers)]
        strong_std  = [np.std([r["cosine_per_layer"][l]  for r in strong]) for l in range(n_layers)]
        weak_std    = [np.std([r["cosine_per_layer"][l]  for r in weak])   for l in range(n_layers)]

        ax = axes[row, 1]
        layers = range(n_layers)
        ax.plot(layers, strong_mean, 'b-o', linewidth=2, label=strong_label, markersize=5)
        ax.fill_between(layers,
                        np.array(strong_mean) - np.array(strong_std),
                        np.array(strong_mean) + np.array(strong_std),
                        alpha=0.2, color='blue')
        ax.plot(layers, weak_mean, 'r-o', linewidth=2, label=weak_label, markersize=5)
        ax.fill_between(layers,
                        np.array(weak_mean) - np.array(weak_std),
                        np.array(weak_mean) + np.array(weak_std),
                        alpha=0.2, color='red')
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Mean Cosine Similarity", fontsize=12)
        ax.set_title(f"Mean Cosine Similarity Across Layers\n{label}: Strong vs Weak Association", fontsize=11)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plot_path = PLOTS_DIR / "cosine_vs_association_all_layers.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {plot_path}")

    for key, label, _ in predictors:
        x = np.array([r[key] for r in results])
        print(f"\nLayer-wise slope summary ({label}):")
        for layer in range(n_layers):
            y = np.array([r["cosine_per_layer"][layer] for r in results])
            slope, _, r_val, p, _ = stats.linregress(x, y)
            sig = " *" if p < 0.05 else ""
            print(f"  Layer {layer:2d}: R^2={r_val**2:.3f}, slope={slope:.4f}, p={p:.3f}{sig}")


# ── PART 1c: SAE Layerwise ──────────────────────────────────────────────

TARGET_LAYERS = [0, 1, 2, 5, 8, 11]

def experiment_sae_vs_association(model, data):
    """
    For each (word1, word2) pair, compute SAE feature overlap between
    compound context and isolation contexts across multiple layers.
    Mirrors experiment1: compares compound vs word2-isolation, word1-isolation,
    and checks whether word1's representation shifts in compound context.
    Uses resid_post to match experiment1.
    """
    print("\n" + "="*70)
    print("SAE FEATURE ANALYSIS (layers {})".format(TARGET_LAYERS))
    print("="*70)

    # load SAEs for all target layers
    saes = {}
    for layer in TARGET_LAYERS:
        print(f"  Loading SAE for layer {layer}...")
        try:
            sae, cfg = ManualSAE.from_pretrained(layer=layer, device=DEVICE)
            sae.eval()
            saes[layer] = sae
            print(f"    Loaded: d_in={cfg['d_in']}, d_sae={cfg['d_sae']}")
        except Exception as e:
            print(f"    Failed for layer {layer}: {e}")

    def get_sae_features(vec, sae):
        """Run a residual stream vector through an SAE, return active feature set."""
        with torch.no_grad():
            tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            acts = sae.encode(tensor).squeeze(0)
        active = set(torch.where(acts > 0)[0].cpu().tolist())
        return active


    results = []

    for entry in data:
        word1 = entry["word1"]
        word2 = entry["word2"]
        normalized_association = entry["normalized_association"]
        log_rank = entry["log_rank"]
        priming_ratio = entry["priming_ratio"]

        # build prompt lists once per compound
        compound_prompts  = [t.format(compound=f"{word1} {word2}") for t in TEMPLATES]
        word2_iso_prompts = [t.format(compound=word2) for t in TEMPLATES]
        word1_iso_prompts = [t.format(compound=word1) for t in TEMPLATES]
        layer_metrics = {}
        skip = False

        for layer, sae in saes.items():
            # word2 in compound context
            compound_vec = get_mean_residual(model, compound_prompts, word2, layer)
            # word2 in isolation
            word2_iso_vec = get_mean_residual(model, word2_iso_prompts, word2, layer)
            # word1 in compound context
            word1_compound_vec = get_mean_residual(model, compound_prompts, word1, layer)
            # word1 in isolation
            word1_iso_vec = get_mean_residual(model, word1_iso_prompts, word1, layer)

            if any(v is None for v in [compound_vec, word2_iso_vec, word1_compound_vec, word1_iso_vec]):
                skip = True
                break

            compound_feats   = get_sae_features(compound_vec, sae)
            word2_iso_feats  = get_sae_features(word2_iso_vec, sae)
            word1_comp_feats = get_sae_features(word1_compound_vec, sae)
            word1_iso_feats  = get_sae_features(word1_iso_vec, sae)

            # word2 metrics: how much does compound context change word2's features?
            unique_to_compound = compound_feats - word2_iso_feats
            overlap_w2 = compound_feats & word2_iso_feats
            union_w2   = compound_feats | word2_iso_feats
            frac_unique_w2 = len(unique_to_compound) / len(compound_feats) if compound_feats else 0
            jaccard_w2     = len(overlap_w2) / len(union_w2) if union_w2 else 0

            # overlap of compound word2 features with word1 isolation features
            overlap_compound_w1 = compound_feats & word1_iso_feats
            frac_overlap_w1 = len(overlap_compound_w1) / len(compound_feats) if compound_feats else 0

            layer_metrics[layer] = {
                    "frac_unique_w2":    round(frac_unique_w2, 4),
                    "jaccard_w2":        round(jaccard_w2, 4),
                    "frac_overlap_w1":   round(frac_overlap_w1, 4),
                    "n_compound_feats":  len(compound_feats),
                    "n_word2_iso_feats": len(word2_iso_feats),
                    "n_word1_comp_feats": len(word1_comp_feats),
                    "n_word1_iso_feats": len(word1_iso_feats),
                }

        if skip:
            print(f"  Skipping ({word1}, {word2}): token not found")
            continue

        results.append({
                "word1": word1,
                "word2": word2,
                "normalized_association": normalized_association,
                "log_rank": log_rank,
                "priming_ratio": priming_ratio,
                "layer_metrics": layer_metrics,
            })

        early = layer_metrics[TARGET_LAYERS[0]]
        late  = layer_metrics[TARGET_LAYERS[-1]]
        print(f"  {word1:12s} {word2:12s} | norm_assoc: {normalized_association:.4f} | "
                  f"frac_unique layer {TARGET_LAYERS[0]}: {early['frac_unique_w2']:.4f} "
                  f"-> layer {TARGET_LAYERS[-1]}: {late['frac_unique_w2']:.4f}")

    with open(RESULTS_DIR / "sae_vs_association.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR / 'sae_vs_association.json'}")
    return results


def plot_sae_vs_association_strength(results):
    """
    For each SAE metric, plot slope of log_rank vs metric across layers
    (mirroring the cosine layerwise plot). Also scatter at early vs late layer.
    """


    metrics = [
        ("frac_unique_w2",          "Frac Unique to Compound (word2)"),
        ("jaccard_w2",              "Jaccard: Compound vs Word2 Isolation"),
        ("frac_overlap_w1",         "Frac Compound Features Overlapping Word1"),
    ]

    fig, axes = plt.subplots(len(metrics), 2, figsize=(14, 5 * len(metrics)))

    x_all = np.array([r["log_rank"] for r in results])

    for row, (metric_key, metric_label) in enumerate(metrics):
        slopes, p_values = [], []
        for layer in TARGET_LAYERS:
            y = np.array([r["layer_metrics"][layer][metric_key] for r in results])
            slope, _, _, p, _ = stats.linregress(x_all, y)
            slopes.append(slope)
            p_values.append(p)

        # Plot 1: slope across layers
        ax = axes[row, 0]
        ax.plot(TARGET_LAYERS, slopes, 'b-o', linewidth=2, markersize=6)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.4)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Slope (log_rank)", fontsize=12)
        ax.set_title(f"Slope of log(Rank) vs {metric_label}", fontsize=11)
        ax.set_xticks(TARGET_LAYERS)

        # Plot 2: scatter at earliest and latest layer
        ax = axes[row, 1]
        early_layer = TARGET_LAYERS[0]
        late_layer  = TARGET_LAYERS[-1]
        y_early = np.array([r["layer_metrics"][early_layer][metric_key] for r in results])
        y_late  = np.array([r["layer_metrics"][late_layer][metric_key] for r in results])

        word1s = list(dict.fromkeys(r["word1"] for r in results))
        colors = plt.cm.tab20(np.linspace(0, 1, len(word1s)))
        word1_color = {w: c for w, c in zip(word1s, colors)}

        for r_entry in results:
            color = word1_color[r_entry["word1"]]
            xi = r_entry["log_rank"]
            ax.scatter(xi, r_entry["layer_metrics"][early_layer][metric_key],
                       color=color, marker='o', s=50, alpha=0.8,
                       edgecolors='k', linewidth=0.3)
            ax.scatter(xi, r_entry["layer_metrics"][late_layer][metric_key],
                       color=color, marker='s', s=50, alpha=0.5,
                       edgecolors='k', linewidth=0.3)

        # regression lines for early and late
        for y_vals, layer, marker_label, ls in [
            (y_early, early_layer, f"Layer {early_layer}", '-'),
            (y_late,  late_layer,  f"Layer {late_layer}",  '--'),
        ]:
            slope, intercept, r_val, p, _ = stats.linregress(x_all, y_vals)
            x_line = np.linspace(x_all.min(), x_all.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, ls,
                    label=f"{marker_label}: R^2={r_val**2:.3f}, p={p:.3f}",
                    linewidth=1.5)

        ax.set_xlabel("log(Rank)", fontsize=12)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f"log(Rank) vs {metric_label}\nLayer {early_layer} (circle) vs Layer {late_layer} (square)", fontsize=11)
        ax.legend(fontsize=8)

        # print summary
        print(f"\n{metric_label} — slope across layers:")
        for layer, slope, p in zip(TARGET_LAYERS, slopes, p_values):
            sig = " *" if p < 0.05 else ""
            print(f"  Layer {layer:2d}: slope={slope:.4f}, p={p:.3f}{sig}")

    plt.tight_layout()
    plot_path = PLOTS_DIR / "sae_vs_association_strength.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to {plot_path}")




# ── PART 2: Sentence-Specific Failure ──────────────────────────────────────────────

SENTENCE_GROUPS = {
    "washing": {
        "majority": "machine",
        "minorities": ["powder", "cycle", "method"],
        "sentences": [
            ("powder", "She measured out a scoop of washing"),
            ("powder", "He bought the wrong brand of washing"),
            ("powder", "The box of washing"),
            ("cycle", "She selected the gentle washing"),
            ("cycle", "He shortened the washing"),
            ("cycle", "The delicate washing"),
            ("method", "The study compared each washing"),
            ("method", "She developed a new washing"),
            ("method", "Researchers evaluated the washing"),
        ]
    },
    "coffee": {
        "majority": "shop",
        "minorities": ["table", "maker", "mug"],
        "sentences": [
            ("table", "He sat on the coffee"),
            ("table", "They grabbed a glass from the coffee"),
            ("table", "She wiped down the coffee"),
            ("maker", "He cleaned the coffee"),
            ("maker", "He emptied the coffee"),
            ("maker", "He used the coffee"),
            ("mug", "She picked up the coffee"),
            ("mug", "He dropped his favorite coffee"),
            ("mug", "She sipped the coffee"),
        ]
    },
    "mountain": {
        "majority": "bike",
        "minorities": ["cabin", "lion", "range"],
        "sentences": [
            ("cabin", "They arrived at the mountain"),
            ("cabin", "She cleaned the mountain"),
            ("cabin", "He built the mountain"),
            ("lion", "He warned them about the mountain"),
            ("lion", "They were afraid of the mountain"),
            ("lion", "The camera captured the mountain"),
            ("range", "The cattle grazed across the mountain"),
            ("range", "She photographed the long mountain"),
            ("range", "Geologists surveyed the mountain"),
        ]
    },
}


def run_sentence_prediction(model):
    group_summaries = {}

    for group_name, group in SENTENCE_GROUPS.items():
        majority = group["majority"]
        minorities = group["minorities"]
        sentences = group["sentences"]

        majority_id = get_token_ids(model, majority)
        if len(majority_id) != 1:
            print(f"Skipping {group_name}: majority '{majority}' is multi-token")
            continue
        majority_id = majority_id[0]


        print(f"\n{'='*70}")
        print(f"GROUP: {group_name} | majority: '{majority}' | minorities: {minorities}")
        print(f"{'='*70}")

        majority_predicted = 0
        majority_beats_intended = 0
        total = 0

        for intended_minority, prompt in sentences:
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                logits = model(tokens)
            probs = F.softmax(logits[0, -1], dim=-1)

            majority_prob = probs[majority_id].item()
            majority_rank = (probs > probs[majority_id]).sum().item() + 1
            majority_wins = majority_rank == 1

            top1_id = torch.argmax(probs).item()
            top1_word = model.to_string([top1_id]).strip()
            top1_prob = probs[top1_id].item()

            # check if intended minority is single token and get its rank
            intended_id_list = get_token_ids(model, intended_minority)
            if len(intended_id_list) == 1:
                intended_prob = probs[intended_id_list[0]].item()
                intended_rank = (probs > probs[intended_id_list[0]]).sum().item() + 1
                maj_beats_intended = majority_prob > intended_prob
            else:
                intended_prob = None
                intended_rank = None
                maj_beats_intended = False

            if majority_wins:
                majority_predicted += 1
            if maj_beats_intended:
                majority_beats_intended += 1
            total += 1

            intended_rank_str = f"rank=#{intended_rank}" if intended_rank is not None else "multi-token"
            print(f"  [{intended_minority}] '{prompt}'")
            print(f"  predicted: '{top1_word}' (p={top1_prob:.4f}) | "
                  f"majority '{majority}': p={majority_prob:.4f} rank=#{majority_rank} | "
                  f"intended '{intended_minority}': p={intended_prob:.4f} {intended_rank_str} | "
                  f"maj>intended: {maj_beats_intended}")

        pct_predicted = round(100 * majority_predicted / total, 1)
        pct_beats_intended = round(100 * majority_beats_intended / total, 1)

        group_summaries[group_name] = {
            "majority": majority,
            "majority_predicted": majority_predicted,
            "majority_beats_intended": majority_beats_intended,
            "total": total,
            "pct_predicted": pct_predicted,
            "pct_beats_intended": pct_beats_intended,
        }
        print(f"\n  SUMMARY: majority '{majority}' | "
              f"predicted: {majority_predicted}/{total} ({pct_predicted}%) | "
              f"beats intended: {majority_beats_intended}/{total} ({pct_beats_intended}%)")

    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    for group_name, s in group_summaries.items():
        print(f"  {group_name:12s} | majority: '{s['majority']:10s}' | "
              f"predicted: {s['majority_predicted']}/{s['total']} ({s['pct_predicted']}%) | "
              f"beats intended: {s['majority_beats_intended']}/{s['total']} ({s['pct_beats_intended']}%)")


if __name__ == "__main__":
    model = load_model()
    scored = score_compounds(model)
    results = experiment_majority_minority_cosine(model, scored)
    plot_cosine_vs_association_all_layers(results)
    sae_results = experiment_sae_vs_association(model, scored)
    plot_sae_vs_association_strength(sae_results)
    run_sentence_prediction(model)
