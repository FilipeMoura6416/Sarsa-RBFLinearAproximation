"""Boxplot das 5 melhores configurações de hiperparâmetros.

Lê o CSV gerado por run.py, agrupa por (learning_rate, n_centers) e ordena as
configurações pela mediana de passos até 10 episódios de sucesso. Execuções que
não chegaram a 10 sucessos dentro do orçamento contam como `total_steps`
(valor censurado, ou seja, uma penalidade) e o número delas aparece no rótulo.

Uso:
    python plot.py
    python plot.py --csv results/results.csv --out plots/boxplot_top5.png --top 5
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    """{(lr, n_centers): [steps_to_success, ...]} e nº de falhas por config."""
    groups = defaultdict(list)
    failures = defaultdict(int)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (float(r["learning_rate"]), int(r["n_centers"]))
            if int(r["reached"]):
                groups[key].append(float(r["steps_to_success"]))
            else:
                groups[key].append(float(r["total_steps"]))  # censurado
                failures[key] += 1
    return groups, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/results.csv")
    parser.add_argument("--out", default="plots/boxplot_top5.png")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    groups, failures = load_results(args.csv)
    # Ordena por mediana; empate desfeito pela média e depois pelo nº de falhas
    ranking = sorted(groups, key=lambda k: (np.median(groups[k]), np.mean(groups[k]), failures[k]))

    print(f"{'lr':>7} {'centers':>7} {'n':>3} {'mediana':>9} {'média':>9} {'desvio':>8} {'falhas':>6}")
    for k in ranking:
        v = np.array(groups[k])
        print(f"{k[0]:>7} {k[1]:>7} {len(v):>3} {np.median(v):>9.0f} {v.mean():>9.0f} "
              f"{v.std():>8.0f} {failures[k]:>6}")

    top = ranking[:args.top]
    data = [groups[k] for k in top]
    labels = [f"α={lr}\n{nc} centros" + (f"\n({failures[(lr, nc)]} falhas)" if failures[(lr, nc)] else "")
              for lr, nc in top]

    color = "#2a78d6"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.boxplot(
        data, widths=0.5, patch_artist=True,
        boxprops=dict(facecolor=color + "40", edgecolor=color, linewidth=1.5),
        medianprops=dict(color=color, linewidth=2.5),
        whiskerprops=dict(color=color, linewidth=1.5),
        capprops=dict(color=color, linewidth=1.5),
        flierprops=dict(marker="o", markersize=0),  # pontos individuais desenhados abaixo
    )
    ax.set_xticks(range(1, len(labels) + 1), labels)
    # Cada seed como um ponto (jitter horizontal) para mostrar a distribuição real
    rng = np.random.default_rng(0)
    for i, v in enumerate(data, 1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=22, color=color,
                   edgecolor="white", linewidth=0.8, zorder=3)

    n_seeds = max(len(v) for v in data)
    ax.set_title(f"SARSA linear (RBF) no MountainCar — {len(top)} melhores configurações "
                 f"({n_seeds} seeds cada)", loc="left", fontsize=12)
    ax.set_ylabel("Passos de ambiente até 10 episódios de sucesso")
    ax.set_xlabel("Configuração (ordenada pela mediana)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nImagem salva em {args.out}")


if __name__ == "__main__":
    main()
