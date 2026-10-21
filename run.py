"""Experimento de hiperparâmetros: SARSA linear + RBF no MountainCar-v0.

Para cada combinação (learning_rate, n_centers) roda N seeds independentes de
`total_steps` passos e registra quantos passos de ambiente foram necessários
até o agente completar 10 episódios alcançando o objetivo.

Os resultados brutos são gravados em CSV (uma linha por execução) logo após
cada execução terminar, então o script pode ser interrompido e retomado:
execuções já presentes no CSV são puladas.

Uso:
    python run.py                     # grade completa (25 configs x 10 seeds)
    python run.py --workers 4         # paraleliza em 4 processos
    python run.py --total-steps 2000 --seeds 2 --out results/teste.csv  # teste rápido
"""

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import gymnasium as gym
import numpy as np

from feature_extractors import RBFFeatureExtractor
from sarsa_linear_agent import LinearSarsaAgent

LR_LIST = [0.005, 0.01, 0.02, 0.05, 0.1]
N_CENTERS_LIST = [9, 16, 25, 36, 49]
TOTAL_STEPS = 50_000
N_SEEDS = 10
N_SUCCESSES = 10  # episódios alcançando o objetivo

# Parâmetros mantidos fixos em todas as configurações
SIGMA = 0.2
EPSILON = 1.0
EPSILON_DECAY = 0.99  # 0.999 por episódio deixa ε≈0.78 após 50k passos (~250 episódios) e o agente quase nunca chega ao objetivo
EPSILON_MIN = 0.01
DISCOUNT_FACTOR = 1.0

FIELDS = [
    "learning_rate", "n_centers", "seed", "steps_to_success", "reached",
    "n_successes", "n_episodes", "total_steps", "sigma", "epsilon_decay",
    "discount_factor", "runtime_s",
]


def run_trial(learning_rate, n_centers, seed, total_steps):
    """Executa uma única execução e devolve um dicionário com o resultado."""
    t0 = time.time()
    np.random.seed(seed)  # act() usa o RNG global do numpy
    env = gym.make("MountainCar-v0")
    feature_extractor = RBFFeatureExtractor(env=env, n_centers=n_centers, sigma=SIGMA)
    agent = LinearSarsaAgent(
        env=env,
        feature_extractor=feature_extractor,
        learning_rate=learning_rate,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN,
        discount_factor=DISCOUNT_FACTOR,
    )
    agent.train(total_steps=total_steps, seed=seed, verbose=False)
    env.close()

    successes = agent.success_steps
    reached = len(successes) >= N_SUCCESSES
    return {
        "learning_rate": learning_rate,
        "n_centers": n_centers,
        "seed": seed,
        # Vazio (NaN) quando não chegou a 10 sucessos dentro do orçamento
        "steps_to_success": successes[N_SUCCESSES - 1] if reached else "",
        "reached": int(reached),
        "n_successes": len(successes),
        "n_episodes": len(agent.episode_lengths),
        "total_steps": total_steps,
        "sigma": SIGMA,
        "epsilon_decay": EPSILON_DECAY,
        "discount_factor": DISCOUNT_FACTOR,
        "runtime_s": round(time.time() - t0, 2),
    }


def load_done(path, total_steps):
    """Chaves (lr, n_centers, seed) já salvas no CSV, para retomar."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {
            (float(r["learning_rate"]), int(r["n_centers"]), int(r["seed"]))
            for r in csv.DictReader(f)
            if int(r["total_steps"]) == total_steps
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="número de seeds por configuração")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", default="results/results.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    seeds = list(range(args.seeds))
    done = load_done(args.out, args.total_steps)
    jobs = [(lr, nc, s) for lr, nc, s in product(LR_LIST, N_CENTERS_LIST, seeds)
            if (lr, nc, s) not in done]
    print(f"{len(done)} execuções já salvas, {len(jobs)} restantes "
          f"({args.workers} processos) -> {args.out}")
    if not jobs:
        return

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="") as f, ProcessPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        futures = {pool.submit(run_trial, lr, nc, s, args.total_steps): (lr, nc, s)
                   for lr, nc, s in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            writer.writerow(row)
            f.flush()  # salva incrementalmente
            sts = row["steps_to_success"] if row["reached"] else "não atingiu"
            print(f"[{i}/{len(jobs)}] lr={row['learning_rate']} centers={row['n_centers']} "
                  f"seed={row['seed']}: {sts} ({row['n_successes']} sucessos, {row['runtime_s']}s)")


if __name__ == "__main__":
    main()
