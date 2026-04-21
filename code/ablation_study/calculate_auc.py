import os
import re
import numpy as np
from scipy import stats
from typing import Dict, List

EPS_PATTERN = re.compile(r"Epsilon = ([0-9.]+)")
ENS_ACC_PATTERN = re.compile(r"Ensemble accuracy:\s+([0-9.]+)%")

class RobustnessAUCAnalyzer:
    def __init__(self, surrogate_path: str, deepens_path: str, random_path: str):
        self.paths = {
            "Surrogate": surrogate_path,
            "DeepEns": deepens_path,
            "RandomSearch": random_path
        }

    def _parse_file(self, filepath: str) -> Dict[str, Dict[str, List[float]]]:
        """
        Returns structure:
        attack -> { eps: acc }
        """
        with open(filepath, 'r') as f:
            text = f.read()

        blocks = re.split(r"=+\nADVERSARIAL ATTACK RESULTS \((.*?)\)\n=+", text)
        results = {}

        # blocks format: [header, ATTACK1, block1, ATTACK2, block2, ...]
        for i in range(1, len(blocks), 2):
            attack = blocks[i].strip()
            block = blocks[i+1]
            eps = []
            acc = []
            for eps_match in EPS_PATTERN.finditer(block):
                e = float(eps_match.group(1))
                start = eps_match.end()
                sub = block[start:start+200]
                acc_match = ENS_ACC_PATTERN.search(sub)
                if acc_match:
                    a = float(acc_match.group(1)) / 100.0
                    eps.append(e)
                    acc.append(a)
            if eps:
                order = np.argsort(eps)
                eps = np.array(eps)[order]
                acc = np.array(acc)[order]
                results[attack] = {"eps": eps, "acc": acc}
        return results

    def _auc(self, eps: np.ndarray, acc: np.ndarray) -> float:
        return np.trapz(acc, eps)

    def load_all(self):
        data = {}
        for name, path in self.paths.items():
            aucs = {}
            for file in os.listdir(path):
                if not file.endswith('.txt'): continue
                parsed = self._parse_file(os.path.join(path, file))
                for attack, vals in parsed.items():
                    auc = self._auc(vals['eps'], vals['acc'])
                    aucs.setdefault(attack, []).append(auc)
            data[name] = aucs
        return data

    def summarize(self, data):
        summary = {}
        for method, attacks in data.items():
            summary[method] = {}
            for attack, aucs in attacks.items():
                aucs = np.array(aucs)
                mean = aucs.mean()
                std = aucs.std(ddof=1)
                ci95 = 1.96 * std / np.sqrt(len(aucs))
                summary[method][attack] = {
                    "mean": mean,
                    "std": std,
                    "ci95": ci95,
                    "values": aucs
                }
        return summary

    def t_tests(self, summary):
        methods = list(summary.keys())
        attacks = next(iter(summary.values())).keys()
        tests = {}
        for attack in attacks:
            tests[attack] = {}
            for i in range(len(methods)):
                for j in range(i+1, len(methods)):
                    m1, m2 = methods[i], methods[j]
                    v1 = summary[m1][attack]['values']
                    v2 = summary[m2][attack]['values']
                    t, p = stats.ttest_ind(v1, v2, equal_var=False)
                    tests[attack][f"{m1} vs {m2}"] = {"t": t, "p": p}
        return tests

    def sign_test(self, summary):
        """
        Критерий знаков (Sign test) для парных сравнений.
        Проверяет гипотезу о равенстве медиан двух выборок.
        Работает с парными наблюдениями (одинаковое количество).
        """
        methods = list(summary.keys())
        attacks = next(iter(summary.values())).keys()
        tests = {}
        
        for attack in attacks:
            tests[attack] = {}
            for i in range(len(methods)):
                for j in range(i+1, len(methods)):
                    m1, m2 = methods[i], methods[j]
                    v1 = summary[m1][attack]['values']
                    v2 = summary[m2][attack]['values']
                    
                    # Уравниваем длины выборок (берем минимальную)
                    min_len = min(len(v1), len(v2))
                    v1_paired = v1[:min_len]
                    v2_paired = v2[:min_len]
                    
                    # Подсчет знаков разностей
                    differences = v1_paired - v2_paired
                    positive = np.sum(differences > 0)
                    negative = np.sum(differences < 0)
                    total_non_zero = positive + negative
                    
                    if total_non_zero == 0:
                        p_value = 1.0
                    else:
                        # Биномиальный тест: H0: p = 0.5
                        p_value = 2 * min(
                            stats.binom.cdf(min(positive, negative), total_non_zero, 0.5),
                            stats.binom.sf(max(positive, negative) - 1, total_non_zero, 0.5)
                        )
                    
                    tests[attack][f"{m1} vs {m2}"] = {
                        "positive": int(positive),
                        "negative": int(negative),
                        "total_pairs": total_non_zero,
                        "p": p_value
                    }
        return tests

    def wilcoxon_test(self, summary):
        """
        Тест Уилкоксона (Mann-Whitney U test) для независимых выборок.
        Непараметрическая альтернатива t-тесту.
        """
        methods = list(summary.keys())
        attacks = next(iter(summary.values())).keys()
        tests = {}
        
        for attack in attacks:
            tests[attack] = {}
            for i in range(len(methods)):
                for j in range(i+1, len(methods)):
                    m1, m2 = methods[i], methods[j]
                    v1 = summary[m1][attack]['values']
                    v2 = summary[m2][attack]['values']
                    
                    # Mann-Whitney U test (непараметрический)
                    statistic, p_value = stats.mannwhitneyu(
                        v1, v2, 
                        alternative='two-sided'
                    )
                    
                    # Эффект размера (rank-biserial correlation)
                    n1, n2 = len(v1), len(v2)
                    effect_size = 1 - (2 * statistic) / (n1 * n2)
                    
                    tests[attack][f"{m1} vs {m2}"] = {
                        "U": statistic,
                        "p": p_value,
                        "effect_size": effect_size
                    }
        return tests

# surrogate_path="/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/fashionmnist"
# deepens_path="/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/fashionmnist_deepens"
# random_path="/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/fashionmnist_random_search"

# surrogate_path="/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/cifar10",
# deepens_path="/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/cifar10_deepens",
# random_path="/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/cifar10_random_search"

# surrogate_path="/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/cifar100",
# deepens_path="/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/deepens_cifar100",
# random_path="/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/cifar100_random_search"

if __name__ == '__main__':
    analyzer = RobustnessAUCAnalyzer(
        surrogate_path="/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/cifar100",
        deepens_path="/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/cifar100_deepens_earlystop",
        random_path="/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/cifar100_random_search"
    )

    raw = analyzer.load_all()
    summary = analyzer.summarize(raw)
    tests = analyzer.t_tests(summary)
    sign_tests = analyzer.sign_test(summary)
    wilcoxon_tests = analyzer.wilcoxon_test(summary)

    print("\n=== AUC SUMMARY ===")
    for method, attacks in summary.items():
        print(f"\n{method}")
        for attack, stats_ in attacks.items():
            print(f"  {attack}: mean={stats_['mean']:.4f}, std={stats_['std']:.4f}, CI95=±{stats_['ci95']:.4f}")

    print("\n=== T-TESTS (Welch's t-test) ===")
    for attack, pairs in tests.items():
        print(f"\n{attack}")
        for pair, res in pairs.items():
            print(f"  {pair}: t={res['t']:.3f}, p={res['p']:.4e}")

    print("\n=== SIGN TEST ===")
    for attack, pairs in sign_tests.items():
        print(f"\n{attack}")
        for pair, res in pairs.items():
            print(f"  {pair}: positive={res['positive']}, negative={res['negative']}, "
                  f"total_pairs={res['total_pairs']}, p={res['p']:.4e}")

    print("\n=== MANN-WHITNEY U TEST (Wilcoxon rank-sum) ===")
    for attack, pairs in wilcoxon_tests.items():
        print(f"\n{attack}")
        for pair, res in pairs.items():
            print(f"  {pair}: U={res['U']:.1f}, p={res['p']:.4e}, effect_size={res['effect_size']:.3f}")