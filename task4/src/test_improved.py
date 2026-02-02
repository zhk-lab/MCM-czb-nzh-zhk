"""
快速测试：用当前最优参数 + 改进规则跑完整 robustness
"""
from pathlib import Path
import sys
import json

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "task4" / "src"))

from robustness_testing import compute_flip_rate_two_key, run_adversarial_attack_test, compute_combined_robustness
from two_key_system import load_all_data, create_weekly_panel

# 测试参数（当前最优 + 新参数）
test_params = {
    'alpha': 0.4,
    'beta': 0.6,
    'ban_consecutive_weeks': 2,
    'save_eligibility_threshold': 0.20,
    'judge_weight': 1.2,
}

print("Testing improved TWO_KEY with params:", test_params)

# 加载数据
fan_df, judge_df, data_df = load_all_data(str(repo_root))
panel = create_weekly_panel(fan_df, judge_df)

# Flip-rate 测试
print("\n[1/2] Flip-rate test...")
flip_result = compute_flip_rate_two_key(
    panel,
    perturbation_levels=[0.01, 0.03, 0.05],
    n_trials=30,
    seed=42,
    params=test_params
)

# 对抗攻击测试
print("\n[2/2] Adversarial attack test...")
attack_result = run_adversarial_attack_test(panel, seed=42, params=test_params)

# 综合
combined = compute_combined_robustness(flip_result, attack_result, weight_flip=0.6, weight_attack=0.4)

print(f"\n\nFINAL ROBUSTNESS: {combined:.3f}")
print(f"  Flip-rate:  {flip_result['robustness_score']:.3f}")
print(f"  Attack:     {attack_result['robustness_score']:.3f}")

# 保存
output = {
    'params': test_params,
    'flip_rate': flip_result['robustness_score'],
    'attack': attack_result['robustness_score'],
    'combined': combined,
}

output_path = repo_root / "task4" / "table" / "test_improved.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nSaved to: {output_path}")
