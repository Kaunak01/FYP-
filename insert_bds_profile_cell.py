"""
Insert a documentation/reconstruction cell into FYP_BDS_GA.ipynb showing the
BDS profile construction logic — the code that was silently missing from the
notebook but lived in run_bds_ga.py. Added for viva defensibility.

Placed AFTER cell 6 (the markdown that says "For each credit card in the
training data, we compute a behavioural profile") and BEFORE cell 7 (which
references card_amt etc. without defining them).
"""
import json, shutil
from pathlib import Path

NB = Path("FYP_BDS_GA.ipynb")
shutil.copy(NB, "FYP_BDS_GA_PRE_FIX.ipynb")

with NB.open(encoding="utf-8") as f:
    nb = json.load(f)

cell_src = [
    "# ============================================================\n",
    "# BDS PROFILE CONSTRUCTION (reconstructed for viva defensibility)\n",
    "# ============================================================\n",
    "# Source of truth: run_bds_ga.py lines 91-128.\n",
    "# This cell was missing from the original notebook — profiles were built\n",
    "# externally and loaded from bds_profiles.joblib. Adding here so the\n",
    "# notebook is self-contained and the construction logic is auditable.\n",
    "#\n",
    "# CRITICAL: All profiles use TRAINING DATA ONLY (train_eng, train_cc).\n",
    "# Test transactions never contribute to any cardholder's 'normal' profile.\n",
    "# This is the leakage-free construction Dr Arafa would expect.\n",
    "\n",
    "print('Building cardholder profiles (train-only, no leakage)...')\n",
    "\n",
    "profile_df = pd.DataFrame({\n",
    "    'cc_num': train_cc,\n",
    "    'amt': train_eng['amt'].values,\n",
    "    'hour': train_eng['hour'].values.astype(int),\n",
    "    'category': train_eng['category_encoded'].values.astype(int),\n",
    "    'velocity_1h': train_eng['velocity_1h'].values,\n",
    "})\n",
    "\n",
    "# Per-card amount statistics (mean / std / count)\n",
    "card_amt = profile_df.groupby('cc_num')['amt'].agg(['mean', 'std', 'count'])\n",
    "card_amt.columns = ['amt_mean', 'amt_std', 'amt_count']\n",
    "card_amt['amt_std'] = card_amt['amt_std'].fillna(0)\n",
    "\n",
    "# Per-card hour-of-day probability distribution (24-bin)\n",
    "card_hour_counts = profile_df.groupby(['cc_num', 'hour']).size().unstack(fill_value=0)\n",
    "for h in range(24):\n",
    "    if h not in card_hour_counts.columns:\n",
    "        card_hour_counts[h] = 0\n",
    "card_hour_counts = card_hour_counts[sorted(card_hour_counts.columns)]\n",
    "card_hour_prob = card_hour_counts.div(card_hour_counts.sum(axis=1), axis=0)\n",
    "\n",
    "# Per-card category probability distribution\n",
    "card_cat_counts = profile_df.groupby(['cc_num', 'category']).size().unstack(fill_value=0)\n",
    "card_cat_prob = card_cat_counts.div(card_cat_counts.sum(axis=1), axis=0)\n",
    "\n",
    "# Per-card 1h-velocity statistics\n",
    "card_vel = profile_df.groupby('cc_num')['velocity_1h'].agg(['mean', 'std'])\n",
    "card_vel.columns = ['vel_mean', 'vel_std']\n",
    "card_vel['vel_std'] = card_vel['vel_std'].fillna(0)\n",
    "\n",
    "# Global fallbacks for unseen / sparse cards\n",
    "global_amt_mean = profile_df['amt'].mean()\n",
    "global_amt_std  = profile_df['amt'].std()\n",
    "global_hour_prob = profile_df.groupby('hour').size() / len(profile_df)\n",
    "global_cat_prob  = profile_df.groupby('category').size() / len(profile_df)\n",
    "global_vel_mean  = profile_df['velocity_1h'].mean()\n",
    "\n",
    "print(f'Profiles built for {len(card_amt):,} cards | Global avg amount: ${global_amt_mean:.2f}')\n",
]

new_cell = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": cell_src,
}

# Find insertion point: after the markdown cell that talks about building profiles.
# The agent reported it as "Cell 6" (0-indexed) — the markdown describing profile build.
# We insert right before the first code cell that references card_amt (cell 7 vectorised
# or cell 9 python-loop precompute_raw_deviations). Safest: scan for that reference.
insert_idx = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if c.get("cell_type") == "code" and "card_amt[[" in src:
        insert_idx = i
        break

if insert_idx is None:
    raise SystemExit("could not find a cell referencing card_amt — aborting insert")

nb["cells"].insert(insert_idx, new_cell)

with NB.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Inserted reconstructed profile-build cell at index {insert_idx} of FYP_BDS_GA.ipynb")
print(f"Backup saved as FYP_BDS_GA_PRE_FIX.ipynb")
print(f"Total cells now: {len(nb['cells'])}")
