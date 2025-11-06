#!/usr/bin/env python
"""Quick script to check immutable_id prefixes in LeMat-Bulk"""

from collections import Counter

from datasets import load_dataset

print("Loading LeMat-Bulk dataset (streaming)...")
dataset = load_dataset(
    "LeMaterial/LeMat-Bulk",
    "compatible_pbe",
    split="train",
    streaming=True
)

print("Sampling 10,000 structures to check prefixes...")
prefixes = []
sample_ids = []

for i, item in enumerate(dataset):
    if i >= 10000:
        break
    
    immutable_id = item.get('immutable_id', '')
    if immutable_id:
        # Extract prefix (everything before first number or up to first dash + word)
        if '-' in immutable_id:
            prefix = immutable_id.split('-')[0] + '-'
        else:
            prefix = immutable_id[:10]  # First 10 chars
        
        prefixes.append(prefix)
        
        # Store some sample IDs for each prefix type
        if len(sample_ids) < 50:
            sample_ids.append(immutable_id)

print("\n" + "="*80)
print("IMMUTABLE_ID PREFIX ANALYSIS")
print("="*80)

# Count prefixes
prefix_counts = Counter(prefixes)
print("\nTop 20 most common prefixes:")
for prefix, count in prefix_counts.most_common(20):
    percentage = (count / len(prefixes)) * 100
    print(f"  {prefix:20s}: {count:6d} ({percentage:5.2f}%)")

print("\n" + "="*80)
print("SAMPLE IMMUTABLE_IDs:")
print("="*80)
for i, sample_id in enumerate(sample_ids[:30]):
    print(f"  {i+1:2d}. {sample_id}")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print(f"Total samples checked: {len(prefixes)}")
print(f"Unique prefixes found: {len(prefix_counts)}")

# Try to identify database sources
mp_count = sum(1 for p in prefixes if p.startswith('mp-'))
oqmd_count = sum(1 for p in prefixes if p.startswith('oqmd-'))
jvasp_count = sum(1 for p in prefixes if p.startswith('jvasp-'))
agm_count = sum(1 for p in prefixes if p.startswith('agm-'))
alex_count = sum(1 for p in prefixes if 'alex' in p.lower())

print(f="\nIdentified sources:")
print(f"  Materials Project (mp-):     {mp_count:6d} ({mp_count/len(prefixes)*100:5.2f}%)")
print(f"  OQMD (oqmd-):                {oqmd_count:6d} ({oqmd_count/len(prefixes)*100:5.2f}%)")
print(f"  JARVIS (jvasp-):             {jvasp_count:6d} ({jvasp_count/len(prefixes)*100:5.2f}%)")
print(f"  AGM (agm-):                  {agm_count:6d} ({agm_count/len(prefixes)*100:5.2f}%)")
print(f"  Alexandria (alex*):          {alex_count:6d} ({alex_count/len(prefixes)*100:5.2f}%)")

print("\n✓ Analysis complete!")

