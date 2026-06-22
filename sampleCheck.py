import json

with open('sample_candidates.json') as f:
    data = json.load(f)

# duplicate to get 100+
extended = (data * 3)[:100]

# give unique IDs to avoid duplicates
for i, c in enumerate(extended):
    c = dict(c)
    c['candidate_id'] = f"CAND_{i+1:07d}"
    extended[i] = c

with open('sample_100.jsonl', 'w') as f:
    for c in extended:
        f.write(json.dumps(c) + '\n')

print(f"Created sample_100.jsonl with {len(extended)} candidates")