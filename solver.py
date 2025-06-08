from ortools.sat.python import cp_model

# ------------------  data  ------------------
# mail
# clusters = {
# 25:	559764,
# 26:	376586,
# 28:	260232,
# 30:	1802,
# 35:	57932,
# 38:	186900,
# 40:	1229618,
# 45:	253992,
# 75:	1160106,
# }
#
# web 
clusters = {
25:	68639,
26:	9841,
28:	340135,
30:	102260,
35:	363,
38:	32611,
40:	885865,
45:	224515,
75:	606504,
}


C = list(clusters.keys())
S = range(5)           # FS1 … FS8
K = [1, 2, 3, 4]       # at most four shards/cluster?
L, H = 400_000, 550_000

model = cp_model.CpModel()
z = {(c,k): model.NewBoolVar(f'z_{c}_{k}') for c in C for k in K}
x = {(c,s,k): model.NewBoolVar(f'x_{c}_{s}_{k}')
     for c in C for s in S for k in K}

# 1. choose exactly one k per cluster
for c in C:
    model.Add(sum(z[c,k] for k in K) == 1)

# 2. put k shards on k distinct stripes
for c in C:
    for k in K:
        model.Add(sum(x[c,s,k] for s in S) == k * z[c,k])

# 3. shard existence implication
for c in C:
    for s in S:
        for k in K:
            model.Add(x[c,s,k] <= z[c,k])

# 4. stripe capacities
for s in S:
    load = sum(int(clusters[c]/k) * x[c,s,k] for c in C for k in K)
    model.Add(load >= L)
    model.Add(load <= H)

# 5. objective (feel free to adjust weights)
clusters_per_stripe = model.NewIntVar(0, len(C)*len(S), 'cps')
model.Add(clusters_per_stripe ==
          sum(x[c,s,k] for c in C for s in S for k in K))

stripes_per_cluster = model.NewIntVar(0, len(C)*max(K), 'spc')
model.Add(stripes_per_cluster ==
          sum(k * z[c,k] for c in C for k in K))

model.Minimize(10*clusters_per_stripe + stripes_per_cluster)

# 6. solve
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60
result = solver.Solve(model)

if result in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    for s in S:
        shards = [(c,k) for c in C for k in K
                  if solver.Value(x[c,s,k])]
        tally = sum(int(clusters[c]/k) for c,k in shards)
        print(f'FS{s+1}: {tally:,} users <-',
              ', '.join(f'cluster{c}/k={k}' for c,k in shards))
else:
    print("No feasible layout with those bounds and K values")

