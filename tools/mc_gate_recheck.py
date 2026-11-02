import sys
import struct
import hashlib
import gzip
import json
import math

def main():
    REPLAY = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qgate/C17/replay/dataset_0.bin"
    DECISIONS = "/tmp/qgate/C17/decisions.jsonl.gz"

    T = 4
    N = 126
    kP = 50.4
    kScale = math.sqrt(50.4)
    q = 0.05

    # Read replay file
    with open(REPLAY, "rb") as f:
        b = f.read()

    total_len = len(b)
    n_bytes = total_len - 32
    n = n_bytes // 8

    # Verify SHA-256
    data_bytes = b[:-32]
    tail_hash = b[-32:]
    computed_hash = hashlib.sha256(data_bytes).digest()
    sha_ok = 1 if computed_hash == tail_hash else 0

    # Unpack doubles
    data = struct.unpack("<%dd" % n, data_bytes)

    # Build strategy returns: strategy i at time t is data[t*N + i]
    # returns[i][t] = data[t*N + i]
    returns = [[0.0] * T for _ in range(N)]
    for t in range(T):
        for i in range(N):
            returns[i][t] = data[t * N + i]

    # 1) Annualized SR for each strategy
    # mu_i = mean(x_i)
    # var_i = sum((x - mu)^2) / (T-1)
    # sd_i = sqrt(var_i)
    # sr_i = (mu_i / sd_i) * kScale
    mu = [0.0] * N
    sd = [0.0] * N
    sr = [0.0] * N

    for i in range(N):
        s = 0.0
        for t in range(T):
            s += returns[i][t]
        mu[i] = s / T

        var_sum = 0.0
        for t in range(T):
            d = returns[i][t] - mu[i]
            var_sum += d * d
        var_i = var_sum / (T - 1)
        sd[i] = math.sqrt(var_i)

        if sd[i] > 0:
            sr[i] = (mu[i] / sd[i]) * kScale
        else:
            sr[i] = 0.0

    # 2) Average absolute correlation m
    # For each pair a < b:
    #   cov = sum_t (x_a[t]-mu_a)(x_b[t]-mu_b)
    #   corr = cov / ((T-1) * sd_a * sd_b), clamp to [-1, 1]
    #   accumulate |corr|
    # m = total / num_pairs
    num_pairs = N * (N - 1) // 2
    abs_corr_sum = 0.0

    for a in range(N):
        for b in range(a + 1, N):
            cov = 0.0
            for t in range(T):
                cov += (returns[a][t] - mu[a]) * (returns[b][t] - mu[b])
            denom = (T - 1) * sd[a] * sd[b]
            if denom > 0:
                corr = cov / denom
            else:
                corr = 0.0
            # Clamp to [-1, 1]
            if corr > 1.0:
                corr = 1.0
            elif corr < -1.0:
                corr = -1.0
            abs_corr_sum += abs(corr)

    m = abs_corr_sum / num_pairs

    # 3) T_eff = clamp(T / (1 + (T-1)*m), 1, T)
    teff_raw = T / (1.0 + (T - 1) * m)
    if teff_raw < 1.0:
        teff = 1.0
    elif teff_raw > T:
        teff = float(T)
    else:
        teff = teff_raw

    # 4) Normalized quantile: Phi(x) = 0.5 * erfc(-x / sqrt(2))
    # Use bisection on [-40, 40] with 200 iterations to find x such that Phi(x) = u
    def phi(x):
        # 0.5 * erfc(-x / sqrt(2))
        # erfc(z) = 1 - erf(z)
        # We need erfc(-x/sqrt(2))
        # math.erfc is available in Python 3.2+
        return 0.5 * math.erfc(-x / math.sqrt(2.0))

    def ppf(u):
        # Find x such that Phi(x) = u via bisection on [-40, 40]
        lo = -40.0
        hi = 40.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if phi(mid) < u:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    ppf_095 = ppf(0.95)
    ppf_1mq = ppf(1.0 - q / T)

    # 5) DSR threshold and BH threshold
    # DSR = kScale * ( sqrt(2*ln(T_eff)/N) + ppf(0.95)/sqrt(N-1) )
    dsr_thr = kScale * (math.sqrt(2.0 * math.log(teff) / N) + ppf_095 / math.sqrt(N - 1))

    # BH = ppf(1 - q/T) / sqrt(N / kP)
    bh_thr = ppf_1mq / math.sqrt(N / kP)

    thr = max(dsr_thr, bh_thr)

    # 6) maxidx = argmax(sr) (smallest index on tie); maxsr = sr[maxidx]
    # above = count of sr_i > thr
    maxidx = 0
    maxsr = sr[0]
    for i in range(1, N):
        if sr[i] > maxsr:
            maxsr = sr[i]
            maxidx = i

    above = 0
    for i in range(N):
        if sr[i] > thr:
            above += 1

    # Read decisions
    selected = 0
    index = -1
    with gzip.open(DECISIONS, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("cell_id") == 1001 and rec.get("rep_id") == 0 and rec.get("rule_id") == "GATE_Full":
                selected = rec.get("selected", 0)
                index = rec.get("index", -1)
                break

    # Output
    print("RECHECK n=%d sha_ok=%d" % (n, sha_ok))
    print("RECHECK m=%.12f" % m)
    print("RECHECK teff=%.12f" % teff)
    print("RECHECK thr=%.9f" % thr)
    print("RECHECK maxidx=%d maxsr=%.6f" % (maxidx, maxsr))
    print("RECHECK above=%d" % above)
    print("MATCH 1001 0 GATE_Full %d %d" % (selected, index))
    print("RECHECK_DONE")


if __name__ == "__main__":
    main()
