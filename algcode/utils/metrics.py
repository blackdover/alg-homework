import numpy as np
import pandas as pd
from typing import Dict, Optional


def _ensure_datetime(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    return pd.to_datetime(s, errors="coerce", utc=True)


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def calculate_sed_metrics(original_df: pd.DataFrame,
                         compressed_df: pd.DataFrame,
                         time_col: str = "BaseDateTime",
                         lat_col: str = "LAT",
                         lon_col: str = "LON",
                         idx_col: str = "orig_idx") -> Dict[str, float]:
    from .geo_utils import GeoUtils

    if len(compressed_df) <= 1 or len(original_df) <= 1:
        return {"mean": 0.0, "max": 0.0, "p95": 0.0}

    if idx_col not in compressed_df.columns:
        raise ValueError(f"compressed_df 缺少 {idx_col}，请让算法输出保留原始索引（orig_idx）")

    o = original_df.reset_index(drop=True).copy()
    o[time_col] = _ensure_datetime(o[time_col])

    c = compressed_df.sort_values(idx_col).reset_index(drop=True).copy()
    c[time_col] = _ensure_datetime(c[time_col])

    o = o.dropna(subset=[time_col, lat_col, lon_col]).reset_index(drop=True)
    c = c.dropna(subset=[time_col, lat_col, lon_col, idx_col]).reset_index(drop=True)

    if len(c) <= 1 or len(o) <= 1:
        return {"mean": 0.0, "max": 0.0, "p95": 0.0}

    sed_values = []

    for i in range(len(c) - 1):
        s = c.iloc[i]
        e = c.iloc[i + 1]

        si = int(s[idx_col])
        ei = int(e[idx_col])
        if ei <= si:
            continue

        t0 = s[time_col]
        t1 = e[time_col]
        if pd.isna(t0) or pd.isna(t1):
            continue

        denom = (t1 - t0).total_seconds()
        if denom <= 0:
            continue

        lat0, lon0 = float(s[lat_col]), float(s[lon_col])
        lat1, lon1 = float(e[lat_col]), float(e[lon_col])

        seg = o.iloc[si + 1: ei].copy()
        if seg.empty:
            continue

        dt = (seg[time_col] - t0).dt.total_seconds().to_numpy(dtype=float)
        alpha = dt / denom
        mask = (alpha >= 0.0) & (alpha <= 1.0)
        if not np.any(mask):
            continue

        alpha = alpha[mask]
        lat_actual = seg.loc[seg.index[mask], lat_col].to_numpy(dtype=float)
        lon_actual = seg.loc[seg.index[mask], lon_col].to_numpy(dtype=float)

        lat_hat = lat0 + alpha * (lat1 - lat0)
        lon_hat = lon0 + alpha * (lon1 - lon0)

        for la, lo, lh, lnh in zip(lat_actual, lon_actual, lat_hat, lon_hat):
            sed = GeoUtils.haversine_distance(la, lo, lh, lnh)
            sed_values.append(sed)

    if not sed_values:
        return {"mean": 0.0, "max": 0.0, "p95": 0.0}

    arr = np.asarray(sed_values, dtype=float)
    return {"mean": float(arr.mean()), "max": float(arr.max()), "p95": float(np.percentile(arr, 95))}


def calculate_navigation_event_recall(original_df: pd.DataFrame,
                                    compressed_df: pd.DataFrame,
                                    cog_threshold: float = 20.0,
                                    match_window_s: int = 30) -> float:
    if len(original_df) < 2:
        return 1.0

    time_col = "BaseDateTime"
    o = original_df.copy()
    c = compressed_df.copy()
    o[time_col] = _ensure_datetime(o[time_col])
    c[time_col] = _ensure_datetime(c[time_col])

    def find_turn_times(df: pd.DataFrame):
        times = []
        for i in range(1, len(df)):
            a = df.iloc[i - 1].get('COG')
            b = df.iloc[i].get('COG')
            if pd.isna(a) or pd.isna(b):
                continue
            try:
                da = float(a)
                db = float(b)
            except Exception:
                continue
            if _angle_diff_deg(da, db) > cog_threshold:
                t = df.iloc[i][time_col]
                if pd.isna(t):
                    continue
                times.append(t)
        return times

    orig_times = find_turn_times(o)
    if len(orig_times) == 0:
        return 1.0
    comp_times = find_turn_times(c)

    matched = 0
    for ot in orig_times:
        found = False
        for ct in comp_times:
            if abs((ct - ot).total_seconds()) <= match_window_s:
                found = True
                break
        if found:
            matched += 1
    return matched / len(orig_times)


def evaluate_compression(original_df: pd.DataFrame,
                        compressed_df: pd.DataFrame,
                        algorithm_name: str,
                        elapsed_time: float = None,
                        calculate_sed: bool = True) -> dict:
    N = len(original_df)
    M = len(compressed_df)
    compression_ratio = (1 - M / N) * 100 if N > 0 else 0.0
    if calculate_sed:
        sed_metrics = calculate_sed_metrics(original_df, compressed_df)
    else:
        sed_metrics = {'mean': 0.0, 'max': 0.0, 'p95': 0.0}
    event_recall = calculate_navigation_event_recall(original_df, compressed_df)

    similarity_score = calculate_trajectory_similarity(original_df, compressed_df, sed_metrics)
    print(f"\n{'='*60}")
    print(f"算法: {algorithm_name}")
    print(f"{'='*60}")
    print(f"原始点数: {N}")
    print(f"压缩后点数: {M}")
    print(f"压缩率: {compression_ratio:.2f}%")
    if calculate_sed:
        print(f"SED均值: {sed_metrics['mean']:.2f} 米")
        print(f"SED最大值: {sed_metrics['max']:.2f} 米")
        print(f"SED 95分位数: {sed_metrics['p95']:.2f} 米")
    print(f"航行事件保留率: {event_recall:.3f}")
    print(f"轨迹相似度: {similarity_score:.3f}")
    if elapsed_time is not None:
        print(f"运行时间: {elapsed_time:.4f} 秒")
    print(f"{'='*60}\n")
    result = {
        'algorithm': algorithm_name,
        'original_points': N,
        'compressed_points': M,
        'compression_ratio': compression_ratio,
        'event_recall': event_recall,
        'trajectory_similarity': similarity_score,
        'elapsed_time': elapsed_time
    }
    if calculate_sed:
        result.update({
            'sed_mean': sed_metrics['mean'],
            'sed_max': sed_metrics['max'],
            'sed_p95': sed_metrics['p95']
        })
    return result


def calculate_trajectory_similarity(original_df: pd.DataFrame,
                                  compressed_df: pd.DataFrame,
                                  sed_metrics: Dict[str, float]) -> float:
    from .geo_utils import GeoUtils

    if len(original_df) <= 1 or len(compressed_df) <= 1:
        return 0.0

    total_distance = 0.0
    for i in range(1, len(original_df)):
        total_distance += GeoUtils.haversine_distance(
            float(original_df.iloc[i - 1]['LAT']), float(original_df.iloc[i - 1]['LON']),
            float(original_df.iloc[i]['LAT']), float(original_df.iloc[i]['LON'])
        )
    total_distance = max(total_distance, 1e-6)

    normalized_sed = sed_metrics.get('mean', 0.0) / total_distance

    keep_ratio = len(compressed_df) / max(len(original_df), 1)
    compression_penalty = 0.0
    if keep_ratio < 0.05:
        compression_penalty = (0.05 - keep_ratio) * 2.0

    score = 1.0 / (1.0 + normalized_sed + compression_penalty)
    return float(np.clip(score, 0.0, 1.0))


