"""
Nickel Drillspace Analysis Web Application
Resource Geology & Financial Analysis Tool
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import io
import os
from scipy.stats import gaussian_kde

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

COG = 1.25  # Cut-off grade %Ni

# ─── Helpers ──────────────────────────────────────────────────────────────────

def classify_blocks(bm_grade, actual_grade, cog=COG):
    """
    Returns classification array:
      'Ore as Ore'      - BM says ore, Actual is ore
      'Ore→Waste'       - BM says ore, Actual is waste
      'Waste→Ore'       - BM says waste, Actual is ore
      'Waste as Waste'  - BM says waste, Actual is waste
    """
    bm_ore = bm_grade >= cog
    actual_ore = actual_grade >= cog
    cats = []
    for bo, ao in zip(bm_ore, actual_ore):
        if bo and ao:
            cats.append('Ore as Ore')
        elif bo and not ao:
            cats.append('Ore → Waste')
        elif not bo and ao:
            cats.append('Waste → Ore')
        else:
            cats.append('Waste as Waste')
    return cats


def compute_financial(df, bm_col, params):
    """
    Compute revenue and cost metrics for a given BM column.
    params keys: meters_drilled, cost_per_meter, price_per_wmt, mining_cost_per_t
    """
    cog = params.get('cog', COG)
    meters = float(params['meters_drilled'])
    cost_per_m = float(params['cost_per_meter'])
    price_ore = float(params['price_per_wmt'])
    mining_cost = float(params['mining_cost_per_t'])

    drilling_cost = meters * cost_per_m

    bm_ore_mask = df[bm_col] >= cog
    actual_ore_mask = df['Ni_Actual%'] >= cog

    # Classify
    oo = (bm_ore_mask & actual_ore_mask)
    ow = (bm_ore_mask & ~actual_ore_mask)
    wo = (~bm_ore_mask & actual_ore_mask)
    ww = (~bm_ore_mask & ~actual_ore_mask)

    # Revenue from correctly classified ore
    tonnage_oo = df.loc[oo, 'Tonnage_Actual_t'].sum()
    tonnage_wo = df.loc[wo, 'Tonnage_Actual_t'].sum()  # missed ore (lost revenue)
    tonnage_ow = df.loc[ow, 'Tonnage_Actual_t'].sum()  # waste mined as ore (false cost)

    # Revenue: sell ore as ore, lost from misclassified
    gross_revenue = tonnage_oo * price_ore
    lost_revenue = tonnage_wo * price_ore          # ore missed
    false_mining_cost = tonnage_ow * mining_cost   # waste mined unnecessarily

    total_ore_tonnage = df.loc[actual_ore_mask, 'Tonnage_Actual_t'].sum()
    total_mining_cost = df['Tonnage_Actual_t'].sum() * mining_cost

    net_revenue = gross_revenue - total_mining_cost - drilling_cost
    recovery_rate = (tonnage_oo / total_ore_tonnage * 100) if total_ore_tonnage > 0 else 0
    precision = (tonnage_oo / (tonnage_oo + tonnage_ow) * 100) if (tonnage_oo + tonnage_ow) > 0 else 0

    return {
        'drilling_cost': round(drilling_cost, 0),
        'gross_revenue': round(gross_revenue, 0),
        'lost_revenue': round(lost_revenue, 0),
        'false_mining_cost': round(false_mining_cost, 0),
        'net_revenue': round(net_revenue, 0),
        'total_ore_tonnage': round(total_ore_tonnage, 1),
        'tonnage_oo': round(tonnage_oo, 1),
        'tonnage_ow': round(tonnage_ow, 1),
        'tonnage_wo': round(tonnage_wo, 1),
        'blocks_oo': int(oo.sum()),
        'blocks_ow': int(ow.sum()),
        'blocks_wo': int(wo.sum()),
        'blocks_ww': int(ww.sum()),
        'recovery_rate': round(recovery_rate, 1),
        'precision': round(precision, 1),
        'total_blocks': len(df),
    }


def compute_stats(predicted, actual):
    """Regression stats: R2, RMSE, bias"""
    n = len(predicted)
    if n == 0:
        return {}
    residuals = np.array(actual) - np.array(predicted)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((np.array(actual) - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean(residuals ** 2))
    bias = np.mean(residuals)
    slope = np.polyfit(predicted, actual, 1)
    return {
        'r2': round(float(r2), 4),
        'rmse': round(float(rmse), 4),
        'bias': round(float(bias), 4),
        'slope': round(float(slope[0]), 4),
        'intercept': round(float(slope[1]), 4),
        'n': n,
    }


def find_tonnage_col(df):
    candidates = ['Tonnage_Actual_t', 'Tonnage_Actual', 'Tonnage', 'TONNAGE', 'tonnes', 'TON']
    for col in candidates:
        if col in df.columns:
            return col
    return None


def compute_weighted_means(df, numeric_cols, weight_col):
    weighted = {}
    if weight_col not in df.columns:
        return weighted
    weights = df[weight_col].astype(float)
    for col in numeric_cols:
        if col not in df.columns:
            continue
        values = df[col]
        valid = values.notna() & weights.notna()
        if valid.sum() == 0 or weights.loc[valid].sum() == 0:
            weighted[col] = None
            continue
        weighted[col] = round(float(np.average(values.loc[valid].astype(float), weights=weights.loc[valid])), 3)
    return weighted


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/load_sample', methods=['POST'])
def load_sample():
    """Load the bundled sample dataset"""
    try:
        sample_path = os.path.join(os.path.dirname(__file__), 'sample_drillspace_data.csv')
        if not os.path.exists(sample_path):
            return jsonify({'error': 'Sample file not found. Run generate_sample.py first.'}), 404
        df = pd.read_csv(sample_path)
        app.config['CURRENT_DF'] = df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        weight_col = find_tonnage_col(df)
        return jsonify({
            'success': True,
            'columns': list(df.columns),
            'numeric_columns': numeric_cols,
            'shape': list(df.shape),
            'preview': df.head(5).to_dict(orient='records'),
            'stats': df[numeric_cols].describe().round(3).to_dict(),
            'weighted_means': compute_weighted_means(df, numeric_cols, weight_col),
            'weight_col': weight_col,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload():
    """Parse uploaded CSV and return column info"""
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        content = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))
        df.columns = [c.strip() for c in df.columns]

        # Auto-detect columns
        cols = list(df.columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        weight_col = find_tonnage_col(df)

        # Store in app context (in-memory for demo; production: use sessions/db)
        app.config['CURRENT_DF'] = df
        app.config['CURRENT_CSV'] = content

        return jsonify({
            'success': True,
            'columns': cols,
            'numeric_columns': numeric_cols,
            'shape': list(df.shape),
            'preview': df.head(5).to_dict(orient='records'),
            'stats': df[numeric_cols].describe().round(3).to_dict(),
            'weighted_means': compute_weighted_means(df, numeric_cols, weight_col),
            'weight_col': weight_col,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/scatter', methods=['POST'])
def scatter():
    """Generate scatter plot data for BM vs Actual"""
    try:
        df = app.config.get('CURRENT_DF')
        if df is None:
            return jsonify({'error': 'No data loaded'}), 400

        body = request.json
        col_bm125 = body.get('col_bm125', 'BM12.5_Ni%')
        col_bm25 = body.get('col_bm25', 'BM25_Ni%')
        col_actual = body.get('col_actual', 'Ni_Actual%')
        cog = float(body.get('cog', COG))

        df = df.dropna(subset=[col_bm125, col_bm25, col_actual])

        cats_125 = classify_blocks(df[col_bm125], df[col_actual], cog)
        cats_25 = classify_blocks(df[col_bm25], df[col_actual], cog)

        stats_125 = compute_stats(df[col_bm125].tolist(), df[col_actual].tolist())
        stats_25 = compute_stats(df[col_bm25].tolist(), df[col_actual].tolist())

        # Confusion matrix counts
        def confusion(cats):
            from collections import Counter
            c = Counter(cats)
            total = len(cats)
            return {k: {'count': v, 'pct': round(v / total * 100, 1)} for k, v in c.items()}

        def kde_density(x_vals, y_vals):
            """Compute per-point KDE density, subsampled for speed, normalized 0-1."""
            xy = np.vstack([x_vals, y_vals])
            # Subsample for KDE estimation if large
            n = len(x_vals)
            if n > 5000:
                idx = np.random.choice(n, 5000, replace=False)
                kde = gaussian_kde(xy[:, idx], bw_method=0.15)
            else:
                kde = gaussian_kde(xy, bw_method=0.15)
            density = kde(xy)
            d_min, d_max = density.min(), density.max()
            if d_max > d_min:
                density = (density - d_min) / (d_max - d_min)
            return density.tolist()

        x125 = df[col_bm125].tolist()
        y125 = df[col_actual].tolist()
        x25  = df[col_bm25].tolist()
        y25  = df[col_actual].tolist()

        dens125 = kde_density(np.array(x125), np.array(y125))
        dens25  = kde_density(np.array(x25),  np.array(y25))

        max_val = max(df[col_bm125].max(), df[col_bm25].max(), df[col_actual].max())
        min_val = min(df[col_bm125].min(), df[col_bm25].min(), df[col_actual].min())

        return jsonify({
            'bm125': {
                'x': x125,
                'y': y125,
                'density': dens125,
                'categories': cats_125,
                'stats': stats_125,
                'confusion': confusion(cats_125),
            },
            'bm25': {
                'x': x25,
                'y': y25,
                'density': dens25,
                'categories': cats_25,
                'stats': stats_25,
                'confusion': confusion(cats_25),
            },
            'cog': cog,
            'axis_range': [round(min_val - 0.1, 2), round(max_val + 0.1, 2)],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/financial', methods=['POST'])
def financial():
    """Compute financial analysis"""
    try:
        df = app.config.get('CURRENT_DF')
        if df is None:
            return jsonify({'error': 'No data loaded'}), 400

        body = request.json
        col_bm125 = body.get('col_bm125', 'BM12.5_Ni%')
        col_bm25 = body.get('col_bm25', 'BM25_Ni%')
        col_actual = body.get('col_actual', 'Ni_Actual%')
        col_tonnage = body.get('col_tonnage', 'Tonnage_Actual_t')

        params_125 = {**body.get('params', {}), 'meters_drilled': body.get('meters_125', 5000)}
        params_25 = {**body.get('params', {}), 'meters_drilled': body.get('meters_25', 2500)}

        df2 = df.dropna(subset=[col_bm125, col_bm25, col_actual, col_tonnage]).copy()
        df2 = df2.rename(columns={
            col_bm125: 'BM12.5_Ni%',
            col_bm25: 'BM25_Ni%',
            col_actual: 'Ni_Actual%',
            col_tonnage: 'Tonnage_Actual_t',
        })

        fin_125 = compute_financial(df2, 'BM12.5_Ni%', params_125)
        fin_25 = compute_financial(df2, 'BM25_Ni%', params_25)

        return jsonify({
            'bm125': fin_125,
            'bm25': fin_25,
            'delta': {
                k: round(fin_125[k] - fin_25[k], 0)
                for k in ['drilling_cost', 'gross_revenue', 'net_revenue', 'lost_revenue']
                if isinstance(fin_125[k], (int, float))
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/grade_distribution', methods=['POST'])
def grade_distribution():
    """Histogram data for grade distributions"""
    try:
        df = app.config.get('CURRENT_DF')
        if df is None:
            return jsonify({'error': 'No data loaded'}), 400

        body = request.json
        col_bm125 = body.get('col_bm125', 'BM12.5_Ni%')
        col_bm25 = body.get('col_bm25', 'BM25_Ni%')
        col_actual = body.get('col_actual', 'Ni_Actual%')

        bins = np.arange(0, 4.0, 0.1).tolist()
        col_tonnage = body.get('col_tonnage', 'Tonnage_Actual_t')

        def weighted_stats(series, weights):
            valid = series.notna() & weights.notna()
            series = series.loc[valid].astype(float)
            weights = weights.loc[valid].astype(float)
            if len(series) == 0 or weights.sum() == 0:
                return {'mean': 0.0, 'std': 0.0}
            mean = np.average(series, weights=weights)
            variance = np.average((series - mean) ** 2, weights=weights)
            return {'mean': round(float(mean), 3), 'std': round(float(np.sqrt(variance)), 3)}

        def hist_data(series, weights, bins):
            valid = series.notna() & weights.notna()
            series = series.loc[valid].astype(float)
            counts, edges = np.histogram(series, bins=bins)
            stats = weighted_stats(series, weights.loc[valid])
            return {
                'counts': counts.tolist(),
                'edges': [round(e, 2) for e in edges.tolist()],
                'mean': stats['mean'],
                'std': stats['std'],
            }

        if col_tonnage not in df.columns:
            return jsonify({'error': f'Tonnage column "{col_tonnage}" not found'}), 400

        tonnage = df[col_tonnage]

        return jsonify({
            'bm125': hist_data(df[col_bm125], tonnage, bins),
            'bm25': hist_data(df[col_bm25], tonnage, bins),
            'actual': hist_data(df[col_actual], tonnage, bins),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/spatial', methods=['POST'])
def spatial():
    """Spatial grade map data"""
    try:
        df = app.config.get('CURRENT_DF')
        if df is None:
            return jsonify({'error': 'No data loaded'}), 400

        body = request.json
        col_x = body.get('col_x', 'X')
        col_y = body.get('col_y', 'Y')
        col_bm125 = body.get('col_bm125', 'BM12.5_Ni%')
        col_bm25 = body.get('col_bm25', 'BM25_Ni%')
        col_actual = body.get('col_actual', 'Ni_Actual%')

        df2 = df.dropna(subset=[col_x, col_y, col_bm125, col_bm25, col_actual])

        return jsonify({
            'x': df2[col_x].tolist(),
            'y': df2[col_y].tolist(),
            'bm125': df2[col_bm125].tolist(),
            'bm25': df2[col_bm25].tolist(),
            'actual': df2[col_actual].tolist(),
            'diff_125': (df2[col_bm125] - df2[col_actual]).round(3).tolist(),
            'diff_25': (df2[col_bm25] - df2[col_actual]).round(3).tolist(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/product_comparison', methods=['POST'])
def product_comparison():
    """Grade comparison breakdown by product"""
    try:
        df = app.config.get('CURRENT_DF')
        if df is None:
            return jsonify({'error': 'No data loaded'}), 400

        body = request.json
        col_bm125 = body.get('col_bm125', 'BM12.5_Ni%')
        col_bm25 = body.get('col_bm25', 'BM25_Ni%')
        col_actual = body.get('col_actual', 'Ni_Actual%')
        col_product = body.get('col_product', 'Actual_Product')
        col_tonnage = body.get('col_tonnage', 'Tonnage_Actual_t')

        if col_product not in df.columns:
            return jsonify({'error': f'Product column "{col_product}" not found'}), 400

        required_cols = [col_bm125, col_bm25, col_actual, col_product, col_tonnage]
        if any(c not in df.columns for c in required_cols):
            return jsonify({'error': 'Missing required columns'}), 400

        df2 = df.dropna(subset=required_cols).copy()
        if len(df2) == 0:
            return jsonify({'error': 'No valid data after filtering'}), 400

        def weighted_mean(group, col, weight_col):
            values = group[col].astype(float)
            weights = group[weight_col].astype(float)
            total = weights.sum()
            return np.average(values, weights=weights) if total > 0 else 0.0

        # Group by product
        products = []
        for product_name, group in df2.groupby(col_product, observed=True):
            bm125_mean = weighted_mean(group, col_bm125, col_tonnage)
            bm25_mean = weighted_mean(group, col_bm25, col_tonnage)
            actual_mean = weighted_mean(group, col_actual, col_tonnage)
            diff_125 = actual_mean - bm125_mean
            diff_25 = actual_mean - bm25_mean
            tonnage_total = group[col_tonnage].sum()
            block_count = len(group)

            products.append({
                'product': str(product_name),
                'bm125': round(float(bm125_mean), 3),
                'bm25': round(float(bm25_mean), 3),
                'actual': round(float(actual_mean), 3),
                'diff_125': round(float(diff_125), 3),
                'diff_25': round(float(diff_25), 3),
                'tonnage': round(float(tonnage_total), 1),
                'blocks': int(block_count),
            })

        # Sort by product name
        products.sort(key=lambda x: x['product'])

        return jsonify({
            'products': products,
            'total_blocks': len(df2),
            'total_tonnage': round(float(df2[col_tonnage].sum()), 1),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/swath', methods=['POST'])
def swath():
    """Swath plot: mean grade by X (Easting), Y (Northing), Z (Elevation) slices"""
    try:
        df = app.config.get('CURRENT_DF')
        if df is None:
            return jsonify({'error': 'No data loaded'}), 400

        body = request.json
        col_x = body.get('col_x', 'X')
        col_y = body.get('col_y', 'Y')
        col_z = body.get('col_z', 'Z')
        col_bm125 = body.get('col_bm125', 'BM12.5_Ni%')
        col_bm25 = body.get('col_bm25', 'BM25_Ni%')
        col_actual = body.get('col_actual', 'Ni_Actual%')
        col_tonnage = body.get('col_tonnage', 'Tonnage_Actual_t')
        bins = int(body.get('bins', 15))

        def weighted_mean(values, weights):
            values = values.astype(float)
            weights = weights.astype(float)
            total = weights.sum()
            return np.average(values, weights=weights) if total > 0 else np.nan

        def swath_by(col):
            """Compute tonnage-weighted mean grade by bins of a coordinate column."""
            required = [col, col_bm125, col_bm25, col_actual, col_tonnage]
            if any(c not in df.columns for c in required):
                return None
            d = df.dropna(subset=required).copy()
            if len(d) < 5:
                return None
            d['bin'] = pd.cut(d[col], bins=bins)
            grp = d.groupby('bin', observed=True).apply(
                lambda g: pd.Series({
                    col_bm125: weighted_mean(g[col_bm125], g[col_tonnage]),
                    col_bm25: weighted_mean(g[col_bm25], g[col_tonnage]),
                    col_actual: weighted_mean(g[col_actual], g[col_tonnage]),
                })
            ).reset_index()
            grp['mid'] = grp['bin'].apply(lambda x: round(x.mid, 1))
            return {
                'x': grp['mid'].tolist(),
                'bm125': [round(float(v), 3) if not pd.isna(v) else None for v in grp[col_bm125]],
                'bm25': [round(float(v), 3) if not pd.isna(v) else None for v in grp[col_bm25]],
                'actual': [round(float(v), 3) if not pd.isna(v) else None for v in grp[col_actual]],
            }

        result = {
            'easting':   swath_by(col_x),
            'northing':  swath_by(col_y),
            'elevation': swath_by(col_z),
        }
        # Fallback keys for backward compat
        if result['easting']:
            result['x']      = result['easting']['x']
            result['bm125']  = result['easting']['bm125']
            result['bm25']   = result['easting']['bm25']
            result['actual'] = result['easting']['actual']

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  Nickel Drillspace Analysis Tool")
    print("  Resource Geology | Financial Analysis")
    print("=" * 60)
    port = int(os.environ.get('PORT', 5050))
    print(f"  Open: http://localhost:{port}")
    print("  Sample data: sample_drillspace_data.csv")
    print("=" * 60)
    app.run(debug=False, port=port, host='0.0.0.0')
