#@title: Forecast Plotting Utility
# This script provides utility functions to prepare and plot forecast data for different horizons.
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, r2_score



# -------------------------------
# Utility: Load prediction data
# -------------------------------
def prepare_forecast_plot_data(index_name: str, file_prefix: str, unit: str):
    horizons = [30, 60, 90]
    dfs = [
        pd.read_csv(f"{file_prefix}/30d_prediction.csv"),
        pd.read_csv(f"{file_prefix}/60d_prediction.csv"),
        pd.read_csv(f"{file_prefix}/90d_prediction.csv"),
    ]

    for df in dfs:
        df["Date"] = pd.to_datetime(df["Date"])

    excluded_keywords = ['Date', 'date', 'Actual_', 'Predicted_']
    all_columns = dfs[0].columns
    features = [col for col in all_columns if not any(key in col for key in excluded_keywords)]
    feature_text = "Features used:\n" + "\n".join(features)

    return {
        "index_name": index_name,
        "dfs": dfs,
        "horizons": horizons,
        "unit": unit,
        "features": features,
        "feature_text": feature_text
    }


def aggregate_mae(df, actual_col, pred_col, freq='W'):
    df = df.copy()
    df = df.set_index('target_date')
    df = df[[actual_col, pred_col]].dropna()
    agg = df.resample(freq).mean()
    return mean_absolute_error(agg[actual_col], agg[pred_col])

def plot_forecast_panel(index_name: str, dfs, horizons, cutoff_date, split_date, unit, feature_text):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[f"{h}-Day Horizon" for h in horizons])

    for i, (horizon, df) in enumerate(zip(horizons, dfs), start=1):
        df = df.copy()
        df['date'] = pd.to_datetime(df['Date'])
        df['target_date'] = df['date'] + pd.Timedelta(days=horizon)
        df = df.sort_values('target_date')
        show_legend = (i == 1)

        cutoff_target_date = (cutoff_date + pd.to_timedelta(horizon, unit='D')).to_pydatetime()

        valid_mask = df[[f'Actual_{horizon}D', f'Predicted_{horizon}D']].notna().all(axis=1)
        df_valid = df[valid_mask]

        test_start_date = df_valid['target_date'].min() if not df_valid.empty else None
        last_actual_date = df.loc[df[f'Actual_{horizon}D'].notna(), 'target_date'].max() if not df_valid.empty else split_date
        has_phase2 = (cutoff_date < split_date) and (split_date < last_actual_date)

        if test_start_date is not None:
            if has_phase2:
                test_period1_mask = (df['target_date'] >= test_start_date) & (df['target_date'] <= split_date) & valid_mask
                test_period2_mask = (df['target_date'] > split_date) & valid_mask
            else:
                test_period1_mask = (df['target_date'] >= test_start_date) & valid_mask
                test_period2_mask = np.array([False] * len(df))
        else:
            test_period1_mask = test_period2_mask = np.array([False] * len(df))

        df_test_period1 = df[test_period1_mask]
        df_test_period2 = df[test_period2_mask]

        mae_1 = r2_1 = mae_2 = r2_2 = np.nan
        if not df_test_period1.empty:
            mae_1 = mean_absolute_error(df_test_period1[f'Actual_{horizon}D'], df_test_period1[f'Predicted_{horizon}D'])
            r2_1 = r2_score(df_test_period1[f'Actual_{horizon}D'], df_test_period1[f'Predicted_{horizon}D'])
        if not df_test_period2.empty:
            mae_2 = mean_absolute_error(df_test_period2[f'Actual_{horizon}D'], df_test_period2[f'Predicted_{horizon}D'])
            r2_2 = r2_score(df_test_period2[f'Actual_{horizon}D'], df_test_period2[f'Predicted_{horizon}D'])

        fig.add_trace(go.Scatter(x=df['date'], y=df[index_name], mode='lines',
                                 name=f'Historical {index_name}', line=dict(color='gray'), showlegend=show_legend),
                      row=i, col=1)
        fig.add_trace(go.Scatter(x=df['target_date'], y=df[f'Actual_{horizon}D'], mode='lines+markers',
                                 name='Actual', line=dict(color='blue'), marker=dict(size=3), showlegend=show_legend),
                      row=i, col=1)
        fig.add_trace(go.Scatter(x=df['target_date'], y=df[f'Predicted_{horizon}D'], mode='lines+markers',
                                 name=f'Predicted ({horizon}D)', line=dict(color='orange'), marker=dict(size=3)),
                      row=i, col=1)

        if test_start_date is not None:
            y_min = df[f'Actual_{horizon}D'].min()
            y_max = df[f'Actual_{horizon}D'].max()
            center_2 = split_date + (last_actual_date - split_date) / 2
            paper_y = 1 - (i - 1) * (1 / 3)
            paper_y_center = 1 - (i - 1) * (1 / 3) - 1/6  # ← halfway within this subplot

            if cutoff_date >= split_date:
                center = test_start_date + (last_actual_date - test_start_date) / 2
                fig.add_shape(type="rect", x0=test_start_date, x1=last_actual_date,
                              y0=y_min, y1=y_max,
                              fillcolor="#2a9d8f", opacity=0.2, line_width=0,
                              row=i, col=1, layer="below")

                fig.add_annotation(
                    text=f"Test Period<br>({test_start_date.date()} to {last_actual_date.date()})<br>MAE: {mae_1:.2f}<br>R²: {r2_1:.2f}",
                    x=center, xref=f'x{i}',
                    y=paper_y, yref='paper',
                    yanchor='top',
                    showarrow=False, bgcolor="#2a9d8f", font=dict(size=10)
                )

                weekly_mae = aggregate_mae(df_test_period1, f'Actual_{horizon}D', f'Predicted_{horizon}D', freq='W')
                monthly_mae = aggregate_mae(df_test_period1, f'Actual_{horizon}D', f'Predicted_{horizon}D', freq='M')

                fig.add_annotation(
                    text=f"Aggregated<br>Weekly MAE: {weekly_mae:.2f}<br>Monthly MAE: {monthly_mae:.2f}",
                    x=center, xref=f'x{i}',
                    y=paper_y_center, yref='paper',
                    showarrow=False, bgcolor="#2a9d8f", font=dict(size=10)
                )
            else:
                # Two phase case
                fig.add_shape(type="rect", x0=test_start_date, x1=split_date,
                              y0=y_min, y1=y_max,
                              fillcolor="LightSalmon", opacity=0.2, line_width=0,
                              row=i, col=1, layer="below")

                fig.add_annotation(
                    text=f"test_period 1<br>({test_start_date.date()} to {split_date.date()})<br>MAE: {mae_1:.2f}<br>R²: {r2_1:.2f}",
                    x=test_start_date + (split_date - test_start_date) / 2, xref=f'x{i}',
                    y=paper_y, yref='paper',
                    yanchor='top',
                    showarrow=False, bgcolor="LightSalmon", font=dict(size=10)
                )

                fig.add_shape(type="rect", x0=split_date + pd.Timedelta(days=1), x1=last_actual_date,
                              y0=y_min, y1=y_max,
                              fillcolor="LightGreen", opacity=0.2, line_width=0,
                              row=i, col=1, layer="below")


                fig.add_annotation(
                    text=f"test_period 2<br>(> {split_date.date()})<br>MAE: {mae_2:.2f}<br>R²: {r2_2:.2f}",
                    x=center_2, xref=f'x{i}',
                    y=paper_y, yref='paper', # ← aligns to top of the subplot's y-axis area
                    yanchor='top',
                    showarrow=False, bgcolor="#2a9d8f", font=dict(size=10)
                )

                weekly_mae_2 = aggregate_mae(df_test_period2, f'Actual_{horizon}D', f'Predicted_{horizon}D', freq='W')
                monthly_mae_2 = aggregate_mae(df_test_period2, f'Actual_{horizon}D', f'Predicted_{horizon}D', freq='M')

                fig.add_annotation(
                    text=f"Aggregated<br>Weekly MAE: {weekly_mae_2:.2f}<br>Monthly MAE: {monthly_mae_2:.2f}",
                    x=center_2, xref=f'x{i}',
                    y=paper_y_center, yref='paper',
                    showarrow=False, bgcolor="#2a9d8f", font=dict(size=10)
                )

        fig.update_yaxes(title_text=f"{index_name} Price ({unit})", row=i, col=1)

    fig.add_annotation(
        text=feature_text.replace('\n', '<br>'),
        x=0.05, y=1.05, xref='paper', yref='paper',
        showarrow=False, align='left', font=dict(size=14),
        bgcolor='rgba(255,255,255,0.7)', bordercolor='black', borderwidth=1
    )

    fig.update_layout(
        height=900,
        title=f"{index_name} Forecast vs Actual by Horizon (train/test cutoff on {cutoff_date.date()})",
        legend=dict(
            x=1.05, y=0.9, xanchor='right', yanchor='top',
            bgcolor='rgba(255,255,255,0.7)', bordercolor='black', borderwidth=1
        ),
        template="plotly_white",
        hovermode='x unified'
    )

    return fig
