import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

# forecast functions (needed for page 2)
from Forecast_Plotting_Functions import prepare_forecast_plot_data, plot_forecast_panel

# --- Streamlit app layout ---
st.set_page_config(page_title="LNG Market Dashboard", layout="wide")

st.markdown("""
    <style>
    /* Try to override Data Editor font size - may only work on some versions */
    section.main .stDataEditorTable {
        font-size: 20px !important;
    }
    /* Try table cell content */
    section.main .stDataEditorTable td {
        font-size: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

tabs = st.tabs(["Market Outlook & Profitability",  "LNG price forecasts",])


with tabs[0]:
    
    # --- Controls in sidebar ---
    df = pd.read_csv("data/processed/destination_netback_us_origin.csv")
    months = df["Period"].unique()
    months = [m for m in months if m != "2025-07"]    

    # Make three columns for selectors at the top
    col_a, spacer, col_b, spacer, col_c = st.columns([1, 0.3, 1, 0.3, 1], gap="medium")
    default_fx = 0.293  # Default conversion rate from €/MWh to $/MMBtu

    with col_a:
        selected_month = st.selectbox("Select Month", sorted(months, reverse=False), key="month_selector_top")
    with col_b:
        currency = st.selectbox("Select Unit", ["€/MWh", "$/MMBtu"], key="unit_selector_top")
    with col_c:
        switch_rate = st.number_input(
            "Unit Conversion Rate (€/MWh to $/MMBtu)",
            value=float(default_fx),
            min_value=0.01, step=0.01,
            key="switch_rate_input_top"
    )    
    st.divider()  # <-- adds a horizontal line and some space
    
    
    # Filter for selected month
    df_month = df[df["Period"] == selected_month].copy()
    df_month['Cargo Cost US origin'] = pd.to_numeric(df_month['Cargo Cost US origin'], errors='coerce')*-1  # Convert to negative for profit calculation

    # Only columns in €/MWh to be converted:
    main_cols = ["Terminal", "Period", 'benchmark price','Total Regas', 'Total Freight From US', 'Netback', 'Cargo Cost US origin','Net Profit']
    euro_cols = ['Prediction_PVB(€/MWh)','Prediction_TTF(€/MWh)','benchmark price']
    dollar_cols = ['Total Regas', 'Basic Slot (Berth)', 'Basic Slot (Unload/Stor/Regas)', 'Basic Slot (B/U/S/R)',
                'Additional Storage', 'Additional Sendout', 'Gas in Kind', 'Entry Capacity', 'Commodity Charge',
                'Total Freight From US', 'Route Cost from US', 'Fuel_loss from US', 'Cargo Cost US origin', 'Prediction_HH($MMBtu)']

    if currency == "€/MWh":
        for col in dollar_cols:
            df_month[col] = pd.to_numeric(df_month[col], errors='coerce')/switch_rate
        # Calculate netback and profit after conversion
        df_month['Netback'] = df_month['benchmark price'] +df_month['Total Regas'] + df_month['Total Freight From US']
        df_month['Net Profit'] = df_month['Netback'] + df_month['Cargo Cost US origin']
        table = df_month[main_cols]
    else:
        df_month_us = df_month.copy()
        for col in euro_cols:
            df_month_us[col] = pd.to_numeric(df_month_us[col], errors='coerce') * switch_rate
        # Calculate after conversion
        df_month_us['Netback'] = df_month_us['benchmark price'] + df_month_us['Total Regas'] + df_month_us['Total Freight From US']
        df_month_us['Net Profit'] = df_month_us['Netback'] + df_month_us['Cargo Cost US origin']
        table = df_month_us[main_cols]
        
    # Identify which columns are numeric and need rounding
    cols_to_round = ["Net Profit", "Netback", "benchmark price", "Total Regas", "Total Freight From US", 'Cargo Cost US origin']

    # Make a copy and round
    table_rounded = table.copy()
    table_rounded[cols_to_round] = table_rounded[cols_to_round].apply(lambda x: np.round(x, 3))

    # --- Two column layout ---
    col1, spacer, col2 = st.columns([2, 0.3, 1])
    with col1:
        st.subheader("Netback & Net Profit per Destination Terminals (US Origin)")
        st.data_editor(table_rounded, hide_index=True, use_container_width=True, height=590)

        # add some space and a divider
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        # Add bar chart below
        columns_options = ['Net Profit', 'Netback', 'benchmark price', 'Total Regas']
        selected_col = st.selectbox("Bar Chart Column:", columns_options, index=0)
        fig = px.bar(
            table,
            x='Terminal',
            y=selected_col,
            height=380,
            labels={selected_col: "Value", "Terminal": "Terminal"},
            title=f"{selected_col} per Terminal for {selected_month}",
            color_discrete_sequence=["#3182bd"]
        )
        
        fig.update_traces(
            textposition="inside",       # or "inside" if you want numbers inside bars
            textfont_size=18              # <-- this will control the annotation size
        )

        # For tick labels (axis), use:
        fig.update_layout(
            xaxis=dict(
                title_text="Terminal",
                title_font = dict(size=20),
                tickfont=dict(size=18),  # Change as you like
            )
        )

        fig.update_layout(
            title={
                'text': f"{selected_col} per Terminal for {selected_month}",  # The title text
                'x': 0.5,                           # Center the title
                'xanchor': 'center',
                'font': {'size': 20}  # Adjust size/color as desired
            },
            font=dict(size=18)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Terminal Profitability Simulation")

        terminal_list = table["Terminal"].unique()
        selected_terminal = st.selectbox("Select Terminal", terminal_list, key="terminal_details")

        row = table[table["Terminal"] == selected_terminal]
        if row.empty:
            st.warning("No data for selected terminal.")
        else:
            row = row.iloc[0]
            benchmark_price = float(row.get("benchmark price", 0))
            regas_fee = float(row.get("Total Regas", 0))
            total_freight = float(row.get("Total Freight From US", 0))
            cargo_costs_base = float(row.get("Cargo Cost US origin", 0))

            # ---- Table "rows" using st.columns ----
            st.markdown("<br>", unsafe_allow_html=True)
            # HEADER (centered)
            st.markdown(
                '''
                <div style="display:flex; font-weight:bold; margin-bottom:6px;">
                    <div style="flex:2; text-align:left;">Item</div>
                    <div style="flex:1; text-align:left;">Value</div>
                </div>
                ''', unsafe_allow_html=True
            )

            # Benchmark Price
            c1, c2 = st.columns([2, 1])
            c1.markdown("Benchmark Price")
            c2.markdown(f"{benchmark_price:.2f}")

            # Adjustment on price (editable inline)
            c1, c2 = st.columns([2, 1])
            c1.markdown("Adjustment on price")
            adj_on_price = c2.number_input(
                "", value=0.0, step=0.01, format="%.2f",
                key=f"adj_price_{selected_terminal}", label_visibility="collapsed"
            )

            # Revenue (calculated, bold)
            revenue = benchmark_price + adj_on_price
            c1, c2 = st.columns([2, 1])
            c1.markdown("**Revenue**")
            c2.markdown(f"**{revenue:.2f}**")

            # Regas fee
            c1, c2 = st.columns([2, 1])
            c1.markdown("Regas fee")
            c2.markdown(f"{regas_fee:.2f}")

            # Total Freight
            c1, c2 = st.columns([2, 1])
            c1.markdown("Total Freight")
            c2.markdown(f"{total_freight:.2f}")

            # Netback (calculated, bold)
            netback = revenue + regas_fee + total_freight
            c1, c2 = st.columns([2, 1])
            c1.markdown("**Netback**")
            c2.markdown(f"**{netback:.2f}**")

            # Cargo Costs US origin
            c1, c2 = st.columns([2, 1])
            c1.markdown("Cargo Cost US origin")
            c2.markdown(f"{cargo_costs_base:.2f}")

            # Adj on cargo prices (editable inline)
            c1, c2 = st.columns([2, 1])
            c1.markdown("Adj on cargo prices")
            adj_on_cargo = c2.number_input(
                "", value=0.0, step=0.01, format="%.2f",
                key=f"adj_cargo_{selected_terminal}", label_visibility="collapsed"
            )

            # Cargo Costs (calculated)
            cargo_costs = cargo_costs_base + adj_on_cargo

            # Net Profit (calculated, styled)
            net_profit = netback + cargo_costs

            # Choose color based on sign
            net_profit_color = "limegreen" if net_profit >= 0 else "tomato"
            c1, c2 = st.columns([2, 1])
            c1.markdown(
                '<span style="font-weight:bold; font-size:1.18em;">Net Profit</span>',
                unsafe_allow_html=True
            )
            c2.markdown(
                f'<span style="font-weight:bold; font-size:1.18em; color:{net_profit_color};">'
                f'{net_profit:+.2f}'
                f'</span>',
                unsafe_allow_html=True
            )

            # add a break and divider
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("---")
            # Use your calculated values for the selected terminal
            waterfall_data = {
                "Revenue": revenue,
                "Regas fee": regas_fee,
                "Total Freight": total_freight,
                "Cargo Costs": cargo_costs,
                "Net Profit": net_profit,
            }

            # Prepare the chart steps
            measure = ["absolute", "relative", "relative", "relative", "total"]
            x_labels = list(waterfall_data.keys())
            y_values = [
                waterfall_data["Revenue"],
                waterfall_data["Regas fee"],
                waterfall_data["Total Freight"],
                waterfall_data["Cargo Costs"],
                waterfall_data["Net Profit"],
            ]

            # Plotly Waterfall chart
            fig_waterfall = go.Figure(go.Waterfall(
                name="Net Profit Calculation",
                orientation="v",
                measure=measure,
                x=x_labels,
                text=[f"{v:.2f}" for v in y_values],
                y=y_values,
                textposition="outside",
                connector={"line":{"color":"rgb(63,63,63)"}},
                decreasing={"marker":{"color":"#fa8072"}},   # Red for negative
                increasing={"marker":{"color":"#7fe881"}},   # Green for positive
                totals={"marker":{"color":"#3182bd"}}        # Blue for total
            ))
            fig_waterfall.update_traces(textposition="auto", textfont_size=16,
                                        insidetextfont=dict(size=16), outsidetextfont=dict(size=16))
            
            fig_waterfall.update_layout(
                title={
                    'text': f"<b>{selected_terminal}</b><br>Net Profit Breakdown",
                    'x': 0.5,  # Center the title
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 18}
                },
                showlegend=False,
                height=400,
                margin=dict(l=30, r=20, t=60, b=30),
                xaxis=dict(tickfont=dict(size=18)),
                font=dict(size=18),
            )

            st.plotly_chart(fig_waterfall, use_container_width=True)

with tabs[1]:
    st.header("Forecasted LNG Benchmark Prices")
    price_option = st.selectbox(
        "Select price to plot:",
        ("TTF", "HH", "PVB", "JKM")
    )
    
    # Depending on the option, load/display relevant data
    if price_option == "TTF":
        data = prepare_forecast_plot_data(
            index_name="TTF_Price",
            file_prefix="reports/TTF_forecast",   # change this path
            unit="EUR/MWh"
        )
        fig =plot_forecast_panel(
            index_name=data["index_name"],
            dfs=data["dfs"],
            horizons=data["horizons"],
            cutoff_date=pd.to_datetime("2021-12-31"),
            split_date=pd.to_datetime("2024-03-31"),
            unit=data["unit"],
            feature_text=data["feature_text"]
        )
        st.plotly_chart(fig, use_container_width=True)
    elif price_option == "HH":
        data = prepare_forecast_plot_data(
            index_name="HH_Price",
            file_prefix="reports/HH_forecast",   # change this path
            unit="$/MMBtu"
        )
        fig =plot_forecast_panel(
            index_name=data["index_name"],
            dfs=data["dfs"],
            horizons=data["horizons"],
            cutoff_date=pd.to_datetime("2021-12-31"),
            split_date=pd.to_datetime("2024-03-31"),
            unit=data["unit"],
            feature_text=data["feature_text"]
        )
        st.plotly_chart(fig, use_container_width=True)
    elif price_option == "PVB":
        data = prepare_forecast_plot_data(
            index_name="PVB",
            file_prefix="reports/PVB_forecast",   # change this path
            unit="EUR/MWh"
        )
        fig =plot_forecast_panel(
            index_name=data["index_name"],
            dfs=data["dfs"],
            horizons=data["horizons"],
            cutoff_date=pd.to_datetime("2024-03-31"),
            split_date=pd.to_datetime("2024-03-31"),
            unit=data["unit"],
            feature_text=data["feature_text"]
        )
        st.plotly_chart(fig, use_container_width=True)
    elif price_option == "JKM":
        data = prepare_forecast_plot_data(
            index_name="JKM_Price",
            file_prefix="reports/JKM_forecast",   # change this path
            unit="$/MMBtu"
        )
        fig =plot_forecast_panel(
            index_name=data["index_name"],
            dfs=data["dfs"],
            horizons=data["horizons"],
            cutoff_date=pd.to_datetime("2024-03-31"),
            split_date=pd.to_datetime("2024-03-31"),
            unit=data["unit"],
            feature_text=data["feature_text"]
        )
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("""
    <div style="background-color:#23272f; padding:18px 22px; border-radius:10px; border: 1.5px solid #444; margin-top:16px;">
    <b>Forecast Model Methodology</b><br>
    <ul>
    <li><b>Features Evaluated:</b> LNG Prices, Oil/Coal prices, TTFMC1_Close, TTFMC2_Close, technical features like return_60, bollinger_width, z_score_30, rolling_std_30, etc.</li>
    <li><b>Model Type:</b> Deep learning LSTM with multiple forecast horizons (30, 60, 90 days).</li>
    <li><b>Train/Test Split:</b> Pre/post cutoff date (e.g., 2021-12-31), test regime post-war event.</li>
    <li><b>Evaluation:</b> MAE and R² annotated for each horizon and regime.</li>
    <li><b>Data Sources:</b> Historical prices from various LNG market reports, weather and other information from publicly available sources.</li>
    </ul>
    <i>Full methodology available upon request to the project team.</i>
    </div>
    """, unsafe_allow_html=True)