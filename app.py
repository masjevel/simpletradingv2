import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import datetime
import requests

# --- SIDKONFIGURATION ---
st.set_page_config(
    page_title="Trend Heatmap Analys",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SIDEBAR: INSTÄLLNINGAR ---
st.sidebar.title("⚙️ Inställningar")

ticker = st.sidebar.text_input("Tillgång (ticker)", value="SPY").upper().strip()

st.sidebar.markdown("---")
st.sidebar.subheader("Teknisk modell")
buy_rsi_threshold = st.sidebar.slider("Köp-RSI (undre gräns)", min_value=10, max_value=45, value=28, step=1)
sell_macd_rsi_filter = st.sidebar.slider("Sälj-RSI (nedre gräns)", min_value=50, max_value=80, value=65, step=1)
cooldown_days = st.sidebar.slider("Cooldown (dagar)", min_value=5, max_value=60, value=15, step=1)
sma_trend_period = st.sidebar.slider("SMA-period (trend)", min_value=50, max_value=300, value=200, step=10)

st.sidebar.markdown("---")
st.sidebar.subheader("Konjunkturmodell (VIX)")
vix_panic_threshold = st.sidebar.slider("VIX-panik gräns", min_value=20.0, max_value=60.0, value=34.0, step=0.5)

st.sidebar.markdown("---")
color_intensity_cap = st.sidebar.slider("Heatmap-intensitet (cap)", min_value=30, max_value=300, value=150, step=10)

st.sidebar.markdown("---")
st.sidebar.subheader("Konjunkturmodell (KI)")
ki_sell_threshold = st.sidebar.slider("KI Överhettning (Sälj-tröskel)", min_value=90, max_value=140, value=110, step=1)
ki_buy_threshold = st.sidebar.slider("KI Lågkonjunktur (Köp-tröskel)", min_value=60, max_value=110, value=90, step=1)
ki_duration_months = st.sidebar.slider("Varaktighet (månader krävs i rad)", min_value=1, max_value=12, value=3, step=1)

# --- DATACACHING ---
@st.cache_data(ttl=3600, show_spinner="Hämtar marknadsdata...")
def load_data(ticker: str, sma_period: int):
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    raw_ticker = yf.download(ticker, start="2000-01-01", end=today, progress=False, auto_adjust=False)
    raw_vix = yf.download("^VIX", start="2000-01-01", end=today, progress=False, auto_adjust=False)
    
    if raw_ticker.empty:
        return None

    df = pd.DataFrame(index=raw_ticker.index)
    
    if isinstance(raw_ticker.columns, pd.MultiIndex):
        df["Close"] = raw_ticker["Close"].iloc[:, 0]
    else:
        df["Close"] = raw_ticker["Close"]
        
    if not raw_vix.empty:
        if isinstance(raw_vix.columns, pd.MultiIndex):
            df["VIX"] = raw_vix["Close"].iloc[:, 0]
        else:
            df["VIX"] = raw_vix["Close"]
        df["VIX"] = df["VIX"].ffill()
    else:
        df["VIX"] = np.nan

    df.dropna(subset=["Close"], inplace=True)
    df["RSI"] = ta.rsi(df["Close"], length=14)
    df["SMA_200"] = ta.sma(df["Close"], length=sma_period)
    
    macd = ta.macd(df["Close"], fast=24, slow=52, signal=9)
    df["MACD"] = macd.iloc[:, 0]
    df["MACD_Signal"] = macd.iloc[:, 2]

    return df


@st.cache_data(ttl=3600, show_spinner="Hämtar OMXS30-data...")
def load_index_data(secondary_ticker: str = "^OMX"):
    today = datetime.date.today().strftime("%Y-%m-%d")
    try:
        raw_index = yf.download(secondary_ticker, start="2000-01-01", end=today, progress=False, auto_adjust=False)
        if raw_index.empty:
            return None

        df_idx = pd.DataFrame(index=raw_index.index)
        if isinstance(raw_index.columns, pd.MultiIndex):
            df_idx["Close"] = raw_index["Close"].iloc[:, 0]
        else:
            df_idx["Close"] = raw_index["Close"]

        df_idx.dropna(subset=["Close"], inplace=True)
        return df_idx
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner="Hämtar Konjunkturbarometer från KI...")
def load_ki_barometer():
    """
    Hämtar Barometerindikatorn via KI:s PXWeb-API med robust felhantering
    och förfalskad User-Agent för att undvika blockering.
    """
    url = "https://statistik.konj.se/PXWeb/api/v1/sv/KonjBar/Barometerindikatorn/BARTOTNMN.px"
    
    # Lägg till en standard User-Agent för att undvika 403 Forbidden från API-brandväggen
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Vi skapar en backup-fråga utifall den specifika koden 'BARTOT' ändrats i framtiden
    queries = [
        {
            "query": [{"code": "ContentsCode", "selection": {"filter": "item", "values": ["BARTOT"]}}],
            "response": {"format": "json"}
        },
        {
            "query": [{"code": "ContentsCode", "selection": {"filter": "all", "values": ["*"]}}],
            "response": {"format": "json"}
        }
    ]

    for query in queries:
        try:
            resp = requests.post(url, json=query, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
                
            data = resp.json()
            columns = data.get("columns", [])
            data_rows = data.get("data", [])

            if not data_rows:
                continue

            # Identifiera tidskolumnen dynamiskt
            time_col_idx = None
            for idx, col in enumerate(columns):
                if col.get("code", "").lower() in ("tid", "month", "manad", "månad", "time"):
                    time_col_idx = idx
                    break
            if time_col_idx is None:
                time_col_idx = 0

            records = []
            for row in data_rows:
                key = row.get("key", [])
                values = row.get("values", [])
                if not key or not values:
                    continue

                period_str = key[time_col_idx]
                # Säkerställ ren tidsträng oavsett om formatet är 2024M01, 2024-01 eller 202401
                clean_str = period_str.replace("M", "").replace("-", "").strip()
                
                try:
                    year = int(clean_str[:4])
                    month = int(clean_str[4:6])
                    period_date = pd.Timestamp(year=year, month=month, day=1)
                except (ValueError, IndexError):
                    continue

                try:
                    # Byt ut eventuella europeiska kommatecken mot punkter innan konvertering till float
                    val_str = values[0].replace(",", ".")
                    value = float(val_str)
                except (ValueError, IndexError, TypeError):
                    continue

                records.append({"Date": period_date, "Barometer": value})

            if records:
                df_ki = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
                df_ki.set_index("Date", inplace=True)
                return df_ki

        except Exception:
            continue

    return None


# --- HÄMTA DATA ---
df = load_data(ticker, sma_trend_period)

if df is None or df.empty:
    st.error(f"❌ Kunde inte hämta data för **{ticker}**. Kontrollera tickern och försök igen.")
    st.stop()

# --- SENASTE VÄRDEN ---
latest = df.iloc[-1]
latest_price = float(latest["Close"])
latest_vix = float(latest["VIX"]) if not np.isnan(latest["VIX"]) else None
latest_rsi = float(latest["RSI"]) if not np.isnan(latest["RSI"]) else None
latest_sma = float(latest["SMA_200"]) if not np.isnan(latest["SMA_200"]) else None
trend_is_positive = (latest_sma is not None) and (latest_price > latest_sma)

# --- TREND HEATMAP BERÄKNING ---
trend_streaks = []
current_streak = 0
current_trend = 0
closes = df["Close"].values
smas = df["SMA_200"].values

for i in range(len(df)):
    price = closes[i]
    sma = smas[i]
    if np.isnan(sma):
        trend_streaks.append(0)
        continue
    trend_now = 1 if price > sma else -1
    if trend_now == current_trend:
        current_streak += 1
    else:
        current_trend = trend_now
        current_streak = 1
    trend_streaks.append(current_streak * current_trend)

df["Streak"] = trend_streaks

# --- SIGNALBERÄKNING ---
visual_last_buy_idx = -999
visual_last_sell_idx = -999

buy_pos_price, buy_pos_date = [], []
buy_neg_price, buy_neg_date = [], []
sell_signals_price, sell_signals_date = [], []
macro_buy_pos_price, macro_buy_pos_date = [], []
macro_buy_neg_price, macro_buy_neg_date = [], []

last_signal_type = None
last_signal_date = None
last_signal_price = None

for i in range(len(df)):
    current_rsi = df["RSI"].iloc[i]
    current_macd = df["MACD"].iloc[i]
    current_signal_val = df["MACD_Signal"].iloc[i]
    prev_macd = df["MACD"].iloc[i - 1] if i > 0 else 0
    prev_signal_val = df["MACD_Signal"].iloc[i - 1] if i > 0 else 0
    current_idx = i
    price = df["Close"].iloc[i]
    current_vix = df["VIX"].iloc[i]
    sma = smas[i]

    if pd.isna(current_rsi) or pd.isna(current_macd) or pd.isna(current_vix):
        continue

    is_buy_signal = False

    if current_vix > vix_panic_threshold:
        if (current_idx - visual_last_buy_idx) > cooldown_days:
            visual_last_buy_idx = current_idx
            is_buy_signal = True
            last_signal_type = "VIX Panik KÖP"
            last_signal_date = df.index[i]
            last_signal_price = price
            if not np.isnan(sma) and price > sma:
                macro_buy_pos_price.append(price)
                macro_buy_pos_date.append(df.index[i])
            else:
                macro_buy_neg_price.append(price)
                macro_buy_neg_date.append(df.index[i])

    if not is_buy_signal:
        if current_rsi < buy_rsi_threshold:
            if (current_idx - visual_last_buy_idx) > cooldown_days:
                visual_last_buy_idx = current_idx
                is_buy_signal = True
                last_signal_type = "RSI KÖP"
                last_signal_date = df.index[i]
                last_signal_price = price
                if not np.isnan(sma) and price > sma:
                    buy_pos_price.append(price)
                    buy_pos_date.append(df.index[i])
                else:
                    buy_neg_price.append(price)
                    buy_neg_date.append(df.index[i])

    if (
        (prev_macd > prev_signal_val)
        and (current_macd < current_signal_val)
        and (current_rsi > sell_macd_rsi_filter)
    ):
        if (current_idx - visual_last_sell_idx) > cooldown_days:
            visual_last_sell_idx = current_idx
            last_signal_type = "MACD SÄLJ"
            last_signal_date = df.index[i]
            last_signal_price = price
            sell_signals_price.append(price)
            sell_signals_date.append(df.index[i])

# --- SIDTITEL ---
st.title(f"📈 Trend Heatmap — {ticker}")

# --- FLIKAR ---
tab1, tab2 = st.tabs(["Teknisk Trendanalys", "Makro- & Konjunkturmodell"])

# ============================================================
# FLIK 1: TEKNISK TRENDANALYS
# ============================================================
with tab1:

    # Rad 1: Nyckelmetrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        trend_label = "📗 POSITIV" if trend_is_positive else "📕 NEGATIV"
        days_in_trend = abs(trend_streaks[-1]) if trend_streaks else 0
        st.metric("Nuvarande trend", trend_label, delta=f"{days_in_trend} dagar")

    with col2:
        if latest_rsi is not None:
            rsi_delta = f"Tröskel: {buy_rsi_threshold}"
            st.metric("RSI (14)", f"{latest_rsi:.1f}", delta=rsi_delta, delta_color="off")
        else:
            st.metric("RSI (14)", "N/A")

    with col3:
        if last_signal_date is not None:
            date_str = pd.Timestamp(last_signal_date).strftime("%Y-%m-%d")
            emoji = "🟢" if "KÖP" in last_signal_type else "🔴"
            st.metric("Senaste signal", f"{emoji} {last_signal_type}", delta=date_str, delta_color="off")
        else:
            st.metric("Senaste signal", "Ingen ännu")

    with col4:
        st.metric(f"Pris ({ticker})", f"${latest_price:.2f}")

    st.markdown("---")

    # Rad 2: VIX-panikmätare
    st.subheader("🌡️ VIX-panikmätare")

    if latest_vix is not None:
        vix_col1, vix_col2 = st.columns([2, 3])
        with vix_col1:
            vix_diff = latest_vix - vix_panic_threshold
            vix_status = "🚨 PANIK — Signal aktiv!" if latest_vix > vix_panic_threshold else "✅ Normalt läge"
            st.metric(
                "VIX just nu",
                f"{latest_vix:.2f}",
                delta=f"{vix_diff:+.2f} vs tröskel ({vix_panic_threshold})",
                delta_color="inverse",
            )
            st.markdown(f"**Status:** {vix_status}")

        with vix_col2:
            vix_pct = min(latest_vix / (vix_panic_threshold * 1.5), 1.0)
            bar_color = "🟥" if latest_vix > vix_panic_threshold else ("🟨" if latest_vix > vix_panic_threshold * 0.75 else "🟩")
            filled = int(vix_pct * 20)
            bar_str = bar_color * filled + "⬜" * (20 - filled)
            st.markdown(f"**VIX-nivå:** `{bar_str}`")
            st.caption(f"0 ──────── {vix_panic_threshold:.0f} (tröskel) ──────── {vix_panic_threshold * 1.5:.0f}")
    else:
        st.info("VIX-data ej tillgänglig.")

    st.markdown("---")

    # --- GRAF ---
    fig = go.Figure()

    pos_mask = df["Streak"] > 0
    fig.add_trace(
        go.Bar(
            x=df.index[pos_mask],
            y=(df["Close"] - df["SMA_200"])[pos_mask],
            base=df["SMA_200"][pos_mask],
            marker=dict(
                color=df["Streak"][pos_mask],
                colorscale=[[0, "#e6ffe6"], [1, "#004d00"]],
                cmin=0,
                cmax=color_intensity_cap,
                line_width=0,
            ),
            name="Positiv Trend",
            hoverinfo="skip",
        )
    )

    neg_mask = df["Streak"] < 0
    fig.add_trace(
        go.Bar(
            x=df.index[neg_mask],
            y=(df["Close"] - df["SMA_200"])[neg_mask],
            base=df["SMA_200"][neg_mask],
            marker=dict(
                color=df["Streak"][neg_mask].abs(),
                colorscale=[[0, "#ffe6e6"], [1, "#800000"]],
                cmin=0,
                cmax=color_intensity_cap,
                line_width=0,
            ),
            name="Negativ Trend",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["SMA_200"], mode="lines",
            name=f"SMA {sma_trend_period}", line=dict(color="royalblue", width=1)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Close"], mode="lines",
            name=f"Pris ({ticker})", line=dict(color="white" if not trend_is_positive else "black", width=1.5)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=buy_neg_date, y=buy_neg_price, mode="markers",
            name="Köp (Negativ Trend)",
            marker=dict(symbol="triangle-up", color="rgba(0, 255, 0, 0.7)", size=10, line=dict(color="black", width=1)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=buy_pos_date, y=buy_pos_price, mode="markers",
            name="Köp (Positiv Trend)",
            marker=dict(symbol="triangle-up", color="deepskyblue", size=12, line=dict(color="white", width=1)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sell_signals_date, y=sell_signals_price, mode="markers",
            name="MACD Vändning (Sälj)",
            marker=dict(symbol="triangle-down", color="rgba(255, 80, 80, 0.8)", size=10, line=dict(color="black", width=1)),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=macro_buy_neg_date, y=macro_buy_neg_price, mode="markers",
            name="VIX Panik (Negativ Trend)",
            marker=dict(symbol="diamond", color="cyan", size=14, line=dict(color="black", width=1.5)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=macro_buy_pos_date, y=macro_buy_pos_price, mode="markers",
            name="VIX Panik (Positiv Trend)",
            marker=dict(symbol="diamond", color="violet", size=14, line=dict(color="white", width=1.5)),
        )
    )

    fig.update_layout(
        yaxis_title="Pris (USD)",
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=False), type="date"),
        bargap=0,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=80, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Data via Yahoo Finance · VIX = ^VIX · Uppdaterad: {datetime.date.today().strftime('%Y-%m-%d')} · "
        f"Cache: 1 timme"
    )

# ============================================================
# FLIK 2: MAKRO- & KONJUNKTURMODELL (KI BAROMETER)
# ============================================================
with tab2:
    st.subheader("📊 Konjunkturbarometermodell (KI)")
    st.caption(
        "Barometerindikatorn från Konjunkturinstitutet jämförs mot tröskelvärden "
        "för att identifiera över- och underhettning i ekonomin."
    )

    df_ki = load_ki_barometer()

    if df_ki is None or df_ki.empty:
        st.warning(
            "⚠️ Kunde inte hämta data från Konjunkturinstitutets API (statistik.konj.se). "
            "Detta kan bero på tillfälliga nätverksproblem eller tillfällig blockering. "
            "Försök ladda om sidan om en liten stund."
        )
    else:
        secondary_ticker = "^OMX"
        df_index = load_index_data(secondary_ticker)

        ki_daily = df_ki.reindex(
            pd.date_range(start=df_ki.index.min(), end=df.index.max(), freq="D")
        )
        ki_daily["Barometer"] = ki_daily["Barometer"].ffill()

        df_combo = df[["Close"]].copy()
        df_combo = df_combo.join(ki_daily["Barometer"], how="left")
        df_combo["Barometer"] = df_combo["Barometer"].ffill()

        if df_index is not None and not df_index.empty:
            df_combo = df_combo.join(df_index["Close"].rename("Index_Close"), how="left")
            df_combo["Index_Close"] = df_combo["Index_Close"].ffill()

        df_combo.dropna(subset=["Barometer"], inplace=True)

        # --- KI-SIGNALLOGIK ---
        ki_signal_dates_buy = []
        ki_signal_values_buy = []
        ki_signal_dates_sell = []
        ki_signal_values_sell = []

        consecutive_over = 0
        consecutive_under = 0
        ki_last_signal_idx_buy = -999
        ki_last_signal_idx_sell = -999

        ki_values = df_ki["Barometer"].values
        ki_dates = df_ki.index

        last_ki_signal_type = None
        last_ki_signal_date = None
        last_ki_signal_value = None

        for i in range(len(df_ki)):
            val = ki_values[i]

            if val > ki_sell_threshold:
                consecutive_over += 1
            else:
                consecutive_over = 0

            if val < ki_buy_threshold:
                consecutive_under += 1
            else:
                consecutive_under = 0

            if consecutive_over >= ki_duration_months:
                if (i - ki_last_signal_idx_sell) > ki_duration_months:
                    ki_last_signal_idx_sell = i
                    ki_signal_dates_sell.append(ki_dates[i])
                    ki_signal_values_sell.append(val)
                    last_ki_signal_type = "SÄLJ/KONTANT"
                    last_ki_signal_date = ki_dates[i]
                    last_ki_signal_value = val

            if consecutive_under >= ki_duration_months:
                if (i - ki_last_signal_idx_buy) > ki_duration_months:
                    ki_last_signal_idx_buy = i
                    ki_signal_dates_buy.append(ki_dates[i])
                    ki_signal_values_buy.append(val)
                    last_ki_signal_type = "KÖP/AKTIER"
                    last_ki_signal_date = ki_dates[i]
                    last_ki_signal_value = val

        # --- NYCKELMETRICS FLIK 2 ---
        ki_col1, ki_col2, ki_col3 = st.columns(3)

        with ki_col1:
            latest_ki_value = float(df_ki["Barometer"].iloc[-1])
            if latest_ki_value > ki_sell_threshold:
                ki_status = "🔴 ÖVERHETTNING"
            elif latest_ki_value < ki_buy_threshold:
                ki_status = "🟢 LÅGKONJUNKTUR"
            else:
                ki_status = "⚪ NEUTRAL"
            st.metric("Barometerindikator (senaste)", f"{latest_ki_value:.1f}", delta=ki_status, delta_color="off")

        with ki_col2:
            latest_ki_date = pd.Timestamp(df_ki.index[-1]).strftime("%Y-%m")
            st.metric("Senaste KI-data", latest_ki_date)

        with ki_col3:
            if last_ki_signal_date is not None:
                date_str = pd.Timestamp(last_ki_signal_date).strftime("%Y-%m")
                emoji = "🟢" if "KÖP" in last_ki_signal_type else "🔴"
                st.metric("Senaste KI-signal", f"{emoji} {last_ki_signal_type}", delta=date_str, delta_color="off")
            else:
                st.metric("Senaste KI-signal", "Ingen ännu")

        st.markdown("---")

        # --- DUAL-AXIS GRAF ---
        fig_ki = make_subplots(specs=[[{"secondary_y": True}]])

        if not df_combo.empty:
            normalized_main = (df_combo["Close"] / df_combo["Close"].iloc[0]) * 100

            fig_ki.add_trace(
                go.Scatter(
                    x=df_combo.index, y=normalized_main, mode="lines",
                    name=f"{ticker} (normaliserad)",
                    line=dict(color="royalblue", width=1.5),
                ),
                secondary_y=False,
            )

            if "Index_Close" in df_combo.columns and df_combo["Index_Close"].notna().any():
                df_idx_valid = df_combo.dropna(subset=["Index_Close"])
                if not df_idx_valid.empty:
                    normalized_index = (df_idx_valid["Index_Close"] / df_idx_valid["Index_Close"].iloc[0]) * 100
                    fig_ki.add_trace(
                        go.Scatter(
                            x=df_idx_valid.index, y=normalized_index, mode="lines",
                            name=f"{secondary_ticker} (normaliserad)",
                            line=dict(color="darkorange", width=1.5),
                        ),
                        secondary_y=False,
                    )

            fig_ki.add_trace(
                go.Scatter(
                    x=df_combo.index, y=df_combo["Barometer"], mode="lines",
                    name="Barometerindikator (KI)",
                    line=dict(color="rgba(150,150,150,0.5)", width=2),
                ),
                secondary_y=True,
            )

            fig_ki.add_hline(
                y=ki_sell_threshold, line_dash="dash", line_color="rgba(200,0,0,0.6)",
                annotation_text=f"Sälj-tröskel ({ki_sell_threshold})",
                annotation_position="top left",
                secondary_y=True,
            )
            fig_ki.add_hline(
                y=ki_buy_threshold, line_dash="dash", line_color="rgba(0,150,0,0.6)",
                annotation_text=f"Köp-tröskel ({ki_buy_threshold})",
                annotation_position="bottom left",
                secondary_y=True,
            )

            if ki_signal_dates_buy:
                buy_y = []
                for d in ki_signal_dates_buy:
                    nearest = normalized_main.index.searchsorted(d)
                    nearest = min(nearest, len(normalized_main) - 1)
                    buy_y.append(normalized_main.iloc[nearest])
                fig_ki.add_trace(
                    go.Scatter(
                        x=ki_signal_dates_buy, y=buy_y, mode="markers",
                        name="KÖP/AKTIER (KI-signal)",
                        marker=dict(symbol="triangle-up", color="lime", size=14, line=dict(color="black", width=1.5)),
                    ),
                    secondary_y=False,
                )

            if ki_signal_dates_sell:
                sell_y = []
                for d in ki_signal_dates_sell:
                    nearest = normalized_main.index.searchsorted(d)
                    nearest = min(nearest, len(normalized_main) - 1)
                    sell_y.append(normalized_main.iloc[nearest])
                fig_ki.add_trace(
                    go.Scatter(
                        x=ki_signal_dates_sell, y=sell_y, mode="markers",
                        name="SÄLJ/KONTANT (KI-signal)",
                        marker=dict(symbol="triangle-down", color="red", size=14, line=dict(color="black", width=1.5)),
                    ),
                    secondary_y=False,
                )

        fig_ki.update_layout(
            hovermode="x unified",
            xaxis=dict(rangeslider=dict(visible=False), type="date"),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            margin=dict(l=10, r=10, t=80, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        fig_ki.update_yaxes(title_text="Pris (normaliserad, start = 100)", secondary_y=False, showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        fig_ki.update_yaxes(title_text="Barometerindikator (KI)", secondary_y=True, showgrid=False)
        fig_ki.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")

        st.plotly_chart(fig_ki, use_container_width=True)

        st.caption(
            f"Data via Konjunkturinstitutet (statistik.konj.se) och Yahoo Finance · "
            f"Jämförelseindex: {secondary_ticker} (OMXS30) · "
            f"Uppdaterad: {datetime.date.today().strftime('%Y-%m-%d')} · Cache: KI 24h / marknadsdata 1h"
        )

```
