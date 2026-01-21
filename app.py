import warnings
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from datetime import datetime
import pytz
import streamlit as st

# تعطيل التحذيرات المزعجة
warnings.filterwarnings("ignore")

# --- القائمة المحدثة: الأصول الأكثر سيولة ---
ASSETS = {
    # العملات الرقمية
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'SOL-USD': 'Solana',
    # الفوركس (الأزواج الرئيسية)
    'EURUSD=X': 'EUR/USD',
    'GBPUSD=X': 'GBP/USD',
    'USDJPY=X': 'USD/JPY',
    'AUDUSD=X': 'AUD/USD',
    'USDCAD=X': 'USD/CAD',
    'USDCHF=X': 'USD/CHF',
    'EURJPY=X': 'EUR/JPY',
    'GBPJPY=X': 'GBP/JPY',
    # المؤشرات والسلع
    'NQ=F': 'Nasdaq 100',
    'YM=F': 'US30 (Dow Jones)',
    '^GSPC': 'S&P 500',
    'GC=F': 'Gold',
    'SI=F': 'Silver',
    'CL=F': 'Crude Oil',
    '^GDAXI': 'DAX 40'
}

# ======== دوال المساعدة ========
def get_morocco_time():
    tz = pytz.timezone('Africa/Casablanca')
    return datetime.now(tz)

def get_session_info():
    now_mar = get_morocco_time()
    h = now_mar.hour
    if 8 <= h < 14: 
        return "LONDON", "🇬🇧 LONDON", "green"
    elif 14 <= h < 18: 
        return "NY-LON", "🌍 NY-LON", "purple"
    elif 18 <= h < 22: 
        return "NEW YORK", "🇺🇸 NEW YORK", "blue"
    else: 
        return "ASIAN", "🌏 ASIAN", "orange"

def is_low_liquidity(asset_name, session):
    if any(x in asset_name for x in ["Nasdaq", "US30", "S&P 500"]):
        return session not in ["NEW YORK", "NY-LON"]
    if "DAX" in asset_name:
        return session not in ["LONDON", "NY-LON"]
    if any(x in asset_name for x in ["Bitcoin", "Ethereum", "Solana"]):
        return False
    return False

def get_relative_strength():
    check_pairs = {'EURUSD=X': 'EUR', 'GBPUSD=X': 'GBP', 'USDJPY=X': 'JPY', 'AUDUSD=X': 'AUD'}
    s_map = {'USD': 0.0}
    for sym, name in check_pairs.items():
        try:
            d = yf.download(sym, period='5d', interval='15m', progress=False)
            if d.empty: continue
            change = ((d['Close'].iloc[-1] - d['Close'].iloc[-20]) / d['Close'].iloc[-20]) * 100
            s_map[name] = -float(change) if name == 'JPY' else float(change)
        except: 
            s_map[name] = 0
    return s_map

def compute_engine_v14_6(df, symbol, s_map):
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)
    
    df['ATR'] = pd.concat([
        df['High']-df['Low'], 
        (df['High']-df['Close'].shift()).abs(), 
        (df['Low']-df['Close'].shift()).abs()
    ], axis=1).max(axis=1).rolling(14).mean()
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Trap'] = np.where((df['Close'].diff(5) > 0) & (df['RSI'].diff(5) < -5), 1, 0)
    df['Trap'] = np.where((df['Close'].diff(5) < 0) & (df['RSI'].diff(5) > 5), -1, df['Trap'])
    
    df['MA50'] = df['Close'].rolling(50).mean()
    df['Returns'] = df['Close'].pct_change(1)
    
    df['CS'] = 0.0
    for curr, val in s_map.items():
        if curr in symbol: 
            df['CS'] = val if symbol.startswith(curr) else -val
            
    df['Target'] = np.select([
        (df['Close'].shift(-4)-df['Close'] > df['ATR']*0.5), 
        (df['Close'].shift(-4)-df['Close'] < -df['ATR']*0.5)
    ], [1, 2], default=0)
    
    return df.dropna()

def get_ai_prediction(df):
    features = ['RSI', 'Returns', 'CS']
    X = df[features].values
    y = df['Target'].values
    
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, verbosity=0, random_state=42)
    model.fit(X[:-4], y[:-4])
    
    probs = model.predict_proba(X[-1:])[0]
    prediction = np.argmax(probs)
    confidence = probs[prediction] * 100
    return prediction, confidence

def run_analysis():
    session_id, session_txt, session_color = get_session_info()
    s_map = get_relative_strength()
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_assets = len(ASSETS)
    
    for idx, (sym, name) in enumerate(ASSETS.items()):
        try:
            status_text.text(f"جاري تحليل {name}...")
            progress_bar.progress((idx + 1) / total_assets)
            
            d1h_raw = yf.download(sym, period='60d', interval='1h', progress=False)
            if d1h_raw.empty: continue
            d1h = compute_engine_v14_6(d1h_raw, sym, s_map)
            t_idx, t_cf = get_ai_prediction(d1h)
            
            d15_raw = yf.download(sym, period='40d', interval='15m', progress=False)
            if d15_raw.empty: continue
            d15 = compute_engine_v14_6(d15_raw, sym, s_map)
            e_idx, e_cf = get_ai_prediction(d15)
            
            current_p = float(d15['Close'].iloc[-1])
            ma50 = float(d15['MA50'].iloc[-1])
            atr = float(d15['ATR'].iloc[-1])
            is_trap = d15['Trap'].iloc[-1] != 0
            
            low_liq = is_low_liquidity(name, session_id)
            
            prec = 5 if current_p < 10 else (2 if current_p > 1000 else 4)
            
            score = int((e_cf * 0.5) + (t_cf * 0.3) + 20)
            if is_trap: score -= 20
            
            trend = "↑" if current_p > ma50 else "↓"
            entry_txt = f"{current_p:,.{prec}f} {trend}"
            
            sig, qual, tp, sl = "WAIT ⏳", "NEUTRAL", "-", "-"
            sig_type = "wait"

            if low_liq:
                sig, qual = "LOW LIQ", "COLD"
                sig_type = "low_liq"
            elif is_trap:
                qual = "⚠️ TRAP"
                sig = "AVOID"
                sig_type = "trap"
            elif e_idx != 0 and e_idx == t_idx:
                is_buy = e_idx == 1
                sig = "BUY 🚀" if is_buy else "SELL 📉"
                sig_type = "buy" if is_buy else "sell"
                tp_v = current_p + (atr * 1.2) if is_buy else current_p - (atr * 1.2)
                sl_v = current_p - (atr * 1.8) if is_buy else current_p + (atr * 1.8)
                tp, sl = f"{tp_v:,.{prec}f}", f"{sl_v:,.{prec}f}"
                
                if score >= 75: qual = "★ ELITE"
                elif score >= 65: qual = "💎 HIGH"
                else: qual = "⚡ SCALP"

            results.append({
                'ASSET': name,
                'ENTRY': entry_txt,
                'TREND': trend,
                'CONF%': f"{e_cf:.1f}%",
                'SCORE': score,
                'SIGNAL': sig,
                'SIGNAL_TYPE': sig_type,
                'TP': tp,
                'SL': sl,
                'QUALITY': qual,
                'is_low': 1 if low_liq else 0
            })
        except Exception as e:
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return results, session_txt, session_color

# ==================== STREAMLIT UI ====================

st.set_page_config(
    page_title="QUANT-CORE ZOZO v14.6.5",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS للخلفية الغامقة
st.markdown("""
<style>
    .stApp { background-color: #1e1e1e; color: #e0e0e0; }
    .main-header {
        background: linear-gradient(135deg, #0f111a 0%, #1e1e2f 100%);
        padding: 20px; border-radius: 12px; margin-bottom: 20px; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .main-header h1 { color: #00ff99; margin:0; font-size:2.5rem; text-shadow:1px 1px 4px #000; }
    .stDataFrame td, .stDataFrame th { color: #e0e0e0 !important; background-color: #1e1e1e !important; }
    div[data-testid="metric-container"] {
        background: #2a2a3d; padding:15px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.3);
        border-left: 4px solid #00ff99; color:#e0e0e0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00ff99 0%, #00cc77 100%);
        color: #1e1e1e; border:none; border-radius:8px; padding:12px 24px; font-weight:bold;
        box-shadow: 0 3px 10px rgba(0,0,0,0.5);
    }
    .stButton > button:hover { background: linear-gradient(135deg, #00cc77 0%, #00ff99 100%); }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏛️ QUANT-CORE ZOZO v14.6.5</h1>
    <p style="color:#99ffcc; margin-top:10px;">نظام التحليل الكمي المتقدم للأسواق المالية</p>
</div>
""", unsafe_allow_html=True)

# عرض الوقت والجلسة
col1, col2, col3 = st.columns(3)
tm_mar = get_morocco_time().strftime('%Y-%m-%d %H:%M:%S')
session_id, session_txt, session_color = get_session_info()
with col1: st.metric("🕐 توقيت المغرب", tm_mar)
with col2: st.metric("📊 الجلسة الحالية", session_txt)
with col3: st.metric("📈 عدد الأصول", len(ASSETS))
st.markdown("---")

# زر التحليل
if st.button("🚀 بدء التحليل", type="primary", use_container_width=True):
    with st.spinner("جاري تحليل الأسواق..."):
        results, session_txt, session_color = run_analysis()
    
    if results:
        df_results = pd.DataFrame(results).sort_values(by=['is_low','SCORE'], ascending=[True, False])
        st.markdown("### 📊 ملخص التحليل")
        col1, col2, col3, col4, col5 = st.columns(5)
        buy_count = len(df_results[df_results['SIGNAL_TYPE']=='buy'])
        sell_count = len(df_results[df_results['SIGNAL_TYPE']=='sell'])
        wait_count = len(df_results[df_results['SIGNAL_TYPE']=='wait'])
        trap_count = len(df_results[df_results['SIGNAL_TYPE']=='trap'])
        low_liq_count = len(df_results[df_results['SIGNAL_TYPE']=='low_liq'])
        with col1: st.metric("🟢 إشارات شراء", buy_count)
        with col2: st.metric("🔴 إشارات بيع", sell_count)
        with col3: st.metric("🔵 انتظار", wait_count)
        with col4: st.metric("⚠️ مصيدة", trap_count)
        with col5: st.metric("⚪ سيولة منخفضة", low_liq_count)
        st.markdown("---")
        st.markdown("### 🎯 تصفية النتائج")
        filter_option = st.selectbox("اختر نوع الإشارة:", ["الكل"])
        df_display = df_results.copy()
        df_show = df_display[['ASSET','ENTRY','CONF%','SCORE','SIGNAL','TP','SL','QUALITY']].copy()

        # تلوين الجدول حسب الإشارة
        def highlight_signals_dark(row):
            if 'BUY' in row['SIGNAL']:
                return ['background-color: #004d00; color: #00ff00'] * len(row)
            elif 'SELL' in row['SIGNAL']:
                return ['background-color: #660000; color: #ff3333'] * len(row)
            elif 'LOW LIQ' in row['SIGNAL']:
                return ['background-color: #333333; color: #cccccc'] * len(row)
            elif 'AVOID' in row['SIGNAL']:
                return ['background-color: #664400; color: #ffcc00'] * len(row)
            else:
                return ['background-color: #222233; color: #99ccff'] * len(row)

        styled_df = df_show.style.apply(highlight_signals_dark, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=600)

        # عرض الإشارات النشطة
        active_signals = df_results[df_results['SIGNAL_TYPE'].isin(['buy','sell'])]
        if not active_signals.empty:
            st.markdown("---")
            st.markdown("### 🎯 الإشارات النشطة - تفاصيل")
            for _, row in active_signals.iterrows():
                with st.expander(f"{'🟢' if row['SIGNAL_TYPE']=='buy' else '🔴'} {row['ASSET']} - {row['SIGNAL']}"):
                    col1,col2,col3,col4 = st.columns(4)
                    with col1: st.metric("سعر الدخول", row['ENTRY'])
                    with col2: st.metric("الهدف (TP)", row['TP'])
                    with col3: st.metric("وقف الخسارة (SL)", row['SL'])
                    with col4: st.metric("الجودة", row['QUALITY'])
                    st.progress(row['SCORE']/100)
                    st.caption(f"Score: {row['SCORE']}/100 | Confidence: {row['CONF%']}")

        st.markdown("---")
        st.warning("⚠️ تنبيه: هذا التحليل للأغراض التعليمية فقط. تداول بحذر واستخدم إدارة مخاطر صارمة.")
    else:
        st.error("❌ لم يتم العثور على بيانات. تحقق من اتصال الإنترنت.")

else:
    st.info("👆 اضغط على زر 'بدء التحليل' لتحليل الأسواق")
    st.markdown("### 📋 الأصول المتاحة للتحليل")
    col1,col2,col3 = st.columns(3)
    assets_list = list(ASSETS.values())
    with col1: 
        st.markdown("*العملات الرقمية:*")
        for asset in assets_list[:3]: st.write(f"• {asset}")
    with col2:
        st.markdown("*الفوركس:*")
        for asset in assets_list[3:11]: st.write(f"• {asset}")
    with col3:
        st.markdown("*المؤشرات والسلع:*")
        for asset in assets_list[11:]: st.write(f"• {asset}")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #e0e0e0; padding: 20px; background: #2a2a3d; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
    <p>🏛️ QUANT-CORE ZOZO v14.6.5 | Powered by XGBoost & YFinance</p>
    <p style="color: #cccccc;">© 2024 - نظام التحليل الكمي المتقدم</p>
</div>
""", unsafe_allow_html=True)