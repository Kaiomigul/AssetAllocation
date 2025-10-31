# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Asset Allocation – Reversão à Média", layout="wide")

# ===============================
# Paleta de cores
# ===============================
PRIMARY_1 = "#008082"   # principal 1
PRIMARY_2 = "#DFAC16"   # principal 2
SECONDARY_RANGE = ["#F7E4AF", "#F2F2F2", "#C00000", PRIMARY_1, PRIMARY_2]

# ===============================
# Utilitários
# ===============================
PT_MONTHS = {
    "jan": "Jan", "fev": "Feb", "mar": "Mar", "abr": "Apr",
    "mai": "May", "jun": "Jun", "jul": "Jul", "ago": "Aug",
    "set": "Sep", "out": "Oct", "nov": "Nov", "dez": "Dec"
}

def parse_pt_date(s: str) -> pd.Timestamp:
    s = s.strip().lower().replace("-", "/")
    for k, v in PT_MONTHS.items():
        if s.startswith(k):
            s = s.replace(k, v, 1)
            break
    parts = s.split("/")
    if len(parts[-1]) == 2:
        yy = int(parts[-1]); year = 2000 + yy
        s = "/".join(parts[:-1] + [str(year)])
    return pd.to_datetime("01 " + s, format="%d %b/%Y")

def parse_pct(col: pd.Series) -> pd.Series:
    s = pd.Series(col, copy=True)
    s = s.map(lambda x: None if pd.isna(x) else str(x).strip())
    def norm(x):
        if x is None: return None
        x = x.replace("\u2013", "-").replace("\u2014", "-")
        if x in {"-", "", "nan", "None"}: return None
        x = x.replace("%", "")
        if "," in x: x = x.replace(".", "")
        x = x.replace(",", ".")
        return x
    s = s.map(norm)
    return pd.to_numeric(s, errors="coerce") / 100.0

def to_pct(x): return f"{100*x:.2f}%"

def cum_base100(returns: pd.Series, start=100.0) -> pd.Series:
    return start * (1 + returns.fillna(0)).cumprod()

def ann_stats(returns: pd.Series, periods_per_year=12):
    m, s = returns.mean(), returns.std()
    ann_ret = (1 + m)**periods_per_year - 1
    ann_vol = s * np.sqrt(periods_per_year)
    sharpe = np.nan if ann_vol == 0 else ann_ret / ann_vol
    return ann_ret, ann_vol, sharpe

# ======= projeção p/ simplex com caixa (respeita MIN/MAX) =======
def project_to_bounds_simplex(v, lo, hi, tol=1e-12, max_iter=50):
    v = np.clip(v, lo, hi).astype(float)
    for _ in range(max_iter):
        s = v.sum()
        if abs(s - 1.0) < tol:
            break
        if s < 1.0:
            head = hi - v
            mask = head > tol
            total_head = head[mask].sum()
            if total_head <= tol:
                break
            v[mask] += (1.0 - s) * head[mask] / total_head
        else:
            surp = v - lo
            mask = surp > tol
            total_surp = surp[mask].sum()
            if total_surp <= tol:
                break
            v[mask] -= (s - 1.0) * surp[mask] / total_surp
        v = np.clip(v, lo, hi)
    return v

# ======= quantização 2,5 p.p. respeitando MIN/MAX e soma 100% =======
def quantize_row_to_step(v, lo, hi, step=0.025, tol=1e-9, max_passes=2000):
    v = project_to_bounds_simplex(v, lo, hi)
    vq = np.round(v / step) * step
    vq = np.clip(vq, lo, hi)
    diff = 1.0 - vq.sum()
    passes = 0
    while abs(diff) >= step/2 and passes < max_passes:
        if diff > 0:
            head = hi - vq
            idxs = np.where(head >= step)[0]
            if idxs.size == 0: break
            for i in idxs:
                add = min(step, head[i], diff)
                vq[i] += add
                diff -= add
                if diff < step/2:
                    break
        else:
            surp = vq - lo
            idxs = np.where(surp >= step)[0]
            if idxs.size == 0: break
            for i in idxs:
                sub = min(step, surp[i], -diff)
                vq[i] -= sub
                diff += sub
                if -diff < step/2:
                    break
        passes += 1
    if abs(1.0 - vq.sum()) > tol:
        if (1.0 - vq.sum()) > 0:
            head = (hi - vq)
            if head.max() > tol:
                j = np.argmax(head)
                vq[j] += min(1.0 - vq.sum(), head[j])
        else:
            surp = (vq - lo)
            if surp.max() > tol:
                j = np.argmax(surp)
                vq[j] -= min(vq.sum() - 1.0, surp[j])
    return np.clip(vq, lo, hi)

# ===============================
# Carregamento dos dados
# ===============================
@st.cache_data
def load_returns(economatica_path: str) -> pd.DataFrame:
    df = pd.read_csv(economatica_path, sep=";", dtype=str)
    df.rename(columns={df.columns[0]: "Data"}, inplace=True)
    df["Data"] = df["Data"].apply(parse_pt_date)
    df = df.set_index("Data").sort_index()
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
    for c in df.columns:
        df[c] = parse_pct(df[c])
    return df

@st.cache_data
def load_weights(weights_path: str) -> pd.DataFrame:
    w = pd.read_csv(weights_path, sep=";", dtype=str)
    w.columns = [c.strip().upper() for c in w.columns]
    for c in ["MIN", "NEUTRO", "MAX"]:
        w[c] = parse_pct(w[c])
    w = w.dropna(subset=["CLASSE"])
    w["CLASSE"] = w["CLASSE"].str.strip()
    w["PERFIL"] = w["PERFIL"].str.strip().str.capitalize()
    return w

ECON_PATH = "economatica.csv"
W_PATH = "PESOS_CLASSES.csv"

missing = [p for p in [ECON_PATH, W_PATH] if not Path(p).exists()]
if missing:
    st.error(f"Arquivo(s) não encontrado(s): {', '.join(missing)}. "
             "Coloque os CSVs na mesma pasta do app e atualize a página.")
    st.stop()

rets_full = load_returns(ECON_PATH)
weights_raw = load_weights(W_PATH)
CLASSES = list(rets_full.columns)

# ===============================
# Filtro de datas
# ===============================
st.sidebar.header("Período de análise")
min_date = rets_full.index.min().to_pydatetime()
max_date = rets_full.index.max().to_pydatetime()
start_date = st.sidebar.date_input("Data inicial", value=min_date, min_value=min_date, max_value=max_date)
end_date   = st.sidebar.date_input("Data final",   value=max_date, min_value=min_date, max_value=max_date)
if start_date > end_date:
    st.sidebar.error("⚠️ A data inicial deve ser anterior à data final.")
    st.stop()

rets = rets_full.loc[(rets_full.index >= pd.Timestamp(start_date)) &
                     (rets_full.index <= pd.Timestamp(end_date))]

# ===============================
# Sidebar – Parâmetros
# ===============================
st.sidebar.header("Parâmetros")

perfil = st.sidebar.selectbox(
    "Perfil da estratégia",
    options=sorted(
        weights_raw["PERFIL"].unique(),
        key=lambda x: ["Ultraconservador","Conservador","Moderado","Agressivo"].index(x)
        if x in ["Ultraconservador","Conservador","Moderado","Agressivo"] else 999
    ),
    index=0
)

classe_focus = st.sidebar.selectbox("Classe foco (Gráf. 0, 1 e 2)", options=CLASSES, index=0)

win_long  = st.sidebar.number_input("Média móvel LONGA (meses)",  min_value=3,  max_value=120, value=36, step=1)
win_short = st.sidebar.number_input("Média móvel CURTA (meses)",   min_value=1,  max_value=60,  value=6,  step=1)

std_win  = st.sidebar.number_input("Janela do desvio padrão (Z-score)", min_value=6,  max_value=120, value=36, step=1)
z_cap    = st.sidebar.slider("Saturação de Z (|Z| máximo)",             min_value=0.5, max_value=3.0,  value=2.0, step=0.1)
z_thresh = st.sidebar.slider("Zona neutra |Z| ≤",                        min_value=0.0, max_value=2.0,  value=0.5, step=0.1)

# >>> bins do histograma
bins_num = st.sidebar.slider("Histograma: nº de bins (Gráf. 0)", min_value=10, max_value=100, value=30, step=1, key="bins_num")

# >>> Estratégia (inclui Alternância)
strategy_mode = st.sidebar.radio(
    "Estratégia de tilt",
    ["Alternância dinâmica (ΔZ)", "Reversão à média (fixa)", "Momentum (fixo)"],
    index=0,
    help="Alternância: ΔZ determina qual regime usar; as opções fixas forçam um único regime.",
    key="strategy_mode"
)

# Parâmetros do ΔZ
delta_lookback = st.sidebar.number_input(
    "ΔZ: nº de observações (lookback)",
    min_value=1, max_value=24, value=2, step=1, key="delta_lookback"
)
delta_threshold = st.sidebar.slider(
    "Limiar |ΔZ| para alternar",
    min_value=0.0, max_value=2.0, value=0.10, step=0.05, key="delta_threshold"
)

# Agressividade global
aggr_factor = st.sidebar.slider(
    "Agressividade do tilt",
    min_value=0.0, max_value=2.0, value=1.0, step=0.1,
    help="Multiplica a intensidade do tilt quando houver regime ativo.",
    key="aggr_factor"
)

st.sidebar.subheader("Pesos táticos")
mode_pesos = st.sidebar.radio(
    "Modo dos pesos",
    options=["Contínuos (livres)", "Degraus de 2,5 p.p."],
    index=0,
    help="Degraus: arredonda a 2,5% respeitando MIN/MAX e mantém a soma em 100%.",
    key="mode_pesos"
)

# Trava de classe
frozen_classes = st.sidebar.multiselect(
    "Travar classe(s) na carteira tática",
    options=CLASSES,
    default=[],
    help="Classes travadas ficam exatamente no peso NEUTRO; somente as demais variam.",
    key="frozen_classes"
)

st.sidebar.subheader("Custo de transação")
use_cost = st.sidebar.checkbox("Aplicar custo na carteira tática", value=False, key="use_cost")
cost_pct = st.sidebar.number_input(
    "Custo (% sobre o valor negociado)",
    min_value=0.0, max_value=2.0, value=0.10, step=0.05,
    help="Turnover mensal = 0,5 × soma(|w_t − w_(t−1)|). Custo = turnover × %custo.",
    disabled=not use_cost,
    key="cost_pct_input"
) / 100.0

base = st.sidebar.number_input("Base inicial (Gráf. 4)", min_value=50.0, max_value=1000.0, value=100.0, step=10.0, key="base")
st.sidebar.caption("Obs.: Z-score = (média curta − média longa) / desvio-padrão (janela).  ΔZ = Z_t − Z_(t−lookback).")

# ===============================
# Gráf. 0 – Distribuição de retornos (classe foco)
# ===============================
serie_focus = rets[classe_focus].dropna()
mu, sigma = float(serie_focus.mean()), float(serie_focus.std())

st.markdown("### 0) Distribuição de retornos – classe selecionada")
hist = alt.Chart(pd.DataFrame({"Retorno": serie_focus})).mark_bar(
    opacity=0.85, color=PRIMARY_1
).encode(
    x=alt.X("Retorno:Q", bin=alt.Bin(maxbins=bins_num), axis=alt.Axis(format="%")),
    y=alt.Y("count():Q", title="Frequência")
)

mean_rule = alt.Chart(pd.DataFrame({"x":[mu]})).mark_rule(color=PRIMARY_2, strokeWidth=2).encode(x="x:Q")
sigma_rules = alt.Chart(pd.DataFrame({"x":[mu - sigma, mu + sigma]})).mark_rule(
    color="#999", strokeDash=[4,4]
).encode(x="x:Q")

st.altair_chart(hist + mean_rule + sigma_rules, use_container_width=True)
st.caption(f"Média = {to_pct(mu)} • Volatilidade (σ) = {to_pct(sigma)}")

# ===============================
# MMs e Z-score (classe foco)
# ===============================
serie    = rets[classe_focus].copy()
ma_long  = serie.rolling(win_long).mean()
ma_short = serie.rolling(win_short).mean()
std_roll = serie.rolling(std_win).std()
z_focus  = (ma_short - ma_long) / std_roll.replace(0, np.nan)

mm_df = pd.DataFrame({
    "Data": rets.index,
    f"MM Curta ({win_short})": ma_short.values,
    f"MM Longa ({win_long})":  ma_long.values
}).set_index("Data")

z_df = pd.DataFrame({"Data": rets.index, "Z-score": z_focus.values}).set_index("Data")

# ===============================
# Gráfico 1 – Médias (classe foco)
# ===============================
st.markdown("### 1) Média de longo prazo vs curto prazo – classe selecionada")
line1 = alt.Chart(mm_df.reset_index()).transform_fold(
    fold=[f"MM Curta ({win_short})", f"MM Longa ({win_long})"],
    as_=["Série", "Valor"]
).mark_line().encode(
    x=alt.X("Data:T", title="Data"),
    y=alt.Y("Valor:Q", title="Retorno médio mensal"),
    color=alt.Color("Série:N", scale=alt.Scale(range=[PRIMARY_1, PRIMARY_2]))
)
st.altair_chart(line1, use_container_width=True)

# ===============================
# Gráfico 2 – Evolução do Z-score (classe foco)
# ===============================
st.markdown("### 2) Evolução do Z-score – classe selecionada")
band_df = z_df.reset_index()
rule_up  = alt.Chart(band_df).mark_rule(color="#999", strokeDash=[3,3]).encode(y=alt.datum(z_thresh))
rule_dn  = alt.Chart(band_df).mark_rule(color="#999", strokeDash=[3,3]).encode(y=alt.datum(-z_thresh))
cap_up   = alt.Chart(band_df).mark_rule(color=PRIMARY_2, strokeDash=[6,4]).encode(y=alt.datum(z_cap))
cap_dn   = alt.Chart(band_df).mark_rule(color=PRIMARY_2, strokeDash=[6,4]).encode(y=alt.datum(-z_cap))
line2 = alt.Chart(band_df).mark_line(color=PRIMARY_1).encode(
    x="Data:T", y=alt.Y("Z-score:Q", title="Z-score")
)
st.altair_chart((line2 + rule_up + rule_dn + cap_up + cap_dn), use_container_width=True)

# ===============================
# Pesos (perfil) + edição
# ===============================
st.markdown("### 3) Pesos por estratégia (Neutro vs Tático) e evolução")
st.caption("Edite os pesos abaixo; os limites MIN/MAX são aplicados ao tático.")

weights_perfil = (
    weights_raw.query("PERFIL == @perfil")
               .set_index("CLASSE")
               .reindex(CLASSES)
)

edit_cols = ["MIN", "NEUTRO", "MAX"]
def _to_float_safe(x):
    try: return float(x)
    except Exception: return 0.0

weights_edit = st.data_editor(
    weights_perfil[edit_cols].rename(columns={"MIN":"Min", "NEUTRO":"Neutro", "MAX":"Max"}).applymap(_to_float_safe),
    num_rows="fixed",
    use_container_width=True
).rename(columns={"Min":"MIN","Neutro":"NEUTRO","Max":"MAX"})

sum_min, sum_max = weights_edit["MIN"].sum(), weights_edit["MAX"].sum()
if sum_min > 1 + 1e-9 or sum_max < 1 - 1e-9:
    st.error(f"As restrições são inviáveis: soma(MIN)={sum_min:.2f}, soma(MAX)={sum_max:.2f}. "
             "Ajuste os limites para que 1 esteja entre elas.")
    st.stop()

if not np.isclose(weights_edit["NEUTRO"].sum(), 1.0):
    weights_edit["NEUTRO"] = weights_edit["NEUTRO"] / weights_edit["NEUTRO"].sum()

# ===== Z e ΔZ para todas as classes =====
z_all = {}
for c in CLASSES:
    s = rets[c]
    ma_l = s.rolling(win_long).mean()
    ma_s = s.rolling(win_short).mean()
    sd   = s.rolling(std_win).std()
    z_all[c] = ((ma_s - ma_l) / sd.replace(0, np.nan)).clip(-10, 10)
z_all_df = pd.DataFrame(z_all).reindex(rets.index)

# ΔZ baseado no lookback selecionado
dz_all_df = z_all_df - z_all_df.shift(int(delta_lookback))

def map_target(row, z_val, dz_val, mode, delta_thr):
    """
    Decide o regime e o alvo:
      - Se |Z| ≤ z_thresh -> NEUTRO
      - Alternância: |ΔZ| <= thr -> NEUTRO; ΔZ>thr -> Momentum; ΔZ<-thr -> Reversão
      - Modos fixos ignoram ΔZ.
    """
    if np.isnan(z_val) or abs(z_val) <= z_thresh:
        return row["NEUTRO"], "Neutro"

    chosen = None
    if mode.startswith("Alternância"):
        if np.isnan(dz_val) or abs(dz_val) <= delta_thr:
            return row["NEUTRO"], "Neutro"
        chosen = "Momentum" if dz_val > 0 else "Reversão"
    elif mode.startswith("Momentum"):
        chosen = "Momentum"
    else:
        chosen = "Reversão"

    if chosen == "Momentum":
        target = row["MAX"] if z_val > 0 else row["MIN"]
    else:  # Reversão
        target = row["MIN"] if z_val > 0 else row["MAX"]
    return target, chosen

def tactical_weight(z_val, dz_val, row, mode, delta_thr, aggr):
    # decide alvo
    target, chosen = map_target(row, z_val, dz_val, mode, delta_thr)
    if target == row["NEUTRO"]:
        return row["NEUTRO"]
    # intensidade (saturada) * agressividade
    frac = min(abs(z_val), z_cap) / z_cap
    frac = max(0.0, min(1.0, aggr * frac))
    return row["NEUTRO"] + (target - row["NEUTRO"]) * frac

# matrizes base
w_neutro = pd.DataFrame(
    np.tile(weights_edit["NEUTRO"].values, (len(rets), 1)),
    index=rets.index, columns=CLASSES
)

# w_tatico_raw com Z e ΔZ
w_tatico_raw = pd.DataFrame(index=rets.index, columns=CLASSES, dtype=float)
for c in CLASSES:
    row = weights_edit.loc[c]
    if c in frozen_classes:
        w_tatico_raw[c] = pd.Series(row["NEUTRO"], index=rets.index)
    else:
        vals = []
        for t in rets.index:
            z_val  = z_all_df.at[t, c]
            dz_val = dz_all_df.at[t, c]
            vals.append(tactical_weight(z_val, dz_val, row, strategy_mode, delta_threshold, aggr_factor))
        w_tatico_raw[c] = vals

# limites (travados: MIN=MAX=NEUTRO)
mins = weights_edit["MIN"].values.copy()
maxs = weights_edit["MAX"].values.copy()
neus = weights_edit["NEUTRO"].values.copy()
for i, c in enumerate(CLASSES):
    if c in frozen_classes:
        mins[i] = neus[i]
        maxs[i] = neus[i]

# contínuo
w_tatico_cont = w_tatico_raw.copy()
for idx in w_tatico_cont.index:
    v = w_tatico_cont.loc[idx].values.astype(float)
    w_tatico_cont.loc[idx] = project_to_bounds_simplex(v, mins, maxs)

# degraus (2,5 p.p.)
if mode_pesos.startswith("Degraus"):
    w_tatico_final = w_tatico_cont.copy()
    for idx in w_tatico_final.index:
        v = w_tatico_final.loc[idx].values.astype(float)
        w_tatico_final.loc[idx] = quantize_row_to_step(v, mins, maxs, step=0.025)
else:
    w_tatico_final = w_tatico_cont.copy()

# ====== Início no 1º ponto válido de Z e ΔZ ======
valid_z  = z_all_df.dropna(how="all").index.min()
valid_dz = dz_all_df.dropna(how="all").index.min()
valid_candidates = [d for d in [valid_z, valid_dz] if d is not None]
valid_start = max(valid_candidates) if valid_candidates else None

if valid_start is not None:
    rets            = rets.loc[rets.index >= valid_start]
    z_all_df        = z_all_df.loc[z_all_df.index >= valid_start]
    dz_all_df       = dz_all_df.loc[dz_all_df.index >= valid_start]
    w_neutro        = w_neutro.loc[w_neutro.index >= valid_start]
    w_tatico_final  = w_tatico_final.loc[w_tatico_final.index >= valid_start]

# Tabela amigável
show_tbl = weights_edit.copy()
show_tbl["NEUTRO"] = show_tbl["NEUTRO"].map(to_pct)
show_tbl["MIN"]    = show_tbl["MIN"].map(to_pct)
show_tbl["MAX"]    = show_tbl["MAX"].map(to_pct)
with st.expander("Ver tabela de pesos (formato %)"):
    st.dataframe(show_tbl, use_container_width=True)

# Evolução dos pesos – área
def weights_area(df_w, title):
    df_plot = df_w.copy()
    df_plot.index.name = "Data"
    df_plot = df_plot.reset_index().melt(id_vars="Data", var_name="Classe", value_name="Peso")
    chart = alt.Chart(df_plot).mark_area(opacity=0.85).encode(
        x=alt.X("Data:T"),
        y=alt.Y("Peso:Q", stack="normalize", title="Peso"),
        color=alt.Color("Classe:N", legend=alt.Legend(columns=2),
                        scale=alt.Scale(range=SECONDARY_RANGE)),
        tooltip=["Data:T", "Classe:N", alt.Tooltip("Peso:Q", format=".2%")]
    ).properties(title=title)
    return chart

col_w1, col_w2 = st.columns(2)
with col_w1:
    st.altair_chart(weights_area(w_neutro,       "Carteira Neutra – evolução (constante)"), use_container_width=True)
with col_w2:
    st.altair_chart(weights_area(w_tatico_final, "Carteira Tática – evolução"),              use_container_width=True)

# --- 3.x) Evolução do Z-score (todas as classes) a partir de valid_start
st.markdown("#### Evolução do Z-score – todas as classes (a partir do 1º Z e ΔZ válidos)")
z_long = (
    z_all_df.reset_index()
           .melt(id_vars="Data", var_name="Classe", value_name="Z")
)
rule_neutral_up = alt.Chart(z_long).mark_rule(color="#999", strokeDash=[3,3]).encode(y=alt.datum(z_thresh))
rule_neutral_dn = alt.Chart(z_long).mark_rule(color="#999", strokeDash=[3,3]).encode(y=alt.datum(-z_thresh))
rule_cap_up     = alt.Chart(z_long).mark_rule(color=PRIMARY_2, strokeDash=[6,4]).encode(y=alt.datum(z_cap))
rule_cap_dn     = alt.Chart(z_long).mark_rule(color=PRIMARY_2, strokeDash=[6,4]).encode(y=alt.datum(-z_cap))

highlight = alt.selection_point(fields=["Classe"], bind="legend")
z_all_chart = (
    alt.Chart(z_long)
       .mark_line(opacity=0.9)
       .encode(
           x=alt.X("Data:T", title="Data"),
           y=alt.Y("Z:Q", title="Z-score"),
           color=alt.Color("Classe:N", legend=alt.Legend(columns=2),
                           scale=alt.Scale(scheme="category20")),
           opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.2)),
           tooltip=["Data:T", "Classe:N", alt.Tooltip("Z:Q", format=".2f")]
       )
       .add_params(highlight)
       .properties(height=320)
)
st.altair_chart(z_all_chart + rule_neutral_up + rule_neutral_dn + rule_cap_up + rule_cap_dn,
                use_container_width=True)

# ===============================
# 4) Evolução Base 100 + métricas (começando em valid_start)
# ===============================
st.markdown("### 4) Evolução Base 100 – Carteira Tática vs Neutra")

ret_neutro       = (w_neutro       * rets).sum(axis=1)
ret_tatico_bruto = (w_tatico_final * rets).sum(axis=1)

turnover = 0.5 * w_tatico_final.diff().abs().sum(axis=1)
turnover.iloc[0] = 0.0
if use_cost:
    cost_series = turnover * cost_pct
    ret_tatico_liquido = ret_tatico_bruto - cost_series
else:
    cost_series = pd.Series(0.0, index=ret_tatico_bruto.index)
    ret_tatico_liquido = ret_tatico_bruto

# Curvas base 100 com paleta
curve = pd.DataFrame({
    "Data": rets.index,
    "Neutra":           cum_base100(ret_neutro, base),
    "Tática (bruta)":   cum_base100(ret_tatico_bruto, base),
})
dom = ["Neutra", "Tática (bruta)"]
rng = [PRIMARY_1, PRIMARY_2]
if use_cost:
    curve["Tática (líquida custo)"] = cum_base100(ret_tatico_liquido, base)
    dom += ["Tática (líquida custo)"]
    rng += ["#C00000"]

curve_melt = curve.melt("Data", var_name="Carteira", value_name="Base 100")
chart_curve = alt.Chart(curve_melt).mark_line().encode(
    x=alt.X("Data:T"),
    y=alt.Y("Base 100:Q"),
    color=alt.Color("Carteira:N", scale=alt.Scale(domain=dom, range=rng))
).properties(height=380)
st.altair_chart(chart_curve, use_container_width=True)

# Métricas (anualizadas)
ann_ret_n,  ann_vol_n,  sh_n  = ann_stats(ret_neutro)
ann_ret_tb, ann_vol_tb, sh_tb = ann_stats(ret_tatico_bruto)
if use_cost:
    ann_ret_tl, ann_vol_tl, sh_tl = ann_stats(ret_tatico_liquido)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Retorno Anualizado (Neutra)",        to_pct(ann_ret_n))
    st.metric("Retorno Anualizado (Tática bruta)",  to_pct(ann_ret_tb))
    if use_cost:
        st.metric("Retorno Anualizado (Tática líquida)", to_pct(ann_ret_tl))
with m2:
    st.metric("Volatilidade Anualizada (Neutra)",       to_pct(ann_vol_n))
    st.metric("Volatilidade Anualizada (Tática bruta)", to_pct(ann_vol_tb))
    if use_cost:
        st.metric("Volatilidade Anualizada (Tática líquida)", to_pct(ann_vol_tl))
with m3:
    st.metric("Sharpe (Neutra)",        f"{sh_n:0.2f}")
    st.metric("Sharpe (Tática bruta)",  f"{sh_tb:0.2f}")
    if use_cost:
        st.metric("Sharpe (Tática líquida)", f"{sh_tl:0.2f}")

# ====== Resumo do período (não anualizado) ======
acc_n  = (1 + ret_neutro).prod() - 1
acc_tb = (1 + ret_tatico_bruto).prod() - 1
volp_n  = ret_neutro.std()
volp_tb = ret_tatico_bruto.std()
if use_cost:
    acc_tl = (1 + ret_tatico_liquido).prod() - 1
    volp_tl = ret_tatico_liquido.std()

st.markdown("#### Resumo do período (não anualizado)")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Retorno acumulado – Neutra", to_pct(acc_n))
    st.metric("Volatilidade do período – Neutra", to_pct(volp_n))
with c2:
    st.metric("Retorno acumulado – Tática (bruta)", to_pct(acc_tb))
    st.metric("Volatilidade do período – Tática (bruta)", to_pct(volp_tb))
with c3:
    if use_cost:
        st.metric("Retorno acumulado – Tática (líquida)", to_pct(acc_tl))
        st.metric("Volatilidade do período – Tática (líquida)", to_pct(volp_tl))

# Scatter risco x retorno
risk_return = pd.DataFrame({
    "Carteira": ["Neutra", "Tática (bruta)"],
    "Retorno Anualizado": [ann_ret_n, ann_ret_tb],
    "Volatilidade Anualizada": [ann_vol_n, ann_vol_tb]
})
if use_cost:
    risk_return = pd.concat([risk_return, pd.DataFrame({
        "Carteira": ["Tática (líquida custo)"],
        "Retorno Anualizado": [ann_ret_tl],
        "Volatilidade Anualizada": [ann_vol_tl]
    })], ignore_index=True)

scatter_colors = { "Neutra": PRIMARY_1, "Tática (bruta)": PRIMARY_2, "Tática (líquida custo)": "#C00000" }
scatter = alt.Chart(risk_return).mark_circle(size=180).encode(
    x=alt.X("Volatilidade Anualizada:Q", axis=alt.Axis(format="%"), title="Risco (Vol anualizada)"),
    y=alt.Y("Retorno Anualizado:Q",     axis=alt.Axis(format="%"), title="Retorno (anualizado)"),
    color=alt.Color("Carteira:N",
                    scale=alt.Scale(domain=list(scatter_colors.keys()),
                                    range=list(scatter_colors.values()))),
    tooltip=[
        "Carteira:N",
        alt.Tooltip("Retorno Anualizado:Q",     format=".2%"),
        alt.Tooltip("Volatilidade Anualizada:Q", format=".2%")
    ]
).properties(height=380)
st.altair_chart(scatter, use_container_width=True)

# ===============================
# Rodapé
# ===============================
with st.expander("Como funciona a Alternância dinâmica (ΔZ)?"):
    st.write(
        f"""
        **Z =** (média curta − média longa) / desvio-padrão.  
        **ΔZ = Z_t − Z_(t−{int(delta_lookback)})**.

        **Regras de decisão**:
        - Se |Z| ≤ {z_thresh}: **sem tilt** (peso neutro).  
        - Se **Alternância (ΔZ)**:  
          • |ΔZ| ≤ {delta_threshold:.2f} → **sem tilt**;  
          • ΔZ > {delta_threshold:.2f} → **Momentum** (se Z>0 vai ao MAX; se Z<0 vai ao MIN);  
          • ΔZ < −{delta_threshold:.2f} → **Reversão** (se Z>0 vai ao MIN; se Z<0 vai ao MAX).  
        - Modos **fixos** ignoram ΔZ e aplicam sempre o regime escolhido.

        A intensidade do ajuste é proporcional a |Z| (saturado em {z_cap}) e
        escalada por **Agressividade = {aggr_factor:.2f}**.  
        Pesos respeitam **MIN/MAX** e a soma = 100%. Classes **travadas** ficam no **NEUTRO**.  
        Custo (se ativado): turnover = 0,5 × soma(|w_t − w_(t−1)|); custo = turnover × {cost_pct:.2%}.

        Toda a comparação Tática×Neutra e os gráficos que dependem de tilt começam no primeiro mês
        em que **Z e ΔZ** estão válidos.
        """
    )
