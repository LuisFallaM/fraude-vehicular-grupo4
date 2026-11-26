import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Detección de Fraude Vehicular — Grupo 4 (SI709)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# METADATOS DE RESPALDO (COINCIDEN 100% CON TF_GRUPO_4_Completo.ipynb)
# ==============================================================================
DEFAULT_METADATA = {
    "nombre_modelo": "XGBoost (GridSearchCV)",
    "umbral_alta_cobertura": 0.26,
    "umbral_max_f1": 0.39,
    "tasa_base_test": 0.06,
    "columnas_entrada": [
        "month", "weekofmonth", "dayofweek", "make", "accidentarea",
        "dayofweekclaimed", "monthclaimed", "weekofmonthclaimed", "sex",
        "maritalstatus", "age", "fault", "vehiclecategory", "vehicleprice",
        "deductible", "driverrating", "days_policy_accident", "days_policy_claim",
        "pastnumberofclaims", "ageofvehicle", "policereportfiled", "witnesspresent",
        "agenttype", "numberofsuppliments", "addresschange_claim", "numberofcars",
        "basepolicy"
    ],
    "perfil_default": {"month": "May", "weekofmonth": 3.0, "dayofweek": "Monday", "make": "Pontiac", "accidentarea": "Urban", "dayofweekclaimed": "Monday", "monthclaimed": "Jan", "weekofmonthclaimed": 3.0, "sex": "Male", "maritalstatus": "Married", "age": 38.0, "fault": "Policy Holder", "vehiclecategory": "Sedan", "vehicleprice": "20000 to 29000", "deductible": "400", "driverrating": 2.0, "days_policy_accident": "more than 30", "days_policy_claim": "more than 30", "pastnumberofclaims": "2 to 4", "ageofvehicle": "7 years", "policereportfiled": "No", "witnesspresent": "No", "agenttype": "External", "numberofsuppliments": "none", "addresschange_claim": "no change", "numberofcars": "1 vehicle", "basepolicy": "Collision"},
    "ord_categories": [["less than 20000", "20000 to 29000", "30000 to 39000", "40000 to 59000", "60000 to 69000", "more than 69000"], ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"], ["none", "8 to 15", "15 to 30", "more than 30"], ["none", "1", "2 to 4", "more than 4"], ["new", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "more than 7"], ["none", "1 to 2", "3 to 5", "more than 5"], ["no change", "under 6 months", "1 year", "2 to 3 years", "4 to 8 years"], ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"]],
    "marcas_disponibles": ["Acura", "BMW", "Chevrolet", "Dodge", "Ferrari", "Ford", "Honda", "Jaguar", "Lexus", "Mazda", "Mercedes", "Mercury", "Nissan", "Pontiac", "Porsche", "Saab", "Saturn", "Toyota", "VW"],
    "metricas_test_operativo": {
        "roc_auc": 0.8265,
        "pr_auc": 0.2315,
        "recall": 0.8541,
        "precision": 0.1330,
        "f1": 0.2302,
        "pct_auditados": 38.52,
        "lift_operativo": 2.22,
        "lift_top10": 3.46,
        "latencia_ms": 12.21
    },
    "bandas_riesgo_test": [{"banda": "Riesgo Bajo (< 26 %)", "decision": "Flujo ordinario de liquidación", "reclamos_test": 1896, "pct_cartera": 61.48, "fraudes_reales": 27, "tasa_fraude_real_pct": 1.42, "lift": 0.24}, {"banda": "Riesgo Estándar (26 % – 39 %)", "decision": "Derivar a auditoría UIE — Cobertura estándar", "reclamos_test": 566, "pct_cartera": 18.35, "fraudes_reales": 50, "tasa_fraude_real_pct": 8.83, "lift": 1.47}, {"banda": "Prioridad Alta (≥ 39 %)", "decision": "Derivar a auditoría UIE — Prioridad alta", "reclamos_test": 622, "pct_cartera": 20.17, "fraudes_reales": 108, "tasa_fraude_real_pct": 17.36, "lift": 2.89}]
}

ETIQUETAS_CAMPOS = {
    "fault": "Responsabilidad del siniestro",
    "basepolicy": "Tipo de cobertura base",
    "vehiclecategory": "Categoría de vehículo",
    "deductible": "Deducible",
    "age": "Edad del conductor",
    "vehicleprice": "Precio del vehículo",
    "pastnumberofclaims": "Reclamos previos",
    "addresschange_claim": "Cambio de dirección",
    "make": "Marca del vehículo",
    "accidentarea": "Zona del accidente",
    "policereportfiled": "Reporte policial",
    "witnesspresent": "Testigo presente",
    "numberofsuppliments": "Número de suplementos",
    "ageofvehicle": "Antigüedad del vehículo",
    "days_policy_accident": "Días póliza → accidente",
    "days_policy_claim": "Días póliza → reclamo",
    "numberofcars": "Número de vehículos",
    "month": "Mes del siniestro",
    "monthclaimed": "Mes del reclamo",
    "dayofweek": "Día del siniestro",
    "dayofweekclaimed": "Día del reclamo",
    "weekofmonth": "Semana del mes (siniestro)",
    "weekofmonthclaimed": "Semana del mes (reclamo)",
    "sex": "Sexo del conductor",
    "maritalstatus": "Estado civil",
    "driverrating": "Calificación del conductor",
    "agenttype": "Tipo de agente"
}


def entrenar_modelo_respaldo(csv_path: str):
    """Entrena el pipeline campeón exacto (prep_B + XGBoost GridSearchCV) si el .joblib no existe o falla por versión."""
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
    from xgboost import XGBClassifier

    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    df["make"] = df["make"].replace({
        "Accura": "Acura", "Mecedes": "Mercedes", "Nisson": "Nissan", "Porche": "Porsche"
    })
    df["age"] = df["age"].astype(float)
    df.loc[(df["age"] == 0) & (df["ageofpolicyholder"] == "16 to 17"), "age"] = 16.5
    df["monthclaimed"] = df["monthclaimed"].replace("0", np.nan)
    df["dayofweekclaimed"] = df["dayofweekclaimed"].replace("0", np.nan)
    df["vehiclecategory"] = df["policytype"].str.split(" - ").str[0]
    df_model = df.drop(columns=["policynumber", "repnumber", "year", "ageofpolicyholder", "policytype"]).copy()

    X = df_model.drop(columns=["fraudfound_p"])
    y = df_model["fraudfound_p"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_train["deductible"] = X_train["deductible"].astype(str)

    cols_num = ["age", "driverrating", "weekofmonth", "weekofmonthclaimed"]
    cols_ord = [
        "vehicleprice", "days_policy_accident", "days_policy_claim",
        "pastnumberofclaims", "ageofvehicle", "numberofsuppliments",
        "addresschange_claim", "numberofcars"
    ]
    cols_nom = [
        "month", "dayofweek", "make", "accidentarea", "dayofweekclaimed",
        "monthclaimed", "sex", "maritalstatus", "fault", "vehiclecategory",
        "policereportfiled", "witnesspresent", "agenttype", "basepolicy", "deductible"
    ]
    prep_B = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), cols_num),
        ("ord", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(
                categories=DEFAULT_METADATA["ord_categories"],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ]), cols_ord),
        ("nom", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cols_nom)
    ])
    pipe = Pipeline([
        ("prep", prep_B),
        ("clf", XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.8, min_child_weight=1,
            scale_pos_weight=5, eval_metric="logloss", random_state=42, n_jobs=-1
        ))
    ])
    pipe.fit(X_train, y_train)
    return pipe


@st.cache_resource(show_spinner="Cargando modelo predictivo y explicador SHAP...")
def cargar_recursos():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Cargar metadatos
    meta = DEFAULT_METADATA.copy()
    for cand in [
        os.path.join(base_dir, "metadata_modelo_final.json"),
        os.path.join(base_dir, "artefactos_hito2", "metadata_modelo_final.json"),
    ]:
        if os.path.exists(cand):
            with open(cand, "r", encoding="utf-8") as f:
                loaded_meta = json.load(f)
            meta.update(loaded_meta)
            break

    # 2. Cargar modelo serializado (.joblib) con respaldo automático si hay cambio de versión o falta archivo
    modelo = None
    origen_modelo = ""
    for cand in [
        os.path.join(base_dir, "modelo_fraude_final.joblib"),
        os.path.join(base_dir, "artefactos_hito2", "modelo_fraude_final.joblib"),
    ]:
        if os.path.exists(cand):
            try:
                modelo = joblib.load(cand)
                # Prueba rápida de inferencia para verificar compatibilidad de versión
                df_test_load = pd.DataFrame([meta["perfil_default"]])[meta["columnas_entrada"]]
                _ = modelo.predict_proba(df_test_load)
                origen_modelo = f"Modelo serializado ({os.path.basename(cand)})"
                break
            except Exception:
                modelo = None

    if modelo is None:
        for csv_cand in [
            os.path.join(base_dir, "fraud_oracle.csv"),
            os.path.join(base_dir, "data", "fraud_oracle.csv"),
        ]:
            if os.path.exists(csv_cand):
                modelo = entrenar_modelo_respaldo(csv_cand)
                origen_modelo = "Entrenado en vivo con hiperparámetros óptimos (fraud_oracle.csv)"
                break

    if modelo is None:
        st.error("No se encontró `modelo_fraude_final.joblib` ni `fraud_oracle.csv` en el repositorio.")
        st.stop()

    # 3. Preparar mapeo de columnas codificadas -> variables originales + explicador SHAP
    prep_fitted = modelo.named_steps["prep"]
    clf_fitted = modelo.named_steps["clf"]
    feature_names_clean = [
        c.replace("num__", "").replace("ord__", "").replace("nom__", "")
        for c in prep_fitted.get_feature_names_out()
    ]
    cols_orig_ordenadas = sorted(list(meta["columnas_entrada"]), key=len, reverse=True)
    mapa_col_a_orig = {}
    for feat_enc in feature_names_clean:
        orig_match = feat_enc
        for col_orig in cols_orig_ordenadas:
            if feat_enc == col_orig or feat_enc.startswith(col_orig + "_"):
                orig_match = col_orig
                break
        mapa_col_a_orig[feat_enc] = orig_match

    explainer = None
    try:
        import shap
        explainer = shap.TreeExplainer(clf_fitted)
    except Exception:
        explainer = None

    return modelo, meta, feature_names_clean, mapa_col_a_orig, explainer, origen_modelo


def evaluar_caso(fila_dict, modelo, meta, feature_names_clean, mapa_col_a_orig, explainer):
    cols = meta["columnas_entrada"]
    df_in = pd.DataFrame([fila_dict])[cols]
    prob = float(modelo.predict_proba(df_in)[0, 1])

    th_cob = float(meta["umbral_alta_cobertura"])
    th_f1 = float(meta["umbral_max_f1"])
    bandas = meta.get("bandas_riesgo_test", DEFAULT_METADATA["bandas_riesgo_test"])

    if prob >= th_f1:
        nivel = "PRIORIDAD ALTA"
        decision = f"DERIVAR A AUDITORÍA (UIE) — PRIORIDAD ALTA (Puntaje ≥ {th_f1*100:.0f} %)"
        color = "error"
        banda_info = bandas[2]
    elif prob >= th_cob:
        nivel = "COBERTURA ESTÁNDAR"
        decision = f"DERIVAR A AUDITORÍA (UIE) — COBERTURA ESTÁNDAR (Puntaje entre {th_cob*100:.0f} % y {th_f1*100:.0f} %)"
        color = "warning"
        banda_info = bandas[1]
    else:
        nivel = "RIESGO BAJO"
        decision = f"FLUJO ORDINARIO DE LIQUIDACIÓN (Puntaje < {th_cob*100:.0f} %)"
        color = "success"
        banda_info = bandas[0]

    # SHAP agregado aditivamente por variable original
    top_shap = []
    if explainer is not None:
        try:
            x_enc = pd.DataFrame(
                modelo.named_steps["prep"].transform(df_in),
                columns=feature_names_clean
            )
            sv = explainer(x_enc)
            vals = sv.values[0, :, 1] if sv.values.ndim == 3 else sv.values[0]
            shap_por_var = {}
            for feat_enc, val_s in zip(feature_names_clean, vals):
                var_orig = mapa_col_a_orig[feat_enc]
                shap_por_var[var_orig] = shap_por_var.get(var_orig, 0.0) + float(val_s)

            top_vars = sorted(shap_por_var.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
            for var_orig, val_s in top_vars:
                val_campo = fila_dict[var_orig]
                if isinstance(val_campo, float) and val_campo.is_integer():
                    val_campo = int(val_campo)
                etiqueta = ETIQUETAS_CAMPOS.get(var_orig, var_orig)
                top_shap.append({
                    "variable": var_orig,
                    "etiqueta": f"{etiqueta} ({var_orig} = {val_campo})",
                    "valor_campo": str(val_campo),
                    "shap": val_s,
                    "efecto": "↑ Aumenta riesgo" if val_s > 0 else "↓ Reduce riesgo"
                })
        except Exception:
            top_shap = []

    return prob, nivel, decision, color, banda_info, top_shap


# ==============================================================================
# CARGA INICIAL Y BARRA LATERAL
# ==============================================================================
modelo, meta, feature_names_clean, mapa_col_a_orig, explainer, origen_modelo = cargar_recursos()
perfil_default = meta["perfil_default"]
ord_cats = meta["ord_categories"]
marcas = meta["marcas_disponibles"]
metricas_test = meta["metricas_test_operativo"]

with st.sidebar:
    st.title("🛡️ Fraude Vehicular")
    st.caption("SI709 — Business Predictive Analytics | Grupo 4")
    st.markdown("---")
    st.markdown(f"**Modelo campeón:** `{meta['nombre_modelo']}`")
    st.markdown(f"**Origen:** {origen_modelo}")
    st.markdown(f"**Umbral Cobertura (Recall ≥ 85 %):** `{meta['umbral_alta_cobertura']*100:.0f} %`")
    st.markdown(f"**Umbral Máximo F1:** `{meta['umbral_max_f1']*100:.0f} %`")
    st.markdown("---")
    st.subheader("Métricas auditadas en TEST")
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", f"{metricas_test['roc_auc']:.4f}")
    c2.metric("PR-AUC", f"{metricas_test['pr_auc']:.4f}")
    c3, c4 = st.columns(2)
    c3.metric("Recall (th=0.26)", f"{metricas_test['recall']*100:.1f} %")
    c4.metric("Lift Top 10 %", f"{metricas_test.get('lift_top10', 3.46):.2f}×")
    st.markdown("---")
    st.info(
        "**Aviso de uso:** El modelo estima un puntaje de riesgo para priorizar auditorías en la UIE "
        "al cierre de la evaluación inicial, antes del desembolso. La decisión final siempre es del auditor."
    )

# ==============================================================================
# CABECERA Y PESTAÑAS
# ==============================================================================
st.title("Sistema de Priorización de Fraude en Seguros Vehiculares")
st.markdown(
    "Herramienta de apoyo para **liquidadores de siniestros** y la **Unidad de Investigaciones Especiales (UIE)**. "
    "Evalúa un reclamo en tiempo real, asigna la cola operativa y explica los factores determinantes con **SHAP agregado por variable**."
)

tab_demo, tab_evidencia, tab_guia = st.tabs([
    "🔍 Evaluar Reclamo (Demo en Vivo)",
    "📊 Evidencia y Confiabilidad del Puntaje",
    "📘 Guía Rápida y FAQ para Liquidadores"
])

# ==============================================================================
# TAB 1: DEMO EN VIVO
# ==============================================================================
with tab_demo:
    st.subheader("1. Cargar un caso de prueba rápido o ingresar un reclamo nuevo")

    # Casos predefinidos para agilizar la demostración en laboratorio
    if "preset" not in st.session_state:
        st.session_state["preset"] = {
            "fault": "Policy Holder", "basepolicy": "All Perils", "vehiclecategory": "Utility",
            "deductible": "500", "age": 32, "vehicleprice": "more than 69000",
            "pastnumberofclaims": "none", "addresschange_claim": "under 6 months",
            "make": "Honda", "accidentarea": "Urban", "policereportfiled": "No", "witnesspresent": "No"
        }

    b_col1, b_col2, b_col3 = st.columns(3)
    if b_col1.button("🔴 Caso 1: Perfil Alto Riesgo", use_container_width=True):
        st.session_state["preset"] = {
            "fault": "Policy Holder", "basepolicy": "All Perils", "vehiclecategory": "Utility",
            "deductible": "500", "age": 32, "vehicleprice": "more than 69000",
            "pastnumberofclaims": "none", "addresschange_claim": "under 6 months",
            "make": "Honda", "accidentarea": "Urban", "policereportfiled": "No", "witnesspresent": "No"
        }
    if b_col2.button("🟠 Caso 2: Perfil Cobertura Estándar", use_container_width=True):
        st.session_state["preset"] = {
            "fault": "Policy Holder", "basepolicy": "Collision", "vehiclecategory": "Sedan",
            "deductible": "400", "age": 38, "vehicleprice": "20000 to 29000",
            "pastnumberofclaims": "1", "addresschange_claim": "no change",
            "make": "Pontiac", "accidentarea": "Urban", "policereportfiled": "No", "witnesspresent": "No"
        }
    if b_col3.button("🟢 Caso 3: Perfil Bajo Riesgo (Legítimo)", use_container_width=True):
        st.session_state["preset"] = {
            "fault": "Third Party", "basepolicy": "Liability", "vehiclecategory": "Sedan",
            "deductible": "400", "age": 45, "vehicleprice": "20000 to 29000",
            "pastnumberofclaims": "none", "addresschange_claim": "no change",
            "make": "Toyota", "accidentarea": "Urban", "policereportfiled": "Yes", "witnesspresent": "Yes"
        }

    p = st.session_state["preset"]

    with st.form("form_reclamo"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            fault = st.selectbox("Responsabilidad del siniestro (fault)", ["Policy Holder", "Third Party"],
                                 index=["Policy Holder", "Third Party"].index(p["fault"]))
            basepolicy = st.selectbox("Cobertura base (basepolicy)", ["All Perils", "Collision", "Liability"],
                                      index=["All Perils", "Collision", "Liability"].index(p["basepolicy"]))
            vehiclecategory = st.selectbox("Categoría de vehículo (vehiclecategory)", ["Sedan", "Sport", "Utility"],
                                           index=["Sedan", "Sport", "Utility"].index(p["vehiclecategory"]))
            deductible = st.selectbox("Deducible (deductible)", ["300", "400", "500", "700"],
                                      index=["300", "400", "500", "700"].index(p["deductible"]))
        with col_b:
            age = st.slider("Edad del conductor (age)", min_value=16, max_value=80, value=int(p["age"]), step=1)
            vehicleprice = st.selectbox("Precio del vehículo (vehicleprice)", ord_cats[0],
                                        index=ord_cats[0].index(p["vehicleprice"]))
            pastnumberofclaims = st.selectbox("Reclamos previos (pastnumberofclaims)", ord_cats[3],
                                              index=ord_cats[3].index(p["pastnumberofclaims"]))
            addresschange_claim = st.selectbox("Cambio de dirección (addresschange_claim)", ord_cats[6],
                                               index=ord_cats[6].index(p["addresschange_claim"]))
        with col_c:
            make_idx = marcas.index(p["make"]) if p["make"] in marcas else 0
            make = st.selectbox("Marca del vehículo (make)", marcas, index=make_idx)
            accidentarea = st.selectbox("Zona del accidente (accidentarea)", ["Urban", "Rural"],
                                        index=["Urban", "Rural"].index(p["accidentarea"]))
            policereportfiled = st.selectbox("Reporte policial (policereportfiled)", ["No", "Yes"],
                                             index=["No", "Yes"].index(p["policereportfiled"]))
            witnesspresent = st.selectbox("Testigo presente (witnesspresent)", ["No", "Yes"],
                                          index=["No", "Yes"].index(p["witnesspresent"]))

        with st.expander("Ajustar campos secundarios del expediente (por defecto usan el perfil típico de TRAIN)"):
            sc1, sc2, sc3 = st.columns(3)
            sex = sc1.selectbox("Sexo (sex)", ["Male", "Female"],
                                index=["Male", "Female"].index(perfil_default.get("sex", "Male")))
            maritalstatus = sc2.selectbox("Estado civil (maritalstatus)", ["Married", "Single", "Divorced", "Widow"],
                                          index=["Married", "Single", "Divorced", "Widow"].index(perfil_default.get("maritalstatus", "Married")))
            numberofsuppliments = sc3.selectbox("Suplementos (numberofsuppliments)", ord_cats[5],
                                                index=ord_cats[5].index(perfil_default.get("numberofsuppliments", "none")))
            ageofvehicle = sc1.selectbox("Antigüedad del vehículo (ageofvehicle)", ord_cats[4],
                                         index=ord_cats[4].index(perfil_default.get("ageofvehicle", "7 years")))
            agenttype = sc2.selectbox("Tipo de agente (agenttype)", ["External", "Internal"],
                                      index=["External", "Internal"].index(perfil_default.get("agenttype", "External")))
            driverrating = sc3.slider("Calificación del conductor (driverrating)", 1, 4, int(perfil_default.get("driverrating", 2)))

        submitted = st.form_submit_button("🔎 Evaluar riesgo de fraude", type="primary", use_container_width=True)

    # Construir registro completo y evaluar (siempre muestra resultado inicial o actualizado)
    fila_actual = perfil_default.copy()
    fila_actual.update({
        "fault": fault,
        "basepolicy": basepolicy,
        "vehiclecategory": vehiclecategory,
        "deductible": str(deductible),
        "age": float(age),
        "vehicleprice": vehicleprice,
        "pastnumberofclaims": pastnumberofclaims,
        "addresschange_claim": addresschange_claim,
        "make": make,
        "accidentarea": accidentarea,
        "policereportfiled": policereportfiled,
        "witnesspresent": witnesspresent,
        "sex": sex,
        "maritalstatus": maritalstatus,
        "numberofsuppliments": numberofsuppliments,
        "ageofvehicle": ageofvehicle,
        "agenttype": agenttype,
        "driverrating": float(driverrating)
    })

    prob, nivel, decision, color, banda_info, top_shap = evaluar_caso(
        fila_actual, modelo, meta, feature_names_clean, mapa_col_a_orig, explainer
    )

    st.markdown("---")
    st.subheader("2. Resultado de evaluación y explicación para el auditor")

    if color == "error":
        st.error(f"**{decision}**")
    elif color == "warning":
        st.warning(f"**{decision}**")
    else:
        st.success(f"**{decision}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Puntaje de riesgo estimado", f"{prob*100:.2f} %")
    m2.metric("Banda operativa", nivel)
    m3.metric("Tasa real de fraude en esta banda (TEST)", f"{banda_info['tasa_fraude_real_pct']:.2f} %")
    m4.metric("Lift empírico vs. azar (6.00 %)", f"{banda_info['lift']:.2f}×")

    st.caption(
        f"**¿Qué significa este valor ({prob*100:.2f} %)?** Como XGBoost se entrenó con `scale_pos_weight = 5` para compensar el desbalance 15.7 : 1, "
        f"este porcentaje es un **puntaje de riesgo ponderado** (no una probabilidad bruta calibrada). La evidencia para confiar en su ordenamiento es que en "
        f"datos nunca vistos (`TEST`, n = 3,084), los reclamos en la banda **{banda_info['banda']}** registraron una tasa real de fraude de "
        f"**{banda_info['tasa_fraude_real_pct']:.2f} %** ({banda_info['fraudes_reales']} de {banda_info['reclamos_test']} reclamos; **Lift {banda_info['lift']:.2f}×**)."
    )

    if top_shap:
        col_left, col_right = st.columns([1.15, 1.0])
        with col_left:
            st.markdown("#### Principales factores explicativos (SHAP agregado por variable)")
            df_shap_show = pd.DataFrame(top_shap)[["etiqueta", "efecto", "shap"]].rename(columns={
                "etiqueta": "Variable y valor ingresado",
                "efecto": "Dirección sobre el riesgo",
                "shap": "Impacto SHAP (log-odds)"
            })
            df_shap_show["Impacto SHAP (log-odds)"] = df_shap_show["Impacto SHAP (log-odds)"].map(lambda x: f"{x:+.3f}")
            st.dataframe(df_shap_show, use_container_width=True, hide_index=True)

        with col_right:
            st.markdown("#### Contribución visual al riesgo de este reclamo")
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            df_plot = pd.DataFrame(top_shap).iloc[::-1]
            colores_bar = ["#c0392b" if v > 0 else "#2980b9" for v in df_plot["shap"]]
            ax.barh(df_plot["etiqueta"], df_plot["shap"], color=colores_bar)
            ax.axvline(0, color="#333333", linewidth=1)
            ax.set_xlabel("Aporte SHAP (positivo = sube riesgo | negativo = reduce riesgo)")
            ax.set_title("Factores determinantes del caso")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ==============================================================================
# TAB 2: EVIDENCIA Y CONFIABILIDAD DEL MODELO
# ==============================================================================
with tab_evidencia:
    st.subheader("¿Por qué confiar en el ordenamiento y las bandas del modelo?")
    st.markdown(
        "El modelo campeón (**XGBoost GridSearchCV** sobre `prep_B`) se evaluó **una sola vez** sobre el conjunto de prueba "
        "(`TEST`, n = 3,084 reclamos; 185 fraudes reales = 6.00 %) con los umbrales fijados exclusivamente en `TRAIN` (`0.26` y `0.39`). "
        "Las métricas en `TEST` (`ROC-AUC = 0.8265`, `PR-AUC = 0.2315`) replican casi exactamente la validación cruzada en `TRAIN` "
        "(`0.8259` y `0.2347`), demostrando generalización sin sobreajuste."
    )

    st.markdown("#### 1. Tasa real de fraude observada por banda de puntaje en `TEST` (n = 3,084)")
    df_bandas = pd.DataFrame(meta.get("bandas_riesgo_test", DEFAULT_METADATA["bandas_riesgo_test"])).rename(columns={
        "banda": "Banda de puntaje (umbrales TRAIN)",
        "decision": "Acción operativa",
        "reclamos_test": "Reclamos en TEST",
        "pct_cartera": "% de la cartera",
        "fraudes_reales": "Fraudes reales",
        "tasa_fraude_real_pct": "Tasa real de fraude (%)",
        "lift": "Lift vs. azar (6.0 %)"
    })
    st.dataframe(df_bandas, use_container_width=True, hide_index=True)

    st.success(
        "**Lectura clave:** Los reclamos clasificados en **Prioridad Alta (≥ 39 %)** tienen una tasa real de fraude de **17.36 %** —**12.2 veces mayor** "
        "que la banda de **Riesgo Bajo (< 26 %)**, donde solo ocurre **1.42 %** de fraude (98.58 % legítimos). Esto respalda usar el puntaje para priorizar la cola de la UIE."
    )

    st.markdown("#### 2. Comparación honesta en `TEST`: Modelo Campeón vs. Línea Base del Hito 1 (Top-K)")
    df_topk_comp = pd.DataFrame([
        {"Capacidad UIE": "Top 5 % (154)",    "Fraudes XGBoost": 31,  "Prec@K XGBoost": "20.13 %", "Lift XGBoost": "3.36×", "Fraudes LogReg H1": 29,  "Lift LogReg H1": "3.14×"},
        {"Capacidad UIE": "Top 10 % (308)",   "Fraudes XGBoost": 64,  "Prec@K XGBoost": "20.78 %", "Lift XGBoost": "3.46×", "Fraudes LogReg H1": 57,  "Lift LogReg H1": "3.09×"},
        {"Capacidad UIE": "Top 20 % (616)",   "Fraudes XGBoost": 108, "Prec@K XGBoost": "17.53 %", "Lift XGBoost": "2.92×", "Fraudes LogReg H1": 106, "Lift LogReg H1": "2.87×"},
        {"Capacidad UIE": "Top 30 % (925)",   "Fraudes XGBoost": 134, "Prec@K XGBoost": "14.49 %", "Lift XGBoost": "2.41×", "Fraudes LogReg H1": 130, "Lift LogReg H1": "2.34×"},
        {"Capacidad UIE": "Top 40 % (1,233)", "Fraudes XGBoost": 162, "Prec@K XGBoost": "13.14 %", "Lift XGBoost": "2.19×", "Fraudes LogReg H1": 163, "Lift LogReg H1": "2.20×"},
    ])
    st.dataframe(df_topk_comp, use_container_width=True, hide_index=True)

    st.markdown("#### 3. Limitaciones transparentes antes de uso comercial real")
    st.markdown(
        "- **Proxy histórico:** El dataset `fraud_oracle.csv` corresponde a 1994–1996 de otro mercado y no contiene monto reclamado, hora exacta del siniestro ni informes periciales.\n"
        "- **Recalibración obligatoria:** Antes de operar en una aseguradora peruana, se debe reentrenar con datos locales vigentes y optimizar el umbral según el costo real por hora de auditoría y el monto reclamado."
    )

# ==============================================================================
# TAB 3: GUÍA RÁPIDA Y FAQ (ABET CAC 6.c2)
# ==============================================================================
with tab_guia:
    st.subheader("Guía operativa para liquidadores y auditores (UIE)")
    st.markdown(
        "1. **Momento de uso:** Ejecutar la evaluación al cierre del registro inicial del siniestro, **antes de autorizar el desembolso** (cuando ya se registraron responsabilidad preliminar `fault` y cobertura `basepolicy`).\n"
        "2. **Ingreso de datos:** Completar los 12 campos principales en la pestaña *Evaluar Reclamo*. Los campos faltantes o secundarios se gestionan de forma segura dentro del pipeline.\n"
        "3. **Acción según cola asignada:**\n"
        "   - **Prioridad Alta (≥ 39 %):** Derivar inmediatamente a investigación prioritaria de la UIE.\n"
        "   - **Cobertura Estándar (26 % – 39 %):** Revisión UIE estándar para asegurar cobertura ≥ 85 % del fraude.\n"
        "   - **Flujo Ordinario (< 26 %):** Aprobar por el flujo regular del liquidador (98.58 % legítimos en `TEST`)."
    )

    st.subheader("Preguntas frecuentes (FAQ)")
    with st.expander("¿Si el sistema deriva el caso a UIE significa que el cliente cometió fraude?"):
        st.write("No. El modelo estima un puntaje de riesgo estadístico para ordenar la cola de auditoría. La decisión final siempre corresponde al auditor humano con respaldo documental.")
    with st.expander("¿Cómo interpreto las explicaciones SHAP en pantalla?"):
        st.write("Cada barra agrega el impacto total de una variable original (`campo = valor`). Las barras rojas (SHAP positivo) suben el riesgo estimado; las azules (SHAP negativo) lo reducen.")
    with st.expander("¿Por qué el puntaje puede decir 58 % si la tasa de fraude de la banda alta es 17.4 %?"):
        st.write("Porque el modelo utiliza `scale_pos_weight = 5` para dar mayor peso a los fraudes raros (6 % de prevalencia). Por ello, la interfaz muestra siempre ambos valores: el puntaje del modelo y la tasa real observada en esa banda sobre `TEST`.")
    with st.expander("¿Qué ocurre si ingresa una marca de vehículo nueva o falta la edad?"):
        st.write("El pipeline fue validado con pruebas unitarias (`TU-04` y `TU-05`): imputa valores faltantes con la mediana/moda de `TRAIN` e ignora categorías desconocidas sin interrumpir el servicio (latencia medida: 12.21 ms).")
