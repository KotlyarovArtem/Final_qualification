"""
Прогнозирование характеристик композиционных материалов (Streamlit-приложение).

Приложение предсказывает по 10 параметрам материала:
1. Модуль упругости при растяжении, ГПа  — модель Random Forest
2. Прочность при растяжении, МПа         — модель Random Forest
3. Соотношение матрица-наполнитель       — нейросеть Keras

Модели обучены в ноутбуке VKR_Composites.ipynb; физические признаки
(тип волокна, объёмная доля, продольный/поперечный модули) вычисляются
модулем composite_physics.py по правилу смесей.

Запуск:  streamlit run app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from composite_physics import (BASE_INPUTS, FIBERS_DF, engineered_features,
                               infer_fiber_type)

st.set_page_config(page_title='Прогноз свойств композитов', page_icon='🔬',
                   layout='centered')

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


@st.cache_resource
def load_models():
    with open(os.path.join(MODEL_DIR, 'artifacts.json'), encoding='utf-8') as f:
        artifacts = json.load(f)
    models = {
        'E': joblib.load(os.path.join(MODEL_DIR, 'ml_modul_uprugosti.joblib')),
        'sigma': joblib.load(os.path.join(MODEL_DIR, 'ml_prochnost.joblib')),
        'scaler_ml': joblib.load(os.path.join(MODEL_DIR, 'scaler_features.joblib')),
        'scaler_X_nn': joblib.load(os.path.join(MODEL_DIR, 'scaler_nn_X.joblib')),
        'scaler_y_nn': joblib.load(os.path.join(MODEL_DIR, 'scaler_nn_y.joblib')),
        'nn': tf.keras.models.load_model(os.path.join(MODEL_DIR, 'nn_mf.keras')),
    }
    return artifacts, models


@st.cache_data
def load_input_ranges():
    bp = pd.read_excel('X_bp.xlsx', index_col=0)
    nup = pd.read_excel('X_nup.xlsx', index_col=0)
    df = bp.join(nup, how='inner')
    ranges = {c: (float(df[c].min()), float(df[c].max()), float(df[c].median()))
              for c in BASE_INPUTS}
    ranges['Угол нашивки, град'] = (0.0, 90.0, 0.0)
    return ranges


try:
    ARTIFACTS, MODELS = load_models()
    st.sidebar.success('Модели загружены')
except Exception as e:  # noqa: BLE001
    st.error(f'Ошибка загрузки моделей: {e}')
    st.info('Выполните основной ноутбук VKR_Composites.ipynb (раздел 8), '
            'чтобы создать папку models/.')
    st.stop()


def predict_all(user_input: dict) -> dict:
    """Прогноз трёх характеристик композита по 10 входным параметрам."""
    eng = engineered_features(user_input['Плотность, кг/м3'])
    row = {**user_input,
           **{k: eng[k] for k in ['fiber_type_id', 'V_f', 'E_f', 'sigma_f',
                                  'rho_f', 'E_1', 'E_2']}}
    features, nn_features = ARTIFACTS['features'], ARTIFACTS['nn_features']
    targets = ARTIFACTS['targets']

    x_ml = pd.DataFrame(MODELS['scaler_ml'].transform(pd.DataFrame([row])[features]),
                        columns=features)
    e_pred = float(MODELS['E'].predict(x_ml)[0])
    s_pred = float(MODELS['sigma'].predict(x_ml)[0])

    row_nn = {**user_input, targets[0]: e_pred, targets[1]: s_pred}
    x_nn = pd.DataFrame(MODELS['scaler_X_nn'].transform(pd.DataFrame([row_nn])[nn_features]),
                        columns=nn_features)
    mf_pred = float(MODELS['scaler_y_nn'].inverse_transform(
        MODELS['nn'].predict(x_nn, verbose=0)).ravel()[0])
    return {'E': e_pred, 'sigma': s_pred, 'mf': mf_pred, 'eng': eng}


st.title('🔬 Прогнозирование характеристик композитных материалов')
st.markdown('Задайте 10 параметров материала — приложение определит тип армирующего волокна '
            'по плотности и выдаст прогноз модуля упругости, прочности и соотношения '
            '«матрица–наполнитель».')

RANGES = load_input_ranges()
LABELS = {
    'Плотность, кг/м3': 'Плотность композита, кг/м³',
    'модуль упругости, ГПа': 'Модуль упругости волокна, ГПа',
    'Количество отвердителя, м.%': 'Количество отвердителя, м.%',
    'Содержание эпоксидных групп,%_2': 'Содержание эпоксидных групп, %',
    'Температура вспышки, С_2': 'Температура вспышки, °C',
    'Поверхностная плотность, г/м2': 'Поверхностная плотность, г/м²',
    'Потребление смолы, г/м2': 'Потребление смолы, г/м²',
    'Угол нашивки, град': 'Угол нашивки, град',
    'Шаг нашивки': 'Шаг нашивки, мм',
    'Плотность нашивки': 'Плотность нашивки',
}

with st.form('input_form'):
    col1, col2 = st.columns(2)
    values = {}
    for i, col_name in enumerate(BASE_INPUTS):
        lo, hi, mid = RANGES[col_name]
        target_col = col1 if i < 5 else col2
        if col_name == 'Угол нашивки, град':
            values[col_name] = target_col.selectbox(LABELS[col_name], [0, 90], index=0)
        elif hi - lo <= 40:
            values[col_name] = target_col.number_input(
                LABELS[col_name], min_value=float(lo), max_value=float(hi),
                value=round(float(mid), 2), step=0.1)
        else:
            values[col_name] = target_col.number_input(
                LABELS[col_name], min_value=float(lo), max_value=float(hi),
                value=round(float(mid), 1), step=1.0)
    submitted = st.form_submit_button('Рассчитать прогноз', type='primary', width='stretch')

if submitted:
    with st.spinner('Расчёт...'):
        result = predict_all(values)
        eng = result['eng']

    st.header('Результаты прогноза')
    m1, m2, m3 = st.columns(3)
    m1.metric('Модуль упругости при растяжении', f"{result['E']:.2f} ГПа",
              help='Прогноз Random Forest')
    m2.metric('Прочность при растяжении', f"{result['sigma']:.1f} МПа",
              help='Прогноз Random Forest')
    m3.metric('Соотношение матрица-наполнитель', f"{result['mf']:.3f}",
              help='Прогноз нейросети Keras')

    with st.expander('Определённые по плотности физические параметры'):
        st.write(f'**Тип армирующего волокна:** {eng["fiber_name"]} '
                 f'(E_f = {eng["E_f"]:.0f} ГПа, σ_f = {eng["sigma_f"]:.0f} МПа, '
                 f'ρ_f = {eng["rho_f"]:.2f} г/см³)')
        st.write(f'**Объёмная доля волокна V_f:** {eng["V_f"]:.3f}')
        st.write(f'**Продольный модуль E₁:** {eng["E_1"]:.2f} ГПа')
        st.write(f'**Поперечный модуль E₂:** {eng["E_2"]:.2f} ГПа')

st.divider()
st.sidebar.header('О приложении')
st.sidebar.info(
    '**Прогнозирование ключевых свойств композитов**\n\n'
    '- ML: Random Forest — модуль упругости (R² = 0.999) и прочность (R² = 0.993)\n'
    '- Нейросеть Keras 128/64/32 — соотношение матрица-наполнитель (R² = 0.982)\n\n'
    'Модели обучены как суррогаты физической модели (правило смесей) '
    'в ноутбуке VKR_Composites.ipynb.')
st.sidebar.header('Справочник волокон')
st.sidebar.dataframe(
    FIBERS_DF[['name', 'E_f', 'sigma_f', 'rho_f']].rename(columns={
        'name': 'Волокно', 'E_f': 'E_f, ГПа', 'sigma_f': 'σ_f, МПа', 'rho_f': 'ρ_f, г/см³'}),
    use_container_width=True, hide_index=True)
