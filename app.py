import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='Прогноз свойств композитов',
    page_icon='🔬',
    layout='wide',
)

st.title('🔬 Прогнозирование свойств композиционных материалов')
st.markdown('ВКР | Data Science Pro | МГТУ им. Н.Э. Баумана')
st.markdown('---')

@st.cache_resource
def load_and_train():
    X_bp  = pd.read_excel('X_bp.xlsx',  index_col=0)
    X_nup = pd.read_excel('X_nup.xlsx', index_col=0)
    df = X_bp.join(X_nup, how='inner')

    df_doe = df.iloc[:40].copy()
    df_doe = df_doe.drop(index=19).reset_index(drop=True)

    FEATURES = [
        'Плотность, кг/м3',
        'модуль упругости, ГПа',
        'Количество отвердителя, м.%',
        'Содержание эпоксидных групп,%_2',
        'Температура вспышки, С_2',
        'Поверхностная плотность, г/м2',
        'Потребление смолы, г/м2',
        'Угол нашивки, град',
        'Шаг нашивки',
        'Плотность нашивки',
    ]

    TARGET_E   = 'Модуль упругости при растяжении, ГПа'
    TARGET_UTS = 'Прочность при растяжении, МПа'
    TARGET_NN  = 'Соотношение матрица-наполнитель'

    X = df_doe[FEATURES]

    scaler = RobustScaler()
    X_sc   = scaler.fit_transform(X)

    rf_e = RandomForestRegressor(n_estimators=400, random_state=42)
    rf_e.fit(X_sc, df_doe[TARGET_E])

    gb_uts = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_uts.fit(X_sc, df_doe[TARGET_UTS])

    nn = MLPRegressor(
        hidden_layer_sizes=(16, 8),
        activation='relu',
        solver='lbfgs',
        alpha=0.1,
        max_iter=2000,
        random_state=42,
    )
    nn.fit(X_sc, df_doe[TARGET_NN])

    return scaler, rf_e, gb_uts, nn, FEATURES

scaler, rf_e, gb_uts, nn, FEATURES = load_and_train()

task = st.radio(
    '**Выберите задачу:**',
    options=[
        '📊 Задание 1 — Модуль упругости и Прочность (ML-модели)',
        '🧠 Задание 2 — Соотношение матрица-наполнитель (Нейронная сеть)',
    ],
)
st.markdown('---')

st.sidebar.header('⚙️ Параметры компонентов')

st.sidebar.subheader('Связующее (матрица)')
density      = st.sidebar.number_input('Плотность, кг/м³',               min_value=1000.0, max_value=3000.0, value=1950.0, step=10.0)
elasticity   = st.sidebar.number_input('Модуль упругости матрицы, ГПа',  min_value=100.0,  max_value=2000.0, value=500.0,  step=10.0)
hardener     = st.sidebar.number_input('Количество отвердителя, м.%',    min_value=0.0,    max_value=300.0,  value=129.0,  step=1.0)
epoxy        = st.sidebar.number_input('Содержание эпоксидных групп, %', min_value=0.0,    max_value=50.0,   value=21.0,   step=0.1)
flash_temp   = st.sidebar.number_input('Температура вспышки, °С',        min_value=50.0,   max_value=500.0,  value=300.0,  step=5.0)

st.sidebar.subheader('Наполнитель')
surface_density = st.sidebar.number_input('Поверхностная плотность, г/м²', min_value=100.0, max_value=2000.0, value=380.0, step=10.0)
resin           = st.sidebar.number_input('Потребление смолы, г/м²',        min_value=50.0,  max_value=500.0,  value=120.0, step=5.0)

st.sidebar.subheader('Параметры нашивки')
stitch_angle   = st.sidebar.selectbox('Угол нашивки, град', options=[0, 90])
stitch_step    = st.sidebar.number_input('Шаг нашивки',       min_value=0.0, max_value=20.0,  value=10.0, step=1.0)
stitch_density = st.sidebar.number_input('Плотность нашивки', min_value=0.0, max_value=100.0, value=47.0, step=1.0)

predict_btn = st.sidebar.button('🚀 Получить прогноз', use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader('📋 Введённые параметры')
    params = pd.DataFrame({
        'Параметр': [
            'Плотность, кг/м³',
            'Модуль упругости матрицы, ГПа',
            'Количество отвердителя, м.%',
            'Содержание эпоксидных групп, %',
            'Температура вспышки, °С',
            'Поверхностная плотность, г/м²',
            'Потребление смолы, г/м²',
            'Угол нашивки, град',
            'Шаг нашивки',
            'Плотность нашивки',
        ],
        'Значение': [
            density, elasticity, hardener, epoxy, flash_temp,
            surface_density, resin, stitch_angle, stitch_step, stitch_density,
        ]
    })
    st.dataframe(params, hide_index=True, use_container_width=True)

with col2:
    st.subheader('📊 Результат')

    if predict_btn:
        X_input = np.array([[
            density, elasticity, hardener, epoxy, flash_temp,
            surface_density, resin, stitch_angle, stitch_step, stitch_density,
        ]])
        X_sc = scaler.transform(X_input)

        if 'Задание 1' in task:
            E_pred   = rf_e.predict(X_sc)[0]
            UTS_pred = gb_uts.predict(X_sc)[0]

            st.success('✅ Прогноз выполнен')
            c1, c2 = st.columns(2)
            with c1:
                st.metric('Модуль упругости при растяжении', f'{E_pred:.2f} ГПа')
            with c2:
                st.metric('Прочность при растяжении', f'{UTS_pred:.1f} МПа')

            st.info(
                f'Модуль упругости: **{E_pred:.2f} ГПа** (RandomForest, R²=0.37)\n\n'
                f'Прочность: **{UTS_pred:.1f} МПа** (GradientBoosting, R²=0.32)'
            )

        else:
            ratio_pred = nn.predict(X_sc)[0]
            st.success('✅ Рекомендация сформирована')
            st.metric('Соотношение матрица-наполнитель', f'{ratio_pred:.3f}')
            st.info(f'Нейронная сеть рекомендует соотношение **{ratio_pred:.3f}**')
            st.warning('⚠️ Результат носит рекомендательный характер (малый объём данных — 39 наблюдений)')
    else:
        st.info('👈 Введите параметры и нажмите **"Получить прогноз"**')

st.markdown('---')
st.caption('Модели: RandomForest (E_target) | GradientBoosting (UTS_target) | MLP lbfgs (ratio) | LOO кросс-валидация | DOE-блок 39 наблюдений')