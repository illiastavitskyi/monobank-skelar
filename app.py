import os
import tempfile
import streamlit as st
from agent_inference import PricingAgent


st.set_page_config(page_title="AI Прайсер | mono x SKELAR", page_icon="🐱", layout="wide")

@st.cache_resource
def load_agent():
    return PricingAgent()

st.title("🐱 AI-Агент Прайсер | mono x SKELAR")
st.markdown("Завантаж опис та фото вживаної речі, щоб отримати рекомендовану ціну на топобазарі.")

with st.spinner("Завантаження нейромереж (RoBERTa, CLIP, XGBoost)... Це може зайняти хвилину."):
    agent = load_agent()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Дані про товар")
    
    description = st.text_area(
        "Опис оголошення:", 
        height=150,
        placeholder="Наприклад: Продам навушники AirPods Pro 2, стан ідеальний, є коробка. Все працює чудово..."
    )
    
    category_id = st.number_input("ID Категорії (якщо відомо):", min_value=1, max_value=100, value=4)
    
    uploaded_files = st.file_uploader(
        "Завантаж фото (від 1 до 5 шт.)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

with col2:
    st.subheader("📊 Результат оцінки агента")
    
    if st.button("🚀 Згенерувати ціну", use_container_width=True, type="primary"):
        if not description:
            st.warning("Будь ласка, додай опис товару.")
        elif not uploaded_files:
            st.warning("Будь ласка, завантаж хоча б одне фото.")
        else:
            with st.spinner("🧠 Агент аналізує текст та шукає візуальні компаративи в базі..."):
                
                first_image = uploaded_files[0]
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(first_image.getvalue())
                    tmp_image_path = tmp_file.name
                
                try:
                    prediction = agent.predict(description, tmp_image_path, category_id=category_id)
                    
                    st.success("Аналіз успішно завершено!")
                    
                    st.metric(
                        label="Рекомендована ціна", 
                        value=f"{prediction['recommended_price']} грн"
                    )
                    st.caption(f"📈 Ціновий діапазон: {prediction['price_range']}")
                    
                    st.divider()
                    
                    st.markdown("### 🎯 Стратегії продажу")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🚀 Fast (Швидко)", f"{prediction['strategies']['fast']} грн")
                    c2.metric("⚖️ Balanced", f"{prediction['strategies']['balanced']} грн")
                    c3.metric("💰 Max Profit", f"{prediction['strategies']['max_profit']} грн")
                    
                    st.divider()
                    
                    st.markdown("### 🕵️‍♂️ Чому така ціна?")
                    st.info(prediction['explanation']['text_analysis_conclusion'])
                    
                    st.write(f"**Візуальний якір (ChromaDB):** Базова ціна схожих товарів ~**{prediction['explanation']['visual_anchor_price']} грн**")
                    st.write(f"**Знайдено компаративів:** {prediction['explanation']['comparatives_found']} шт.")
                    
                    if prediction['explanation']['similar_items']:
                        with st.expander("Подивитись схожі товари (ID)"):
                            for item in prediction['explanation']['similar_items']:
                                st.write(f"- ID: `{item['id']}` | Продано за: **{item['price']} грн**")
                                
                except Exception as e:
                    st.error(f"Виникла помилка під час обробки: {e}")
                finally:
                    if os.path.exists(tmp_image_path):
                        os.remove(tmp_image_path)
