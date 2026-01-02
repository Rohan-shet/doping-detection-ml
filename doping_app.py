import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import io
from fpdf import FPDF
import os
import plotly.io as pio
pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width = 700
pio.kaleido.scope.default_height = 400

# 🎨 Page config
st.set_page_config(page_title="Doping Detection", layout="wide")
st.markdown("<h1 style='color:#4CAF50;'>🏋‍♂️ Doping Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Analyze athlete metrics and detect suspicious performance patterns using a Random Forest classifier.")

# 📂 Upload
uploaded_file = st.file_uploader("📥 Upload athlete dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("### 🔍 Data Preview")
    st.dataframe(df.head())

    required_cols = ['vo2_max', 'hemoglobin', 'testosterone', 'is_doped']
    if all(col in df.columns for col in required_cols):

        df.dropna(subset=required_cols, inplace=True)
        X = df[['vo2_max', 'hemoglobin', 'testosterone']]
        y = df['is_doped']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        df['is_predicted_doped'] = rf_model.predict(X)
        accuracy = accuracy_score(y_test, rf_model.predict(X_test))
        st.success(f"✅ Model Accuracy: {accuracy:.2f} using Random Forest Classifier")

        st.markdown("### 🧠 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        fig_imp = px.bar(importance_df, x='Feature', y='Importance', color='Feature', title="Feature Importance")
        st.plotly_chart(fig_imp, use_container_width=True)

        st.sidebar.markdown("## 🔎 Filters")
        gender_filter = st.sidebar.multiselect("Gender", df['gender'].dropna().unique(), default=list(df['gender'].dropna().unique()))
        sport_filter = st.sidebar.multiselect("Sport", df['sport'].dropna().unique(), default=list(df['sport'].dropna().unique()))
        age_range = st.sidebar.slider("Age", int(df['age'].min()), int(df['age'].max()), (int(df['age'].min()), int(df['age'].max())))
        test_range = st.sidebar.slider("Testosterone", float(df['testosterone'].min()), float(df['testosterone'].max()), (float(df['testosterone'].min()), float(df['testosterone'].max())))
        if 'missed_tests' in df.columns:
            missed_range = st.sidebar.slider("Missed Tests", int(df['missed_tests'].min()), int(df['missed_tests'].max()), (int(df['missed_tests'].min()), int(df['missed_tests'].max())))
        else:
            missed_range = (0, 100)

        filtered_df = df[
            (df['gender'].isin(gender_filter)) &
            (df['sport'].isin(sport_filter)) &
            (df['age'].between(age_range[0], age_range[1])) &
            (df['testosterone'].between(test_range[0], test_range[1])) &
            (df['missed_tests'].between(missed_range[0], missed_range[1]) if 'missed_tests' in df.columns else True)
        ]

        clean_df = filtered_df[filtered_df['is_predicted_doped'] == 0]
        suspicious_df = filtered_df[filtered_df['is_predicted_doped'] == 1]

        st.markdown("### ✅ Athlete Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🟢 Clean Athletes")
            st.dataframe(clean_df.reset_index(drop=True))
        with col2:
            st.markdown("#### 🔴 Suspicious Athletes")
            st.dataframe(suspicious_df.reset_index(drop=True))

        st.markdown("---")
        st.markdown("### 📊 Summary Charts")

        pie_data = filtered_df['is_predicted_doped'].value_counts().rename({0: 'Clean', 1: 'Suspicious'}).reset_index()
        pie_data.columns = ['Status', 'Count']
        fig_pie = px.pie(pie_data, names='Status', values='Count', title='Clean vs Suspicious Athletes', color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_bar = None
        if 'sport' in df.columns:
            bar_data = suspicious_df['sport'].value_counts().reset_index()
            bar_data.columns = ['Sport', 'Count']
            fig_bar = px.bar(bar_data, x='Sport', y='Count', title='Suspicious Athletes by Sport', color='Sport', color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### 📈 Individual Athlete Metrics")
        athlete_ids = df['athlete_id'].unique() if 'athlete_id' in df.columns else df.index.tolist()
        selected_athlete = st.selectbox("Select Athlete", athlete_ids)
        athlete_data = df[df['athlete_id'] == selected_athlete] if 'athlete_id' in df.columns else df.iloc[[selected_athlete]]
        metrics = ['vo2_max', 'hemoglobin', 'testosterone']
        melted = pd.melt(athlete_data[metrics], var_name='Metric', value_name='Value')
        fig_ind = px.bar(melted, x='Metric', y='Value', title='Physiology Metrics', color='Metric', color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig_ind, use_container_width=True)

        st.markdown("---")
        filtered_csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered Predictions CSV", data=filtered_csv, file_name="filtered_doping_predictions.csv", mime="text/csv")

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name="Predictions")
        st.download_button("📊 Download Excel Report", data=excel_buffer.getvalue(), file_name="filtered_predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        class PDF(FPDF):
            def header(self):
                self.set_font("Arial", 'B', 14)
                self.set_text_color(0, 102, 204)
                self.cell(0, 10, "Doping Detection Report", ln=True, align='C')
                self.ln(12)

            def add_summary(self, total, suspicious):
                self.set_font("Arial", '', 12)
                self.set_text_color(0)
                self.cell(0, 10, f"Total Filtered Athletes: {total}", ln=True)
                self.cell(0, 10, f"Suspicious Athletes: {suspicious}", ln=True)
                self.ln(10)

            def add_table(self, title, dataframe):
                self.ln(5)
                self.set_font("Arial", 'B', 12)
                self.set_text_color(220, 50, 50)
                self.cell(0, 10, title, ln=True)
                self.ln(3)
                self.set_text_color(0)
                self.set_font("Arial", '', 9)
                epw = self.w - 2 * self.l_margin
                col_width = epw / len(dataframe.columns)
                header_map = {
                    'athlete_id': 'ID', 'age': 'Age', 'gender': 'Gen', 'sport': 'Sprt',
                    'vo2_max': 'VO2', 'hemoglobin': 'Hemo', 'testosterone': 'Testo',
                    'abnormal_blood_passport': 'ABP', 'missed_tests': 'Miss',
                    'is_doped': 'Dope?', 'is_predicted_doped': 'Pred?'
                }
                short_cols = [header_map.get(col, col[:6]) for col in dataframe.columns]
                for col in short_cols:
                    self.cell(col_width, 8, str(col), border=1)
                self.ln()
                for _, row in dataframe.iterrows():
                    for item in row:
                        self.cell(col_width, 8, str(item)[:15], border=1)
                    self.ln()
                self.ln(8)

            def add_image(self, path, title):
                self.ln(5)
                self.set_font("Arial", 'B', 12)
                self.cell(0, 10, title, ln=True)
                self.ln(4)
                self.image(path, w=180)
                self.ln(12)

        pie_path = "pie_chart_temp.png"
        bar_path = "bar_chart_temp.png"
        ind_path = "athlete_metrics_temp.png"

        pio.write_image(fig_pie, pie_path)
        if fig_bar:
            pio.write_image(fig_bar, bar_path)
        pio.write_image(fig_ind, ind_path)

        pdf = PDF()
        pdf.add_page()
        pdf.add_summary(len(filtered_df), len(suspicious_df))
        pdf.add_table("Clean Athletes (Top 10)", clean_df.head(10))
        pdf.add_table("Suspicious Athletes (Top 10)", suspicious_df.head(10))
        pdf.add_image(pie_path, "Chart: Clean vs Suspicious")
        if fig_bar:
            pdf.add_image(bar_path, "Chart: Suspicious by Sport")
        pdf.add_image(ind_path, "Chart: Athlete Metrics")

        pdf_bytes = pdf.output(dest='S').encode('latin1')
        st.download_button("📄 Download Full PDF Report", data=pdf_bytes, file_name="doping_report.pdf", mime="application/pdf")

        for f in [pie_path, bar_path, ind_path]:
            if os.path.exists(f):
                os.remove(f)

    else:
        st.error("❌ Your CSV must include: vo2_max, hemoglobin, testosterone, is_doped")
else:
    st.info("⬆ Upload a file to begin.")
