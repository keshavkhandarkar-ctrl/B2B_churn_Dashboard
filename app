import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("B2B Client Risk & Churn Prediction Dashboard")

# Load dataset
df = pd.read_csv("B2B_Client_Churn_5000.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# ----------------------------
# Risk Score Logic
# ----------------------------

def calculate_risk(row):
    score = 0
    
    if row["Payment_Delay_Days"] > 30:
        score += 3
        
    if row["Monthly_Usage"] < 50:
        score += 2
        
    if row["Contract_Length"] < 6:
        score += 2
        
    if row["Support_Tickets"] > 5:
        score += 3
        
    return score

df["Risk_Score"] = df.apply(calculate_risk, axis=1)

# Risk Category

def risk_category(score):
    
    if score <= 2:
        return "Low Risk"
        
    elif score <= 5:
        return "Medium Risk"
        
    else:
        return "High Risk"

df["Risk_Category"] = df["Risk_Score"].apply(risk_category)

# ----------------------------
# KPIs
# ----------------------------

st.subheader("Key Metrics")

total_clients = len(df)
high_risk_clients = len(df[df["Risk_Category"] == "High Risk"])
avg_revenue = df["Revenue"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("Total Clients", total_clients)
col2.metric("High Risk Clients", high_risk_clients)
col3.metric("Average Revenue", round(avg_revenue,2))

# ----------------------------
# Risk Distribution Chart
# ----------------------------

st.subheader("Risk Category Distribution")

risk_counts = df["Risk_Category"].value_counts()

st.bar_chart(risk_counts)

# ----------------------------
# Revenue vs Risk Scatter
# ----------------------------

st.subheader("Revenue vs Risk Score")

fig, ax = plt.subplots()

ax.scatter(df["Revenue"], df["Risk_Score"])

ax.set_xlabel("Client Revenue")
ax.set_ylabel("Risk Score")

st.pyplot(fig)

# ----------------------------
# Machine Learning Model
# ----------------------------

st.subheader("Churn Prediction Model")

df["Renewal_Status"] = df["Renewal_Status"].map({"Yes":1,"No":0})

X = df[["Monthly_Usage","Payment_Delay_Days","Contract_Length","Support_Tickets","Revenue"]]

y = df["Renewal_Status"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = DecisionTreeClassifier()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test,predictions)

st.write("Model Accuracy:", round(accuracy*100,2), "%")

# Confusion Matrix

st.write("Confusion Matrix")

st.write(confusion_matrix(y_test,predictions))

# ----------------------------
# High Risk Clients Table
# ----------------------------

st.subheader("Top 20 High Risk Clients")

high_risk = df[df["Risk_Category"]=="High Risk"]

st.dataframe(high_risk.head(20))

# ----------------------------
# Retention Strategy
# ----------------------------

if st.button("Generate Retention Strategy"):

    st.write("Offer payment flexibility for clients with delays above 30 days")
    
    st.write("Assign account managers to high revenue clients")
    
    st.write("Encourage longer contracts through incentives")
    
    st.write("Improve customer support for clients with frequent complaints")

# ----------------------------
# Responsible AI
# ----------------------------

st.subheader("Ethical Implications of Predicting Client Churn")

st.write("""
Predictive models may introduce bias depending on historical data patterns.
Labeling customers as high risk may influence business decisions and customer treatment.

Organizations should ensure transparency, protect client data privacy, and use
AI predictions responsibly to support better decision-making rather than unfair targeting.
""")
