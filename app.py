import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------
# PAGE SETTINGS
# -----------------------------------

st.set_page_config(
    page_title="B2B Client Intelligence Lab",
    layout="wide"
)

# -----------------------------------
# CUSTOM STYLE
# -----------------------------------

st.markdown("""
<style>

body{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

.big{
font-size:44px;
font-weight:900;
font-family:Trebuchet MS;
}

.subtitle{
font-size:18px;
opacity:0.8;
}

.metric-box{
background:#1f2b38;
padding:15px;
border-radius:12px;
text-align:center;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big">B2B Client Intelligence Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Risk Monitoring • Churn Insights • Strategic Retention</div>', unsafe_allow_html=True)

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv("B2B_Client_Churn_5000.csv")

# -----------------------------------
# RISK SCORE (Different logic)
# -----------------------------------

def risk_calc(row):

    risk = 0

    if row["Payment_Delay_Days"] > 20:
        risk += 2

    if row["Monthly_Usage_Score"] < 40:
        risk += 2

    if row["Support_Tickets_Last30Days"] > 4:
        risk += 3

    if row["Contract_Length_Months"] < 8:
        risk += 2

    return risk


df["Risk_Score"] = df.apply(risk_calc, axis=1)

def risk_label(x):

    if x >= 6:
        return "High"
    elif x >= 3:
        return "Medium"
    else:
        return "Low"

df["Risk_Level"] = df["Risk_Score"].apply(risk_label)

df["Churn"] = df["Renewal_Status"].map({"Yes":0,"No":1})

# -----------------------------------
# KPI METRICS
# -----------------------------------

col1,col2,col3,col4 = st.columns(4)

col1.metric("Total Clients",len(df))

col2.metric(
"High Risk Clients",
len(df[df["Risk_Level"]=="High"])
)

col3.metric(
"Average Revenue",
round(df["Monthly_Revenue_USD"].mean(),2)
)

col4.metric(
"Churn Rate %",
round(df["Churn"].mean()*100,2)
)

st.divider()

# -----------------------------------
# TABS LAYOUT
# -----------------------------------

tab1,tab2,tab3,tab4 = st.tabs(
["Risk Overview","Business Segments","Churn Prediction","Client Explorer"]
)

# -----------------------------------
# TAB 1
# -----------------------------------

with tab1:

    st.subheader("Risk Distribution")

    counts = df["Risk_Level"].value_counts()

    fig1 = plt.figure()

    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%"
    )

    plt.title("Risk Level Share")

    st.pyplot(fig1)

    st.subheader("Revenue Distribution by Risk")

    fig2 = plt.figure()

    plt.boxplot([
        df[df["Risk_Level"]=="Low"]["Monthly_Revenue_USD"],
        df[df["Risk_Level"]=="Medium"]["Monthly_Revenue_USD"],
        df[df["Risk_Level"]=="High"]["Monthly_Revenue_USD"]
    ])

    plt.xticks([1,2,3],["Low","Medium","High"])
    plt.ylabel("Revenue")

    st.pyplot(fig2)

# -----------------------------------
# TAB 2
# -----------------------------------

with tab2:

    st.subheader("Industry vs Region Risk Heatmap")

    pivot = pd.pivot_table(
        df,
        index="Industry",
        columns="Region",
        values="Risk_Score",
        aggfunc="mean"
    )

    fig3 = plt.figure()

    plt.imshow(pivot,aspect="auto")

    plt.colorbar(label="Risk Score")

    plt.xticks(range(len(pivot.columns)),pivot.columns,rotation=45)

    plt.yticks(range(len(pivot.index)),pivot.index)

    st.pyplot(fig3)

    st.subheader("Contract Length vs Churn")

    bins = pd.cut(df["Contract_Length_Months"],bins=6)

    churn = df.groupby(bins)["Churn"].mean()*100

    fig4 = plt.figure()

    plt.fill_between(range(len(churn)),churn.values)

    plt.xticks(range(len(churn)),[str(x) for x in churn.index],rotation=45)

    plt.ylabel("Churn %")

    st.pyplot(fig4)

# -----------------------------------
# TAB 3 (ML MODEL)
# -----------------------------------

with tab3:

    st.subheader("Churn Prediction Model")

    features = [
    "Monthly_Usage_Score",
    "Payment_Delay_Days",
    "Contract_Length_Months",
    "Support_Tickets_Last30Days",
    "Monthly_Revenue_USD",
    "Risk_Score"
    ]

    X = df[features]

    y = df["Churn"]

    X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2
    )

    model = DecisionTreeClassifier(max_depth=6)

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test,pred)

    st.metric("Model Accuracy",round(acc*100,2))

    imp = pd.Series(
    model.feature_importances_,
    index=features
    ).sort_values(ascending=False)

    st.bar_chart(imp)

# -----------------------------------
# TAB 4 CLIENT EXPLORER
# -----------------------------------

with tab4:

    st.subheader("Client Lookup")

    client = st.selectbox(
    "Choose Client",
    df["Client_ID"]
    )

    row = df[df["Client_ID"]==client]

    st.write(row)

    st.subheader("Recommended Actions")

    if row["Risk_Level"].values[0]=="High":

        st.warning("Immediate account manager engagement recommended")

        st.write("• Offer renewal incentive")
        st.write("• Provide dedicated support")
        st.write("• Review payment delays")

    elif row["Risk_Level"].values[0]=="Medium":

        st.info("Monitor engagement")

        st.write("• Encourage product usage")
        st.write("• Provide feature training")

    else:

        st.success("Client stable")

        st.write("• Maintain relationship")
        st.write("• Offer loyalty perks")
