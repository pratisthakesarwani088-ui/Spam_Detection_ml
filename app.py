import streamlit as st
import joblib



# PAGE CONFIGURATIOn


st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="centered"
)


# ==============================
# LOAD TRAINED MODEL
# ==============================

model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

with open("model/accuracy.txt", "r") as f:
    model_accuracy = float(f.read()) * 100


# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
}

.block-container {
    max-width: 850px;
    padding-top: 45px;
}


/* Header */

.title {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    color: white;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 17px;
    margin-bottom: 35px;
}


/* Result area */

.result-heading {
    text-align: center;
    color: white;
    font-size: 18px;
    font-weight: 600;
    margin-top: 25px;
    margin-bottom: 15px;
}


/* Footer */

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 45px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)



# HEADER


st.markdown(
    '<div class="title">🛡️ SpamGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'SMS Spam Detection using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)



# MESSAGE INPUT


st.markdown("### 📩 Enter your message")

message = st.text_area(
    "Message",
    placeholder="Type or paste your SMS here...",
    height=150,
    label_visibility="collapsed"
)



# BUTTONS


col1, col2 = st.columns(2)

with col1:

    analyze = st.button(
        "🔍 Analyze Message",
        use_container_width=True
    )

with col2:

    clear = st.button(
        "🗑️ Clear",
        use_container_width=True
    )

# CLEAR BUTTON


if clear:
    st.rerun()


# ==============================
# PREDICTION
# ==============================

if analyze:

    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")

    else:

        # Convert text into TF-IDF
        message_vector = vectorizer.transform([message])

        # Prediction
        prediction = model.predict(message_vector)[0]

        # Probability
        probabilities = model.predict_proba(message_vector)[0]

        # Confidence
        confidence = max(probabilities) * 100


        # ==============================
        # SPAM RESULT
        # ==============================

        if prediction == "spam":

            st.error("🚨 SPAM MESSAGE")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            with col2:
                st.metric(
                    "Model Accuracy",
                    f"{model_accuracy:.2f}%"
                )

            st.info(
                "💡 Suggestion: This message may be suspicious. "
                "Avoid clicking unknown links or sharing personal "
                "or financial information."
            )


        # ==============================
        # NOT SPAM RESULT
        # ==============================

        else:

            st.success("✅ NOT SPAM")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            with col2:
                st.metric(
                    "Model Accuracy",
                    f"{model_accuracy:.2f}%"
                )

            st.info(
                "💡 Suggestion: This message appears to be normal. "
                "Proceed if you recognize the sender."
            )


# ==============================
# FOOTER
# ==============================

st.markdown(
    """
    <div class="footer">
        SpamGuard AI • Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)