
# Import the required libraries
import streamlit as st
import pickle

# Load the trained machine learning model
with open("spam_detector_model.pkl", "rb") as file:
    model = pickle.load(file)

# Configure the web page
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# Display the heading
st.title("📧 Email Spam Detector")

st.write(
    "Enter an email or message below to check "
    "whether it is spam or legitimate."
)

# Take message input from the user
message = st.text_area(
    "Enter your message here:",
    height=180,
    placeholder="Example: Congratulations! You have won a free prize..."
)

# Check the message when the button is clicked
if st.button("Check Message", type="primary"):

    if message.strip() == "":
        st.warning("Please enter a message first.")

    else:
        # Predict the message category
        prediction = model.predict([message])[0]

        # Display the result
        if prediction == "spam":
            st.error("⚠️ This message looks like SPAM!")
        else:
            st.success("✅ This message looks like NOT SPAM.")

st.caption(
    "This is a machine-learning prediction. "
    "Always verify suspicious emails before clicking links."
)
