import streamlit as st
import openai
import os
# Replace with your OpenAI API key
openai.api_key = st.secrets("OPENAI_API_KEY")


title_style = """
    color: #008080; /* Dark cyan color */
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 30px;
"""
text_style = """
    font-size: 18px;
    color: #800080; /* Dark gray color */
"""

# Define CSS style for success message
success_style = """
    font-size: 18px;
    color: #008000; /* Green color */
"""
created_style = """
    color: #888888; /* Light gray color */
    font-size: 99px; /* Increased font size */
"""
# Define CSS style for loading spinner
spinner_style = """
    width: 50px;
    height: 50px;
"""

def get_embeddings(text):
  """Fetches an embedding vector for a given text using OpenAI's Embeddings API."""
  response = openai.embeddings.create(
      model="text-embedding-3-small",  # Consider using a more suitable engine
      input=text
  )
  return response.data[0].embedding

def cosine_similarity(embedding1, embedding2):
  """Calculates the cosine similarity between two embedding vectors."""
  dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
  magnitude1 = sum(x**2 for x in embedding1) ** 0.5
  magnitude2 = sum(x**2 for x in embedding2) ** 0.5
  if magnitude1 == 0 or magnitude2 == 0:
    return 0  # Avoid division by zero
  return dot_product / (magnitude1 * magnitude2)

def main():
  """Streamlit app to calculate text similarity."""
  st.set_page_config(page_title="Text Similarity Checker", page_icon=":chart_with_upwards_trend:")
  st.markdown("<p style='{}'>➡️created by 'Muhammad Zain Attiq'</p>".format(created_style), unsafe_allow_html=True)

  st.markdown("<h1 style='{}'>Text Similarity Checker</h1>".format(title_style), unsafe_allow_html=True)


  text1 = st.text_input("Enter Text 1:", "")
  text2 = st.text_input("Enter Text 2:", "")

  if st.button("Compare"):
    if text1 and text2:
        try:
            with st.spinner("Creating embeddings for text 1..."):
                embedding1 = get_embeddings(text1)
            st.success("Done")
            st.markdown(f"<p style='{text_style}'>Embedding for Text 1 created successfully.</p>", unsafe_allow_html=True)

            # Text 2
            with st.spinner("Creating embeddings for text 2..."):
                embedding2 = get_embeddings(text2)
            st.success("Done")
            st.markdown(f"<p style='{text_style}'>Embedding for Text 2 created successfully.</p>", unsafe_allow_html=True)

            # Similarity calculation
            with st.spinner("Calculating similarity..."):
                similarity = cosine_similarity(embedding1, embedding2)
            st.success("Done")
            st.markdown(f"<p style='{success_style}'>Similarity Score: {similarity:.4f}</p>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.write("Please enter both texts before clicking 'Compare'")

if __name__ == "__main__":
    main()
