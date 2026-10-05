import streamlit as st
import pandas as pd
import requests
import json

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Product Recommendation System",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# LOAD PRODUCT DATABASE
# =========================================================

@st.cache_data
def load_products():

    df = pd.read_csv("products.csv")

    # Convert numeric columns if they exist
    for column in ["price", "rating", "reviews"]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


products = load_products()


# =========================================================
# SESSION STATE
# =========================================================

if "selected_product" not in st.session_state:

    st.session_state.selected_product = None


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# =========================================================
# TITLE
# =========================================================

st.title("🤖 AI Product Recommendation System")

st.write(
    "Search products, compare products and ask an AI assistant "
    "anything about the recommended product."
)

st.divider()


# =========================================================
# FIND YOUR PRODUCT
# =========================================================

st.header("🔎 Find Your Product")

product_search = st.text_input(
    "Enter any product",
    placeholder="Example: laptop, mobile, refrigerator..."
)


# =========================================================
# FILTERS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    category = st.selectbox(
        "📂 Category",
        ["All"] +
        sorted(
            products["category"]
            .dropna()
            .unique()
            .tolist()
        )
    )


with col2:

    min_price = st.number_input(
        "💰 Minimum Price (₹)",
        min_value=0,
        max_value=500000,
        value=0,
        step=500
    )


with col3:

    max_price = st.number_input(
        "💰 Maximum Price (₹)",
        min_value=500,
        max_value=500000,
        value=100000,
        step=500
    )


with col4:

    min_rating = st.slider(
        "⭐ Minimum Rating",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.1
    )


# =========================================================
# SEARCH FUNCTION
# =========================================================

def search_products():

    results = products.copy()

    # -----------------------------------------------
    # PRODUCT SEARCH
    # -----------------------------------------------

    if product_search.strip():

        search = product_search.lower().strip()

        search_columns = [
            "name",
            "category",
            "subcategory",
            "brand",
            "purpose",
            "description",
            "features",
            "recommended_for"
        ]

        mask = pd.Series(
            False,
            index=results.index
        )

        for column in search_columns:

            if column in results.columns:

                mask = (
                    mask
                    |
                    results[column]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        search,
                        na=False
                    )
                )

        results = results[mask]

    # -----------------------------------------------
    # CATEGORY
    # -----------------------------------------------

    if category != "All":

        results = results[
            results["category"] == category
        ]

    # -----------------------------------------------
    # PRICE
    # -----------------------------------------------

    results = results[
        (results["price"] >= min_price)
        &
        (results["price"] <= max_price)
    ]

    # -----------------------------------------------
    # RATING
    # -----------------------------------------------

    results = results[
        results["rating"] >= min_rating
    ]

    # -----------------------------------------------
    # SORT
    # -----------------------------------------------

    results = results.sort_values(
        by=["rating", "reviews"],
        ascending=False
    )

    return results


# =========================================================
# FIND PRODUCTS BUTTON
# =========================================================

if st.button(
    "🔍 Find Products",
    use_container_width=True
):

    results = search_products()

    st.session_state.search_results = results


# =========================================================
# SHOW SEARCH RESULTS
# =========================================================

if "search_results" in st.session_state:

    results = st.session_state.search_results

    st.divider()

    st.subheader("🏆 Recommended Products")

    if len(results) == 0:

        st.warning(
            "😕 No products found. Try changing your search "
            "or filters."
        )

    else:

        st.success(
            f"✅ {len(results)} product(s) found!"
        )

        product_columns = st.columns(3)

        for index, (_, product) in enumerate(
            results.iterrows()
        ):

            with product_columns[index % 3]:

                st.markdown(
                    f"""
                    ### 🛍️ {product['name']}

                    **Brand:** {product['brand']}

                    💰 **₹{product['price']:,.0f}**

                    ⭐ **{product['rating']}/5**

                    👥 **{product['reviews']:,} reviews**

                    📂 **{product['category']}**

                    🎯 **Best for:** {product['purpose']}
                    """
                )

                st.write(
                    product["description"]
                )

                st.caption(
                    "🔧 Features: "
                    + str(product["features"])
                )


        # =================================================
        # SELECT PRODUCT FOR AI CHAT
        # =================================================

        st.divider()

        st.subheader(
            "💬 Choose a Product to Ask the AI About"
        )

        product_names = results["name"].tolist()

        selected_name = st.selectbox(
            "Select a product",
            product_names
        )

        selected_row = results[
            results["name"] == selected_name
        ].iloc[0]

        st.session_state.selected_product = (
            selected_row.to_dict()
        )

        st.success(
            f"✅ AI is now ready to answer questions "
            f"about **{selected_name}**."
        )


# =========================================================
# PRODUCT INFORMATION BUILDER
# =========================================================

def get_complete_product_information(product):

    information = []

    for column, value in product.items():

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value == "":
            continue

        # Make column names readable
        readable_column = column.replace(
            "_",
            " "
        ).title()

        information.append(
            f"{readable_column}: {value}"
        )

    return "\n".join(information)


# =========================================================
# OLLAMA AI FUNCTION
# =========================================================

def ask_ollama(
    question,
    product_information,
    chat_history
):

    # -----------------------------------------------------
    # SYSTEM PROMPT
    # -----------------------------------------------------

    system_prompt = f"""
You are an AI Product Assistant.

You are answering questions about ONE specific product.

IMPORTANT RULES:

1. Use the product information provided below.
2. Answer questions using the information available about
   this product.
3. Do not invent specifications.
4. If the requested information is not available in the
   product information, clearly say:
   "This information is not available in the product
   database."
5. You can explain the usefulness, features, advantages,
   purpose and suitability of the product using the
   information provided.
6. Give clear and easy-to-understand answers.
7. If the user asks "tell me everything", organize all
   available product information into sections.
8. If the user asks whether the product is good for a
   particular purpose, explain using the available
   specifications and purpose.
9. Never create fake specifications.

PRODUCT INFORMATION:

{product_information}
"""

    # -----------------------------------------------------
    # BUILD MESSAGES
    # -----------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Add previous conversation
    for item in chat_history:

        messages.append(
            {
                "role": item["role"],
                "content": item["content"]
            }
        )

    # Add current question
    messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # -----------------------------------------------------
    # OLLAMA REQUEST
    # -----------------------------------------------------

    url = "http://localhost:11434/api/chat"

    payload = {
        "model": "llama3.2",
        "messages": messages,
        "stream": False
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:

            return (
                "❌ Ollama returned an error.\n\n"
                f"Status code: {response.status_code}\n\n"
                f"{response.text}"
            )

        data = response.json()

        return data["message"]["content"]

    except requests.exceptions.ConnectionError:

        return (
            "❌ I could not connect to Ollama.\n\n"
            "Make sure Ollama is running and the model "
            "`llama3.2` is installed."
        )

    except Exception as e:

        return (
            f"❌ Error while communicating with AI: {e}"
        )


# =========================================================
# AI CHATBOT
# =========================================================

st.divider()

st.header("🤖 AI Product Chatbot")

if st.session_state.selected_product is None:

    st.info(
        "👆 First search for a product and select a "
        "product above. Then you can ask the AI anything "
        "about that product."
    )

else:

    product = st.session_state.selected_product

    product_name = product.get(
        "name",
        "Selected Product"
    )

    st.markdown(
        f"### 🛍️ Currently discussing: **{product_name}**"
    )

    st.caption(
        "You can ask multiple questions about this product."
    )

    # -----------------------------------------------------
    # SHOW PREVIOUS CHAT
    # -----------------------------------------------------

    for message in st.session_state.chat_history:

        if message["role"] == "user":

            with st.chat_message("user"):

                st.write(
                    message["content"]
                )

        else:

            with st.chat_message("assistant"):

                st.markdown(
                    message["content"]
                )

    # -----------------------------------------------------
    # CHAT INPUT
    # -----------------------------------------------------

    question = st.chat_input(
        f"Ask anything about {product_name}..."
    )

    if question:

        # Display user question
        with st.chat_message("user"):

            st.write(question)

        # Complete product information
        product_information = (
            get_complete_product_information(
                product
            )
        )

        # Ask AI
        answer = ask_ollama(
            question,
            product_information,
            st.session_state.chat_history
        )

        # Display answer
        with st.chat_message("assistant"):

            st.markdown(answer)

        # Save conversation
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
