# Cropify
# Crop Recommender with Claude Sonnet 🧑‍🌾🌾

This project is a Streamlit application that recommends suitable crops based on various environmental parameters using the Claude model from Anthropic.

## Features

- Input environmental parameters to get crop recommendations.
- Update or create vector stores for document ingestion.
- Uses Langchain and AWS Bedrock for embeddings and LLM services.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repository-url
    cd your-repository
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Configure AWS CLI with your credentials:
    ```bash
    aws configure
    ```

    You will be prompted to enter your AWS Access Key ID, Secret Access Key, default region name, and default output format. Make sure you have the necessary permissions to access Bedrock services.

## Usage

1. Prepare your PDF documents and place them in a directory named `data`.

2. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

3. Use the web interface to input environmental parameters and get crop recommendations.

## File Structure

- `app.py`: Main application file.
- `requirements.txt`: List of required packages.
- `README.md`: This readme file.

## How It Works

1. **Data Ingestion**: Load PDF documents and split them into chunks using `PyPDFDirectoryLoader` and `RecursiveCharacterTextSplitter`.
2. **Vector Embedding and Vector Store**: Create embeddings using the Titan Embeddings Model and store them using FAISS.
3. **LLM Integration**: Use the Claude model to generate responses based on the input query and context retrieved from the vector store.

## License

This project is licensed under the MIT License.
