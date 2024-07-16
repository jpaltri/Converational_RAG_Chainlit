# AI Chatbot for Air Imports and CargoWise

This project is an AI-powered chatbot designed to assist with questions related to air imports and CargoWise. Leveraging advanced language models, the chatbot provides accurate and concise answers, maintaining a user-friendly and interactive interface. It's built to handle complex queries by contextualizing previous interactions, making it a powerful tool for industry professionals.

## Features

### Memory Management
- **Session Management**: The chatbot utilizes a stateful approach to manage chat history, ensuring that context is preserved throughout a user's session. This means the chatbot can remember previous questions and answers within a single session, allowing for more coherent and contextually relevant conversations.
- **History Awareness**: The chatbot can reformulate user questions based on the chat history. This feature ensures that follow-up questions are understood correctly even if they reference previous parts of the conversation.

### User Interface
- **Custom Chat Profiles**: Users can select from different chat profiles. Each profile represents a different underlying language model, such as GPT-3.5 or GPT-4, allowing users to choose the model that best suits their needs.
- **Interactive Settings**: The chatbot interface includes customizable settings. Users can select which model to use, enable or disable token streaming for real-time responses, and adjust the temperature setting to control the creativity of the responses.
- **Predefined Starters**: The chatbot includes predefined starter prompts to help users quickly get answers to common questions related to air imports and CargoWise.

### Retrieval-Augmented Generation (RAG)
- **Document Integration**: The chatbot can load and process PDF documents, using them as a knowledge base for answering user queries. This integration allows the chatbot to provide precise information from the documents directly in its responses.
- **Advanced Retrieval**: The chatbot employs RAG techniques to enhance its responses. By retrieving relevant information from a vector store before generating answers, the chatbot ensures that responses are not only contextually relevant but also backed by accurate data from the provided documents.

### Chainlit Integration
- **Chainlit Framework**: The chatbot is built using the Chainlit framework, which provides a robust infrastructure for developing and deploying conversational AI applications. Chainlit facilitates the creation of dynamic and interactive user interfaces, enhancing the overall user experience.

## Setup

### Prerequisites
- Python 3.7+
- `pip` (Python package installer)

### Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Create a `.env` file in the project root directory and add your environment variables:**

    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    ANTHROPIC_API_KEY=your_anthropic_api_key
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=your_langchain_api_key
    ```

## Usage

1. **Run the chatbot:**

    ```sh
    python your_script_name.py
    ```

2. **Interact with the chatbot through the provided UI.**

## Project Structure

- `Converational_RAG_Chainlit.py`: Main script to run the chatbot.
- `requirements.txt`: Lists the dependencies required for the project.
- `.env`: Contains environment variables (not included in the repository for security reasons).
- `README.md`: Project documentation.


