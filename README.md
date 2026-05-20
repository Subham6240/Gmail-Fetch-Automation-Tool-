# AI-Powered Gmail Reply Assistant

This project is a Python-based Gmail assistant that uses the Gmail API, Google OAuth 2.0, LangChain, and OpenAI LLMs to read emails, summarize content, generate AI-based reply suggestions, and send replies through a simple Flask web interface.

## Features

- Gmail OAuth authentication
- Read and filter Gmail messages
- Generate AI-powered email replies
- Multiple reply tones
- Send threaded replies
- Send new emails with attachments
- Summarize email and attachment content
- Flask-based web interface

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Subham6240/Gmail-Fetch-Automation-Tool.git
cd Gmail-Fetch-Automation-Tool
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create the `.env` file

Create a `.env` file in the project root and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
HOST=127.0.0.1
PORT=5000
```

### 4. Download `credentials.json` from Google Gmail API

This project uses Google OAuth, so you must download a Gmail API OAuth client file and save it as `credentials.json`.

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project, or select an existing project.
3. Go to **APIs & Services > Library**.
4. Search for **Gmail API** and click **Enable**.
5. Go to **APIs & Services > OAuth consent screen** and complete the required app setup.
   - For personal testing, choose **External** and add your own Gmail account as a test user if required.
6. Go to **APIs & Services > Credentials**.
7. Click **Create Credentials > OAuth client ID**.
8. Select **Desktop app** as the application type.
9. Click **Create**.
10. Download the JSON file.
11. Rename the downloaded file to:

```text
credentials.json
```

12. Move `credentials.json` into the root folder of this project, next to files like `main.py`, `auth.py`, and `config.py`.

> Do not upload or share `credentials.json`. It contains private OAuth client details. The project already ignores `credentials.json` and `token.json` through `.gitignore`.

### 5. Run the app

```bash
python main.py
```

On the first run, a browser window will open and ask you to log in with your Gmail account. After successful login, a `token.json` file will be created automatically for future use.

## Notes

- Keep `credentials.json` private.
- Delete `token.json` if you change Gmail API scopes or want to re-authenticate.
- Make sure the Gmail account you use is added as a test user in the OAuth consent screen if your app is still in testing mode.
