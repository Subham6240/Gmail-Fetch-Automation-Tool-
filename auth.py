# auth.py

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.exceptions import RefreshError

from config import SCOPES, TOKEN_PATH, CLIENT_SECRET_PATH


class GmailAuth:
    """Creates an authenticated Gmail service client."""

    def __init__(self, client_secret_path: str = CLIENT_SECRET_PATH, token_path: str = TOKEN_PATH):
        self.client_secret_path = client_secret_path
        self.token_path = token_path
        self.service = None

    def get_service(self):
        if self.service is not None:
            return self.service

        creds = None

        def fresh_login() -> Credentials:
            flow = InstalledAppFlow.from_client_secrets_file(self.client_secret_path, SCOPES)
            new_creds = flow.run_local_server(port=0)
            with open(self.token_path, "w", encoding="utf-8") as token:
                token.write(new_creds.to_json())
            return new_creds

        # Load existing token if present
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
            # If scopes changed, force re-login
            if not creds or not creds.scopes or not set(SCOPES).issubset(set(creds.scopes)):
                try:
                    os.remove(self.token_path)
                except OSError:
                    pass
                creds = None

        # Refresh or login
        if not creds or not creds.valid:
            try:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if os.path.exists(self.token_path):
                        try:
                            os.remove(self.token_path)
                        except OSError:
                            pass
                    creds = fresh_login()
            except RefreshError:
                if os.path.exists(self.token_path):
                    try:
                        os.remove(self.token_path)
                    except OSError:
                        pass
                creds = fresh_login()

        self.service = build("gmail", "v1", credentials=creds)
        return self.service