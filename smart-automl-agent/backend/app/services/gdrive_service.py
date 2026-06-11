"""Google Drive service — with automatic token refresh."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DRIVE_FILES_URL  = "https://www.googleapis.com/drive/v3/files"
DRIVE_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"
FOLDER_MIME      = "application/vnd.google-apps.folder"
APP_FOLDER_NAME  = "Model Gen X"
TOKEN_URL        = "https://oauth2.googleapis.com/token"


class GDriveService:
    def __init__(self, token: str, user_id: int | None = None):
        """
        token: either a plain access_token string OR a JSON string with
               {access_token, refresh_token, client_id, client_secret}
        """
        self._token_data: dict = {}
        self._access_token: str = ""

        # Try to parse as JSON first
        if token and token.strip().startswith("{"):
            try:
                self._token_data = json.loads(token)
                self._access_token = self._token_data.get("access_token", "")
            except Exception:
                self._access_token = token
        else:
            self._access_token = token or ""

        self.user_id = user_id

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

    def _refresh_token(self) -> bool:
        """Try to refresh access token using refresh_token."""
        refresh_token = self._token_data.get("refresh_token")
        client_id     = self._token_data.get("client_id")
        client_secret = self._token_data.get("client_secret")

        if not refresh_token:
            # Try to get from settings
            try:
                from app.core.config import settings
                client_id     = client_id or settings.google_client_id
                client_secret = client_secret or settings.google_client_secret
            except Exception:
                pass

        if not refresh_token or not client_id or not client_secret:
            return False

        try:
            r = httpx.post(TOKEN_URL, data={
                "grant_type":    "refresh_token",
                "refresh_token": refresh_token,
                "client_id":     client_id,
                "client_secret": client_secret,
            }, timeout=15)
            if r.status_code == 200:
                new_data = r.json()
                self._access_token = new_data.get("access_token", "")
                self._token_data["access_token"] = self._access_token
                logger.info("Drive token refreshed successfully")

                # Save new token to DB
                if self.user_id:
                    try:
                        from app.db.session import SessionLocal
                        from app.models.user import User
                        db = SessionLocal()
                        user = db.query(User).filter(User.id == self.user_id).first()
                        if user:
                            user.drive_token = json.dumps(self._token_data)
                            db.commit()
                        db.close()
                    except Exception as e:
                        logger.warning("Could not save refreshed token: %s", e)
                return True
            logger.error("Token refresh failed: %s", r.text[:200])
        except Exception as e:
            logger.error("Token refresh error: %s", e)
        return False

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make request with automatic token refresh on 401."""
        r = httpx.request(method, url, headers=self.headers, timeout=15, **kwargs)
        if r.status_code == 401:
            logger.info("Drive 401 — attempting token refresh...")
            if self._refresh_token():
                r = httpx.request(method, url, headers=self.headers, timeout=15, **kwargs)
        return r

    # ── Folders ────────────────────────────────────────────────────────

    def get_or_create_folder(self, name: str, parent_id: str | None = None) -> str:
        query = f"name='{name}' and mimeType='{FOLDER_MIME}' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        r = self._request("GET", DRIVE_FILES_URL,
                          params={"q": query, "fields": "files(id,name)"})
        files = r.json().get("files", [])
        if files:
            return files[0]["id"]
        body: dict[str, Any] = {"name": name, "mimeType": FOLDER_MIME}
        if parent_id:
            body["parents"] = [parent_id]
        r = self._request("POST", DRIVE_FILES_URL, json=body)
        data = r.json()
        if "id" not in data:
            raise Exception(f"Failed to create folder '{name}': {data}")
        return data["id"]

    def get_app_folders(self) -> dict[str, str]:
        root_id     = self.get_or_create_folder(APP_FOLDER_NAME)
        datasets_id = self.get_or_create_folder("datasets", root_id)
        models_id   = self.get_or_create_folder("models",   root_id)
        return {"root": root_id, "datasets": datasets_id, "models": models_id}

    # ── List ───────────────────────────────────────────────────────────

    def list_folder(self, folder_id: str) -> list[dict]:
        r = self._request("GET", DRIVE_FILES_URL, params={
            "q":      f"'{folder_id}' in parents and trashed=false",
            "fields": "files(id,name,size,modifiedTime,webViewLink)",
        })
        return r.json().get("files", [])

    # ── Upload ─────────────────────────────────────────────────────────

    def upload_file(self, local_path: str | Path, folder_id: str,
                    filename: str | None = None) -> dict[str, str]:
        p       = Path(local_path)
        fname   = filename or p.name
        content = p.read_bytes()
        mime    = "text/csv" if fname.endswith(".csv") else "application/octet-stream"

        # Metadata upload
        meta = {"name": fname, "parents": [folder_id]}
        r = httpx.post(
            f"{DRIVE_UPLOAD_URL}?uploadType=multipart",
            headers={"Authorization": f"Bearer {self._access_token}"},
            files={
                "metadata": (None, json.dumps(meta), "application/json; charset=UTF-8"),
                "file":     (fname, content, mime),
            },
            timeout=120,
        )
        if r.status_code == 401:
            if self._refresh_token():
                r = httpx.post(
                    f"{DRIVE_UPLOAD_URL}?uploadType=multipart",
                    headers={"Authorization": f"Bearer {self._access_token}"},
                    files={
                        "metadata": (None, json.dumps(meta), "application/json; charset=UTF-8"),
                        "file":     (fname, content, mime),
                    },
                    timeout=120,
                )
        data = r.json()
        return {"id": data.get("id",""), "name": fname,
                "webViewLink": data.get("webViewLink","")}

    # ── Download ───────────────────────────────────────────────────────

    def download_file(self, file_id: str, dest_path: str | Path) -> Path:
        r = self._request("GET", f"{DRIVE_FILES_URL}/{file_id}?alt=media")
        Path(dest_path).write_bytes(r.content)
        return Path(dest_path)

    # ── Delete ─────────────────────────────────────────────────────────

    def delete_file(self, file_id: str) -> bool:
        r = self._request("DELETE", f"{DRIVE_FILES_URL}/{file_id}")
        return r.status_code == 204


def _get_drive_service(user, user_id: int | None = None) -> GDriveService:
    """Helper to build GDriveService from user.drive_token."""
    token = user.drive_token if hasattr(user, "drive_token") else user
    return GDriveService(token or "", user_id=user_id)