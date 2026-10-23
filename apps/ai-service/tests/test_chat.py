import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from google.genai import errors

from app.core.config import Settings, get_settings
from app.main import app


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(gemini_api_key="test-secret", gemini_model="test-model")
        app.dependency_overrides[get_settings] = lambda: self.settings
        self.addCleanup(app.dependency_overrides.clear)
        self.client = self.enterContext(TestClient(app))
        self.factory = self.enterContext(patch("app.services.llm_service.genai.Client"))
        self.gemini = self.factory.return_value.__enter__.return_value
        self.gemini.models.generate_content.return_value = SimpleNamespace(text="Generated answer")

    def test_chat_success_uses_configured_model_and_closes_client(self):
        response = self.client.post("/chat", json={"message": " Explain RAG "})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"response": "Generated answer"})
        self.gemini.models.generate_content.assert_called_once_with(
            model="test-model", contents="Explain RAG"
        )
        self.assertEqual(self.factory.call_args.kwargs["api_key"], "test-secret")
        self.assertFalse(self.factory.call_args.kwargs["vertexai"])
        self.assertEqual(self.factory.call_args.kwargs["http_options"].timeout, 30000)
        self.factory.return_value.__exit__.assert_called_once()

    def test_invalid_requests_do_not_call_gemini(self):
        for payload in ({}, {"message": ""}, {"message": "  \n"},
                        {"message": 123}, {"message": None}, {"message": []},
                        {"message": "x" * 10001}, {"message": "hi", "extra": True}):
            with self.subTest(payload_type=type(payload.get("message"))):
                self.assertEqual(self.client.post("/chat", json=payload).status_code, 422)
        self.assertEqual(self.client.post("/chat", content="{", headers={"Content-Type": "application/json"}).status_code, 422)
        self.factory.assert_not_called()

    def test_missing_configuration(self):
        for settings in (Settings(gemini_model="test-model"), Settings(gemini_api_key="test-secret")):
            self.settings = settings
            response = self.client.post("/chat", json={"message": "Hello"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json(), {"detail": "Chat service is not configured."})
        self.factory.assert_not_called()

    def test_provider_errors_are_sanitized(self):
        for code, expected in ((400, 502), (401, 502), (403, 502), (404, 502), (429, 503), (500, 502), (503, 502)):
            with self.subTest(code=code):
                self.gemini.models.generate_content.side_effect = errors.APIError(
                    code, {"error": {"message": "test-secret raw internal details", "code": code}}
                )
                response = self.client.post("/chat", json={"message": "Hello"})
                self.assertEqual(response.status_code, expected)
                self.assertNotIn("test-secret", response.text)
                self.assertNotIn("raw internal", response.text)

    def test_transport_and_unexpected_errors_are_sanitized(self):
        for error, expected in ((httpx.ReadTimeout("test-secret"), 504),
                                (httpx.ConnectError("test-secret"), 502),
                                (RuntimeError("test-secret"), 502)):
            with self.subTest(error=type(error).__name__):
                self.gemini.models.generate_content.side_effect = error
                response = self.client.post("/chat", json={"message": "Hello"})
                self.assertEqual(response.status_code, expected)
                self.assertNotIn("test-secret", response.text)

    def test_client_initialization_failure_is_sanitized(self):
        self.factory.side_effect = ValueError("test-secret")
        response = self.client.post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("test-secret", response.text)

    def test_empty_or_blocked_model_response(self):
        for text in (None, "", "   "):
            self.gemini.models.generate_content.return_value = SimpleNamespace(text=text)
            response = self.client.post("/chat", json={"message": "Hello"})
            self.assertEqual(response.status_code, 502)
            self.assertEqual(response.json(), {"detail": "Gemini returned no text response."})

    def test_health_and_docs_without_configuration(self):
        self.settings = Settings()
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service": "ai-service"})
        self.assertEqual(self.client.get("/docs").status_code, 200)
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/chat", schema["paths"])
        self.factory.assert_not_called()


class ConfigurationTests(unittest.TestCase):
    def test_dotenv_loading_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("GEMINI_API_KEY=file-test-secret\nGEMINI_MODEL=file-model\n")
            with patch("app.core.config.ENV_FILE", env_file), patch.dict(os.environ, {}, clear=True):
                get_settings.cache_clear()
                self.addCleanup(get_settings.cache_clear)
                settings = get_settings()
                self.assertEqual(settings.gemini_api_key, "file-test-secret")
                self.assertEqual(settings.gemini_model, "file-model")
                self.assertNotIn("file-test-secret", repr(settings))
                os.environ["GEMINI_MODEL"] = " environment-model "
                get_settings.cache_clear()
                self.assertEqual(get_settings().gemini_model, "environment-model")


if __name__ == "__main__":
    unittest.main()
