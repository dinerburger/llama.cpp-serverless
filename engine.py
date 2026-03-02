import logging
import os
from typing import Any, AsyncGenerator, Dict

import aiohttp

from utils import DummyRequest, JobInput, create_error_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaCppEngine:
    """Engine that wraps llama-server HTTP client for OpenAI-compatible routes."""
    
    def __init__(self):
        self.host = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
        self.port = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
        self.base_url = f"http://{self.host}:{self.port}"
        self.api_key = os.getenv("LLAMA_API_KEY", "").strip()
        
        self._session: aiohttp.ClientSession = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def generate(self, job_input: JobInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Route to appropriate handler based on openai_route."""
        try:
            if job_input.openai_route == "/v1/models":
                result = await self._handle_models()
                yield result
            elif job_input.openai_route == "/v1/chat/completions":
                async for response in self._handle_chat_completion(job_input):
                    yield response
            elif job_input.openai_route == "/v1/completions":
                async for response in self._handle_completion(job_input):
                    yield response
            else:
                yield create_error_response(f"Invalid route: {job_input.openai_route}").model_dump()
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            yield create_error_response(str(e)).model_dump()
    
    async def _handle_models(self) -> Dict[str, Any]:
        """Handle /v1/models route."""
        session = await self._get_session()
        url = f"{self.base_url}/v1/models"
        
        try:
            async with session.get(url, timeout=30) as response:
                if response.ok:
                    return await response.json()
                else:
                    return create_error_response(
                        f"Failed to fetch models: {response.status}"
                    ).model_dump()
        except Exception as e:
            return create_error_response(f"Failed to fetch models: {e}").model_dump()
    
    async def _handle_chat_completion(self, job_input: JobInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle /v1/chat/completions route."""
        async for response in self._handle_completion_request(job_input, "/v1/chat/completions"):
            yield response
    
    async def _handle_completion(self, job_input: JobInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle /v1/completions route."""
        async for response in self._handle_completion_request(job_input, "/v1/completions"):
            yield response
    
    async def _handle_completion_request(self, job_input: JobInput, endpoint: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle completion requests with streaming support."""
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        stream = job_input.stream
        
        logger.info(f"Making request to {url} with payload: {job_input.openai_input}")
        
        try:
            async with session.post(
                url,
                json=job_input.openai_input,
                timeout=600,
                headers={"Accept": "text/event-stream"} if stream else {}
            ) as response:
                logger.info(f"Response status: {response.status}")
                
                if not response.ok:
                    error_text = await response.text()
                    logger.error(f"Request failed: {error_text}")
                    yield create_error_response(
                        f"Request failed: {response.status} - {error_text}"
                    ).model_dump()
                    return
                
                if stream:
                    # Streaming response - buffer chunks and split by newlines
                    buffer = ""
                    async for chunk in response.content.iter_any():
                        buffer += chunk.decode('utf-8')
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line_str = line.strip()
                            logger.debug(f"Received SSE line: {line_str}")
                            if line_str:
                                yield line_str
                    # Handle any remaining data in buffer
                    if buffer.strip():
                        line_str = buffer.strip()
                        logger.debug(f"Received SSE line (final): {line_str}")
                        yield line_str
                else:
                    # Non-streaming response - return JSON directly
                    result = await response.json()
                    logger.info(f"Non-streaming response: {result}")
                    yield result
                    
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {e}")
            yield create_error_response(f"HTTP client error: {e}").model_dump()
        except Exception as e:
            logger.error(f"Request failed: {e}")
            yield create_error_response(f"Request failed: {e}").model_dump()
