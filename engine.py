import json
import logging
import os
from typing import Any, AsyncGenerator, Dict

from openai import AsyncOpenAI

from utils import DummyRequest, JobInput, create_error_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaCppEngine:
    """Engine that wraps llama-server using AsyncOpenAI client for OpenAI-compatible routes."""
    
    def __init__(self):
        self.host = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
        self.port = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
        self.base_url = f"http://{self.host}:{self.port}"
        self.api_key = os.getenv("LLAMA_API_KEY", "").strip() or "not-needed"
        
        # Create AsyncOpenAI client for communicating with llama-server
        self._client: AsyncOpenAI | None = None
    
    def _get_client(self) -> AsyncOpenAI:
        """Get or create AsyncOpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        return self._client
    
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
        client = self._get_client()
        
        try:
            models = await client.models.list()
            return {"object": "list", "data": [m.model_dump() for m in models.data]}
        except Exception as e:
            return create_error_response(f"Failed to fetch models: {e}").model_dump()
    
    async def _handle_chat_completion(self, job_input: JobInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle /v1/chat/completions route."""
        async for response in self._handle_chat_completion_request(job_input):
            yield response
    
    async def _handle_completion(self, job_input: JobInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle /v1/completions route."""
        async for response in self._handle_completion_request(job_input):
            yield response
    
    async def _handle_chat_completion_request(self, job_input: JobInput) -> AsyncGenerator[Any, None]:
        """Handle chat completion requests with streaming support."""
        client = self._get_client()
        
        # logger.debug(f"Making chat request with payload: {job_input.openai_input}")
        
        try:
            # Create the chat completion request
            chat_kwargs = {k: v for k, v in job_input.openai_input.items()}
            
            if job_input.stream:
                chat_kwargs.pop("stream")
                
                # Streaming mode - use async iteration
                stream = await client.chat.completions.create(
                    **chat_kwargs,
                    stream=True
                )
                async for chunk in stream:
                    # Convert the chunk to a dict and format as SSE
                    chunk_dict = chunk.model_dump()
                    # logger.debug(f"Received chunk: {chunk_dict}")
                    # Format as SSE: "data: {...}\n\n"
                    sse_line = f"data: {json.dumps(chunk_dict)}\n\n"
                    yield sse_line
                
                # Send [DONE] marker to signal end of stream
                yield "data: [DONE]\n\n"
            else:
                # Non-streaming mode - yield raw dict
                response = await client.chat.completions.create(**chat_kwargs)
                yield response.model_dump()
                    
        except Exception as e:
            logger.error(f"Chat completion request failed: {e}")
            yield create_error_response(f"Chat completion request failed: {e}").model_dump()
    
    async def _handle_completion_request(self, job_input: JobInput) -> AsyncGenerator[Any, None]:
        """Handle legacy completion requests with streaming support."""
        client = self._get_client()
        
        # logger.debug(f"Making completion request with payload: {job_input.openai_input}")
        
        try:
            completion_kwargs = {k: v for k, v in job_input.openai_input.items()}
            
            if job_input.stream:
                completion_kwargs.pop("stream")
                # Streaming mode - use async iteration
                stream = await client.completions.create(**completion_kwargs, stream=True)
                async for chunk in stream:
                    chunk_dict = chunk.model_dump()
                    # logger.debug(f"Received chunk: {chunk_dict}")
                    # Format as SSE: "data: {...}\n\n"
                    sse_line = f"data: {json.dumps(chunk_dict)}\n\n"
                    yield sse_line
                
                # Send [DONE] marker to signal end of stream
                yield "data: [DONE]\n\n"
            else:
                # Non-streaming mode - yield raw dict
                response = await client.completions.create(**completion_kwargs)
                yield response.model_dump()
                    
        except Exception as e:
            logger.error(f"Completion request failed: {e}")
            yield create_error_response(f"Completion request failed: {e}").model_dump()
