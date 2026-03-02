import os
import logging
from http import HTTPStatus
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO)


class JobInput:
    """Parse job input for OpenAI-compatible routes."""
    
    def __init__(self, job: Dict[str, Any]):
        job_input = job.get("input", {})
        
        self.openai_route = job_input.get("openai_route", "/v1/chat/completions")
        self.openai_input = job_input.get("openai_input", {})
        # Check stream at both top level and inside openai_input
        self.stream = job_input.get("stream", self.openai_input.get("stream", False))
        self.max_batch_size = job_input.get("max_batch_size")
        self.batch_size_growth_factor = job_input.get("batch_size_growth_factor")
        self.min_batch_size = job_input.get("min_batch_size")


class BatchSize:
    """Dynamic batch size controller for streaming responses."""
    
    def __init__(self, max_batch_size: int, min_batch_size: int, batch_size_growth_factor: float):
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.batch_size_growth_factor = batch_size_growth_factor
        self.is_dynamic = (
            batch_size_growth_factor > 1 
            and min_batch_size >= 1 
            and max_batch_size > min_batch_size
        )
        if self.is_dynamic:
            self.current_batch_size = min_batch_size
        else:
            self.current_batch_size = max_batch_size
    
    def update(self):
        """Update batch size for next batch."""
        if self.is_dynamic:
            self.current_batch_size = min(
                self.current_batch_size * self.batch_size_growth_factor,
                self.max_batch_size
            )


class DummyState:
    """Dummy state for compatibility with vLLM-style serving."""
    
    def __init__(self):
        self.request_metadata = None


class DummyRequest:
    """Dummy request object for compatibility with vLLM-style serving."""
    
    def __init__(self):
        self.headers = {}
        self.state = DummyState()
    
    async def is_disconnected(self):
        return False


class ErrorResponse:
    """OpenAI-compatible error response."""
    
    def __init__(self, error: Dict[str, Any]):
        self.error = error
    
    def model_dump(self) -> Dict[str, Any]:
        return {"error": self.error}


def create_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST
) -> ErrorResponse:
    """Create an OpenAI-compatible error response."""
    return ErrorResponse(error={
        "message": message,
        "type": err_type,
        "code": status_code.value
    })