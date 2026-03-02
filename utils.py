import logging
from http import HTTPStatus
from typing import Any, Dict

logging.basicConfig(level=logging.INFO)


class JobInput:
    """Parse job input for OpenAI-compatible routes."""
    
    def __init__(self, job: Dict[str, Any]):
        job_input = job.get("input", {})
        
        self.openai_route = job_input.get("openai_route", "/v1/chat/completions")
        self.openai_input = job_input.get("openai_input", {})
        # Check stream at both top level and inside openai_input
        self.stream = job_input.get("stream", self.openai_input.get("stream", False))


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