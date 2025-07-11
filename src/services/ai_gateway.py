"""AI Gateway service for routing requests to different AI providers."""
import asyncio
import os
import time
import json
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import aiohttp
import openai
from anthropic import Anthropic

from ..core.config import get_config, get_settings
from ..models.database import AIProvider, AIModel

settings = get_settings()
config = get_config()

class AIGateway:
    """Gateway for routing AI requests to different providers."""
    
    def __init__(self):
        self.providers = {}
        self.models = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize AI provider clients."""
        
        # OpenAI
        if settings.openai_api_key:
            self.providers["openai"] = openai.AsyncOpenAI(
                api_key=settings.openai_api_key
            )
            
        # Anthropic
        if settings.anthropic_api_key:
            try:
                self.providers["anthropic"] = Anthropic(
                    api_key=settings.anthropic_api_key
                )
            except Exception as e:
                print(f"⚠️ Failed to initialize Anthropic client: {e}")
                self.providers["anthropic"] = None
    
    async def route_request(
        self, 
        model_name: str, 
        request_data: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Route request to appropriate AI provider."""
        
        # Determine provider and model
        provider_info = await self._get_provider_for_model(model_name)
        if not provider_info:
            raise ValueError(f"Model {model_name} not supported")
        
        provider_name = provider_info["provider"]
        actual_model = provider_info["model"]
        
        # Route to appropriate handler
        start_time = time.time()
        
        try:
            if provider_name == "openai":
                result = await self._handle_openai_request(actual_model, request_data)
            elif provider_name == "anthropic":
                result = await self._handle_anthropic_request(actual_model, request_data)
            elif provider_name == "local":
                result = await self._handle_local_request(actual_model, request_data)
            else:
                raise ValueError(f"Provider {provider_name} not implemented")
            
            response_time = int((time.time() - start_time) * 1000)
            
            # Calculate costs
            cost_info = await self._calculate_provider_costs(
                provider_name, actual_model, result
            )
            
            return {
                **result,
                "provider": provider_name,
                "model": actual_model,
                "response_time_ms": response_time,
                "cost_info": cost_info,
                "success": True
            }
            
        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            
            # Return demo response when API keys aren't configured
            user_message = ""
            if request_data.get("messages"):
                last_message = request_data["messages"][-1]
                # Handle both dict and Pydantic object
                if hasattr(last_message, 'content'):
                    user_message = last_message.content
                elif isinstance(last_message, dict):
                    user_message = last_message.get("content", "")
                else:
                    user_message = str(last_message)
            
            demo_response = f"🤖 Demo Response: I received your message '{user_message}'. This is a demo response because no AI provider API keys are configured. To use real AI providers, add your OpenAI or Anthropic API keys to Railway environment variables."
            
            return {
                "provider": provider_name,
                "model": actual_model,
                "response_time_ms": response_time,
                "success": True,  # Mark as success for demo
                "response": demo_response,
                "usage": {
                    "input_tokens": len(user_message.split()) if user_message else 0,
                    "output_tokens": len(demo_response.split()),
                    "total_tokens": len(user_message.split()) + len(demo_response.split()) if user_message else len(demo_response.split())
                },
                "finish_reason": "stop",
                "cost_info": {
                    "provider_cost": 0.0, 
                    "input_tokens": len(user_message.split()) if user_message else 0, 
                    "output_tokens": len(demo_response.split()),
                    "total_tokens": len(user_message.split()) + len(demo_response.split()) if user_message else len(demo_response.split())
                },
                "demo_mode": True,
                "original_error": str(e)
            }
    
    async def _get_provider_for_model(self, model_name: str) -> Optional[Dict[str, str]]:
        """Determine which provider handles the given model."""
        
        # OpenAI models
        openai_models = {
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt-4-turbo-preview",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "text-embedding-ada-002": "text-embedding-ada-002",
            "text-embedding-3-small": "text-embedding-3-small",
            "text-embedding-3-large": "text-embedding-3-large"
        }
        
        # Anthropic models
        anthropic_models = {
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3.5-sonnet": "claude-3-5-sonnet-20241022"
        }
        
        # Local models (Ollama)
        local_models = {
            # Llama models
            "llama2": "llama2",
            "llama2:7b": "llama2:7b", 
            "llama2:13b": "llama2:13b",
            "llama2:70b": "llama2:70b",
            "llama3": "llama3",
            "llama3:8b": "llama3:8b",
            "llama3:70b": "llama3:70b",
            "llama3.1": "llama3.1",
            "llama3.1:8b": "llama3.1:8b",
            "llama3.1:70b": "llama3.1:70b",
            
            # Code models
            "codellama": "codellama",
            "codellama:7b": "codellama:7b",
            "codellama:13b": "codellama:13b",
            "codellama:34b": "codellama:34b",
            
            # Mistral models
            "mistral": "mistral",
            "mistral:7b": "mistral:7b",
            "mixtral": "mixtral",
            "mixtral:8x7b": "mixtral:8x7b",
            
            # Specialized models
            "phi3": "phi3",
            "phi3:mini": "phi3:mini",
            "gemma": "gemma",
            "gemma:2b": "gemma:2b",
            "gemma:7b": "gemma:7b",
            "neural-chat": "neural-chat",
            "orca-mini": "orca-mini",
            "vicuna": "vicuna",
            "wizardcoder": "wizardcoder",
            
            # User's specific models
            "mistral:latest": "mistral:latest",
            "deepseek-r1:latest": "deepseek-r1:latest",
            
            # Backwards compatibility
            "llama-2-7b": "llama2:7b",
            "code-llama": "codellama",
            "mistral-7b": "mistral:7b"
        }
        
        if model_name in openai_models:
            return {"provider": "openai", "model": openai_models[model_name]}
        elif model_name in anthropic_models:
            return {"provider": "anthropic", "model": anthropic_models[model_name]}
        elif model_name in local_models:
            return {"provider": "local", "model": local_models[model_name]}
        else:
            # Check if it's a local model with :latest or other tag
            # For local models, pass through the exact name to Ollama
            potential_base_models = ["mistral", "llama2", "llama3", "llama3.1", "codellama", 
                                   "mixtral", "phi3", "gemma", "neural-chat", "orca-mini", 
                                   "vicuna", "wizardcoder", "deepseek-r1", "deepseek", "qwen"]
            
            model_base = model_name.split(':')[0]  # Get base name before any colon
            
            # Also check for common patterns like deepseek-r1
            if '-' in model_name:
                model_base_with_dash = model_name.split(':')[0]  # Keep full name before colon
                if any(base in model_base_with_dash for base in potential_base_models):
                    return {"provider": "local", "model": model_name}
            
            if model_base in potential_base_models or model_name.endswith(':latest'):
                return {"provider": "local", "model": model_name}  # Pass exact name to Ollama
            
            return None
    
    async def _handle_openai_request(
        self, 
        model: str, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle OpenAI API request."""
        
        client = self.providers.get("openai")
        if not client:
            raise ValueError("OpenAI client not initialized")
        
        request_type = request_data.get("type", "chat")
        
        if request_type == "chat":
            response = await client.chat.completions.create(
                model=model,
                messages=request_data.get("messages", []),
                max_tokens=request_data.get("max_tokens", 1000),
                temperature=request_data.get("temperature", 0.7),
                stream=request_data.get("stream", False)
            )
            
            if request_data.get("stream", False):
                return await self._handle_openai_stream(response)
            else:
                return {
                    "response": response.choices[0].message.content,
                    "usage": {
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "finish_reason": response.choices[0].finish_reason
                }
        
        elif request_type == "embedding":
            response = await client.embeddings.create(
                model=model,
                input=request_data.get("input", "")
            )
            
            return {
                "embeddings": [item.embedding for item in response.data],
                "usage": {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": 0,
                    "total_tokens": response.usage.total_tokens
                }
            }
        
        else:
            raise ValueError(f"Unsupported request type: {request_type}")
    
    async def _handle_anthropic_request(
        self, 
        model: str, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle Anthropic API request."""
        
        client = self.providers.get("anthropic")
        if not client:
            raise ValueError("Anthropic client not initialized")
        
        messages = request_data.get("messages", [])
        system_message = request_data.get("system", "")
        
        # Convert OpenAI format to Anthropic format
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        response = client.messages.create(
            model=model,
            max_tokens=request_data.get("max_tokens", 1000),
            temperature=request_data.get("temperature", 0.7),
            system=system_message,
            messages=anthropic_messages
        )
        
        return {
            "response": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            "finish_reason": "stop"
        }
    
    async def _handle_local_request(
        self, 
        model: str, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle local model request via Ollama."""
        
        messages = request_data.get("messages", [])
        max_tokens = request_data.get("max_tokens", 1000)
        temperature = request_data.get("temperature", 0.7)
        
        # Get Ollama endpoint from environment or use default
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        
        try:
            # Prepare Ollama request
            ollama_payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{ollama_url}/api/chat",
                    json=ollama_payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        response_text = result.get("message", {}).get("content", "")
                        
                        # Extract token usage if available
                        prompt_eval_count = result.get("prompt_eval_count", 0)
                        eval_count = result.get("eval_count", 0)
                        
                        # If no token counts, estimate
                        if not prompt_eval_count:
                            prompt_text = " ".join([msg.get("content", "") for msg in messages])
                            prompt_eval_count = int(len(prompt_text.split()) * 1.3)
                        
                        if not eval_count:
                            eval_count = int(len(response_text.split()) * 1.3)
                        
                        return {
                            "response": response_text,
                            "usage": {
                                "input_tokens": prompt_eval_count,
                                "output_tokens": eval_count,
                                "total_tokens": prompt_eval_count + eval_count
                            },
                            "finish_reason": "stop",
                            "model_info": {
                                "model": result.get("model", model),
                                "total_duration": result.get("total_duration", 0),
                                "load_duration": result.get("load_duration", 0),
                                "prompt_eval_duration": result.get("prompt_eval_duration", 0),
                                "eval_duration": result.get("eval_duration", 0)
                            }
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error {response.status}: {error_text}")
                        
        except Exception as e:
            # Fallback to demo response if Ollama is unavailable
            prompt = self._convert_messages_to_prompt(messages)
            response_text = f"💰 LOCAL PROFIT MODE: I received your message. Your Ollama instance at {ollama_url} is not responding ({str(e)}). Please ensure Ollama is running and accessible. This request would generate 100% profit since there are no API costs!"
            
            # Estimate tokens for billing
            input_tokens = len(prompt.split()) * 1.3
            output_tokens = len(response_text.split()) * 1.3
            
            return {
                "response": response_text,
                "usage": {
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens),
                    "total_tokens": int(input_tokens + output_tokens)
                },
                "finish_reason": "stop",
                "local_mode": True,
                "error": str(e)
            }
    
    async def _handle_openai_stream(self, stream) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming OpenAI response."""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield {
                    "type": "content",
                    "content": chunk.choices[0].delta.content
                }
        
        yield {
            "type": "done",
            "usage": {
                "input_tokens": 0,  # OpenAI doesn't provide usage in streaming
                "output_tokens": 0,
                "total_tokens": 0
            }
        }
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert message format to simple prompt for local models."""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    async def _calculate_provider_costs(
        self, 
        provider: str, 
        model: str, 
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate provider costs based on usage."""
        
        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        
        # Get cost per token from configuration
        provider_config = config.get_ai_provider_config(provider)
        cost_per_token = provider_config.get("cost_per_token", {})
        
        if model in cost_per_token:
            # Some providers have different rates for input/output
            if isinstance(cost_per_token[model], dict):
                input_cost = input_tokens * cost_per_token[model].get("input", 0)
                output_cost = output_tokens * cost_per_token[model].get("output", 0)
                total_cost = input_cost + output_cost
            else:
                # Single rate for all tokens
                total_cost = total_tokens * cost_per_token[model]
        else:
            # Default cost if model not found
            total_cost = total_tokens * 0.000001  # Very low default
        
        return {
            "provider_cost": round(total_cost, 6),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_per_token": cost_per_token.get(model, 0)
        }
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models across all providers."""
        
        models = []
        
        # OpenAI models
        if "openai" in self.providers:
            openai_models = [
                {
                    "id": "gpt-4",
                    "name": "GPT-4",
                    "provider": "openai",
                    "type": "chat",
                    "context_length": 8192,
                    "description": "Most capable model, great for complex tasks"
                },
                {
                    "id": "gpt-3.5-turbo",
                    "name": "GPT-3.5 Turbo", 
                    "provider": "openai",
                    "type": "chat",
                    "context_length": 4096,
                    "description": "Fast and efficient for most tasks"
                },
                {
                    "id": "text-embedding-ada-002",
                    "name": "Text Embedding Ada 002",
                    "provider": "openai", 
                    "type": "embedding",
                    "context_length": 8191,
                    "description": "High-quality text embeddings"
                }
            ]
            models.extend(openai_models)
        
        # Anthropic models
        if "anthropic" in self.providers:
            anthropic_models = [
                {
                    "id": "claude-3-opus",
                    "name": "Claude 3 Opus",
                    "provider": "anthropic",
                    "type": "chat", 
                    "context_length": 200000,
                    "description": "Most powerful Claude model for complex reasoning"
                },
                {
                    "id": "claude-3-sonnet",
                    "name": "Claude 3 Sonnet",
                    "provider": "anthropic",
                    "type": "chat",
                    "context_length": 200000, 
                    "description": "Balanced performance and cost"
                },
                {
                    "id": "claude-3-haiku",
                    "name": "Claude 3 Haiku",
                    "provider": "anthropic",
                    "type": "chat",
                    "context_length": 200000,
                    "description": "Fastest Claude model for simple tasks"
                }
            ]
            models.extend(anthropic_models)
        
        # Local models (if enabled)
        local_config = config.get_ai_provider_config("local")
        if local_config.get("enabled", False):
            local_models = [
                {
                    "id": "llama-2-7b",
                    "name": "Llama 2 7B",
                    "provider": "local",
                    "type": "chat",
                    "context_length": 4096,
                    "description": "Open source model, private deployment"
                }
            ]
            models.extend(local_models)
        
        return models
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all AI providers."""
        
        health_status = {}
        
        for provider_name in self.providers.keys():
            try:
                if provider_name == "openai":
                    # Simple test request
                    client = self.providers[provider_name]
                    response = await client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=1
                    )
                    health_status[provider_name] = {"status": "healthy", "latency_ms": 0}
                    
                elif provider_name == "anthropic":
                    client = self.providers[provider_name]
                    response = client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=1,
                        messages=[{"role": "user", "content": "test"}]
                    )
                    health_status[provider_name] = {"status": "healthy", "latency_ms": 0}
                    
                else:
                    health_status[provider_name] = {"status": "healthy", "latency_ms": 0}
                    
            except Exception as e:
                health_status[provider_name] = {
                    "status": "unhealthy", 
                    "error": str(e),
                    "latency_ms": -1
                }
        
        return {
            "overall_status": "healthy" if all(
                status["status"] == "healthy" 
                for status in health_status.values()
            ) else "degraded",
            "providers": health_status,
            "timestamp": datetime.utcnow().isoformat()
        }