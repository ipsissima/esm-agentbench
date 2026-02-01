# Hexagonal Architecture Guide

## Overview

This project follows the **Hexagonal Architecture** (also known as Ports and Adapters architecture) to maintain clean separation of concerns between:
- **Domain logic** (core business rules)
- **Application layer** (orchestration and use cases)
- **Infrastructure** (external systems and frameworks)

## Architecture Layers

### 1. Ports (`/ports`)

**Ports** are interfaces that define contracts for external dependencies. They are protocol-based (using Python's `Protocol` from `typing`) to enable duck typing and easier testing.

Available Ports:
- `EmbedderPort` - Interface for text embedding implementations
- `InferencePort` - Interface for LLM inference implementations
- `KernelPort` - Interface for verified kernel execution
- `SignerPort` - Interface for cryptographic signing
- `TraceStoragePort` - Interface for trace persistence

Example port definition:
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbedderPort(Protocol):
    """Port for embedding implementations."""
    
    def embed(self, text: str) -> np.ndarray:
        """Embed a single piece of text."""
        ...
```

### 2. Adapters (`/adapters`)

**Adapters** are concrete implementations of the port interfaces. They handle communication with external systems (databases, ML models, APIs, etc.).

Available Adapters:
- `SentenceTransformersEmbedder` - Implements `EmbedderPort` using sentence-transformers
- `HuggingFaceInferenceAdapter` - Implements `InferencePort` using HuggingFace models
- `KernelClientAdapter` - Implements `KernelPort` for kernel execution
- `GpgSignerAdapter` - Implements `SignerPort` using GPG
- `FilesystemTraceStorage` - Implements `TraceStoragePort` using filesystem

Example adapter:
```python
from ports.embedder import EmbedderPort

class SentenceTransformersEmbedder(EmbedderPort):
    """Adapter for sentence-transformers models."""
    
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None
    
    def embed(self, text: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Embedder not loaded. Call load() first.")
        return np.asarray(self._model.encode(text))
```

### 3. Core (`/core`)

**Core** contains the domain logic - pure business rules that don't depend on external systems. The core depends only on port interfaces, never on concrete adapters.

Example:
```python
from ports.kernel import KernelPort

def compute_certificate_from_trace(
    trace: Mapping[str, Any],
    *,
    kernel: Optional[KernelPort] = None,  # Depends on port, not adapter
    ...
) -> Dict[str, Any]:
    # Pure domain logic
    embeddings = extract_embeddings(trace)
    certificate = compute_certificate(embeddings, ...)
    
    # Optional kernel verification via port
    if kernel is not None:
        kernel_output = kernel.run_kernel_and_verify(...)
        certificate["kernel_output"] = kernel_output
    
    return certificate
```

### 4. Services (`/services`)

**Services** provide application-level orchestration and dependency management. They coordinate between domain logic and adapters.

#### AdapterFactory

Factory for creating adapter instances with consistent configuration:

```python
from services import AdapterFactory

# Create adapters using factory
embedder = AdapterFactory.create_embedder("all-MiniLM-L6-v2")
inference = AdapterFactory.create_inference_adapter("gpt2")
kernel = AdapterFactory.create_kernel_adapter()
```

#### CertificateService

High-level service for certificate generation:

```python
from services import CertificateService
from services import create_kernel_adapter, create_signer

# Configure service with adapters
service = CertificateService(
    kernel=create_kernel_adapter(),
    signer=create_signer("my-id", "my-gpg-key"),
)

# Generate and sign certificate
certificate = service.generate_and_sign(
    trace=trace_data,
    verify_with_kernel=True,
)
```

### 5. Applications (`/apps`)

**Applications** are entry points that wire everything together. They should use the service layer rather than directly instantiating adapters.

## Usage Examples

### Basic Certificate Generation

```python
from services import CertificateService

# Simple usage without verification
service = CertificateService()
certificate = service.generate_certificate(trace)
```

### With Kernel Verification

```python
from services import CertificateService, create_kernel_adapter

# Use kernel for verified computation
service = CertificateService(
    kernel=create_kernel_adapter()
)

certificate = service.generate_certificate(
    trace,
    verify_with_kernel=True,
)
```

### With Signing and Storage

```python
from pathlib import Path
from services import CertificateService, create_kernel_adapter, create_signer, create_storage

# Full-featured service
service = CertificateService(
    kernel=create_kernel_adapter(),
    signer=create_signer("certifier-id", "gpg-key-id"),
    storage=create_storage(Path("/var/traces")),
)

# Generate, sign, and persist
certificate = service.generate_and_sign(trace)
```

## Benefits of Hexagonal Architecture

1. **Testability**: Easy to mock ports for testing core logic
2. **Flexibility**: Swap implementations without changing core logic
3. **Independence**: Core logic doesn't depend on frameworks or external systems
4. **Clarity**: Clear separation of concerns and dependencies
5. **Maintainability**: Changes to infrastructure don't affect domain logic

## Testing Strategy

### Unit Testing Core Logic

Mock the ports to test domain logic in isolation:

```python
from unittest.mock import Mock
from core.certificate import compute_certificate_from_trace

def test_certificate_generation():
    mock_kernel = Mock()
    mock_kernel.run_kernel_and_verify.return_value = {"status": "verified"}
    
    certificate = compute_certificate_from_trace(
        trace,
        kernel=mock_kernel,
    )
    
    assert mock_kernel.run_kernel_and_verify.called
```

### Integration Testing Adapters

Test adapters with real dependencies:

```python
from adapters import SentenceTransformersEmbedder

def test_embedder_integration():
    embedder = SentenceTransformersEmbedder("all-MiniLM-L6-v2")
    embedder.load()
    
    embedding = embedder.embed("test text")
    
    assert embedding.shape[0] == 384  # Model dimension
```

## Adding New Adapters

To add a new adapter implementation:

1. Create adapter class in `/adapters`:
   ```python
   from ports.embedder import EmbedderPort
   
   class MyCustomEmbedder(EmbedderPort):
       def embed(self, text: str) -> np.ndarray:
           # Implementation
           ...
   ```

2. Export from `/adapters/__init__.py`:
   ```python
   from .my_custom_embedder import MyCustomEmbedder
   
   __all__ = [..., "MyCustomEmbedder"]
   ```

3. Add factory method in `/services/adapter_factory.py`:
   ```python
   @staticmethod
   def create_embedder(model_name: str, *, adapter_type: str = "..."):
       if adapter_type == "custom":
           from adapters import MyCustomEmbedder
           return MyCustomEmbedder(model_name)
       # ... existing code
   ```

4. Write tests in `/tests`:
   ```python
   def test_custom_embedder():
       from adapters import MyCustomEmbedder
       embedder = MyCustomEmbedder("model-name")
       # Test implementation
   ```

## Migration Guide

For existing code that directly instantiates adapters:

**Before:**
```python
from adapters.kernel_client import KernelClientAdapter
kernel = KernelClientAdapter()
```

**After:**
```python
from services import create_kernel_adapter
kernel = create_kernel_adapter()
```

This makes it easier to:
- Change implementations via configuration
- Mock dependencies in tests
- Apply consistent initialization logic

## Architecture Diagram

```
┌─────────────────────────────────────────┐
│         Applications (/apps)            │
│  Entry points, CLI, web servers, etc.  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         Services (/services)            │
│  Orchestration, dependency injection    │
│  AdapterFactory, CertificateService     │
└────────────┬────────────────────────────┘
             │
             ▼
┌────────────────────────────────┐
│        Core (/core)            │
│    Domain logic & rules        │
│  Depends only on Ports (↓)    │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│        Ports (/ports)          │
│  Interfaces (Protocols)        │
│  EmbedderPort, KernelPort...   │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│      Adapters (/adapters)      │
│  Concrete implementations      │
│  SentenceTransformers, GPG...  │
└────────────────────────────────┘
             │
             ▼
┌────────────────────────────────┐
│   External Systems             │
│  ML models, databases, APIs    │
└────────────────────────────────┘
```

## References

- [Hexagonal Architecture (Alistair Cockburn)](https://alistair.cockburn.us/hexagonal-architecture/)
- [Ports and Adapters Pattern](https://herbertograca.com/2017/09/14/ports-adapters-architecture/)
- [Clean Architecture (Robert C. Martin)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
