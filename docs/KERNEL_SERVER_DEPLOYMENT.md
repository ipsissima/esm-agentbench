# Kernel Server Deployment Guide

This document describes the deployment configuration and security features of the ESM-AgentBench kernel server.

## Overview

The kernel server (`kernel_server.py`) runs the verified kernel in an isolated process behind a Unix socket RPC interface. This architecture provides:

1. **Process isolation**: If the kernel crashes, only the server process dies
2. **Security**: Socket permissions, HMAC authentication, and rate limiting
3. **Restart semantics**: Clear failure modes and retry logic for clients

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ESM_KERNEL_SOCKET` | `/tmp/esm_kernel.sock` | Path to Unix domain socket |
| `ESM_KERNEL_SOCKET_PERMS` | `0600` | Socket file permissions (octal) |
| `ESM_KERNEL_AUTH_TOKEN` | (none) | HMAC authentication token |
| `ESM_KERNEL_MAX_REQUESTS` | `100` | Max requests per client per second |
| `VERIFIED_KERNEL_PATH` | (auto-detect) | Path to verified kernel shared library |

### Security Features

#### 1. Socket File Permissions

**Purpose**: Restrict access to the kernel server socket.

**Default**: `0600` (owner read/write only)

**Configuration**:
```bash
# Owner read/write only (most secure)
export ESM_KERNEL_SOCKET_PERMS=0600

# Owner and group read/write
export ESM_KERNEL_SOCKET_PERMS=0660

# World-readable (NOT recommended for production)
export ESM_KERNEL_SOCKET_PERMS=0666
```

**Rationale**:  
Kernel operations have access to sensitive computation. Limiting socket access prevents unauthorized clients from submitting malicious inputs.

---

#### 2. HMAC Authentication

**Purpose**: Authenticate client requests using HMAC-SHA256.

**Configuration**:
```bash
# Generate a secure token (32+ random bytes recommended)
export ESM_KERNEL_AUTH_TOKEN=$(openssl rand -hex 32)

# Start server with authentication
python certificates/kernel_server.py
```

**Client Usage**:
```python
import hashlib
import hmac
import json

auth_token = os.environ["ESM_KERNEL_AUTH_TOKEN"]

# Construct request
request = {
    "method": "compute_certificate",
    "params": {...},
    "id": 1,
}

# Compute HMAC over method + params + id
message = json.dumps(request, sort_keys=True).encode("utf-8")
request_hmac = hmac.new(
    auth_token.encode("utf-8"),
    message,
    hashlib.sha256
).hexdigest()

# Add HMAC to request
request["hmac"] = request_hmac

# Send to server
# ...
```

**Rationale**:  
HMAC authentication ensures only authorized clients can submit requests. The token is never transmitted; only the HMAC digest is sent, preventing token leakage.

---

#### 3. Rate Limiting

**Purpose**: Prevent denial-of-service attacks.

**Default**: 100 requests per client per second

**Configuration**:
```bash
# Lower for production (e.g., 10/s)
export ESM_KERNEL_MAX_REQUESTS=10

# Higher for testing (e.g., 1000/s)
export ESM_KERNEL_MAX_REQUESTS=1000
```

**Behavior**:  
- Tracks requests per client (identified by socket connection)
- Sliding window: counts requests in the last 1 second
- Exceeding limit returns error: `{"error": "Rate limit exceeded"}`

**Rationale**:  
Prevents a single client from monopolizing the kernel server and ensures fair resource allocation.

---

## Deployment Patterns

### 1. Basic Deployment (Development/Testing)

No authentication, default permissions:

```bash
# Start server
python certificates/kernel_server.py

# Use from client
python -c "
from certificates.kernel_client import KernelClient
client = KernelClient('/tmp/esm_kernel.sock')
result = client.ping()
print(result)
"
```

---

### 2. Secure Deployment (Production)

Full security enabled:

```bash
# Generate and store authentication token
export ESM_KERNEL_AUTH_TOKEN=$(openssl rand -hex 32)
echo "$ESM_KERNEL_AUTH_TOKEN" > /etc/esm/kernel_token.txt
chmod 600 /etc/esm/kernel_token.txt

# Configure server
export ESM_KERNEL_SOCKET=/var/run/esm/kernel.sock
export ESM_KERNEL_SOCKET_PERMS=0600
export ESM_KERNEL_MAX_REQUESTS=10

# Start server with systemd
systemctl start esm-kernel-server
```

**Systemd Unit** (`/etc/systemd/system/esm-kernel-server.service`):

```ini
[Unit]
Description=ESM Kernel Server
After=network.target

[Service]
Type=simple
User=esm-kernel
Group=esm-kernel
WorkingDirectory=/opt/esm-agentbench
EnvironmentFile=/etc/esm/kernel.env
ExecStart=/opt/esm-agentbench/.venv/bin/python certificates/kernel_server.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Environment File** (`/etc/esm/kernel.env`):

```bash
ESM_KERNEL_SOCKET=/var/run/esm/kernel.sock
ESM_KERNEL_SOCKET_PERMS=0600
ESM_KERNEL_AUTH_TOKEN=<token-from-file>
ESM_KERNEL_MAX_REQUESTS=10
VERIFIED_KERNEL_PATH=/opt/esm-agentbench/UELAT/kernel_verified.so
```

---

### 3. Supervised Deployment (High Availability)

Use a supervisor to automatically restart the server on failure:

**Supervisord Config** (`/etc/supervisor/conf.d/esm-kernel.conf`):

```ini
[program:esm-kernel-server]
command=/opt/esm-agentbench/.venv/bin/python certificates/kernel_server.py
directory=/opt/esm-agentbench
user=esm-kernel
autostart=true
autorestart=true
startretries=10
stderr_logfile=/var/log/esm/kernel_server.err.log
stdout_logfile=/var/log/esm/kernel_server.out.log
environment=
    ESM_KERNEL_SOCKET="/var/run/esm/kernel.sock",
    ESM_KERNEL_SOCKET_PERMS="0600",
    ESM_KERNEL_AUTH_TOKEN="<token>",
    ESM_KERNEL_MAX_REQUESTS="10"
```

---

## Client Retry Semantics

When the kernel server dies or restarts, clients should implement retry logic:

```python
import time
from certificates.kernel_client import KernelClient, KernelConnectionError

def call_with_retry(client_fn, max_retries=3, backoff=1.0):
    """Call client function with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return client_fn()
        except KernelConnectionError:
            if attempt == max_retries - 1:
                raise
            wait_time = backoff * (2 ** attempt)
            print(f"Connection failed, retrying in {wait_time}s...")
            time.sleep(wait_time)

# Usage
client = KernelClient('/var/run/esm/kernel.sock', auth_token=token)
result = call_with_retry(lambda: client.compute_certificate(...))
```

**Recommended Retry Strategy**:
- **Max retries**: 3-5
- **Backoff**: Exponential (1s, 2s, 4s, ...)
- **Timeout**: 30s per attempt
- **Error handling**: Log failures, alert on repeated failures

---

## Monitoring

### Health Check

```bash
# Ping the server
echo '{"method": "ping", "params": {}, "id": 1}' | \
  socat - UNIX-CONNECT:/var/run/esm/kernel.sock
```

Expected response:
```json
{"result": "pong", "error": null, "id": 1}
```

### Self-Test

```bash
# Run built-in self-test
echo '{"method": "selftest", "params": {}, "id": 2}' | \
  socat - UNIX-CONNECT:/var/run/esm/kernel.sock
```

Expected response:
```json
{
  "result": {
    "ok": true,
    "message": "Self-test passed: res=0.123456, bound=0.234567",
    "verified": true
  },
  "error": null,
  "id": 2
}
```

### Logs

Server logs to stderr by default:

```bash
# View logs with systemd
journalctl -u esm-kernel-server -f

# View logs with supervisord
tail -f /var/log/esm/kernel_server.out.log
```

---

## Failure Modes

### Server Crashes

**Symptom**: Client receives connection refused or socket not found.

**Recovery**:
- **Systemd**: Automatic restart (configured in unit file)
- **Supervisord**: Automatic restart (configured in conf)
- **Manual**: `systemctl restart esm-kernel-server`

**Client behavior**: Retry with exponential backoff.

---

### Socket Permission Denied

**Symptom**: Client receives "Permission denied" on socket connection.

**Diagnosis**:
```bash
ls -l /var/run/esm/kernel.sock
# Expected: srw------- 1 esm-kernel esm-kernel 0 ... kernel.sock
```

**Fix**:
- Check socket permissions: `ESM_KERNEL_SOCKET_PERMS`
- Add client user to `esm-kernel` group (if using 0660 permissions)

---

### Authentication Failure

**Symptom**: Server returns `{"error": "Authentication failed"}`.

**Diagnosis**:
- Verify token matches: `echo $ESM_KERNEL_AUTH_TOKEN`
- Check HMAC computation in client

**Fix**: Ensure client uses correct token and HMAC algorithm.

---

### Rate Limit Exceeded

**Symptom**: Server returns `{"error": "Rate limit exceeded"}`.

**Diagnosis**: Client is sending too many requests.

**Fix**:
- Reduce client request rate
- Increase server limit: `ESM_KERNEL_MAX_REQUESTS`
- Use request batching where possible

---

## Performance Considerations

### Throughput

- **Single-threaded**: Requests are processed sequentially per connection
- **Multi-threaded**: Multiple connections handled in parallel
- **Typical latency**: 1-10ms per request (depending on matrix size)

### Scalability

For high-throughput scenarios, consider:
- **Multiple server instances**: Run one server per CPU core
- **Load balancing**: Distribute clients across server instances
- **Connection pooling**: Reuse connections to reduce overhead

---

## Security Checklist

- [ ] Set socket permissions to `0600` or `0660` (production)
- [ ] Enable HMAC authentication (`ESM_KERNEL_AUTH_TOKEN`)
- [ ] Configure appropriate rate limit (`ESM_KERNEL_MAX_REQUESTS`)
- [ ] Use systemd/supervisord for automatic restart
- [ ] Log to centralized logging system
- [ ] Set up monitoring and alerting
- [ ] Regularly rotate authentication tokens
- [ ] Restrict file system access (run as dedicated user)
- [ ] Use AppArmor/SELinux profiles (optional)

---

## Troubleshooting

### Server won't start

**Check**:
1. Socket path exists and is writable: `ls -ld /var/run/esm/`
2. Verified kernel exists: `ls -l $VERIFIED_KERNEL_PATH`
3. Python environment is activated
4. Dependencies are installed: `pip list | grep numpy`

### Server starts but clients can't connect

**Check**:
1. Socket file exists: `ls -l $ESM_KERNEL_SOCKET`
2. Socket permissions: `stat -c %a $ESM_KERNEL_SOCKET`
3. Client has access: `sudo -u <client-user> ls -l $ESM_KERNEL_SOCKET`

### Server crashes frequently

**Check**:
1. Kernel library is valid: `file $VERIFIED_KERNEL_PATH`
2. System has sufficient memory: `free -h`
3. OCaml runtime is available (for verified kernel)
4. Logs for error patterns: `journalctl -u esm-kernel-server | grep ERROR`

---

## Version History

- **v1.0** (2026-02-01): Initial deployment guide with security features
- Future: Add token rotation mechanism, mutual TLS support
