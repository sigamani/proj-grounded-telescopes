# OPA Security Policies

This directory contains Open Policy Agent (OPA) policies for governance, risk management, and compliance (GRC) of the Ray + vLLM inference infrastructure.

## 🛡️ Policy Categories

### Docker Security (`docker/security.rego`)
- **Container Privileges**: Prevents privileged containers and dangerous capabilities
- **User Security**: Enforces non-root execution 
- **Network Security**: Restricts host network access and dangerous network modes
- **Volume Security**: Controls host path mounts and dangerous filesystem access
- **Resource Limits**: Enforces memory and CPU constraints

### Resource Management (`docker/resource_limits.rego`)  
- **Memory Limits**: Prevents excessive memory allocation
- **CPU Constraints**: Enforces CPU limits in production
- **GPU Resources**: Validates GPU resource allocation and monitoring
- **Ulimit Controls**: Manages process and file descriptor limits
- **Restart Policies**: Ensures appropriate restart behavior

### Kubernetes Security (`kubernetes/pod_security.rego`)
- **Pod Security Standards**: Enforces Kubernetes Pod Security Standards
- **Security Contexts**: Mandates proper security contexts
- **Capability Management**: Controls Linux capabilities
- **Resource Quotas**: Enforces resource limits and requests
- **Volume Security**: Validates volume mounts and host path access

### Compliance (`security/compliance.rego`)
- **SOC2 Type II**: Audit logging and access controls
- **GDPR**: Data protection and encryption requirements  
- **PCI DSS**: Payment data security (when applicable)
- **ISO 27001**: Information security management
- **HIPAA**: Healthcare data protection (when applicable)

## 🚀 Usage

### Local Validation
```bash
# Install OPA
curl -L -o opa https://openpolicyagent.org/downloads/v0.58.0/opa_linux_amd64_static
chmod +x ./opa && sudo mv opa /usr/local/bin

# Test policies
opa fmt --diff policies/
opa test policies/

# Validate configurations
opa eval -d policies/ -i compose.dev.yaml "data.docker.security.deny[_]"
```

### CI/CD Integration
The policies are automatically validated in GitHub Actions:

1. **Format Check**: `opa fmt --diff policies/`
2. **Unit Tests**: `opa test policies/`  
3. **Configuration Validation**: Against both dev and prod compose files
4. **Policy Reports**: Generated as artifacts for audit trails

## 📋 Policy Rules

### Security Enforcement
- ❌ **DENY**: Root containers, privileged mode, host networking
- ❌ **DENY**: Dangerous capabilities (SYS_ADMIN, NET_ADMIN, etc.)
- ❌ **DENY**: Excessive resource allocation (>64GB RAM, >16 CPU)
- ❌ **DENY**: Host path mounts to sensitive directories
- ⚠️ **WARN**: Missing security contexts, high resource usage

### Compliance Checks
- ✅ **REQUIRE**: Audit logging for SOC2 compliance
- ✅ **REQUIRE**: Encryption for GDPR data protection
- ✅ **REQUIRE**: Access controls and authentication
- ✅ **REQUIRE**: Vulnerability scanning for production
- ✅ **REQUIRE**: Specific version tags (no :latest in prod)

### Resource Governance  
- ✅ **ENFORCE**: Memory limits on all containers
- ✅ **ENFORCE**: CPU limits in production environments
- ✅ **ENFORCE**: Proper restart policies
- ✅ **ENFORCE**: GPU resource monitoring

## 🔧 Configuration

### Environment-Specific Policies
```bash
# Development environment (lenient)
opa eval -d policies/ -i compose.dev.yaml "data.docker.security.warn[_]"

# Production environment (strict)
opa eval -d policies/ -i compose.prod.yaml "data.docker.security.deny[_]"
```

### Custom Policy Extensions
Add new policies by creating `.rego` files in appropriate directories:

```rego
package custom.policy

import future.keywords.if

deny[msg] if {
    # Your custom rule
    input.something == "forbidden"
    msg := "Custom policy violation"
}
```

## 📊 Monitoring & Reporting

### Policy Violations
All policy violations are:
- ❌ **Blocked** in CI/CD pipeline
- 📝 **Logged** in OPA policy reports  
- 📋 **Tracked** as GitHub Actions artifacts
- 🔍 **Auditable** for compliance reviews

### Audit Trail
- Policy evaluation results stored as artifacts
- Timestamped policy reports for compliance audits
- Integration with GitHub security tab for vulnerability tracking

## 🎯 Benefits

### Security Assurance
- **Defense in Depth**: Multiple policy layers prevent security issues
- **Automated Enforcement**: No manual security reviews required
- **Consistent Standards**: Same policies across all environments

### Compliance Automation  
- **Regulatory Compliance**: Automated SOC2, GDPR, PCI DSS checks
- **Audit Readiness**: Complete policy audit trails
- **Risk Reduction**: Proactive policy enforcement

### Operational Excellence
- **Shift-Left Security**: Catch issues in CI/CD before deployment
- **Self-Documenting**: Policies serve as security documentation
- **Scalable Governance**: Consistent policies across all infrastructure

## 🔗 References

- [Open Policy Agent Documentation](https://www.openpolicyagent.org/docs/)
- [OPA GitHub Actions Integration](https://www.openpolicyagent.org/docs/cicd/)
- [Kubernetes Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)