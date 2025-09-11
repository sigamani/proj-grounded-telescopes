# Kubernetes Pod Security Policy
# Enforces security standards for Ray cluster deployment

package kubernetes.security

import future.keywords.if
import future.keywords.in

# DENY: Privileged pods
deny[msg] if {
    some container in input.spec.containers
    container.securityContext.privileged == true
    msg := sprintf("Container '%s' must not run in privileged mode", [container.name])
}

# DENY: Host network access
deny[msg] if {
    input.spec.hostNetwork == true
    msg := "Pod must not use host network"
}

# DENY: Host PID namespace
deny[msg] if {
    input.spec.hostPID == true
    msg := "Pod must not use host PID namespace"
}

# DENY: Host IPC namespace  
deny[msg] if {
    input.spec.hostIPC == true
    msg := "Pod must not use host IPC namespace"
}

# DENY: Root filesystem access
deny[msg] if {
    some container in input.spec.containers
    not container.securityContext.readOnlyRootFilesystem
    not allows_writable_filesystem(container)
    msg := sprintf("Container '%s' should use read-only root filesystem", [container.name])
}

allows_writable_filesystem(container) if {
    # Ray containers need writable filesystem
    container.name == "ray-head"
}

allows_writable_filesystem(container) if {
    container.name == "ray-worker"
}

# DENY: Running as root
deny[msg] if {
    some container in input.spec.containers
    container.securityContext.runAsUser == 0
    msg := sprintf("Container '%s' must not run as root (UID 0)", [container.name])
}

# DENY: Privilege escalation
deny[msg] if {
    some container in input.spec.containers
    container.securityContext.allowPrivilegeEscalation == true
    msg := sprintf("Container '%s' must not allow privilege escalation", [container.name])
}

# DENY: Dangerous capabilities
dangerous_caps := {
    "SYS_ADMIN",
    "NET_ADMIN",
    "SYS_PTRACE", 
    "DAC_OVERRIDE",
    "SETUID",
    "SETGID"
}

deny[msg] if {
    some container in input.spec.containers
    some cap in container.securityContext.capabilities.add
    cap in dangerous_caps
    msg := sprintf("Container '%s' cannot add dangerous capability '%s'", [container.name, cap])
}

# REQUIRE: Drop ALL capabilities by default
warn[msg] if {
    some container in input.spec.containers
    not "ALL" in container.securityContext.capabilities.drop
    msg := sprintf("Container '%s' should drop ALL capabilities by default", [container.name])
}

# DENY: Host path volumes in sensitive locations
sensitive_paths := {
    "/",
    "/boot", 
    "/dev",
    "/etc",
    "/lib",
    "/proc", 
    "/sys",
    "/usr"
}

deny[msg] if {
    some volume in input.spec.volumes
    volume.hostPath
    volume.hostPath.path in sensitive_paths
    msg := sprintf("Volume cannot mount sensitive host path '%s'", [volume.hostPath.path])
}

# DENY: Pods without resource limits
deny[msg] if {
    some container in input.spec.containers
    not container.resources.limits.memory
    msg := sprintf("Container '%s' must specify memory limits", [container.name])
}

deny[msg] if {
    some container in input.spec.containers  
    not container.resources.limits.cpu
    msg := sprintf("Container '%s' must specify CPU limits", [container.name])
}

# DENY: Excessive resource requests
deny[msg] if {
    some container in input.spec.containers
    to_number(trim_suffix(container.resources.limits.memory, "Gi")) > 64
    msg := sprintf("Container '%s' memory limit exceeds 64Gi", [container.name])
}

deny[msg] if {
    some container in input.spec.containers
    to_number(container.resources.limits.cpu) > 16
    msg := sprintf("Container '%s' CPU limit exceeds 16 cores", [container.name])
}

# REQUIRE: Security context constraints
deny[msg] if {
    some container in input.spec.containers
    not container.securityContext
    msg := sprintf("Container '%s' must specify securityContext", [container.name])
}

# DENY: Empty security context
deny[msg] if {
    some container in input.spec.containers
    container.securityContext == {}
    msg := sprintf("Container '%s' securityContext cannot be empty", [container.name])
}

# REQUIRE: Pod security context
warn[msg] if {
    not input.spec.securityContext
    msg := "Pod should specify securityContext for enhanced security"
}

# GPU Security
warn[msg] if {
    some container in input.spec.containers
    container.resources.limits["nvidia.com/gpu"]
    not container.securityContext.capabilities
    msg := sprintf("GPU container '%s' should explicitly manage capabilities", [container.name])
}