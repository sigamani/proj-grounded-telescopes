# Docker Security Policy for Ray + vLLM Infrastructure
# Enforces security best practices for container deployment

package docker.security

import future.keywords.if
import future.keywords.in

# DENY: Running as root user
deny[msg] if {
    input.User == "root"
    msg := "Container must not run as root user"
}

deny[msg] if {
    input.User == "0"
    msg := "Container must not run as UID 0 (root)"
}

# DENY: Privileged containers
deny[msg] if {
    input.HostConfig.Privileged == true
    msg := "Container must not run in privileged mode"
}

# DENY: Host network mode (security risk)
deny[msg] if {
    input.HostConfig.NetworkMode == "host"
    msg := "Container must not use host network mode"
}

# DENY: Host PID namespace
deny[msg] if {
    input.HostConfig.PidMode == "host"
    msg := "Container must not share host PID namespace"
}

# ALLOW: Specific required capabilities for Ray
required_capabilities := {
    "SYS_RESOURCE",  # For Ray resource management
    "IPC_LOCK"       # For GPU memory locking
}

# DENY: Dangerous capabilities
dangerous_capabilities := {
    "SYS_ADMIN",
    "NET_ADMIN", 
    "SYS_PTRACE",
    "DAC_OVERRIDE",
    "SETUID",
    "SETGID"
}

deny[msg] if {
    some cap in input.HostConfig.CapAdd
    cap in dangerous_capabilities
    msg := sprintf("Dangerous capability '%s' is not allowed", [cap])
}

# DENY: All capabilities granted
deny[msg] if {
    some cap in input.HostConfig.CapAdd
    cap == "ALL"
    msg := "Granting ALL capabilities is not allowed"
}

# WARN: Missing security options
warn[msg] if {
    not input.HostConfig.SecurityOpt
    msg := "Container should specify security options (e.g., no-new-privileges)"
}

# DENY: Writable root filesystem (when not required)
deny[msg] if {
    input.HostConfig.ReadonlyRootfs == false
    not allows_writable_root
    msg := "Container should use read-only root filesystem where possible"
}

allows_writable_root if {
    # Ray needs writable filesystem for temp files and logs
    some env in input.Config.Env
    startswith(env, "RAY_")
}

# DENY: Excessive resource limits
deny[msg] if {
    input.HostConfig.Memory > 64000000000  # 64GB limit
    msg := "Container memory limit exceeds maximum allowed (64GB)"
}

deny[msg] if {
    input.HostConfig.CpuShares > 4096  # 4 CPU limit
    msg := "Container CPU limit exceeds maximum allowed"
}

# DENY: Host volume mounts in dangerous paths
dangerous_host_paths := {
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
    some mount in input.HostConfig.Binds
    some dangerous_path in dangerous_host_paths
    startswith(mount, sprintf("%s:", [dangerous_path]))
    msg := sprintf("Mounting host path '%s' is not allowed", [dangerous_path])
}

# ALLOW: Required volume mounts for Ray
allowed_volumes := {
    "/app",
    "/tmp/ray", 
    "/var/log",
    "/home"
}

# DENY: Unrestricted volume mounts
deny[msg] if {
    some mount in input.HostConfig.Binds
    host_path := split(mount, ":")[0]
    not volume_is_allowed(host_path)
    msg := sprintf("Host volume mount '%s' is not in allowlist", [host_path])
}

volume_is_allowed(path) if {
    some allowed in allowed_volumes
    startswith(path, allowed)
}

volume_is_allowed(path) if {
    # Allow relative paths (current directory mounts)
    not startswith(path, "/")
}