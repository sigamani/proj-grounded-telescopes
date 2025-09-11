# Docker Resource Limits Policy
# Ensures containers have appropriate resource constraints

package docker.resources

import future.keywords.if
import future.keywords.in

# DENY: Containers without memory limits
deny[msg] if {
    not input.HostConfig.Memory
    msg := "Container must specify memory limits"
}

deny[msg] if {
    input.HostConfig.Memory == 0
    msg := "Container must specify non-zero memory limits"
}

# DENY: Excessive memory allocation
deny[msg] if {
    input.HostConfig.Memory > 32000000000  # 32GB
    msg := "Container memory limit exceeds 32GB maximum"
}

# WARN: Low memory allocation for Ray workloads
warn[msg] if {
    input.HostConfig.Memory < 2000000000  # 2GB
    is_ray_container
    msg := "Ray container should have at least 2GB memory"
}

is_ray_container if {
    some env in input.Config.Env
    contains(env, "RAY_ADDRESS")
}

is_ray_container if {
    some cmd in input.Config.Cmd
    contains(cmd, "ray start")
}

# DENY: Containers without CPU limits in production
deny[msg] if {
    not input.HostConfig.CpuShares
    is_production_env
    msg := "Production containers must specify CPU limits"
}

is_production_env if {
    some env in input.Config.Env
    env == "ENV=production"
}

# DENY: Swap unlimited
deny[msg] if {
    input.HostConfig.MemorySwap == -1
    msg := "Container must not have unlimited swap"
}

# DENY: No restart policy
deny[msg] if {
    not input.HostConfig.RestartPolicy.Name
    msg := "Container must specify restart policy"
}

# WARN: Always restart policy in development
warn[msg] if {
    input.HostConfig.RestartPolicy.Name == "always"
    not is_production_env
    msg := "Development containers should use 'unless-stopped' restart policy"
}

# DENY: Excessive PID limits
deny[msg] if {
    input.HostConfig.PidsLimit > 10000
    msg := "Container PID limit should not exceed 10000"
}

# GPU Resource Validation
deny[msg] if {
    input.HostConfig.DeviceRequests
    not has_gpu_monitoring
    msg := "GPU containers must have monitoring enabled"
}

has_gpu_monitoring if {
    some env in input.Config.Env
    contains(env, "NVIDIA_VISIBLE_DEVICES")
}

# DENY: Ulimit modifications without justification
dangerous_ulimits := {
    "nofile",
    "nproc", 
    "core"
}

warn[msg] if {
    some ulimit in input.HostConfig.Ulimits
    ulimit.Name in dangerous_ulimits
    ulimit.Soft > 65536
    msg := sprintf("High ulimit for '%s' may impact system stability", [ulimit.Name])
}