# Security Compliance Policy
# Enforces regulatory and security compliance requirements

package security.compliance

import future.keywords.if
import future.keywords.in

# SOC2 Compliance Requirements
deny[msg] if {
    not has_audit_logging
    msg := "SOC2: System must have audit logging enabled"
}

has_audit_logging if {
    some env in input.Config.Env
    startswith(env, "AUDIT_LOG")
}

has_audit_logging if {
    some volume in input.HostConfig.Binds
    contains(volume, "/var/log")
}

# GDPR Data Protection
deny[msg] if {
    processes_personal_data
    not has_data_encryption
    msg := "GDPR: Systems processing personal data must have encryption"
}

processes_personal_data if {
    some env in input.Config.Env
    contains(env, "PERSONAL_DATA=true")
}

has_data_encryption if {
    some env in input.Config.Env
    startswith(env, "ENCRYPTION_KEY")
}

# PCI DSS Requirements (if applicable)
deny[msg] if {
    handles_payment_data
    not has_network_segmentation
    msg := "PCI DSS: Payment systems must have network segmentation"
}

handles_payment_data if {
    some env in input.Config.Env
    contains(env, "PAYMENT_PROCESSING=true")
}

has_network_segmentation if {
    input.HostConfig.NetworkMode != "host"
    input.HostConfig.NetworkMode != "bridge"
}

# ISO 27001 Information Security
deny[msg] if {
    not has_security_monitoring
    msg := "ISO 27001: Systems must have security monitoring"
}

has_security_monitoring if {
    some port in input.NetworkSettings.Ports
    port == "9090/tcp"  # Prometheus monitoring
}

# Data Retention Policies
warn[msg] if {
    not has_data_retention_policy
    msg := "Compliance: Data retention policy should be defined"
}

has_data_retention_policy if {
    some env in input.Config.Env
    startswith(env, "DATA_RETENTION_DAYS")
}

# Access Control Requirements
deny[msg] if {
    not has_access_control
    msg := "Compliance: Access control mechanisms must be implemented"
}

has_access_control if {
    some env in input.Config.Env
    contains(env, "AUTH_ENABLED=true")
}

has_access_control if {
    # Ray dashboard should have authentication in production
    some env in input.Config.Env
    contains(env, "RAY_DASHBOARD_AUTH")
}

# Incident Response Capabilities
warn[msg] if {
    not has_incident_response
    msg := "Compliance: Incident response capabilities should be enabled"
}

has_incident_response if {
    some env in input.Config.Env
    contains(env, "INCIDENT_WEBHOOK")
}

# Backup and Recovery
warn[msg] if {
    not has_backup_strategy
    msg := "Compliance: Backup and recovery strategy should be implemented"
}

has_backup_strategy if {
    some volume in input.HostConfig.Binds
    contains(volume, "backup")
}

# Vulnerability Management
deny[msg] if {
    not has_vulnerability_scanning
    is_production_container
    msg := "Compliance: Production containers must be vulnerability scanned"
}

is_production_container if {
    some env in input.Config.Env
    env == "ENV=production"
}

has_vulnerability_scanning if {
    some label in input.Config.Labels
    label == "vulnerability_scan=passed"
}

# Container Image Security
deny[msg] if {
    uses_latest_tag
    is_production_container
    msg := "Compliance: Production containers must use specific version tags"
}

uses_latest_tag if {
    contains(input.Config.Image, ":latest")
}

uses_latest_tag if {
    not contains(input.Config.Image, ":")
}

# Secrets Management
deny[msg] if {
    has_hardcoded_secrets
    msg := "Compliance: Secrets must not be hardcoded in containers"
}

has_hardcoded_secrets if {
    some env in input.Config.Env
    contains(env, "PASSWORD=")
}

has_hardcoded_secrets if {
    some env in input.Config.Env
    contains(env, "SECRET=")
}

has_hardcoded_secrets if {
    some env in input.Config.Env
    contains(env, "KEY=")
    not contains(env, "KEY_FILE=")  # Key files are acceptable
}