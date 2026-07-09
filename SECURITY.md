# Security Policy

edge-dit.cpp is currently in v0.1-alpha. The HTTP server and Python development
console are intended for local development and trusted environments unless you
add your own authentication, authorization, network isolation, and request
limits.

## Reporting a Vulnerability

Please report security issues privately before public disclosure. If the public
repository does not yet have private vulnerability reporting enabled, contact
the maintainers directly through the project owner or repository administrator.

When reporting, include:

- affected commit or release;
- build options and backend;
- reproduction steps;
- expected impact;
- any relevant logs, excluding secrets or private model/data paths.

## Scope

Security reports may cover:

- memory safety issues in native code;
- unsafe file or path handling;
- denial-of-service behavior in local servers;
- dependency or build-chain vulnerabilities;
- unsafe defaults that expose local generation services.

Model behavior, generated content quality, and upstream model license issues
are important but are not handled as security vulnerabilities in this policy.
