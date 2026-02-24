# CIS Controls v8

**Source:** https://www.cisecurity.org/controls/v8
**Released:** May 2021

The CIS Controls are a prioritized set of actions that collectively form a defense-in-depth set
of best practices that mitigate the most common attacks against systems and networks.

## Implementation Groups

- **IG1 (Basic Cyber Hygiene):** Essential for all organizations. ~56 safeguards.
- **IG2 (Foundational):** For organizations with more IT complexity. Adds ~74 safeguards.
- **IG3 (Organizational):** For organizations with dedicated security teams facing sophisticated threats. Adds ~23 safeguards.

---

## CIS-1: Inventory and Control of Enterprise Assets
**Implementation Group:** IG1

Actively manage all enterprise assets (end-user devices, network devices, IoT devices, servers)
connected to the infrastructure — physically, virtually, remotely, and in cloud environments —
to accurately know the totality of assets that need to be monitored and protected.

**Key Safeguards:**
- 1.1: Establish and Maintain Detailed Enterprise Asset Inventory
- 1.2: Address Unauthorized Assets
- 1.3: Utilize an Active Discovery Tool
- 1.4: Use Dynamic Host Configuration Protocol (DHCP) Logging
- 1.5: Use a Passive Asset Discovery Tool

---

## CIS-2: Inventory and Control of Software Assets
**Implementation Group:** IG1

Actively manage all software on the network so that only authorized software is installed and
can execute, and unauthorized and unmanaged software is found and prevented from installation or execution.

**Key Safeguards:**
- 2.1: Establish and Maintain a Software Inventory
- 2.2: Ensure Authorized Software is Currently Supported
- 2.3: Address Unauthorized Software
- 2.4: Utilize Automated Software Inventory Tools
- 2.5: Allowlist Authorized Software
- 2.6: Allowlist Authorized Libraries
- 2.7: Allowlist Authorized Scripts

---

## CIS-3: Data Protection
**Implementation Group:** IG1

Develop processes and technical controls to identify, classify, securely handle, retain, and dispose of data.

**Key Safeguards:**
- 3.1: Establish and Maintain a Data Management Process
- 3.2: Establish and Maintain a Data Inventory
- 3.3: Configure Data Access Control Lists
- 3.4: Enforce Data Retention
- 3.5: Securely Dispose of Data
- 3.6: Encrypt Data on End-User Devices
- 3.7: Establish and Maintain a Data Classification Scheme
- 3.8: Document Data Flows
- 3.9: Encrypt Data on Removable Media
- 3.10: Encrypt Sensitive Data in Transit
- 3.11: Encrypt Sensitive Data at Rest
- 3.12: Segment Data Processing and Storage Based on Sensitivity
- 3.13: Deploy a Data Loss Prevention Solution
- 3.14: Log Sensitive Data Access

---

## CIS-4: Secure Configuration of Enterprise Assets and Software
**Implementation Group:** IG1

Establish and maintain the secure configuration of enterprise assets and software.

**Key Safeguards:**
- 4.1: Establish and Maintain a Secure Configuration Process
- 4.2: Establish and Maintain a Secure Configuration Process for Network Infrastructure
- 4.3: Configure Automatic Session Locking on Enterprise Assets
- 4.4: Implement and Manage a Firewall on Servers
- 4.5: Implement and Manage a Firewall on End-User Devices
- 4.6: Securely Manage Enterprise Assets and Software
- 4.7: Manage Default Accounts on Enterprise Assets and Software
- 4.8: Uninstall or Disable Unnecessary Services on Enterprise Assets and Software
- 4.9: Configure Trusted DNS Servers on Enterprise Assets
- 4.10: Enforce Automatic Device Lockout on Portable End-User Devices
- 4.11: Enforce Remote Wipe Capability on Portable End-User Devices
- 4.12: Separate Enterprise Workspaces on Mobile End-User Devices

---

## CIS-5: Account Management
**Implementation Group:** IG1

Use processes and tools to assign and manage authorization to credentials for user accounts,
including administrator accounts, as well as service accounts.

**Key Safeguards:**
- 5.1: Establish and Maintain an Inventory of Accounts
- 5.2: Use Unique Passwords
- 5.3: Disable Dormant Accounts
- 5.4: Restrict Administrator Privileges to Dedicated Administrator Accounts
- 5.5: Establish and Maintain an Inventory of Service Accounts
- 5.6: Centralize Account Management

---

## CIS-6: Access Control Management
**Implementation Group:** IG1

Use processes and tools to create, assign, manage, and revoke access credentials and privileges
for user, administrator, and service/application accounts.

**Key Safeguards:**
- 6.1: Establish an Access Granting Process
- 6.2: Establish an Access Revoking Process
- 6.3: Require MFA for Externally-Exposed Applications
- 6.4: Require MFA for Remote Network Access
- 6.5: Require MFA for Administrative Access
- 6.6: Establish and Maintain an Inventory of Authentication and Authorization Systems
- 6.7: Centralize Access Control
- 6.8: Define and Maintain Role-Based Access Control

---

## CIS-7: Continuous Vulnerability Management
**Implementation Group:** IG1

Develop a plan to continuously assess and track vulnerabilities on all enterprise assets within
the enterprise's infrastructure, to remediate and minimize the window of opportunity for attackers.

**Key Safeguards:**
- 7.1: Establish and Maintain a Vulnerability Management Process
- 7.2: Establish and Maintain a Remediation Process
- 7.3: Perform Automated Operating System Patch Management
- 7.4: Perform Automated Application Patch Management
- 7.5: Perform Automated Vulnerability Scans of Internal Enterprise Assets
- 7.6: Perform Automated Vulnerability Scans of Externally-Exposed Enterprise Assets
- 7.7: Remediate Detected Vulnerabilities

---

## CIS-8: Audit Log Management
**Implementation Group:** IG1

Collect, alert, review, and retain audit logs of events that could help detect, understand,
or recover from an attack.

**Key Safeguards:**
- 8.1: Establish and Maintain an Audit Log Management Process
- 8.2: Collect Audit Logs
- 8.3: Ensure Adequate Audit Log Storage
- 8.4: Standardize Time Synchronization
- 8.5: Collect Detailed Audit Logs
- 8.6: Collect DNS Query Audit Logs
- 8.7: Collect URL Request Audit Logs
- 8.8: Collect Command-Line Audit Logs
- 8.9: Centralize Audit Logs
- 8.10: Retain Audit Logs
- 8.11: Conduct Audit Log Reviews
- 8.12: Collect Service Provider Logs

---

## CIS-9: Email and Web Browser Protections
**Implementation Group:** IG1

Improve protections and detections of threats from email and web vectors.

**Key Safeguards:**
- 9.1: Ensure Use of Only Fully Supported Browsers and Email Clients
- 9.2: Use DNS Filtering Services
- 9.3: Maintain and Enforce Network-Based URL Filters
- 9.4: Restrict Unnecessary or Unauthorized Browser and Email Client Extensions
- 9.5: Implement DMARC
- 9.6: Block Unnecessary File Types
- 9.7: Deploy and Maintain Email Server Anti-Malware Protections

---

## CIS-10: Malware Defenses
**Implementation Group:** IG1

Prevent or control the installation, spread, and execution of malicious applications, code, or scripts.

**Key Safeguards:**
- 10.1: Deploy and Maintain Anti-Malware Software
- 10.2: Configure Automatic Anti-Malware Signature Updates
- 10.3: Disable Autorun and Autoplay for Removable Media
- 10.4: Configure Automatic Anti-Malware Scanning of Removable Media
- 10.5: Enable Anti-Exploitation Features
- 10.6: Centrally Manage Anti-Malware Software
- 10.7: Use Behavior-Based Anti-Malware Software

---

## CIS-11: Data Recovery
**Implementation Group:** IG1

Establish and maintain data recovery practices sufficient to restore in-scope enterprise assets
to a pre-incident and trusted state.

**Key Safeguards:**
- 11.1: Establish and Maintain a Data Recovery Process
- 11.2: Perform Automated Backups
- 11.3: Protect Recovery Data
- 11.4: Establish and Maintain an Isolated Instance of Recovery Data
- 11.5: Test Data Recovery

---

## CIS-12: Network Infrastructure Management
**Implementation Group:** IG2

Establish and maintain the secure management of network infrastructure.

**Key Safeguards:**
- 12.1: Ensure Network Infrastructure is Up-to-Date
- 12.2: Establish and Maintain a Secure Network Architecture
- 12.3: Securely Manage Network Infrastructure
- 12.4: Establish and Maintain Architecture Diagram(s)
- 12.5: Centralize Network Authentication, Authorization, and Auditing (AAA)
- 12.6: Use of Secure Network Management and Communication Protocols
- 12.7: Ensure Remote Devices Utilize a VPN and are Connecting to an Enterprise's AAA Infrastructure
- 12.8: Establish and Maintain Dedicated Computing Resources for All Administrative Work

---

## CIS-13: Network Monitoring and Defense
**Implementation Group:** IG2

Operate processes and tooling to establish and maintain comprehensive network monitoring and defense.

**Key Safeguards:**
- 13.1: Centralize Security Event Alerting
- 13.2: Deploy a Host-Based Intrusion Detection Solution
- 13.3: Deploy a Network Intrusion Detection Solution
- 13.4: Perform Traffic Filtering Between Network Segments
- 13.5: Manage Access Control for Remote Assets
- 13.6: Collect Network Traffic Flow Logs
- 13.7: Deploy a Host-Based Intrusion Prevention Solution
- 13.8: Deploy a Network Intrusion Prevention Solution
- 13.9: Deploy Port-Level Access Control
- 13.10: Perform Application Layer Filtering
- 13.11: Tune Security Event Alerting Thresholds

---

## CIS-14: Security Awareness and Skills Training
**Implementation Group:** IG1

Establish and maintain a security awareness program to influence behavior among the workforce.

**Key Safeguards:**
- 14.1: Establish and Maintain a Security Awareness Program
- 14.2: Train Workforce Members to Recognize Social Engineering Attacks
- 14.3: Train Workforce Members on Authentication Best Practices
- 14.4: Train Workforce on Data Handling Best Practices
- 14.5: Train Workforce Members on Causes of Unintentional Data Exposure
- 14.6: Train Workforce Members on Recognizing and Reporting Security Incidents
- 14.7: Train Workforce on How to Identify and Report if Their Enterprise Assets are Missing Security Updates
- 14.8: Train Workforce on the Dangers of Connecting to and Transmitting Data Over Insecure Networks
- 14.9: Conduct Role-Specific Security Awareness and Skills Training

---

## CIS-15: Service Provider Management
**Implementation Group:** IG2

Develop a process to evaluate service providers who hold sensitive data or are responsible
for critical IT platforms or processes.

**Key Safeguards:**
- 15.1: Establish and Maintain an Inventory of Service Providers
- 15.2: Establish and Maintain a Service Provider Management Policy
- 15.3: Classify Service Providers
- 15.4: Ensure Service Provider Contracts Include Security Requirements
- 15.5: Assess Service Providers
- 15.6: Monitor Service Providers
- 15.7: Securely Decommission Service Providers

---

## CIS-16: Application Software Security
**Implementation Group:** IG2

Manage the security life cycle of in-house developed, hosted, or acquired software.

**Key Safeguards:**
- 16.1: Establish and Maintain a Secure Application Development Process
- 16.2: Establish and Maintain a Process to Accept and Address Software Vulnerabilities
- 16.3: Perform Root Cause Analysis on Security Vulnerabilities
- 16.4: Establish and Maintain a Process for Filtering New or Unrecognized Libraries
- 16.5: Use Up-to-Date and Trusted Third-Party Software Components
- 16.6: Establish and Maintain a Severity Rating System and Process for Application Vulnerabilities
- 16.7: Use Standard Hardening Configuration Templates for Application Infrastructure
- 16.8: Separate Production and Non-Production Systems
- 16.9: Train Developers in Application Security Concepts and Secure Coding
- 16.10: Apply Secure Design Principles in Application Architectures
- 16.11: Leverage Vetted Modules or Services for Application Security Components
- 16.12: Implement Code-Level Security Checks
- 16.13: Conduct Application Penetration Testing
- 16.14: Conduct Threat Modeling

---

## CIS-17: Incident Response Management
**Implementation Group:** IG2

Establish a program to develop and maintain an incident response capability.

**Key Safeguards:**
- 17.1: Designate Personnel to Manage Incident Handling
- 17.2: Establish and Maintain Contact Information for Reporting Security Incidents
- 17.3: Establish and Maintain an Enterprise Process for Reporting Incidents
- 17.4: Establish and Maintain an Incident Response Process
- 17.5: Assign Key Roles and Responsibilities for Incident Response
- 17.6: Define Mechanisms for Communicating During Incident Response
- 17.7: Conduct Routine Incident Response Exercises
- 17.8: Conduct Post-Incident Reviews
- 17.9: Establish and Maintain Security Incident Thresholds

---

## CIS-18: Penetration Testing
**Implementation Group:** IG2

Test the effectiveness and resiliency of enterprise assets through identifying and exploiting
weaknesses in controls (people, processes, and technology).

**Key Safeguards:**
- 18.1: Establish and Maintain a Penetration Testing Program
- 18.2: Perform Periodic External Penetration Tests
- 18.3: Remediate Penetration Test Findings
- 18.4: Validate Security Measures
- 18.5: Perform Periodic Internal Penetration Tests
