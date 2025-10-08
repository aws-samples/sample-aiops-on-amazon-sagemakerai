# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

log_data = [
    {"id": "TICKET-001", "error_name": "Connection Timeout"},
    {"id": "TICKET-002", "error_name": "Database Authentication Failed"},
    {"id": "TICKET-003", "error_name": "Memory Overflow"},
    {"id": "TICKET-004", "error_name": "API Rate Limit Exceeded"},
    {"id": "TICKET-005", "error_name": "Invalid SSL Certificate"},
    {"id": "TICKET-006", "error_name": "Disk Space Full"},
    {"id": "TICKET-007", "error_name": "Network Connectivity Lost"},
    {"id": "TICKET-008", "error_name": "Permission Denied"},
    {"id": "TICKET-009", "error_name": "Service Unavailable"},
    {"id": "TICKET-010", "error_name": "Configuration File Missing"}
]

log_data_set = set([i['id'] for i in log_data])
