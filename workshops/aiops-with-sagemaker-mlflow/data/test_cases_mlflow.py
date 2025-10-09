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

test_case_1 = {
    "user_prompt":"Can you help me solve the ticket: TICKET-001?",
    "error_name": "Connection Timeout",
    "solution": [
            "1. Check network connectivity between client and server",
            "2. Verify if the server is running and accessible",
            "3. Increase the connection timeout settings",
            "4. Check for firewall rules blocking the connection",
            "5. Monitor network latency and bandwidth"
    ],
    "expected_tools" : ["log_identifier", "information_retriever"],
    "expected_arguments": [{'ticket_id': 'TICKET-001'}, {'error_type': 'Connection Timeout'}]
}

test_case_2 = {
    "user_prompt": "Can you help me with this ticket id: TICKET-002?",
    "error_name": "Database Authentication Failed",
    "solution": [
            "1. Verify database credentials are correct",
            "2. Check if the database user account is locked",
            "3. Ensure database service is running",
            "4. Review database access permissions",
            "5. Check for recent password changes"
    ],
    "expected_tools" : ["log_identifier", "information_retriever"],
    "expected_arguments": [{'ticket_id': 'TICKET-002'}, {'error_type': 'Database Authentication Failed'}]
}

test_case_3 = {
    "user_prompt": "I got a ticket: TICKET-003, can you help me with this?",
    "error_name": "Disk Space Full",
    "solution":  [
            "1. Remove temporary and unnecessary files",
            "2. Implement log rotation",
            "3. Archive old data",
            "4. Expand disk space",
            "5. Monitor disk usage trends"
    ],
    "expected_tools" : ["log_identifier", "information_retriever"],
    "expected_arguments": [{'ticket_id': 'TICKET-006'}, {'error_type': 'Disk Space Full'}]
}

test_case_4 = {
    "user_prompt": "Help me solve the ticekt: TICKET-004",
    "error_name": "Permission Denied",
    "solution": [
            "1. Review user access rights",
            "2. Check file and directory permissions",
            "3. Verify group memberships",
            "4. Update security policies",
            "5. Audit access control lists"
    ],
    "expected_tools" : ["log_identifier", "information_retriever", "notify_team"],
    "expected_arguments": [{'ticket_id': 'TICKET-008'}, {'error_type': 'Permission Denied'}]
}
