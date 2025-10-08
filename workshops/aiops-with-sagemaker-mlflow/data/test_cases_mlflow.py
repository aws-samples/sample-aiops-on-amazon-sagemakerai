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

TEST_CASES = [
    {
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
},
{
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
},
{
    "user_prompt": "I got a ticket: TICKET-003, can you help me with this?",
    "error_name": "Memory Overflow",
    "solution":  [
    "1. Analyze application memory usage patterns",
    "2. Increase available memory or swap space",
    "3. Look for memory leaks in the application",
    "4. Optimize database queries and caching",
    "5. Consider implementing memory pooling"
    ],
    "expected_tools" : ["log_identifier", "information_retriever"],
    "expected_arguments": [{'ticket_id': 'TICKET-003'}, {'error_type': 'Disk Space Full'}]
},
{
    "user_prompt": "Help me solve the ticekt: TICKET-004",
    "error_name": "API Rate Limit Exceeded",
    "solution": [
        "1. Implement request throttling",
        "2. Use caching to reduce API calls",
        "3. Review API usage patterns",
        "4. Contact service provider for limit increase",
        "5. Optimize API call frequency"
    ],
    "expected_tools" : ["log_identifier", "information_retriever", "notify_team"],
    "expected_arguments": [{'ticket_id': 'TICKET-004'}, {'error_type': 'Permission Denied'}]
},
{
    "user_prompt": "How do I solve this ticket: TICKET-005?",
    "error_name": "Invalid SSL Certificate",
    "solution": [
        "1. Check certificate expiration date",
        "2. Verify certificate chain is complete",
        "3. Ensure certificate matches domain name",
        "4. Update SSL certificate if expired",
        "5. Check certificate authority validity"
    ],
    "expected_tools" : ["log_identifier", "information_retriever"],
    "expected_arguments": [{'ticket_id': 'TICKET-005'}, {'error_type': 'Invalid SSL Certificate'}]
},
{
    "user_prompt": "I need help with this ticket: TICKET-006",
    "error_name": "Disk Space Full",
    "solution": [
        "1. Remove temporary and unnecessary files",
        "2. Implement log rotation",
        "3. Archive old data",
        "4. Expand disk space",
        "5. Monitor disk usage trends"
    ],
    "expected_tools" : ["log_identifier", "information_retriever"],
    "expected_arguments": [{'ticket_id': 'TICKET-006'}, {'error_type': 'Disk Space Full'}]
},
{
    "user_prompt": "What do I do to resolve TICKET-007",
    "error_name": "Network Connectivity Lost",
    "solution": [
        "1. Check physical network connections",
        "2. Verify router and switch status",
        "3. Test DNS resolution",
        "4. Check for network interface errors",
        "5. Monitor network traffic patterns"
    ],
    "expected_tools" : ["log_identifier", "information_retriever"],
    "expected_arguments": [{'ticket_id': 'TICKET-007'}, {'error_type': 'Network Connectivity Lost'}]
},
{
    "user_prompt": "I need help with this ticket: TICKET-008",
    "error_name": "Permission Denied",
    "solution": [
        "1. Review user access rights",
        "2. Check file and directory permissions",
        "3. Verify group memberships",
        "4. Update security policies",
        "5. Audit access control lists"
    ],
    "expected_tools" : ["log_identifier", "information_retriever"],
    "expected_arguments": [{'ticket_id': 'TICKET-008'}, {'error_type': 'Permission Denied'}]
},
{
    "user_prompt": "How should I fix TICKET-009?",
    "error_name": "Service Unavailable",
    "solution": [
        "1. Check service status and logs",
        "2. Restart the service",
        "3. Verify dependencies are running",
        "4. Check system reyour source documents. Monitor service health metrics"
    ],
    "expected_tools" : ["log_identifier", "information_retriever"],
    "expected_arguments": [{'ticket_id': 'TICKET-009'}, {'error_type': 'Service Unavailable'}]
},
 {
    "user_prompt": "I was just assigned TICKET-010, what do I do?",
    "error_name": "Configuration File Missing",
    "solution": [
        "1. Locate backup configuration files",
        "2. Restore from version control",
        "3. Create new configuration file with default settings",
        "4. Check file path and permissions",
        "5. Verify application deployment process"
    ],
    "expected_tools" : ["log_identifier", "information_retriever"],
    "expected_arguments": [{'ticket_id': 'TICKET-010'}, {'error_type': 'Configuration File Missing'}]
},
{
    "user_prompt": "I need help with this ticket",
    "error_name": "",
    "solution": ["I apologize, but I can only provide ETL resolution steps for specific ticket IDs. Please provide a ticket ID."],
    "expected_tools" : [],
    "expected_arguments": []
}

]

# Convenience functions for accessing test cases
def get_test_case(case_id):
    """Get a specific test case by ID"""
    return next((case for case in TEST_CASES if case["id"] == case_id), None)

def get_all_test_cases():
    """Get all test cases"""
    return TEST_CASES

def get_test_cases_by_error_type(error_type):
    """Get test cases filtered by error type"""
    return [case for case in TEST_CASES if case["error_name"] == error_type]