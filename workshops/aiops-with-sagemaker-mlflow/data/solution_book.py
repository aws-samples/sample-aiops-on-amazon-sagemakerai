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

solution_book = {
    "Connection Timeout": """1. Check network connectivity between client and server
2. Verify if the server is running and accessible
3. Increase the connection timeout settings
4. Check for firewall rules blocking the connection
5. Monitor network latency and bandwidth
    """,
    
    "Database Authentication Failed": """1. Verify database credentials are correct
2. Check if the database user account is locked
3. Ensure database service is running
4. Review database access permissions
5. Check for recent password changes
    """,
    
    "Memory Overflow": """1. Analyze application memory usage patterns
2. Increase available memory or swap space
3. Look for memory leaks in the application
4. Optimize database queries and caching
5. Consider implementing memory pooling
    """,
    
    "Invalid SSL Certificate": """1. Check certificate expiration date
2. Verify certificate chain is complete
3. Ensure certificate matches domain name
4. Update SSL certificate if expired
5. Check certificate authority validity
    """,
    
    "Disk Space Full": """1. Remove temporary and unnecessary files
2. Implement log rotation
3. Archive old data
4. Expand disk space
5. Monitor disk usage trends
    """,
    
    "Network Connectivity Lost": """1. Check physical network connections
2. Verify router and switch status
3. Test DNS resolution
4. Check for network interface errors
5. Monitor network traffic patterns
    """,
    
    "Permission Denied": """1. Review user access rights
2. Check file and directory permissions
3. Verify group memberships
4. Update security policies
5. Audit access control lists
    """,
    
    "Service Unavailable": """1. Check service status and logs
2. Restart the service
3. Verify dependencies are running
4. Check system source documents. Monitor service health metrics
    """,
    
    "Configuration File Missing": """1. Locate backup configuration files
2. Restore from version control
3. Create new configuration file with default settings
4. Check file path and permissions
5. Verify application deployment process
    """
}
