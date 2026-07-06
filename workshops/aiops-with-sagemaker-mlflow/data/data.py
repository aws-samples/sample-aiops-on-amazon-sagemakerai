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

patient_records = [
    {"id": "PT-001", "symptoms": "chest pain, shortness of breath, sweating", "age": 62, "severity": "high"},
    {"id": "PT-002", "symptoms": "persistent cough, mild fever, fatigue", "age": 35, "severity": "low"},
    {"id": "PT-003", "symptoms": "severe headache, blurred vision, nausea", "age": 48, "severity": "high"},
    {"id": "PT-004", "symptoms": "joint pain, swelling, morning stiffness", "age": 55, "severity": "medium"},
    {"id": "PT-005", "symptoms": "abdominal pain, bloating, irregular bowel movements", "age": 40, "severity": "medium"},
    {"id": "PT-006", "symptoms": "dizziness, rapid heartbeat, fainting", "age": 70, "severity": "high"},
    {"id": "PT-007", "symptoms": "sore throat, runny nose, mild headache", "age": 28, "severity": "low"},
    {"id": "PT-008", "symptoms": "lower back pain, numbness in legs, difficulty walking", "age": 52, "severity": "high"},
    {"id": "PT-009", "symptoms": "skin rash, itching, swelling around eyes", "age": 22, "severity": "medium"},
    {"id": "PT-010", "symptoms": "frequent urination, excessive thirst, unexplained weight loss", "age": 45, "severity": "medium"},
    {"id": "PT-011", "symptoms": "high fever, severe body aches, chills", "age": 38, "severity": "high"},
    {"id": "PT-012", "symptoms": "persistent heartburn, difficulty swallowing, chest discomfort after eating", "age": 56, "severity": "medium"},
    {"id": "PT-013", "symptoms": "sudden confusion, slurred speech, weakness on one side", "age": 72, "severity": "high"},
    {"id": "PT-014", "symptoms": "chronic fatigue, pale skin, shortness of breath on exertion", "age": 30, "severity": "medium"},
    {"id": "PT-015", "symptoms": "ankle swelling, weight gain, shortness of breath when lying down", "age": 68, "severity": "high"},
]

patient_ids = set([p['id'] for p in patient_records])
