#!/bin/bash

# OpenEnv Submission Validation Script
# Usage: ./validate-submission.sh <SPACE_URL>

if [ -z "$1" ]; then
    echo "Usage: ./validate-submission.sh <SPACE_URL>"
    echo "Example: ./validate-submission.sh https://atharwa1602-my-openenv.hf.space"
    exit 1
fi

URL=$1
# Remove trailing slash if present
URL=${URL%/}

echo "============================================="
echo "🔍 Validating OpenEnv Space: $URL"
echo "============================================="

# 1. Health Check
echo -n "[1/4] Checking /health endpoint... "
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$URL/health")
if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "✅ PASS (200 OK)"
else
    echo "❌ FAIL (HTTP $HTTP_STATUS)"
    exit 1
fi

# 2. Reset Endpoint
echo -n "[2/4] Checking POST /reset endpoint... "
HTTP_STATUS=$(curl -s -X POST -o /tmp/openenv_resp.json -w "%{http_code}" "$URL/reset")
if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "✅ PASS (200 OK)"
    cat /tmp/openenv_resp.json | grep -q 'ticket_id'
    if [ $? -eq 0 ]; then
        echo "      └─ Valid Observation returned."
    else
        echo "      └─ ⚠️ Warning: JSON did not contain expected observation structure."
    fi
else
    echo "❌ FAIL (HTTP $HTTP_STATUS)"
    exit 1
fi

# 3. State Endpoint
echo -n "[3/4] Checking GET /state endpoint... "
HTTP_STATUS=$(curl -s -X GET -o /tmp/openenv_resp.json -w "%{http_code}" "$URL/state")
if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "✅ PASS (200 OK)"
else
    echo "❌ FAIL (HTTP $HTTP_STATUS)"
    exit 1
fi

# 4. Step Endpoint
echo -n "[4/4] Checking POST /step endpoint... "
HTTP_STATUS=$(curl -X POST -s -o /tmp/openenv_resp.json -w "%{http_code}" "$URL/step" \
    -H "Content-Type: application/json" \
    -d '{"category":"billing","priority":"medium","response_snippet":"invoice charge"}')
if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "✅ PASS (200 OK)"
    cat /tmp/openenv_resp.json | grep -q 'reward'
    if [ $? -eq 0 ]; then
        echo "      └─ Valid StepResult (with reward) returned."
    else
        echo "      └─ ⚠️ Warning: JSON did not contain 'reward' key."
    fi
else
    echo "❌ FAIL (HTTP $HTTP_STATUS)"
    exit 1
fi

echo "============================================="
echo "🎉 ALL VALIDATION CHECKS PASSED!"
echo "Your space is fully compliant with OpenEnv endpoints."
echo "============================================="
