#!/usr/bin/env bash
# Smoke-test one full episode against the live HF Space.
URL="${1:-https://anilpaliwal132-panacea.hf.space}"

echo "=== RESET ==="
curl -s -X POST -H "Content-Type: application/json" -d '{}' "$URL/reset" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); o=d['observation']; print('patient:', o['claim']['patient_id']); print('dept:', o['claim']['department']); print('amount: \$%.2f' % o['claim']['claimed_amount'])"

echo
echo "=== STEP 1: TOOL_REGISTRY ==="
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"action":{"type":"tool_call","tool_name":"TOOL_REGISTRY","verdict":"REJECTED","reasoning":""}}' \
  "$URL/step" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('reward:', d['reward']); print('evidence:', d['observation']['last_tool_evidence'])"

echo
echo "=== STEP 2: TOOL_BILLING ==="
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"action":{"type":"tool_call","tool_name":"TOOL_BILLING","verdict":"REJECTED","reasoning":""}}' \
  "$URL/step" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('reward:', d['reward']); print('evidence:', d['observation']['last_tool_evidence'])"

echo
echo "=== STEP 3: VERDICT ==="
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"action":{"type":"verdict","verdict":"REJECTED","reasoning":"patient ID returned NO RECORD in registry"}}' \
  "$URL/step" \
  | python3 -c "import json,sys,pprint; d=json.load(sys.stdin); print('total reward:', d['reward']); print('done:', d['done']); pprint.pprint(d['observation'].get('metadata', {}))"
