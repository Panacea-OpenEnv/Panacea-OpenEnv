import json
import random

def generate_sft_data(output_file="sft_drift_recovery.jsonl", num_examples=75):
    """
    Generates SFT (Supervised Fine-Tuning) data for the Oversight Agent.
    Teaches the model to gracefully recover from schema drift (ProgrammingError).
    Format: conversational JSONL (OpenAI / HuggingFace standard)
    """
    data = []
    
    tables = ['vitals', 'comorbidities', 'protocols', 'patients']
    
    for i in range(num_examples):
        drifted_table = random.choice(tables)
        new_table_name = f"{drifted_table}_v{random.randint(10, 99)}"
        
        # We simulate the interaction flow
        messages = [
            {"role": "system", "content": "You are the Project Panacea Oversight Agent. Parse claims and verify truths against the database. Handle errors gracefully."},
            {"role": "user", "content": f"New Claim Submitted. Please verify.\nClaim ID: {random.randint(100, 999)}, Patient: P{random.randint(1000, 9999)}"}
        ]
        
        # Initial Attempt (Fails due to drift)
        initial_query = f"SELECT * FROM {drifted_table} WHERE patient_id = 'P1002';"
        messages.append({"role": "assistant", "content": f"<thought>I need to check the patient's {drifted_table} to verify this claim. I will query the {drifted_table} table.</thought>\n<query>{initial_query}</query>"})
        
        # Environment returns ProgrammingError
        error_msg = f"ProgrammingError: relation \"{drifted_table}\" does not exist"
        messages.append({"role": "user", "content": f"Query Execution Failed:\n{error_msg}"})
        
        # Agent queries Information Schema to recover
        info_query = f"SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name LIKE '{drifted_table}%';"
        messages.append({"role": "assistant", "content": f"<thought>The '{drifted_table}' table is missing. Schema drift must have occurred. I should query the information_schema to find the new table name.</thought>\n<query>{info_query}</query>"})
        
        # Environment returns schema query results
        messages.append({"role": "user", "content": f"Query Result:\n[('{new_table_name}',)]"})
        
        # Agent successfully reissues query
        corrected_query = f"SELECT * FROM {new_table_name} WHERE patient_id = 'P1002';"
        messages.append({"role": "assistant", "content": f"<thought>Found the new table name: '{new_table_name}'. Reissuing my original query.</thought>\n<query>{corrected_query}</query>"})

        data.append({"messages": messages})
        
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
            
    print(f"Generated {num_examples} SFT examples in {output_file}")

if __name__ == "__main__":
    generate_sft_data()
