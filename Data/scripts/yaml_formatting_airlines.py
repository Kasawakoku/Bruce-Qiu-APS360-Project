import re

def format_airline_list(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    out_lines = []
    
    for line in lines:
        # Preserve blank lines for readability
        if not line.strip():
            out_lines.append(line)
            continue
            
        stripped = line.strip()
        
        # A properly formatted line either ends with ':' (The True Airline),
        # starts with '-' (An already formatted alias), or is a '#' comment.
        is_proper = stripped.endswith(':') or stripped.startswith('-') or stripped.startswith('#')
        
        if is_proper:
            out_lines.append(line.rstrip() + '\n')
        else:
            # It's an improperly formatted alias.
            # Regex removes the comma and any trailing numbers (e.g., ",19")
            cleaned_alias = re.sub(r',\s*\d+$', '', stripped).strip()
            
            # Indent with 2 spaces and a dash
            formatted_line = "  - " + cleaned_alias + "\n"
            out_lines.append(formatted_line)
            
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)
    print(f"Cleaned airline list saved to {output_filepath}")

# Run it
if __name__ == "__main__":
    format_airline_list("../class_definitions/raw_airlines.txt", "../class_definitions/airline_mapping.yaml")