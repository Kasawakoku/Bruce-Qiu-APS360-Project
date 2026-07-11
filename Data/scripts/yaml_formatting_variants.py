# automatically format aircraft yaml file


import re

def format_aircraft_list(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    out_lines = []
    last_proper_indent = 0
    
    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue
            
        stripped = line.strip()
        
        # Your rule: Starts with '-', ends with ':', or is a comment
        is_proper = stripped.startswith('-') or stripped.endswith(':') or stripped.startswith('#')
        
        if is_proper:
            out_lines.append(line.rstrip() + '\n')
            # Track the indentation of this proper line
            last_proper_indent = len(line) - len(line.lstrip(' '))
        else:
            # It's an improperly formatted model line.
            # Regex removes the comma and any trailing numbers (e.g., ",47" or ",1")
            cleaned_model = re.sub(r',\s*\d+$', '', stripped).strip()
            
            # If the last proper line was a Variant (4 spaces), we indent the Model by +2 (6 spaces).
            # If the last proper line was already a Model (6 spaces), we keep it at 6 spaces.
            indent_to_use = last_proper_indent + 2 if last_proper_indent < 6 else last_proper_indent
            
            formatted_line = (" " * indent_to_use) + "- " + cleaned_model + "\n"
            out_lines.append(formatted_line)
            
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)
    print(f"Cleaned and formatted list saved to {output_filepath}")

# Run it (Make sure to save your messy block in 'raw_aircraft.txt')
if __name__ == "__main__":
    format_aircraft_list("../class_definitions/raw_aircraft.txt", "../class_definitions/aircraft_hierarchy.yaml")