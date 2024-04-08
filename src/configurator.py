import sys
from ast import literal_eval

# Loop through all command-line arguments starting from index 1 (ignore script name at index 0)
for argument in sys.argv[1:]:
    if '=' not in argument:
        # If no '=' in the argument, assume it's a config file path
        assert not argument.startswith('--'), "Invalid argument format for config file"
        config_filepath = argument
        print(f"Loading config from file: {config_filepath}")
        try:
            with open(config_filepath) as config_file:
                print(config_file.read())  # Display the config file contents
            exec(open(config_filepath).read())  # Execute the config file
        except FileNotFoundError:
            print(f"Error: Config file {config_filepath} not found.")
    else:
        # Assume it's a --key=value argument for configuration override
        assert argument.startswith('--'), "Invalid argument format for key=value pair"
        key, value = argument.split('=', 1)
        key = key[2:]  # Remove the '--' prefix from the key
        
        if key in globals():
            try:
                # Attempt to parse the value into a literal (bool, int, list, etc.)
                parsed_value = literal_eval(value)
            except (SyntaxError, ValueError):
                # If parsing fails, treat it as a string
                parsed_value = value
            
            # Ensure that the types match between the original global variable and the new value
            if isinstance(parsed_value, type(globals()[key])):
                print(f"Overriding: {key} = {parsed_value}")
                globals()[key] = parsed_value  # Override the global variable
            else:
                print(f"Type mismatch for {key}: expected {type(globals()[key])}, got {type(parsed_value)}")
        else:
            raise ValueError(f"Unknown configuration key: {key}")
