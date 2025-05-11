def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return data

def save_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(data)

def log_message(message):
    print(f"[LOG] {message}")

def debug_info(variable_name, value):
    print(f"[DEBUG] {variable_name}: {value}")