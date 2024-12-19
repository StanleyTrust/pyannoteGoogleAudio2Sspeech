import os

# Указываем путь к директории
directory = "."

# Проходим по всем файлам в директории
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        file_path = os.path.join(directory, filename)
        os.remove(file_path)
        print(f"Файл {filename} успішно видалено.")