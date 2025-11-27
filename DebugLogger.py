'''
import datetime
import os


class DebugLogger:
    _file = None
    _log_file = "debug-python.txt"

    @classmethod
    def init(cls):
        try:
            cls._file = open(cls._log_file, 'w', encoding='utf-8')
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cls._file.write(f"=== DEBUG LOG PYTHON - {timestamp} ===\n")
            cls._file.flush()
        except Exception as e:
            print(f"Erro ao inicializar log: {e}")

    @classmethod
    def log(cls, message):
        if cls._file:
            cls._file.write(f"{message}\n")
            cls._file.flush()  # Force write immediately
        # Também mostra no console (opcional)
        print(message)

    @classmethod
    def close(cls):
        if cls._file:
            cls._file.write("=== FIM DO LOG ===\n")
            cls._file.close()
'''

import datetime
import json

class DebugLogger:
    _file = None
    _log_file = "debug-python.txt"

    @classmethod
    def init(cls):
        try:
            cls._file = open(cls._log_file, 'w', encoding='utf-8')
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cls._file.write(f"=== DEBUG LOG PYTHON - {timestamp} ===\n")
            cls._file.flush()
        except Exception as e:
            print(f"Erro ao inicializar log: {e}")

    @classmethod
    def log(cls, message):
        if cls._file:
            cls._file.write(f"{message}\n")
            cls._file.flush()
        print(message)

    @classmethod
    def log_json(cls, phase, event, **fields):
        try:
            payload = {"phase": phase, "event": event}
            payload.update(fields)
            line = json.dumps(payload, ensure_ascii=False)
        except Exception:
            # fallback simples
            parts = [f"\"{k}\":{json.dumps(v, ensure_ascii=False)}" for k, v in fields.items()]
            line = "{" + f"\"phase\":\"{phase}\",\"event\":\"{event}\"," + ",".join(parts) + "}"
        cls.log(line)

    @classmethod
    def close(cls):
        if cls._file:
            cls._file.write("=== FIM DO LOG ===\n")
            cls._file.close()
